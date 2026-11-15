"""RT-DETR-L tile-detector training — the Kaggle run as a script, plus declared variants.

``models/rtdetr_fold0_best.pt`` was trained in a Kaggle notebook that is not in the
repository. Its recipe is not lost, though: ultralytics stores the full ``train_args``
in every checkpoint, and ``python -m training.inspect_checkpoint`` reads them without
unpickling anything. The ``baseline`` profile below is that recipe, value for value.
Only two things differ, on purpose:

* ``device`` — the original ran on two Kaggle GPUs (``'0,1'``); here it is the single
  local GPU (``0``), or CPU when there is none. The total batch size is unchanged.
* ``data`` / ``project`` / ``name`` — local paths instead of ``/kaggle/working``.

Every other profile is the baseline with a small, named set of changes, so a result
can always be read as "baseline, except …". A profile is an *experiment*, not an
upgrade: whether it is better is decided by ``training.eval_detector`` on a held-out
external game, never by this script and never on the provided match. One-off changes
go through ``--set key=value`` and are recorded the same way.

Each run writes ``<run dir>/train_meta.json``: the resolved arguments, the profile and
overrides, package versions, the device, the git commit, and the tile dataset's
``provenance.json`` (dataset identity, licence, split). That file is what ties a
checkpoint to the data it saw — keep it with the weights.

    python -m training.train_rtdetr --data data/external/tiles_fold0/data.yaml --name fold_0
    python -m training.train_rtdetr --data … --name fold_0_long --profile long
    python -m training.train_rtdetr --data … --name probe --set epochs=1 --set fraction=0.05
    python -m training.train_rtdetr --data … --name fold_0 --resume
    python -m training.train_rtdetr --data … --name fold_0 --dry-run     # print, do nothing

ultralytics and torch are imported only when a run actually starts, so ``--dry-run``
and the unit tests need neither.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from training.compliance import REPO_ROOT, assert_external

# The recipe recovered from models/rtdetr_fold0_best.pt (ultralytics 8.4.212,
# saved 2026-11-14). Regenerate with:  python -m training.inspect_checkpoint <pt>
BASELINE = {
    "model": "rtdetr-l.pt",
    "epochs": 12,
    "batch": 16,
    "imgsz": 320,
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "cos_lr": True,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "amp": True,
    "seed": 0,
    "deterministic": True,
    "pretrained": True,
    "patience": 100,
    "workers": 2,
    "close_mosaic": 10,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "erasing": 0.4,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 64,
    "plots": True,
    "val": True,
}

# Declared variants: each is BASELINE plus these overrides. The "why" is part of the
# record and is copied into train_meta.json.
PROFILES = {
    "baseline": {
        "why": "the recovered Kaggle recipe, unchanged (reproduction run)",
        "overrides": {},
    },
    "baseline_bs8": {
        "why": "single-GPU analogue of the original two-GPU run: with device='0,1'"
               " and batch=16 each GPU saw 8 images, so BatchNorm statistics came"
               " from batches of 8; batch=8 on one GPU matches that (nbs=64 keeps the"
               " effective optimisation batch). Run next to `baseline` to learn which"
               " reproduces the recorded metrics more closely",
        "overrides": {"batch": 8},
    },
    "no_bias_warmup": {
        "why": "ultralytics warms the bias parameters up from warmup_bias_lr=0.1,"
               " which under AdamW with lr0=1e-4 is 1000x the base rate; the recorded"
               " curve shows the bias group at lr 0.067 and 0.033 in epochs 1-2 while"
               " mAP50 sat at 0.11 / 0.10. Starting the bias warm-up at 0 removes"
               " that spike — one changed value",
        "overrides": {"warmup_bias_lr": 0.0},
    },
    "no_mosaic": {
        "why": "with epochs=12 and close_mosaic=10 mosaic was on for epochs 1-2 only,"
               " the two weakest epochs of the recorded run (confounded with the"
               " warm-up). Mosaic shrinks an already tiny ball; this run turns it off"
               " entirely to see whether those epochs were being wasted",
        "overrides": {"mosaic": 0.0},
    },
    "upsample2x": {
        "why": "feed each 320 px tile to the network at 640 px: RT-DETR's finest"
               " feature map is stride 8, so a ~6 px ball covers less than one cell"
               " at native scale and about one and a half at 2x. Same tiles, same"
               " labels, same schedule — only the input size changes. Costs roughly"
               " 4x the pixels per tile (lower the batch if memory runs out), and"
               " inference must then also run at imgsz=640 (eval_detector and"
               " smoke_inference read the size from the checkpoint)",
        "overrides": {"imgsz": 640},
    },
    "long": {
        "why": "same recipe on a 3x schedule: the 12-epoch run spent 3 epochs warming"
               " up and closed mosaic for the last 10, so mosaic was active for only"
               " 2 epochs; 36 epochs with the same close_mosaic=10 keeps it on for"
               " 26. The recorded validation mAP was flat from epoch 5, so expect a"
               " small gain at best — this run tests schedule length, nothing else",
        "overrides": {"epochs": 36, "patience": 12},
    },
    "finetune": {
        "why": "continue from an existing ball checkpoint (pass --weights) on a new"
               " or larger tile set, at a third of the learning rate and with a"
               " short warm-up — for adding games or hard negatives without"
               " relearning from the COCO weights",
        "overrides": {"epochs": 12, "lr0": 3e-5, "warmup_epochs": 1.0, "patience": 6},
    },
}

RUNS_DIR = REPO_ROOT / "runs" / "rtdetr"


def parse_override(text: str):
    """``key=value`` -> ``(key, value)`` with the value read as JSON when it parses.

    ``epochs=36`` -> int, ``lr0=3e-5`` -> float, ``cos_lr=false`` -> bool,
    ``optimizer=AdamW`` -> str.
    """
    if "=" not in text:
        raise ValueError(f"override must be key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"override has an empty key: {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.strip().lower()
        value = {"true": True, "false": False, "none": None}.get(lowered, raw.strip())
    return key, value


def resolve_args(profile: str = "baseline", overrides: dict | None = None,
                 weights: str | None = None) -> dict:
    """Training kwargs for a profile: BASELINE + the profile's overrides + ``--set``.

    ``weights`` replaces the starting model (the ``finetune`` profile requires it).
    ``data``, ``project``, ``name`` and ``device`` are added by :func:`plan_run`.
    """
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    if profile == "finetune" and not weights:
        raise ValueError("the finetune profile needs --weights <checkpoint.pt>")
    args = dict(BASELINE)
    args.update(PROFILES[profile]["overrides"])
    args.update(overrides or {})
    if weights:
        args["model"] = str(weights)
    reserved = {"data", "project", "name", "device", "resume", "exist_ok"} & set(
        overrides or {})
    if reserved:
        raise ValueError(f"{sorted(reserved)} are set by flags, not by --set")
    return args


def _yaml_scalar(text: str) -> str:
    """A YAML scalar without its quotes or trailing ``# comment``."""
    text = text.strip()
    if text[:1] in ("'", '"'):
        quote = text[0]
        end = text.find(quote, 1)
        if end == -1:
            raise ValueError(f"unterminated quoted value: {text!r}")
        inner = text[1:end]
        return json.loads(text[:end + 1]) if quote == '"' else inner.replace("''", "'")
    hash_at = text.find(" #")
    return (text[:hash_at] if hash_at != -1 else text).strip()


def dataset_paths(data_yaml) -> list[Path]:
    """The YAML plus every folder it names (``path``, ``train``, ``val``, ``test``).

    A small parser for the flat YAML ultralytics uses, so the provided-data guard can
    look inside the YAML without PyYAML (not installed with ultralytics 8.4.212). It
    handles quotes, inline ``# comments`` and ``- item`` lists, and fails closed:
    a structure it does not understand under one of those keys raises instead of
    being skipped.
    """
    data_yaml = Path(data_yaml).expanduser().resolve()
    fields: dict[str, list[str]] = {}
    current = None
    for number, line in enumerate(data_yaml.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            if ":" not in line:
                raise ValueError(f"{data_yaml}:{number}: cannot parse {line!r}")
            key, value = line.split(":", 1)
            current = key.strip()
            value = _yaml_scalar(value)
            fields[current] = [value] if value else []
            continue
        if current in ("path", "train", "val", "test"):
            if stripped.startswith("- "):
                fields[current].append(_yaml_scalar(stripped[2:]))
            else:
                raise ValueError(f"{data_yaml}:{number}: unsupported structure under"
                                 f" {current!r}: {line!r}")
    root_values = fields.get("path") or []
    root = Path(root_values[0]).expanduser() if root_values else data_yaml.parent
    if not root.is_absolute():
        root = data_yaml.parent / root
    paths = [data_yaml, root]
    for key in ("train", "val", "test"):
        for value in fields.get(key, []):
            entry = Path(value).expanduser()
            paths.append(entry if entry.is_absolute() else root / entry)
    return paths


def plan_run(data_yaml, name: str, profile: str = "baseline",
             overrides: dict | None = None, weights: str | None = None,
             device=None, project=None) -> dict:
    """Everything a run needs, validated, without importing torch or ultralytics.

    Refuses provided-match material anywhere in the dataset YAML, and a ``weights``
    path that does not exist.
    """
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    assert_external(*dataset_paths(data_yaml), what="training dataset")
    if weights and not Path(weights).expanduser().is_file():
        raise FileNotFoundError(f"weights not found: {weights}")

    args = resolve_args(profile, overrides, weights)
    project = Path(project).expanduser().resolve() if project else RUNS_DIR
    args.update({"data": str(data_yaml), "project": str(project), "name": name,
                 "exist_ok": True})
    if device is not None:
        args["device"] = device
    return {"train_args": args, "profile": profile,
            "profile_why": PROFILES[profile]["why"],
            "overrides": dict(overrides or {}), "run_dir": str(project / name),
            "data_yaml": str(data_yaml)}


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tile_provenance(data_yaml: str):
    candidate = Path(data_yaml).parent / "provenance.json"
    if candidate.is_file():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"error": f"unreadable {candidate}"}
    return None


def _results_summary(run_dir: Path):
    """Best and last epoch of ultralytics' results.csv, or None if it is not there."""
    results = run_dir / "results.csv"
    if not results.is_file():
        return None
    import pandas as pd

    table = pd.read_csv(results)
    table.columns = [c.strip() for c in table.columns]
    key = "metrics/mAP50-95(B)"
    if key not in table.columns or table.empty:
        return {"epochs_logged": int(len(table))}
    best = table.loc[table[key].idxmax()]
    keep = [c for c in ("epoch", "metrics/precision(B)", "metrics/recall(B)",
                        "metrics/mAP50(B)", key) if c in table.columns]
    return {"epochs_logged": int(len(table)),
            "best": {c: float(best[c]) for c in keep},
            "last": {c: float(table.iloc[-1][c]) for c in keep}}


def write_meta(plan: dict, started: str, finished: str | None, status: str,
               environment: dict | None = None) -> Path:
    """Write ``<run dir>/train_meta.json`` and return its path."""
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": status,
        "started_utc": started,
        "finished_utc": finished,
        "git_sha": _git_sha(),
        "profile": plan["profile"],
        "profile_why": plan["profile_why"],
        "overrides": plan["overrides"],
        "train_args": plan["train_args"],
        "environment": environment or {},
        "tile_dataset_provenance": _tile_provenance(plan["data_yaml"]),
        "results": _results_summary(run_dir),
        "best_weights": str(run_dir / "weights" / "best.pt"),
    }
    target = run_dir / "train_meta.json"
    target.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return target


def _environment() -> dict:
    """Package versions and the device the run will use."""
    import torch
    import ultralytics

    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "ultralytics": ultralytics.__version__,
           "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["gpu_memory_gb"] = round(props.total_memory / 2 ** 30, 1)
        env["cuda"] = torch.version.cuda
    return env


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(plan: dict, resume: bool = False):
    """Start (or resume) the training run described by :func:`plan_run`."""
    import torch
    from ultralytics import RTDETR

    args = dict(plan["train_args"])
    if "device" not in args:
        args["device"] = 0 if torch.cuda.is_available() else "cpu"
        plan["train_args"]["device"] = args["device"]
    environment = _environment()
    started = _now()
    write_meta(plan, started, None, "running", environment)

    last = Path(plan["run_dir"]) / "weights" / "last.pt"
    try:
        if resume:
            if not last.is_file():
                raise FileNotFoundError(f"nothing to resume: {last} does not exist")
            model = RTDETR(str(last))
            results = model.train(resume=True)
        else:
            model = RTDETR(args.pop("model"))
            results = model.train(**args)
    except BaseException:
        write_meta(plan, started, _now(), "failed", environment)
        raise
    write_meta(plan, started, _now(), "finished", environment)
    return results


def main(argv: list[str] | None = None) -> int:
    """Train the RT-DETR-L ball tile detector on an external tile dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.train_rtdetr",
        description="Train RT-DETR-L on a tile dataset built by training.tiles.")
    parser.add_argument("--data", required=True, help="dataset YAML (data.yaml)")
    parser.add_argument("--name", required=True, help="run name, e.g. fold_0")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="baseline")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="one-off override (repeatable)")
    parser.add_argument("--weights", default=None,
                        help="start from this checkpoint instead of rtdetr-l.pt")
    parser.add_argument("--device", default=None,
                        help="0, cpu, ... (default: GPU 0 if available)")
    parser.add_argument("--project", default=None,
                        help=f"runs folder (default {RUNS_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run of this name from weights/last.pt")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved plan as JSON and exit")
    args = parser.parse_args(argv)

    overrides = dict(parse_override(text) for text in args.overrides)
    device = args.device
    if device is not None and device.isdigit():
        device = int(device)
    plan = plan_run(args.data, args.name, args.profile, overrides, args.weights,
                    device, args.project)

    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return 0

    print(f"profile {plan['profile']}: {plan['profile_why']}")
    if plan["overrides"]:
        print(f"overrides: {plan['overrides']}")
    run(plan, resume=args.resume)
    meta = json.loads((Path(plan["run_dir"]) / "train_meta.json").read_text())
    print(f"run dir     : {plan['run_dir']}")
    print(f"best weights: {meta['best_weights']}")
    if meta.get("results"):
        print(f"results     : {json.dumps(meta['results'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
