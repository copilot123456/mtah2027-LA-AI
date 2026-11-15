"""Preflight for the training machine: is everything in place before a long run?

Checks, in order, and reports each as PASS / WARN / FAIL with the fact it found:

* Python version and the packages the training path needs (torch, torchvision,
  ultralytics, cv2, numpy, pandas), with their versions;
* the GPU: CUDA available to torch, device name, total memory, driver / CUDA runtime;
* a real GPU computation (a small convolution, forward and backward) — a driver that
  loads but cannot compute fails here, not an hour into training;
* video decoding: OpenCV can write and read back an MP4 (the codec path the external
  videos need);
* the committed checkpoints: present, and clean under the static import audit of
  :mod:`training.inspect_checkpoint` (nothing is unpickled);
* ultralytics settings that matter on a shared or private machine: usage analytics
  (``sync``) should be off;
* free disk space where the datasets and runs will be written.

Exit code 0 when nothing FAILed. ``--json`` prints the full report for the run log.
``--require-gpu`` turns a missing GPU from WARN into FAIL (use it on the training PC).

    python -m training.env_check --require-gpu
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from training.compliance import REPO_ROOT

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
# The detector checkpoint was trained with this ultralytics version; other versions
# usually load it, but an exact match removes one variable from a reproduction.
CHECKPOINT_ULTRALYTICS = "8.4.212"
MIN_FREE_GB = 30.0


def _check(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}


def check_packages() -> list[dict]:
    """Importability and versions of the training dependencies."""
    out = [_check("python", PASS if sys.version_info >= (3, 11) else FAIL,
                  sys.version.split()[0])]
    for module in ("numpy", "pandas", "torch", "torchvision", "cv2", "ultralytics"):
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "unknown")
            status = PASS
            detail = str(version)
            if module == "ultralytics" and version != CHECKPOINT_ULTRALYTICS:
                status = WARN
                detail = (f"{version} (the committed detector was trained with"
                          f" {CHECKPOINT_ULTRALYTICS})")
            out.append(_check(module, status, detail))
        except Exception as exc:  # noqa: BLE001 - report any import failure
            out.append(_check(module, FAIL, f"import failed: {exc}"))
    return out


def check_gpu(require_gpu: bool) -> list[dict]:
    """CUDA visibility, device facts and one real forward/backward pass."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return [_check("gpu", FAIL, f"torch not importable: {exc}")]

    if not torch.cuda.is_available():
        return [_check("gpu", FAIL if require_gpu else WARN,
                       "torch.cuda.is_available() is False — CPU only. Check the"
                       " NVIDIA driver (nvidia-smi) and that torch is a CUDA build"
                       f" (this one is {torch.__version__})")]
    out = []
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 2 ** 30
    out.append(_check("gpu", PASS, f"{props.name}, {memory_gb:.1f} GB,"
                      f" CUDA runtime {torch.version.cuda}, torch {torch.__version__}"))
    try:
        layer = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        batch = torch.randn(4, 3, 320, 320, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = layer(batch).float().mean()
        loss.backward()
        torch.cuda.synchronize()
        out.append(_check("gpu_compute", PASS, "conv forward/backward with AMP ran"))
    except Exception as exc:  # noqa: BLE001
        out.append(_check("gpu_compute", FAIL, f"GPU computation failed: {exc}"))
    return out


def check_video_io() -> dict:
    """Write a tiny MP4 with OpenCV and read it back."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return _check("video_io", FAIL, f"cv2/numpy not importable: {exc}")
    with tempfile.TemporaryDirectory() as folder:
        target = str(Path(folder) / "probe.mp4")
        writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
        if not writer.isOpened():
            return _check("video_io", WARN, "cannot open an mp4v writer (reading real"
                          " videos may still work; frame folders always do)")
        for value in (10, 120, 240):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        writer.release()
        cap = cv2.VideoCapture(target)
        frames = 0
        while cap.read()[0]:
            frames += 1
        cap.release()
    if frames != 3:
        return _check("video_io", FAIL, f"wrote 3 frames, read back {frames}")
    return _check("video_io", PASS, f"OpenCV {cv2.__version__} wrote and read an MP4"
                  " (mp4v only — the codec of the real videos is proven by the first"
                  " tiles/eval run on them)")


def check_checkpoints() -> list[dict]:
    """Committed checkpoints exist and pass the static import audit."""
    from training.inspect_checkpoint import audit_imports

    out = []
    for name in ("rtdetr_fold0_best.pt", "fot_v2_fold0.pt"):
        path = REPO_ROOT / "models" / name
        if not path.is_file():
            out.append(_check(f"checkpoint:{name}", WARN, "not found in models/"))
            continue
        try:
            audit = audit_imports(path)
        except Exception as exc:  # noqa: BLE001 - a corrupt file is a FAIL row
            out.append(_check(f"checkpoint:{name}", FAIL, f"unreadable: {exc}"))
            continue
        if audit["safe"]:
            out.append(_check(f"checkpoint:{name}", PASS,
                              f"{path.stat().st_size / 1e6:.1f} MB, import audit clean"))
        else:
            out.append(_check(f"checkpoint:{name}", FAIL,
                              f"import audit flagged {audit['outside_allowlist']}"
                              f" {audit['unresolved']} — do NOT load this file"))
    return out


def check_ultralytics_settings() -> dict:
    """Usage analytics should be off on this machine."""
    try:
        from ultralytics import settings
    except Exception as exc:  # noqa: BLE001
        return _check("ultralytics_sync", WARN, f"cannot read settings: {exc}")
    if settings.get("sync", True):
        return _check("ultralytics_sync", WARN,
                      "analytics sync is ON — run: yolo settings sync=False")
    return _check("ultralytics_sync", PASS, "analytics sync is off")


def check_disk(path=None) -> dict:
    """Free space on the volume that will hold datasets and runs."""
    target = Path(path) if path else REPO_ROOT
    try:
        free_gb = shutil.disk_usage(target).free / 2 ** 30
    except OSError as exc:
        return _check("disk", FAIL, f"cannot read {target}: {exc}")
    status = PASS if free_gb >= MIN_FREE_GB else WARN
    return _check("disk", status, f"{free_gb:.0f} GB free at {target}"
                  + ("" if status == PASS else f" (under {MIN_FREE_GB:.0f} GB)"))


def run_checks(require_gpu: bool = False, disk_path=None) -> list[dict]:
    """All checks, in report order."""
    report = check_packages()
    report += check_gpu(require_gpu)
    report.append(check_video_io())
    report += check_checkpoints()
    report.append(check_ultralytics_settings())
    report.append(check_disk(disk_path))
    return report


def main(argv: list[str] | None = None) -> int:
    """Print the preflight report; exit 1 if any check FAILed."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.env_check",
        description="Preflight checks for the detector-training environment.")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail (not warn) when no CUDA GPU is usable")
    parser.add_argument("--disk-path", default=None,
                        help="where datasets/runs will live (default: the repo)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_checks(args.require_gpu, args.disk_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report:
            print(f"[{row['status']}] {row['check']:28s} {row['detail']}")
    failed = [row for row in report if row["status"] == FAIL]
    if not args.json:
        print(f"\n{len(failed)} failed, "
              f"{sum(r['status'] == WARN for r in report)} warnings")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
