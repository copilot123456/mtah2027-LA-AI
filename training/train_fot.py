"""FoT v2 training — a documented RECONSTRUCTION, not the original Kaggle loop.

What is known for certain (from ``src/fot_video.py`` and ``models/fot_v2_fold0.pt``):
the architecture (``build_fot_model``: 12 features -> 160-d, 4 layers, 8 heads,
window 64; the committed weights load into it strictly), the input features
(``build_window_features``), the two outputs (a position = interpolation baseline +
residual, and a ``hit5`` logit), the frame step (2) and the train / validation games
(4, 6 / 9) recorded in ``models/fold0_train_meta.json``.

What is NOT in the repository and is therefore assumed here — confirm each with the
author before treating a checkpoint from this script as a replacement:

  A1  ``hit5`` means "the predicted position is within 5 px of the true ball"
      (``HIT_RADIUS_PX``). Inference uses its sigmoid as the confidence of a bridged
      frame, which is consistent with this reading but does not prove it.
  A2  The position loss is a smooth-L1 on the pixel error scaled by that radius, and
      the two losses are summed with equal weight. The recorded
      ``best_fot_loss = 1.022`` cannot be compared with this script's loss.
  A3  Supervision is on every annotated step (observed or not).
  A4  Occlusions are simulated by removing a random run of observations per window.
  A5  The detector's outputs on the training games are an adequate stand-in for its
      outputs at inference (they are optimistic: the detector has seen those games).

Because of A1-A5 the script always reports, on the held-out game, the model next to
the plain interpolation baseline it starts from. A checkpoint that does not beat the
baseline on bridged steps is not an improvement, whatever its loss says.

Inputs: a dataset manifest and the candidates cache written by
``python -m training.eval_detector --frame-step 1`` for the training and validation
games (detections on consecutive frames; sub-sampling to the model's frame step
happens here, exactly as at inference).

    python -m training.train_fot --manifest data/external/manifest.json \
        --candidates-dir runs/eval/cands --label local \
        --train-games 4 6 --val-games 9 --name fot_fold0

Output: ``runs/fot/<name>/fot_v2.pt`` (a bare ``state_dict``, the format
``build_fot_model`` loads) and ``train_meta.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import src.fot_video as F
from training.compliance import REPO_ROOT, assert_external
from training.eval_detector import candidates_path, frames_path, load_candidates
from training.fot_data import WindowSet, bridgeable_mask, concat, windows_for_game
from training.manifest import load_annotations, load_manifest, resolve_split

HIT_RADIUS_PX = 5.0          # assumption A1
RUNS_DIR = REPO_ROOT / "runs" / "fot"
DEFAULTS = {"epochs": 40, "batch": 128, "lr": 3e-4, "weight_decay": 1e-2,
            "gap_prob": 0.6, "max_gap": 6, "frame_step": 2, "seed": 0}


def load_game_windows(manifest, game_id: str, candidates_dir: Path, label: str,
                      frame_step: int, profile: str, seed: int | None,
                      gap_prob: float, max_gap: int) -> WindowSet:
    """FoT windows of one external game from its cached detector candidates."""
    from training.frames import frame_size

    game = manifest.game(game_id)
    cands_file = candidates_path(candidates_dir, label, game.id)
    frames_file = frames_path(candidates_dir, label, game.id)
    for needed in (cands_file, frames_file):
        if not needed.is_file():
            raise FileNotFoundError(
                f"{needed} not found — run `python -m training.eval_detector"
                f" --frame-step 1 --games {game.id} --labels {label} ...` first")
    frames = pd.read_csv(frames_file)["frame"].to_numpy()
    if len(frames) > 1 and int(np.median(np.diff(np.sort(frames)))) != 1:
        raise ValueError(
            f"{frames_file}: detections are not on consecutive frames; FoT windows"
            " need eval_detector --frame-step 1")
    frame_w, frame_h = frame_size(game)
    return windows_for_game(load_candidates(cands_file), frames, load_annotations(game),
                            frame_w, frame_h, frame_step=frame_step, profile=profile,
                            seed=seed, gap_prob=gap_prob, max_gap=max_gap)


def step_losses(pred_xy, hit_logit, target, target_mask, frame_w: int, frame_h: int):
    """``(loss, err_px, hit_target)`` for one batch (torch tensors).

    Position: smooth-L1 of the pixel error in units of the hit radius. Hit head:
    binary cross-entropy against "this prediction is within the radius" (the target
    is computed from the detached prediction, so the head learns to estimate the
    model's own accuracy). Both are averaged over annotated steps only.
    """
    import torch
    import torch.nn.functional as fn

    scale = torch.tensor([float(frame_w), float(frame_h)], device=pred_xy.device)
    err_px = torch.linalg.vector_norm((pred_xy - target) * scale, dim=-1)
    mask = target_mask.float()
    denom = mask.sum().clamp(min=1.0)
    loc = fn.smooth_l1_loss(err_px / HIT_RADIUS_PX, torch.zeros_like(err_px),
                            reduction="none", beta=1.0)
    hit_target = (err_px.detach() <= HIT_RADIUS_PX).float()
    bce = fn.binary_cross_entropy_with_logits(hit_logit, hit_target, reduction="none")
    loss = ((loc + bce) * mask).sum() / denom
    return loss, err_px, hit_target


def evaluate(model, data: WindowSet, device, batch: int = 256,
             max_gap: int | None = None) -> dict:
    """Held-out metrics: the model against the interpolation baseline it starts from.

    ``bridged`` steps are the annotated steps inference would actually take from the
    model: unobserved, inside a gap of at most ``max_gap`` steps (the profile's limit)
    with a detection on both sides. ``all_unobserved`` also counts edges and long gaps,
    which inference leaves to other stages.
    """
    if max_gap is None:
        max_gap = F.profile_params("max_coverage")[1]
    import torch

    model.eval()
    errs, base_errs, probs, losses = [], [], [], []
    scale = np.array([data.frame_w, data.frame_h], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(data), batch):
            sl = slice(start, start + batch)
            feats = torch.from_numpy(data.feats[sl]).to(device)
            base = torch.from_numpy(data.baseline[sl]).to(device)
            target = torch.from_numpy(data.target[sl]).to(device)
            mask = torch.from_numpy(data.target_mask[sl]).to(device)
            pred_xy, hit_logit = model(feats, base)
            loss, err_px, _ = step_losses(pred_xy, hit_logit, target, mask,
                                          data.frame_w, data.frame_h)
            losses.append(float(loss) * int(mask.sum()))
            errs.append(err_px.cpu().numpy())
            probs.append(torch.sigmoid(hit_logit).cpu().numpy())
            base_errs.append(np.linalg.norm(
                (data.baseline[sl] - data.target[sl]) * scale, axis=-1))
    err = np.concatenate(errs)
    base_err = np.concatenate(base_errs)
    prob = np.concatenate(probs)
    annotated = data.target_mask
    bridged = annotated & bridgeable_mask(data.observed, max_gap)
    unobserved = annotated & ~data.observed

    def stats(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"n": 0, "median_px": None, "p90_px": None, "hit_rate": None}
        return {"n": int(values.size), "median_px": float(np.median(values)),
                "p90_px": float(np.percentile(values, 90)),
                "hit_rate": float((values <= HIT_RADIUS_PX).mean())}

    hit = err <= HIT_RADIUS_PX
    out = {
        "loss": float(sum(losses) / max(1, int(annotated.sum()))),
        "windows": len(data),
        "all_steps_model": stats(err[annotated]),
        "all_steps_baseline": stats(base_err[annotated]),
        "bridged_steps_model": stats(err[bridged]),
        "bridged_steps_baseline": stats(base_err[bridged]),
        "all_unobserved_model": stats(err[unobserved]),
        "all_unobserved_baseline": stats(base_err[unobserved]),
        "hit_prob_mean_on_hits": float(prob[annotated & hit].mean())
        if (annotated & hit).any() else None,
        "hit_prob_mean_on_misses": float(prob[annotated & ~hit].mean())
        if (annotated & ~hit).any() else None,
    }
    model_med = out["bridged_steps_model"]["median_px"]
    base_med = out["bridged_steps_baseline"]["median_px"]
    out["beats_baseline_on_bridged"] = (
        None if model_med is None or base_med is None else bool(model_med < base_med))
    return out


def train(train_data: WindowSet, val_data: WindowSet, run_dir: Path, epochs: int,
          batch: int, lr: float, weight_decay: float, seed: int, device=None,
          init=None, log=print, max_gap: int | None = None) -> dict:
    """Train FoT v2 and save the best (lowest held-out loss) ``state_dict``.

    The held-out game picks the epoch, so the metrics reported on it are slightly
    optimistic; a clean verdict needs a game used for neither training nor selection.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = F.build_fot_model(F.WINDOW_SIZE, weights=init, device=device)
    if init is None:
        # The model predicts baseline + residual. With a random last layer the
        # residual is noise and training starts tens of pixels WORSE than plain
        # interpolation; zeroing it makes step 0 exactly the baseline, so the model
        # can only be kept if it learns something on top of it.
        last = model.coord_head[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(train_data) / batch)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)

    tensors = [torch.from_numpy(a) for a in (train_data.feats, train_data.baseline,
                                             train_data.target, train_data.target_mask)]
    generator = torch.Generator().manual_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "fot_v2.pt"
    best = {"loss": float("inf"), "epoch": None, "metrics": None}
    history = []
    before = evaluate(model, val_data, device, max_gap=max_gap)
    log(f"before training: held-out loss {before['loss']:.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_data), generator=generator)
        running, seen = 0.0, 0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            feats, base, target, mask = (t[index].to(device) for t in tensors)
            pred_xy, hit_logit = model(feats, base)
            loss, _, _ = step_losses(pred_xy, hit_logit, target, mask,
                                     train_data.frame_w, train_data.frame_h)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.detach()) * len(index)
            seen += len(index)
        metrics = evaluate(model, val_data, device, max_gap=max_gap)
        history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                        "val_loss": metrics["loss"]})
        log(f"epoch {epoch:3d}/{epochs}  train {running / max(1, seen):.4f}"
            f"  held-out {metrics['loss']:.4f}")
        if np.isfinite(metrics["loss"]) and metrics["loss"] < best["loss"]:
            best = {"loss": metrics["loss"], "epoch": epoch, "metrics": metrics}
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       best_path)
    if best["epoch"] is None:
        raise RuntimeError("no epoch produced a finite held-out loss; nothing was"
                           " saved — check the data (NaN targets?) and the learning rate")
    return {"best": best, "before_training": before, "history": history,
            "weights": str(best_path), "device": str(device)}


def main(argv: list[str] | None = None) -> int:
    """Train FoT v2 on external games from cached detector candidates."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.train_fot",
        description="FoT v2 training (reconstruction — see the module docstring).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates-dir", required=True,
                        help="folder written by training.eval_detector --frame-step 1")
    parser.add_argument("--label", required=True,
                        help="detector label used in that folder")
    parser.add_argument("--train-games", nargs="+", required=True)
    parser.add_argument("--val-games", nargs="+", required=True)
    parser.add_argument("--name", required=True, help="run name")
    parser.add_argument("--init", default=None,
                        help="start from this FoT state_dict (e.g. models/fot_v2_fold0.pt)")
    parser.add_argument("--profile", choices=sorted(F.PROFILES), default="max_coverage")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--device", default=None)
    parser.add_argument("--runs-dir", default=None,
                        help=f"where runs are written (default {RUNS_DIR})")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    train_ids, val_ids = resolve_split(manifest, args.train_games, args.val_games)
    candidates_dir = Path(args.candidates_dir).expanduser()
    run_dir = (Path(args.runs_dir).expanduser() if args.runs_dir else RUNS_DIR) / args.name
    assert_external(candidates_dir, run_dir, what="FoT training input/output")

    train_data = concat([load_game_windows(manifest, g, candidates_dir, args.label,
                                           args.frame_step, args.profile, args.seed + i,
                                           args.gap_prob, args.max_gap)
                         for i, g in enumerate(train_ids)])
    val_data = concat([load_game_windows(manifest, g, candidates_dir, args.label,
                                         args.frame_step, args.profile, 10_000 + i,
                                         args.gap_prob, args.max_gap)
                       for i, g in enumerate(val_ids)])
    print(f"train windows {len(train_data)} (games {train_ids}),"
          f" held-out windows {len(val_data)} (games {val_ids})")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = train(train_data, val_data, run_dir, args.epochs, args.batch, args.lr,
                   args.weight_decay, args.seed, args.device, args.init,
                   max_gap=F.profile_params(args.profile)[1])
    meta = {
        "reconstruction": True,
        "assumptions": "A1-A5 in training/train_fot.py — confirm with the author",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": manifest.dataset, "license": manifest.license,
        "source_url": manifest.source_url,
        "train_games": train_ids, "val_games": val_ids,
        "detector_label": args.label, "init": args.init, "profile": args.profile,
        "hit_radius_px": HIT_RADIUS_PX,
        "selection_note": "the held-out game also selected the best epoch: its"
                          " metrics are optimistic",
        "hyperparameters": {k: getattr(args, k) for k in DEFAULTS},
        **result,
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best = result["best"]["metrics"] or {}
    print(f"best epoch {result['best']['epoch']}, held-out loss {result['best']['loss']:.4f}")
    print(f"bridged steps  model   : {best.get('bridged_steps_model')}")
    print(f"bridged steps  baseline: {best.get('bridged_steps_baseline')}")
    verdict = best.get("beats_baseline_on_bridged")
    print("verdict: " + ("beats the interpolation baseline on bridged steps" if verdict
                         else "does NOT beat the interpolation baseline on bridged steps"
                         if verdict is False else "no bridged steps to judge on"))
    print("note: the held-out game also picked the epoch — confirm on another game")
    print(f"weights: {result['weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
