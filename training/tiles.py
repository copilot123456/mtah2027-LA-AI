"""Tile dataset builder: external games -> the 320 px tiles RT-DETR is trained on.

Inference (``src.fot_video.detect_window``) cuts every frame into ``TILE_SIZE`` tiles
on the ``compute_tile_coords`` grid and runs the detector per tile, so the detector
must be trained on tiles cut the same way. This module reuses that exact grid — the
constants and the function are imported from ``src.fot_video``, not copied — and
writes an ultralytics detection dataset:

    <out>/images/{train,val}/g<game>_f<frame>_x<tx>_y<ty>.jpg
    <out>/labels/{train,val}/<same stem>.txt     "0 cx cy w h" (normalised to the tile)
    <out>/data.yaml                              what `training.train_rtdetr --data` takes
    <out>/tiles_index.csv                        one row per tile, for audits and evaluation
    <out>/provenance.json                        dataset identity, split, arguments, counts

Per sampled frame:

* **positive tiles** — every grid tile that contains the ball centre (the 60 px
  overlap puts a ball in up to four tiles; inference sees all of them, so training
  does too). The label box is the annotated ``w x h`` when the dataset has one,
  otherwise a ``box_px`` square, clipped to the tile.
* **background tiles** — ``neg_per_pos`` random tiles that do not touch the ball
  (or ``neg_per_empty`` on a frame with no visible ball). They have no label file,
  which is how ultralytics represents a background image.
* **hard negatives** (optional) — tiles around false alarms of an existing detector,
  taken from a ``training.eval_detector`` candidates CSV: the boots, penalty spots
  and line marks the current model already confuses with the ball.

Train and validation tiles come from different games, never from the same one: the
split is by game (``training.manifest.resolve_split``).

Not reproduced, because it is unknown: whether the original Kaggle tiles were
pitch-masked (inference blacks out everything outside the pitch when a homography
is valid). External games usually have no homography, so tiles are written
unmasked; this train/inference difference is recorded in ``provenance.json``.

The planning functions are pure numpy and unit-tested; only :func:`build` touches
cv2 and the disk.

    python -m training.tiles --manifest data/external/manifest.json \
        --out data/external/tiles_fold0 --train-games 4 6 --val-games 9
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.fot_video import TILE_SIZE, TILE_STRIDE, compute_tile_coords
from training.compliance import REPO_ROOT, assert_external
from training.manifest import (Manifest, fingerprint, load_annotations, load_manifest,
                               resolve_split)

DEFAULT_FRAME_STEP = 2      # models/fold0_train_meta.json: "frame_step": 2
DEFAULT_BOX_PX = 10.0       # ASSUMPTION — used only when the dataset has no box sizes
DEFAULT_NEG_PER_POS = 1.0
DEFAULT_NEG_PER_EMPTY = 1.0
DEFAULT_SEED = 0

KIND_POS = "pos"
KIND_NEG = "neg"
KIND_HARD_NEG = "hard_neg"


@dataclass(frozen=True)
class TilePlan:
    """One tile to write: its top-left corner, kind, and YOLO label (None = background)."""

    tx: int
    ty: int
    kind: str
    label: tuple[float, float, float, float] | None = None


def ball_box(x: float, y: float, w: float, h: float,
             box_px: float = DEFAULT_BOX_PX) -> tuple[float, float, float, float]:
    """``(cx, cy, bw, bh)`` in frame pixels; ``box_px`` fills in a missing size."""
    bw = float(w) if np.isfinite(w) and w > 0 else float(box_px)
    bh = float(h) if np.isfinite(h) and h > 0 else float(box_px)
    return float(x), float(y), bw, bh


def label_for_tile(tx: int, ty: int, box: tuple[float, float, float, float],
                   tile_w: int = TILE_SIZE, tile_h: int = TILE_SIZE):
    """YOLO label ``(cx, cy, w, h)`` normalised to the tile, or None.

    A tile owns the ball when the ball *centre* lies inside it (left/top edge
    inclusive, right/bottom exclusive). The box is clipped to the tile, so a ball on
    the border keeps a valid, smaller box instead of one that leaves the image.
    """
    cx, cy, bw, bh = box
    if not (tx <= cx < tx + tile_w and ty <= cy < ty + tile_h):
        return None
    x0 = max(cx - bw / 2.0, tx)
    x1 = min(cx + bw / 2.0, tx + tile_w)
    y0 = max(cy - bh / 2.0, ty)
    y1 = min(cy + bh / 2.0, ty + tile_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (((x0 + x1) / 2.0 - tx) / tile_w, ((y0 + y1) / 2.0 - ty) / tile_h,
            (x1 - x0) / tile_w, (y1 - y0) / tile_h)


def tile_touches_box(tx: int, ty: int, box: tuple[float, float, float, float],
                     tile_w: int = TILE_SIZE, tile_h: int = TILE_SIZE) -> bool:
    """True when any part of the ball box falls inside the tile."""
    cx, cy, bw, bh = box
    return (cx + bw / 2.0 > tx and cx - bw / 2.0 < tx + tile_w
            and cy + bh / 2.0 > ty and cy - bh / 2.0 < ty + tile_h)


def _stochastic_round(value: float, rng: np.random.Generator) -> int:
    """Round ``value`` up with probability equal to its fractional part."""
    base = int(np.floor(value))
    return base + int(rng.random() < (value - base))


def plan_frame(frame_w: int, frame_h: int, ball, rng: np.random.Generator,
               neg_per_pos: float = DEFAULT_NEG_PER_POS,
               neg_per_empty: float = DEFAULT_NEG_PER_EMPTY,
               hard_points=(), box_px: float = DEFAULT_BOX_PX,
               tile_size: int = TILE_SIZE, stride: int = TILE_STRIDE) -> list[TilePlan]:
    """Tiles to cut from one frame.

    ``ball`` is ``(x, y, w, h)`` in frame pixels or None when no ball is visible.
    ``hard_points`` are ``(x, y)`` false-alarm positions; each adds the background
    tile in which the point is most central, unless that tile touches the ball.
    Frames smaller than a tile produce nothing (inference could not tile them either).
    """
    if frame_w < tile_size or frame_h < tile_size:
        return []
    grid = compute_tile_coords(frame_w, frame_h, tile_size, stride)
    plans: list[TilePlan] = []
    used: set[tuple[int, int]] = set()

    box = None
    if ball is not None:
        box = ball_box(*ball, box_px=box_px)
        for tx, ty in grid:
            label = label_for_tile(tx, ty, box, tile_size, tile_size)
            if label is not None:
                plans.append(TilePlan(tx, ty, KIND_POS, label))
                used.add((tx, ty))

    def is_background(tx: int, ty: int) -> bool:
        return (tx, ty) not in used and (
            box is None or not tile_touches_box(tx, ty, box, tile_size, tile_size))

    for px, py in hard_points:
        holders = [(tx, ty) for tx, ty in grid
                   if tx <= px < tx + tile_size and ty <= py < ty + tile_size]
        holders = [t for t in holders if is_background(*t)]
        if not holders:
            continue
        centre = tile_size / 2.0
        tx, ty = min(holders, key=lambda t: (abs(px - t[0] - centre)
                                             + abs(py - t[1] - centre), t))
        plans.append(TilePlan(tx, ty, KIND_HARD_NEG, None))
        used.add((tx, ty))

    n_pos = sum(1 for p in plans if p.kind == KIND_POS)
    wanted = (_stochastic_round(neg_per_pos * n_pos, rng) if n_pos
              else _stochastic_round(neg_per_empty, rng))
    pool = [t for t in grid if is_background(*t)]
    if wanted and pool:
        picks = rng.choice(len(pool), size=min(wanted, len(pool)), replace=False)
        for index in sorted(int(i) for i in picks):
            plans.append(TilePlan(pool[index][0], pool[index][1], KIND_NEG, None))
    return plans


def tile_stem(game_id: str, frame: int, tx: int, ty: int) -> str:
    """File stem of one tile; encodes everything needed to find it in the game."""
    return f"g{game_id}_f{int(frame):07d}_x{int(tx):04d}_y{int(ty):04d}"


def load_hard_negatives(path, min_conf: float, radius_px: float,
                        annotations: dict[str, pd.DataFrame]) -> dict:
    """``{(game, frame): [(x, y), ...]}`` false alarms from a candidates CSV.

    The CSV is what ``training.eval_detector`` writes: ``game, frame, gx, gy, conf``.
    A candidate is a false alarm when its confidence is at least ``min_conf`` and it
    is farther than ``radius_px`` from that frame's annotated ball (or the frame has
    no visible ball).
    """
    assert_external(path, what="hard-negative candidates")
    table = pd.read_csv(path, dtype={"game": str})
    table = table[table["conf"] >= min_conf]
    out: dict = {}
    for game_id, group in table.groupby("game"):
        ann = annotations.get(str(game_id))
        if ann is None:
            continue
        truth = ann.set_index("frame")[["x", "y"]]
        for row in group.itertuples(index=False):
            frame = int(row.frame)
            if frame in truth.index:
                gx, gy = truth.loc[frame, "x"], truth.loc[frame, "y"]
                if np.hypot(row.gx - gx, row.gy - gy) <= radius_px:
                    continue
            out.setdefault((str(game_id), frame), []).append((float(row.gx),
                                                              float(row.gy)))
    return out


def check_annotation_range(game, truth: pd.DataFrame, frame_w: int, frame_h: int,
                           n_frames: int) -> dict:
    """Refuse annotations that cannot belong to these frames; return the ranges.

    Catches the silent failure modes of frame numbering and geometry: annotated frame
    numbers outside ``frame_offset .. frame_offset + n_frames - 1`` (a wrong
    ``frame_offset``, 1-based vs 0-based), and ball centres outside the image (labels
    made for another resolution). A few frames past the end of a *video* are
    tolerated because container frame counts are approximate.
    """
    first = int(game.frame_offset)
    last = first + int(n_frames) - 1
    frames = truth.index.to_numpy()
    tolerance = 0 if game.frames_dir is not None else 5
    outside = frames[(frames < first) | (frames > last + tolerance)]
    if len(outside):
        raise ValueError(
            f"game {game.id}: {len(outside)} annotated frame(s) outside the frame range"
            f" {first}..{last} (e.g. {outside[:3].tolist()}) — check 'frame_offset'")
    off_image = int(((truth["x"] > frame_w) | (truth["y"] > frame_h)).sum())
    if off_image > 0.01 * max(len(truth), 1):
        raise ValueError(
            f"game {game.id}: {off_image} ball centres lie outside the"
            f" {frame_w}x{frame_h} frame — annotations for another resolution?")
    return {"frames": [first, last],
            "annotated": [int(frames.min()), int(frames.max())] if len(frames) else None}


def write_data_yaml(out_dir: Path) -> Path:
    """The ultralytics dataset YAML (written by hand: no PyYAML dependency)."""
    target = out_dir / "data.yaml"
    # json.dumps gives a double-quoted YAML scalar: a path with " #" or ": " in it
    # would otherwise be truncated or make the YAML invalid.
    target.write_text(
        f"path: {json.dumps(str(out_dir.resolve()))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: ball\n", encoding="utf-8")
    return target


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(manifest: Manifest, out_dir, train_games: list[str], val_games: list[str],
          frame_step: int = DEFAULT_FRAME_STEP, neg_per_pos: float = DEFAULT_NEG_PER_POS,
          neg_per_empty: float = DEFAULT_NEG_PER_EMPTY, box_px: float = DEFAULT_BOX_PX,
          seed: int = DEFAULT_SEED, max_frames_per_game: int | None = None,
          hard_negatives=None, hard_min_conf: float = 0.18, hard_radius_px: float = 15.0,
          ext: str = "jpg", quality: int = 95, force: bool = False,
          progress_every: int = 500) -> dict:
    """Cut the tiles, write the dataset, return the provenance dict.

    Per-frame sampling is seeded by ``(seed, game position, frame)``, so the result
    does not depend on how the run is chunked and a re-run reproduces the same tiles.
    """
    import cv2

    from training.frames import frame_size, iter_frames

    from training.frames import frame_count

    out_dir = Path(out_dir).expanduser()
    assert_external(out_dir, what="tile dataset output")
    if ext not in ("jpg", "png"):
        raise ValueError("ext must be jpg or png")
    if (out_dir / "images").exists():
        if not force:
            raise FileExistsError(
                f"{out_dir}/images already exists; choose a new --out or pass --force"
                " to delete it and rebuild")
        if not (out_dir / "tiles_index.csv").is_file():
            raise FileExistsError(
                f"{out_dir} was not written by training.tiles (no tiles_index.csv);"
                " refusing to delete it")
        # Rebuild from scratch: merging would keep stale tiles and label files that
        # ultralytics would still train on (including games that moved split).
        for stale in ("images", "labels"):
            shutil.rmtree(out_dir / stale, ignore_errors=True)

    split_of = {g: "train" for g in train_games}
    split_of.update({g: "val" for g in val_games})
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = {g: load_annotations(manifest.game(g)) for g in split_of}
    hard = {}
    if hard_negatives is not None:
        hard = load_hard_negatives(hard_negatives, hard_min_conf, hard_radius_px,
                                   annotations)
    hard_used: set = set()

    params = ([cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext == "jpg" else [])
    rows = []
    counts = {s: {KIND_POS: 0, KIND_NEG: 0, KIND_HARD_NEG: 0, "frames": 0,
                  "frames_with_ball": 0} for s in ("train", "val")}
    boxes_from_default = 0

    for game_pos, game_id in enumerate(manifest.ids()):
        if game_id not in split_of:
            continue
        split = split_of[game_id]
        game = manifest.game(game_id)
        frame_w, frame_h = frame_size(game)
        if frame_w < TILE_SIZE or frame_h < TILE_SIZE:
            raise ValueError(f"game {game_id}: frames are {frame_w}x{frame_h}, smaller"
                             f" than one {TILE_SIZE} px tile")
        truth = annotations[game_id].set_index("frame")
        check_annotation_range(game, truth, frame_w, frame_h, frame_count(game))
        print(f"game {game_id} [{split}] {frame_w}x{frame_h},"
              f" {len(truth)} annotated frames", flush=True)

        for n_seen, (frame, image) in enumerate(
                iter_frames(game, step=frame_step, max_frames=max_frames_per_game), 1):
            ball = None
            if frame in truth.index:
                record = truth.loc[frame]
                ball = (record["x"], record["y"], record["w"], record["h"])
                if not (np.isfinite(record["w"]) and record["w"] > 0):
                    boxes_from_default += 1
            rng = np.random.default_rng([int(seed), game_pos, int(frame)])
            hard_points = hard.get((game_id, int(frame)), ())
            if hard_points:
                hard_used.add((game_id, int(frame)))
            plans = plan_frame(frame_w, frame_h, ball, rng, neg_per_pos, neg_per_empty,
                               hard_points, box_px)
            counts[split]["frames"] += 1
            counts[split]["frames_with_ball"] += int(ball is not None)

            for plan in plans:
                stem = tile_stem(game_id, frame, plan.tx, plan.ty)
                crop = image[plan.ty:plan.ty + TILE_SIZE, plan.tx:plan.tx + TILE_SIZE]
                image_path = out_dir / "images" / split / f"{stem}.{ext}"
                if not cv2.imwrite(str(image_path), crop, params):
                    raise IOError(f"could not write {image_path}")
                if plan.label is not None:
                    label_path = out_dir / "labels" / split / f"{stem}.txt"
                    label_path.write_text(
                        "0 " + " ".join(f"{v:.6f}" for v in plan.label) + "\n",
                        encoding="utf-8")
                counts[split][plan.kind] += 1
                rows.append({
                    "split": split, "game": game_id, "frame": int(frame),
                    "tx": plan.tx, "ty": plan.ty, "kind": plan.kind,
                    "ball_x": float(ball[0]) if ball is not None else np.nan,
                    "ball_y": float(ball[1]) if ball is not None else np.nan,
                    "file": f"images/{split}/{stem}.{ext}"})
            if progress_every and n_seen % progress_every == 0:
                print(f"  game {game_id}: {n_seen} frames,"
                      f" {sum(counts[split][k] for k in (KIND_POS, KIND_NEG, KIND_HARD_NEG))}"
                      f" {split} tiles so far", flush=True)

    unused = set(hard) - hard_used
    if unused:
        raise ValueError(
            f"{len(unused)} of {len(hard)} hard-negative frames were never visited"
            f" (tiles --frame-step {frame_step} vs the stride they were mined at, or a"
            " game outside this split) — mine with the same --frame-step")
    for split in ("train", "val"):
        if counts[split]["frames"] and counts[split][KIND_POS] == 0:
            raise ValueError(f"the {split} split has no positive tile — check the"
                             " annotations (empty file, wrong ids or frame numbering)")

    index = pd.DataFrame(rows, columns=["split", "game", "frame", "tx", "ty", "kind",
                                        "ball_x", "ball_y", "file"])
    index.to_csv(out_dir / "tiles_index.csv", index=False)
    data_yaml = write_data_yaml(out_dir)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "dataset": manifest.dataset, "license": manifest.license,
        "source_url": manifest.source_url, "manifest": str(manifest.path),
        "manifest_fingerprint": fingerprint(manifest),
        "train_games": list(train_games), "val_games": list(val_games),
        "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE,
        "frame_step": frame_step, "neg_per_pos": neg_per_pos,
        "neg_per_empty": neg_per_empty, "box_px_default": box_px,
        "label_boxes_from_default_size": boxes_from_default,
        "seed": seed, "max_frames_per_game": max_frames_per_game,
        "hard_negatives": str(hard_negatives) if hard_negatives is not None else None,
        "hard_negative_frames_used": len(hard_used),
        "hard_min_conf": hard_min_conf, "hard_radius_px": hard_radius_px,
        "image_ext": ext, "jpeg_quality": quality if ext == "jpg" else None,
        "pitch_masked": False,
        "pitch_mask_note": "inference masks non-pitch pixels when a homography is"
                           " valid; these tiles are unmasked (no homography for"
                           " external games)",
        "counts": counts, "data_yaml": str(data_yaml.resolve()),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2),
                                             encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    """Build a tile dataset from an external-games manifest."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.tiles",
        description="Cut external games into the 320 px tile dataset RT-DETR trains on.")
    parser.add_argument("--manifest", required=True, help="dataset manifest (JSON)")
    parser.add_argument("--out", required=True, help="output dataset folder")
    parser.add_argument("--train-games", nargs="*", default=None, help="game ids")
    parser.add_argument("--val-games", nargs="*", default=None, help="game ids")
    parser.add_argument("--fold", type=int, default=None,
                        help="leave-one-game-out fold index (instead of the two lists)")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--neg-per-pos", type=float, default=DEFAULT_NEG_PER_POS)
    parser.add_argument("--neg-per-empty", type=float, default=DEFAULT_NEG_PER_EMPTY)
    parser.add_argument("--box-px", type=float, default=DEFAULT_BOX_PX,
                        help="label box edge when the dataset has no box sizes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-frames-per-game", type=int, default=None)
    parser.add_argument("--hard-negatives", default=None,
                        help="candidates CSV from training.eval_detector")
    parser.add_argument("--hard-min-conf", type=float, default=0.18)
    parser.add_argument("--hard-radius-px", type=float, default=15.0)
    parser.add_argument("--ext", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--force", action="store_true",
                        help="delete an existing tile dataset in --out and rebuild it")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the manifest and split, write nothing")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    train, val = resolve_split(manifest, args.train_games, args.val_games, args.fold)
    print(f"dataset {manifest.dataset!r} ({manifest.license}) — train games {train},"
          f" validation games {val}")
    if args.dry_run:
        from training.frames import frame_count, frame_size

        for game_id in train + val:
            game = manifest.game(game_id)
            ann = load_annotations(game)
            sized = int(np.isfinite(ann["w"]).sum())
            frame_w, frame_h = frame_size(game)
            ranges = check_annotation_range(game, ann.set_index("frame"), frame_w,
                                            frame_h, frame_count(game))
            print(f"  game {game_id}: {frame_w}x{frame_h}, frames"
                  f" {ranges['frames'][0]}..{ranges['frames'][1]}, {len(ann)} annotated"
                  f" frames ({ranges['annotated']}), {sized} with a box size")
        print("dry run: nothing written")
        return 0

    provenance = build(
        manifest, args.out, train, val, frame_step=args.frame_step,
        neg_per_pos=args.neg_per_pos, neg_per_empty=args.neg_per_empty,
        box_px=args.box_px, seed=args.seed,
        max_frames_per_game=args.max_frames_per_game,
        hard_negatives=args.hard_negatives, hard_min_conf=args.hard_min_conf,
        hard_radius_px=args.hard_radius_px, ext=args.ext, quality=args.quality,
        force=args.force)
    for split, c in provenance["counts"].items():
        print(f"{split}: {c['frames']} frames ({c['frames_with_ball']} with ball) ->"
              f" {c[KIND_POS]} positive, {c[KIND_NEG]} background,"
              f" {c[KIND_HARD_NEG]} hard-negative tiles")
    if provenance["label_boxes_from_default_size"]:
        print(f"NOTE: {provenance['label_boxes_from_default_size']} labels used the"
              f" default {args.box_px:g} px box (the dataset has no box size there)")
    print(f"data yaml: {provenance['data_yaml']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
