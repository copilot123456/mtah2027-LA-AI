"""Tile-level validation of one or more checkpoints on the same tile dataset.

Two uses:

* **Dataset integrity.** ``models/rtdetr_fold0_best.pt`` recorded its validation
  metrics when it was trained (precision 0.732, recall 0.699, mAP50 0.646,
  mAP50-95 0.271 on the fold-0 validation tiles — read them back with
  ``python -m training.inspect_checkpoint``). If the tile dataset on this machine is
  the one it was trained on, validating that checkpoint here reproduces those
  numbers closely. A large difference means the tiles differ (other games, other
  sampling, other labels) — find out why before training on them.
* **Like-for-like comparison** of checkpoints when only a prepared tile dataset is
  available and the full-frame evaluation (``training.eval_detector``) cannot be run
  because the raw games are not on the machine.

Tile-level mAP is what ultralytics optimises, not what the pipeline uses; prefer
``training.eval_detector`` whenever the raw external games are available.

    python -m training.val_tiles --data data/external/tiles_fold0/data.yaml \
        --weights models/rtdetr_fold0_best.pt runs/rtdetr/fold_0/weights/best.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from training.compliance import assert_external
from training.train_rtdetr import dataset_paths

# Recorded in models/rtdetr_fold0_best.pt (best epoch of the original run).
RECORDED_FOLD0 = {"precision": 0.73161, "recall": 0.69933, "mAP50": 0.64649,
                  "mAP50-95": 0.27119}


def validate(weights, data_yaml, device=None, batch: int = 16, imgsz: int | None = None,
             split: str = "val") -> dict:
    """Run ultralytics validation and return the four box metrics as floats.

    ``imgsz`` defaults to the size the checkpoint recorded at training time, so a
    checkpoint trained on up-sampled tiles (profile ``upsample2x``) is validated at
    its own size, not at 320.
    """
    import torch
    from ultralytics import RTDETR

    from training.eval_detector import recorded_imgsz

    imgsz = int(imgsz or recorded_imgsz(weights))
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    model = RTDETR(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch, device=device,
                        split=split, plots=False, verbose=False)
    box = metrics.box
    return {"imgsz": imgsz, "precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50-95": float(box.map)}


def main(argv: list[str] | None = None) -> int:
    """Validate each checkpoint on the dataset's validation tiles and print a table."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.val_tiles",
        description="Tile-level validation of checkpoints on one tile dataset.")
    parser.add_argument("--data", required=True, help="dataset YAML")
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="input size for every checkpoint (default: each"
                             " checkpoint's recorded training size)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    data_yaml = Path(args.data).expanduser().resolve()
    assert_external(*dataset_paths(data_yaml), what="validation dataset")
    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    rows = []
    for weights in args.weights:
        result = validate(weights, data_yaml, device, args.batch, imgsz=args.imgsz,
                          split=args.split)
        rows.append({"weights": str(weights), **result})

    if args.json:
        print(json.dumps({"recorded_fold0": RECORDED_FOLD0, "results": rows}, indent=2))
        return 0
    columns = ["precision", "recall", "mAP50", "mAP50-95"]
    print(f"{'checkpoint':60s} {'imgsz':>6s} " + " ".join(f"{c:>10s}" for c in columns))
    print(f"{'(recorded in the fold-0 checkpoint)':60s} {320:6d} "
          + " ".join(f"{RECORDED_FOLD0[c]:10.3f}" for c in columns))
    for row in rows:
        print(f"{row['weights'][-60:]:60s} {row['imgsz']:6d} "
              + " ".join(f"{row[c]:10.3f}" for c in columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
