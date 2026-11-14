"""Experiment declaration: the FoT v2 video detector as the `ball_fill` stage.

This is a SECOND spec file, added alongside the frozen `pipeline_spec.py` — the
framework change process (PIPELINE_STANDARD_AI.md §7) does not apply, because
nothing shared is modified: the default spec, the runner, the handlers, config and
the tests are all untouched. This file only *uses* them.

What it swaps (standard recipe 4.1):

    ball_fill:  S.ball_track        -> src.fot_video.fot_video_track
    everything: load, emission, decode, x_policy, summary — identical handlers

Run it explicitly (the default `python -m src.submit` still runs pipeline_spec.SPEC):

    from config import CFG
    from src.pipeline import run
    labels, summary = run(CFG, spec=FOT_VIDEO_SPEC)

Requirements beyond the default run: the gated tactical video (games.require),
models/rtdetr_fold0_best.pt + models/fot_v2_fold0.pt, and torch/ultralytics/cv2.

Known adoption gap, deliberately not papered over: src/emission.py gives
SOURCE_DETECTOR zero team confidence (its docstring says "confidence TBD when that
tier is implemented"), so video detections steer the dead-ball features and the
kinematic confidence decay but contribute no H/A evidence on their own. Closing
that gap means a config.py + emission.py change, which is a framework change and
follows PIPELINE_STANDARD_AI.md §7 — not this branch.
"""

from src import fot_video as F
from src import steps as S
from src.pipeline import Step

FOT_VIDEO_SPEC: list[Step] = [
    Step("load",      S.load,           provides=("arrays",)),
    Step("ball_fill", F.fot_video_track, requires=("arrays",), provides=("track",)),
    Step("emission",  S.emission,       requires=("arrays", "track"),
         provides=("emission",)),
    Step("decode",    S.decode_step,    requires=("arrays", "track", "emission"),
         provides=("decoded",)),
    Step("x_policy",  S.x_policy,       requires=("arrays", "emission", "decoded"),
         provides=("labels",)),
    Step("summary",   S.summarize,      requires=("arrays", "track", "decoded", "labels"),
         provides=("summary",)),
]
