"""The pipeline declaration — EDIT THIS FILE to swap, reorder or add steps.

Quick start (the point of the framework):
  * Swap a stage: replace its handler in SPEC below. Example — gap filling with
    ViT instead of the current handler:
        from src.vit_fill import ball_track       # your fn(ctx, cfg) -> None
        Step("ball_fill", ball_track, requires=("arrays",), provides=("track",))
  * Add a stage: write any function `fn(ctx, cfg) -> None` WITH A DOCSTRING
    (in its own module), then append
        Step("name", fn, requires=(...), provides=(...))
    to SPEC. `requires` must name fields earlier steps `provides`.
  * The default SPEC below is the frozen pipeline: it must keep reproducing
    LA+AI.csv byte-for-byte (md5 aeef0058a08860c9e2407423b1172684).

Rules that bind (full list in PIPELINE_STANDARD_AI.md):
  * No tuning/fitting/selecting on the provided match, its ground truth, the
    video, or the public window — external evidence only (IDSSE, >0.02 paired,
    pre-declared in dossier.md §7) may change scoring behaviour.
  * Every change to this standard must be shared with the team before it
    merges (adoption step, see PIPELINE_STANDARD.html).
"""

from src import steps as S
from src.pipeline import Step

SPEC: list[Step] = [
    Step("load",      S.load,        provides=("arrays",)),
    Step("ball_fill", S.ball_track,  requires=("arrays",), provides=("track",)),
    Step("emission",  S.emission,    requires=("arrays", "track"),
         provides=("emission",)),
    Step("decode",    S.decode_step, requires=("arrays", "track", "emission"),
         provides=("decoded",)),
    Step("x_policy",  S.x_policy,    requires=("arrays", "emission", "decoded"),
         provides=("labels",)),
    Step("summary",   S.summarize,   requires=("arrays", "track", "decoded", "labels"),
         provides=("summary",)),
]
