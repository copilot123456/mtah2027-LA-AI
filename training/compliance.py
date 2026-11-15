"""Seatbelt for the one rule training can break silently: external data only.

Competition rule (README / dossier): any training, fitting or tuning must use
EXTERNAL data — never the provided match (its video, tracking, Smart Tagging or
ground truth). The inference code in ``src/fot_video.py`` legitimately reads the
provided video as *input*; everything under ``training/`` must not, and calls
:func:`assert_external` on every path it is handed before opening it.

What counts as provided material here:

* anything inside a provided-source folder of the match bundle
  (``<DATA_DIR>/tactical``, ``<DATA_DIR>/broadcast`` — ``games.SOURCES``), where
  ``DATA_DIR`` is ``./data`` or ``$MTAH_DATA_DIR``;
* any path with a component that contains a provided stem (``BAR-ATH`` — the keys
  of ``games.GAMES`` and the stems of ``games.SOURCES``), wherever it lives, so a
  copy of the video under another folder is still refused while it keeps its name;
* anything under ``evaluation/gt`` (ground truth).

Other folders under ``data/`` (for example ``data/external/…`` or ``data/idsse``)
are fine: ``data/*`` is git-ignored, so that is the natural place for external
datasets.

Limit, stated plainly: this is a name-and-location check. It cannot recognise a
renamed copy of the provided video. It prevents accidents; it does not prove
compliance — the dataset manifest's ``source_url`` / ``license`` fields and the
provenance file each run writes are the actual record.
"""

from __future__ import annotations

import os
from pathlib import Path

from games import DATA_DIR, GAMES, SOURCES

REPO_ROOT = Path(__file__).resolve().parent.parent


class ProvidedDataError(RuntimeError):
    """Raised when training/tuning code is pointed at provided-match material."""


def provided_stems() -> tuple[str, ...]:
    """Lower-cased name fragments that identify provided-match files."""
    stems = {game.lower() for game in GAMES}
    stems.update(entry["stem"].lower() for entry in SOURCES.values())
    return tuple(sorted(stems))


def _bundle_roots() -> list[Path]:
    """Match-bundle roots to protect: the import-time DATA_DIR and the live env var."""
    roots = {Path(DATA_DIR)}
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        roots.add(Path(env))
    roots.add(REPO_ROOT / "data")
    return [root.expanduser().resolve() for root in roots]


def provided_reason(path) -> str | None:
    """Why ``path`` is provided-match material, or None when it is not.

    Checks both the path as written and its fully resolved form, so a symlink that
    points into the bundle is caught as well as a symlink that lives inside it.
    """
    raw = os.path.expanduser(str(path))
    written = Path(os.path.abspath(raw))
    # realpath of the RAW path: the OS resolves "link/.." through the link, while
    # abspath would collapse it lexically first and hide where it really points.
    resolved = Path(os.path.realpath(raw))
    stems = provided_stems()

    for candidate in (written, resolved):
        for part in candidate.parts:
            lowered = part.lower()
            for stem in stems:
                if stem in lowered:
                    return (f"path component {part!r} contains the provided-match"
                            f" stem {stem!r}")

        for root in _bundle_roots():
            for source in SOURCES:
                # the bundle folder itself may be a symlink to somewhere else
                for folder in {root / source, Path(os.path.realpath(root / source))}:
                    if candidate == folder or folder in candidate.parents:
                        return f"inside the provided-match bundle folder {folder}"

        gt = (REPO_ROOT / "evaluation" / "gt").resolve()
        if candidate == gt or gt in candidate.parents:
            return f"inside the ground-truth folder {gt}"
    return None


def is_provided(path) -> bool:
    """True when ``path`` is provided-match material (see the module docstring)."""
    return provided_reason(path) is not None


def assert_external(*paths, what: str = "training input") -> None:
    """Raise :class:`ProvidedDataError` if any path is provided-match material.

    Call it on every video, frame folder, annotation file, dataset folder and
    dataset YAML before training or tuning code opens it.
    """
    for path in paths:
        if path is None:
            continue
        reason = provided_reason(path)
        if reason is not None:
            raise ProvidedDataError(
                f"refusing to use {path} as {what}: {reason}.\n"
                "Competition rule: any training, fitting or tuning must use EXTERNAL"
                " data only — never the provided match video, tracking or ground"
                " truth. Point this tool at an external dataset (for example under"
                " data/external/) and record its source and licence in the dataset"
                " manifest.")
