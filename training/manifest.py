"""External-dataset contract for detector training: one manifest, one annotation format.

The training tools do not know (and must not guess) which public dataset is used.
They read a small JSON *manifest* that names the dataset, its licence and source —
the provenance the technical report has to disclose — and lists the games:

    {
      "dataset": "<public dataset name>",
      "license": "<licence name or URL>",
      "source_url": "<where it was downloaded from>",
      "notes": "optional free text",
      "games": [
        {"id": "4", "video": "game4/video.mp4", "annotations": "game4/ball.csv"},
        {"id": "6", "frames_dir": "game6/img1", "annotations": "game6/ball.csv",
         "frame_offset": 1, "fps": 25}
      ]
    }

Paths are relative to the manifest file (absolute paths are accepted). A game is
either a ``video`` file or a ``frames_dir`` of images whose natural sort order
(``2.jpg`` before ``10.jpg``) is the frame order; ``frame_offset`` is the annotation
frame number of the first frame. It defaults to 0 for a video and is REQUIRED for a
``frames_dir`` (MOT-style ``img1/000001.jpg`` folders are 1-based: a wrong offset
shifts every label by one frame, silently). Game ids may use letters, digits,
``.``, ``_`` and ``-`` only (they become file names).

Canonical ball annotation — one CSV per game, one row per frame where the ball is
visible, coordinates in pixels of the game's native resolution:

    frame,x,y[,w,h][,visible]

``x, y`` is the ball centre. ``w, h`` (box size in pixels) is optional; rows with
``visible == 0`` are dropped. :func:`convert_mot` and :func:`convert_csv` write
this format from MOT-style ``gt.txt`` files and from arbitrary CSV column layouts.

Every path is passed through :func:`training.compliance.assert_external`, and a
manifest without ``dataset`` / ``license`` / ``source_url`` is rejected: an
undocumented dataset cannot be defended in the technical report.

Only the standard library, numpy and pandas are needed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from training.compliance import assert_external

REQUIRED_TOP = ("dataset", "license", "source_url", "games")
# A value that IS one of these (whole value, case-insensitive) is a placeholder. A
# substring test would reject real text such as "uso academico" or "todorov-lab".
PLACEHOLDER_RE = re.compile(r"(todo|tbd|unknown|n/?a|none|null|\?+|x{2,}|-+|fill me.*)")
GAME_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
CANONICAL_COLUMNS = ("frame", "x", "y", "w", "h")


class ManifestError(ValueError):
    """The manifest or one of its annotation files breaks the contract."""


@dataclass(frozen=True)
class Game:
    """One external game: where its frames and its ball annotations live."""

    id: str
    annotations: Path
    video: Path | None = None
    frames_dir: Path | None = None
    frame_offset: int = 0
    fps: float | None = None

    @property
    def source(self) -> Path:
        """The frame source, whichever of video / frames_dir is set."""
        return self.video if self.video is not None else self.frames_dir


@dataclass(frozen=True)
class Manifest:
    """A validated dataset manifest."""

    path: Path
    dataset: str
    license: str
    source_url: str
    games: tuple[Game, ...]
    notes: str = ""
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    def game(self, game_id) -> Game:
        """The game with this id, or a readable error naming the ones that exist."""
        wanted = str(game_id)
        for game in self.games:
            if game.id == wanted:
                return game
        raise ManifestError(f"no game {wanted!r} in {self.path};"
                            f" have {[g.id for g in self.games]}")

    def ids(self) -> list[str]:
        """Game ids in manifest order."""
        return [game.id for game in self.games]


def _is_placeholder(value) -> bool:
    """True for a missing, non-text or template value (``<name>``, ``TBD``, ``n/a``...)."""
    if not isinstance(value, str):
        return True
    text = value.strip().lower()
    return ((not text) or (text.startswith("<") and text.endswith(">"))
            or PLACEHOLDER_RE.fullmatch(text) is not None)


def load_manifest(path, check_files: bool = True) -> Manifest:
    """Read and validate a dataset manifest.

    Raises :class:`ManifestError` listing every problem at once, and
    ``ProvidedDataError`` if any path is provided-match material. With
    ``check_files=False`` the files are not required to exist (useful for a dry run
    before the dataset is downloaded); the compliance check still runs.
    """
    path = Path(path).expanduser().resolve()
    assert_external(path, what="dataset manifest")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: the manifest must be a JSON object")

    problems = []
    for key in REQUIRED_TOP:
        if key not in raw:
            problems.append(f"missing top-level key {key!r}")
    for key in ("dataset", "license", "source_url"):
        if key in raw and _is_placeholder(raw[key]):
            problems.append(f"{key!r} is empty or still a placeholder: {raw[key]!r}"
                            " — the dataset's identity, licence and source must be"
                            " written down before anything is trained on it")
    entries = raw.get("games", [])
    if not isinstance(entries, list) or not entries:
        problems.append("'games' must be a non-empty list")
        entries = []

    base = path.parent
    games: list[Game] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"games[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{label}: must be an object")
            continue
        if "id" not in entry:
            problems.append(f"{label}: missing 'id'")
            continue
        game_id = str(entry["id"])
        if not GAME_ID_RE.fullmatch(game_id):
            problems.append(f"{label}: game id {game_id!r} may only use letters, digits,"
                            " '.', '_' and '-'")
            continue
        if game_id in seen:
            problems.append(f"{label}: duplicate game id {game_id!r}")
        seen.add(game_id)

        has_video, has_dir = "video" in entry, "frames_dir" in entry
        if has_video == has_dir:
            problems.append(f"{label} (id {game_id}): give exactly one of"
                            " 'video' or 'frames_dir'")
            continue
        if "annotations" not in entry:
            problems.append(f"{label} (id {game_id}): missing 'annotations'")
            continue

        def resolve(value):
            candidate = Path(str(value)).expanduser()
            return candidate if candidate.is_absolute() else (base / candidate)

        video = resolve(entry["video"]) if has_video else None
        frames_dir = resolve(entry["frames_dir"]) if has_dir else None
        annotations = resolve(entry["annotations"])
        assert_external(video, frames_dir, annotations,
                        what=f"external game {game_id}")

        if check_files:
            if video is not None and not video.is_file():
                problems.append(f"{label} (id {game_id}): video not found: {video}")
            if frames_dir is not None and not frames_dir.is_dir():
                problems.append(f"{label} (id {game_id}): frames_dir not found:"
                                f" {frames_dir}")
            if not annotations.is_file():
                problems.append(f"{label} (id {game_id}): annotations not found:"
                                f" {annotations}")

        if has_dir and "frame_offset" not in entry:
            problems.append(f"{label} (id {game_id}): 'frame_offset' is required for a"
                            " frames_dir — the annotation frame number of the first"
                            " image (1 for MOT-style img1/000001.jpg)")
            continue
        offset = entry.get("frame_offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            problems.append(f"{label} (id {game_id}): 'frame_offset' must be an integer"
                            f" >= 0, got {offset!r}")
            continue
        fps = entry.get("fps")
        if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float))
                                or fps <= 0):
            problems.append(f"{label} (id {game_id}): 'fps' must be a positive number,"
                            f" got {fps!r}")
            continue
        games.append(Game(
            id=game_id, annotations=annotations, video=video, frames_dir=frames_dir,
            frame_offset=offset, fps=float(fps) if fps is not None else None))

    if problems:
        raise ManifestError(f"{path}: invalid manifest:\n  - " + "\n  - ".join(problems))

    return Manifest(path=path, dataset=str(raw["dataset"]), license=str(raw["license"]),
                    source_url=str(raw["source_url"]), games=tuple(games),
                    notes=str(raw.get("notes", "")), raw=raw)


def load_annotations(game: Game) -> pd.DataFrame:
    """Canonical ball annotations of one game: ``frame, x, y, w, h`` sorted by frame.

    ``w`` / ``h`` are NaN when the file has no box sizes. Rows with ``visible == 0``
    or a non-finite centre are dropped. A frame annotated more than once is an error
    (the detector is single-ball), as is a negative coordinate or box size.
    """
    assert_external(game.annotations, what=f"annotations of game {game.id}")
    table = pd.read_csv(game.annotations)
    table.columns = [str(c).strip().lower() for c in table.columns]
    missing = [c for c in ("frame", "x", "y") if c not in table.columns]
    if missing:
        raise ManifestError(
            f"{game.annotations}: missing column(s) {missing}; the canonical format"
            " is frame,x,y[,w,h][,visible] — see training.manifest.convert_csv /"
            " convert_mot")

    if "visible" in table.columns:
        table = table[table["visible"].astype(float) > 0]
    for column in ("w", "h"):
        if column not in table.columns:
            table[column] = np.nan

    out = table[list(CANONICAL_COLUMNS)].copy()
    for column in CANONICAL_COLUMNS:
        try:
            out[column] = pd.to_numeric(out[column], errors="raise").astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{game.annotations}: column {column!r} is not"
                                f" numeric ({exc})") from exc
    out = out[np.isfinite(out["frame"]) & np.isfinite(out["x"]) & np.isfinite(out["y"])]
    if (out["frame"] % 1 != 0).any():
        raise ManifestError(f"{game.annotations}: non-integer frame numbers")
    if (out["frame"] < 0).any():
        raise ManifestError(f"{game.annotations}: negative frame numbers")
    out["frame"] = out["frame"].astype(np.int64)

    if (out[["x", "y"]] < 0).any().any():
        raise ManifestError(f"{game.annotations}: negative ball coordinates")
    if (out[["w", "h"]] < 0).any().any():
        raise ManifestError(f"{game.annotations}: negative box size")
    duplicated = out["frame"].duplicated()
    if duplicated.any():
        frames = out.loc[duplicated, "frame"].head(5).tolist()
        raise ManifestError(
            f"{game.annotations}: {int(duplicated.sum())} frame(s) annotated more"
            f" than once (e.g. {frames}); one ball per frame is required")
    return out.sort_values("frame").reset_index(drop=True)


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def list_frame_files(frames_dir) -> list[Path]:
    """Image files of a frames_dir in frame order.

    Natural sort (``2.jpg`` before ``10.jpg``; zero-padded names are unaffected), and
    hidden files such as macOS ``._0001.jpg`` are skipped.
    """
    folder = Path(frames_dir)
    files = [p for p in folder.iterdir()
             if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")]
    return sorted(files, key=_natural_key)


def leave_one_game_out(manifest: Manifest) -> list[dict]:
    """Fold table: fold k validates on game k (manifest order) and trains on the rest."""
    ids = manifest.ids()
    if len(ids) < 2:
        raise ManifestError("leave-one-game-out needs at least two games")
    return [{"fold": k, "val_games": [held_out],
             "train_games": [g for g in ids if g != held_out]}
            for k, held_out in enumerate(ids)]


def resolve_split(manifest: Manifest, train_games=None, val_games=None,
                  fold: int | None = None) -> tuple[list[str], list[str]]:
    """``(train_ids, val_ids)`` from explicit lists or a leave-one-game-out fold.

    The two lists must be disjoint and non-empty: a game on both sides leaks the
    validation set into training and makes every reported metric meaningless.
    """
    if fold is not None:
        if train_games or val_games:
            raise ManifestError("give either --fold or explicit train/val games")
        table = leave_one_game_out(manifest)
        if not 0 <= fold < len(table):
            raise ManifestError(f"fold {fold} out of range 0..{len(table) - 1}")
        train, val = table[fold]["train_games"], table[fold]["val_games"]
    else:
        train = [str(g) for g in (train_games or [])]
        val = [str(g) for g in (val_games or [])]
    for game_id in train + val:
        manifest.game(game_id)
    if not train or not val:
        raise ManifestError("both a train and a validation game list are required")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ManifestError(f"game(s) {overlap} are in both train and validation")
    return train, val


def fingerprint(manifest: Manifest) -> str:
    """sha256 over the manifest content and every annotation file's bytes.

    Recorded in each run's provenance so a trained checkpoint can be tied to the
    exact labels it saw. Videos are identified by name and size only (hashing
    gigabytes on every run is not worth it).
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest.raw, sort_keys=True).encode("utf-8"))
    for game in manifest.games:
        digest.update(game.id.encode("utf-8"))
        if game.annotations.is_file():
            digest.update(game.annotations.read_bytes())
        source = game.source
        if source is not None and source.is_file():
            digest.update(f"{source.name}:{source.stat().st_size}".encode("utf-8"))
        elif source is not None and source.is_dir():
            files = list_frame_files(source)
            if files:
                digest.update(f"{len(files)}:{files[0].name}:{files[-1].name}"
                              .encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Converters to the canonical annotation CSV
# --------------------------------------------------------------------------- #
def _write_canonical(table: pd.DataFrame, out_csv) -> pd.DataFrame:
    out_csv = Path(out_csv)
    assert_external(out_csv, what="converted annotations")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table = table.sort_values("frame").reset_index(drop=True)
    table.to_csv(out_csv, index=False)
    return table


def convert_csv(src, out_csv, frame: str, x: str, y: str, w: str | None = None,
                h: str | None = None, visible: str | None = None,
                top_left: bool = False, normalized_size: tuple[int, int] | None = None,
                frame_add: int = 0) -> pd.DataFrame:
    """Rewrite any CSV as the canonical format by naming its columns.

    ``top_left=True`` means ``x, y`` are the box's top-left corner (needs ``w, h``)
    and are shifted to the centre. ``normalized_size=(width, height)`` means the
    coordinates are fractions of the frame and are scaled to pixels. ``frame_add``
    is added to the frame number (e.g. -1 to turn 1-based into 0-based).
    """
    assert_external(src, what="source annotations")
    source = pd.read_csv(src)
    for column in (frame, x, y, w, h, visible):
        if column is not None and column not in source.columns:
            raise ManifestError(f"{src}: no column {column!r};"
                                f" have {list(source.columns)}")
    if top_left and (w is None or h is None):
        raise ManifestError("top_left=True needs the w and h columns")

    out = pd.DataFrame({"frame": source[frame].astype(np.int64) + int(frame_add),
                        "x": source[x].astype(np.float64),
                        "y": source[y].astype(np.float64)})
    out["w"] = source[w].astype(np.float64) if w is not None else np.nan
    out["h"] = source[h].astype(np.float64) if h is not None else np.nan
    if normalized_size is not None:
        width, height = normalized_size
        out["x"] *= width
        out["w"] *= width
        out["y"] *= height
        out["h"] *= height
    if top_left:
        out["x"] += out["w"] / 2.0
        out["y"] += out["h"] / 2.0
    if visible is not None:
        out = out[source[visible].astype(float).to_numpy() > 0]
    return _write_canonical(out, out_csv)


def convert_mot(src, out_csv, track_ids=None, class_ids=None, class_column: int = 7,
                frame_add: int = 0) -> pd.DataFrame:
    """Rewrite a MOT-style ``gt.txt`` (frame,id,left,top,w,h,conf,class,...) as canonical.

    Keep the rows whose track id is in ``track_ids`` and/or whose class (column
    ``class_column``, 0-based) is in ``class_ids`` — whichever identifies the ball in
    that dataset. If several ball rows share a frame, the largest box is kept.
    """
    assert_external(src, what="source annotations")
    raw = pd.read_csv(src, header=None)
    if raw.shape[1] < 6:
        raise ManifestError(f"{src}: expected at least 6 MOT columns, got {raw.shape[1]}")
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in range(6)):
        raise ManifestError(f"{src}: non-numeric MOT columns — a header row? Remove it,"
                            " or use convert_csv with column names")
    keep = np.ones(len(raw), dtype=bool)
    if track_ids is not None:
        keep &= raw[1].isin(list(track_ids)).to_numpy()
    if class_ids is not None:
        if raw.shape[1] <= class_column:
            raise ManifestError(f"{src}: no class column {class_column}")
        keep &= raw[class_column].isin(list(class_ids)).to_numpy()
    if track_ids is None and class_ids is None:
        raise ManifestError("give track_ids and/or class_ids to pick the ball rows")
    rows = raw[keep]
    if not len(rows):
        raise ManifestError(
            f"{src}: no row matched track_ids={track_ids} class_ids={class_ids}"
            f" (track ids present: {sorted(raw[1].unique().tolist())[:10]})")

    out = pd.DataFrame({
        "frame": rows[0].astype(np.int64) + int(frame_add),
        "x": rows[2].astype(np.float64) + rows[4].astype(np.float64) / 2.0,
        "y": rows[3].astype(np.float64) + rows[5].astype(np.float64) / 2.0,
        "w": rows[4].astype(np.float64),
        "h": rows[5].astype(np.float64),
    })
    out["_area"] = out["w"] * out["h"]
    out = (out.sort_values(["frame", "_area"], ascending=[True, False])
              .drop_duplicates("frame").drop(columns="_area"))
    return _write_canonical(out, out_csv)
