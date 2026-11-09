"""Data loading and alignment for the BAR-ATH tactical recording.

Alignment contract (DATA.md, enforced by tests/test_alignment.py):
  * ATD: 199,319 records, frame_id 1..199319, video_frame = frame_id - 1
  * graded video frames: 11185..90651 (half 1) and 110808..190197 (half 2)
  * homography rows: frame 0..199318 == submission rows; tactical valid everywhere
  * smart tagging: frame = round(seconds * 25); 638 instances

Caches: parsed kloppy frames are pickled under data/.cache/ (gitignored via data/*).
Nothing here reads ground truth or the video.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import asset_paths, half_video_blocks  # noqa: E402

N_FRAMES = 199_319
FPS = 25
GAME = "BAR-ATH"
N_SLOTS = 46
_CACHE_DIR = os.path.join(ROOT, "data", ".cache")


def _cache_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, name)


def _read_cache(name: str):
    path = _cache_path(name)
    if os.path.exists(path):
        return pd.read_pickle(path)


def _write_cache(name: str, obj) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    pd.to_pickle(obj, _cache_path(name))


def load_tracking(force: bool = False) -> pd.DataFrame:
    """Tactical ATD as a DataFrame, one row per video frame (cached)."""
    if not force:
        cached = _read_cache("atd_tactical.pkl")
        if cached is not None:
            return cached
    from kloppy import metrica

    paths = asset_paths()
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    df = ds.to_df()
    _write_cache("atd_tactical.pkl", df)
    return df


def load_slot_teams() -> np.ndarray:
    """(46,) int8: 0 = home slot, 1 = away slot. Slots are tracklets, not players."""
    import xml.etree.ElementTree as ET

    cached = _read_cache("slot_teams.pkl")
    if cached is not None:
        return cached.to_numpy()
    root = ET.parse(asset_paths()["atd_meta"]).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if len(slots) != N_SLOTS:
        raise ValueError(f"expected {N_SLOTS} track slots, found {len(slots)}")
    out = np.array([0 if tid == team_ids[0] else 1 for tid in slots], dtype=np.int8)
    _write_cache("slot_teams.pkl", pd.Series(out))
    return out


def load_homography(force: bool = False) -> pd.DataFrame:
    if not force:
        cached = _read_cache("homography.pkl")
        if cached is not None:
            return cached
    df = pd.read_csv(asset_paths()["homography"])
    _write_cache("homography.pkl", df)
    return df


def load_smart_tagging(force: bool = False) -> pd.DataFrame:
    """638 instances with start_frame/end_frame = round(seconds * 25)."""
    if not force:
        cached = _read_cache("smart_tagging.pkl")
        if cached is not None:
            return cached
    from kloppy import sportscode

    df = sportscode.load(asset_paths()["smart_tagging"]).to_df()
    start_s = pd.to_timedelta(df["timestamp"]).dt.total_seconds()
    end_s = pd.to_timedelta(df["end_timestamp"]).dt.total_seconds()
    df = df.copy()
    df["start_frame"] = (start_s * FPS).round().astype(np.int64)
    df["end_frame"] = (end_s * FPS).round().astype(np.int64)
    _write_cache("smart_tagging.pkl", df)
    return df


def graded_mask(n: int = N_FRAMES) -> np.ndarray:
    """True on the 158,857 graded video frames (both halves)."""
    mask = np.zeros(n, dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        mask[first : last + 1] = True
    return mask


def half_labels(n: int = N_FRAMES) -> np.ndarray:
    """int8 per frame: 0 = ungraded, 1 = first half, 2 = second half."""
    out = np.zeros(n, dtype=np.int8)
    for half, first, last in half_video_blocks(GAME):
        out[first : last + 1] = half
    return out


@dataclass(frozen=True)
class MatchArrays:
    """Numpy view of the match. Coordinates stay normalized (0-1); multiply by
    pitch_length_m/pitch_width_m (config.CFG) for metres. NaN = missing."""

    ball: np.ndarray        # (N, 2) float64
    players: np.ndarray     # (N, 46, 2) float64
    players_s: np.ndarray   # (N, 46) float64, m/s as reported by the tracker
    frame_id: np.ndarray    # (N,) int64, 1..N
    slot_team: np.ndarray   # (46,) int8, 0 home / 1 away
    graded: np.ndarray      # (N,) bool
    half: np.ndarray        # (N,) int8, 0/1/2
    setpiece_starts: np.ndarray  # (M,) int64, smart-tagging SET PIECES start frames (E7)

    @property
    def n_frames(self) -> int:
        return len(self.frame_id)


def set_piece_starts() -> np.ndarray:
    """Start frames of the smart-tagging SET PIECES instances (E7 timing input)."""
    tags = load_smart_tagging()
    sel = tags[tags["code"] == "SET PIECES"]
    return np.sort(sel["start_frame"].to_numpy(dtype=np.int64))


def load_arrays(force: bool = False) -> MatchArrays:
    df = load_tracking(force=force)
    if len(df) != N_FRAMES:
        raise ValueError(f"expected {N_FRAMES} rows, got {len(df)}")

    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=np.float64)
    xs = df[[f"{i}_x" for i in range(1, N_SLOTS + 1)]].to_numpy(dtype=np.float64)
    ys = df[[f"{i}_y" for i in range(1, N_SLOTS + 1)]].to_numpy(dtype=np.float64)
    ss = df[[f"{i}_s" for i in range(1, N_SLOTS + 1)]].to_numpy(dtype=np.float64)
    players = np.stack([xs, ys], axis=-1)

    return MatchArrays(
        ball=ball,
        players=players,
        players_s=ss,
        frame_id=df["frame_id"].to_numpy(dtype=np.int64),
        slot_team=load_slot_teams(),
        graded=graded_mask(len(df)),
        half=half_labels(len(df)),
        setpiece_starts=set_piece_starts(),
    )
