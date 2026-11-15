"""Data loading and alignment for the BAR-ATH tactical recording.

Alignment contract (DATA.md, enforced by tests/test_alignment.py):
  * ATD: 199,319 records, frame_id 1..199319, video_frame = frame_id - 1
  * graded video frames: 11185..90651 (half 1) and 110808..190197 (half 2)
  * homography rows: frame 0..199318 == submission rows; tactical valid everywhere
  * smart tagging: frame = round(seconds * 25); 638 instances

Caches: parsed kloppy frames are pickled under data/.cache/ (gitignored via data/*).
Nothing here reads ground truth or the video.
"""

from __future__ import annotations  # postponed annotation evaluation

import os  # path helpers
import sys  # sys.path bootstrap
from dataclasses import dataclass  # plain data container

import numpy as np  # array math
import pandas as pd  # tables and pickles

_SRC = os.path.dirname(os.path.abspath(__file__))  # this file's directory
ROOT = os.path.dirname(_SRC)  # repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # make repo root importable

from games import asset_paths, half_video_blocks  # noqa: E402  # file paths + graded blocks

N_FRAMES = 199_319  # submission row count
FPS = 25  # tactical video frame rate
GAME = "BAR-ATH"  # game key used by games.py
N_SLOTS = 46  # player track slots per frame
_CACHE_DIR = os.path.join(ROOT, "data", ".cache")  # pickled cache for fast reloads


def _cache_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, name)  # cache file location


def _read_cache(name: str):
    path = _cache_path(name)  # resolve cache path
    if os.path.exists(path):  # cache file present
        return pd.read_pickle(path)  # load cached object
    # cache miss: falls through returning None


def _write_cache(name: str, obj) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)  # ensure cache dir exists
    pd.to_pickle(obj, _cache_path(name))  # persist for next run


def load_tracking(force: bool = False) -> pd.DataFrame:
    """Tactical ATD as a DataFrame, one row per video frame (cached)."""
    if not force:
        cached = _read_cache("atd_tactical.pkl")  # try cache first
        if cached is not None:
            return cached  # reuse pickled ATD
    from kloppy import metrica  # ATD parser, imported lazily

    paths = asset_paths()  # locate ATD files
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])  # parse feed
    df = ds.to_df()  # flatten to DataFrame
    _write_cache("atd_tactical.pkl", df)  # cache for reloads
    return df  # parsed tracking table


def load_slot_teams() -> np.ndarray:
    """(46,) int8: 0 = home slot, 1 = away slot. Slots are tracklets, not players."""
    import xml.etree.ElementTree as ET  # meta XML parser

    cached = _read_cache("slot_teams.pkl")  # try cache first
    if cached is not None:
        return cached.to_numpy()  # reuse cached team map
    root = ET.parse(asset_paths()["atd_meta"]).getroot()  # read ATD meta XML
    team_ids = [team.get("id") for team in root.iter("Team")]  # the two team ids
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams, found {len(team_ids)}")  # contract check
    slots = [player.get("teamId") for player in root.iter("Player")]  # team id per slot
    if len(slots) != N_SLOTS:
        raise ValueError(f"expected {N_SLOTS} track slots, found {len(slots)}")  # contract check
    out = np.array([0 if tid == team_ids[0] else 1 for tid in slots], dtype=np.int8)  # first team = home
    _write_cache("slot_teams.pkl", pd.Series(out))  # cache result
    return out  # home/away per slot


def load_homography(force: bool = False) -> pd.DataFrame:
    if not force:
        cached = _read_cache("homography.pkl")  # try cache first
        if cached is not None:
            return cached  # reuse pickled homography
    df = pd.read_csv(asset_paths()["homography"])  # one row per frame
    _write_cache("homography.pkl", df)  # cache for reloads
    return df  # homography table


def load_smart_tagging(force: bool = False) -> pd.DataFrame:
    """638 instances with start_frame/end_frame = round(seconds * 25)."""
    if not force:
        cached = _read_cache("smart_tagging.pkl")  # try cache first
        if cached is not None:
            return cached  # reuse pickled tags
    from kloppy import sportscode  # smart-tagging parser, lazy

    df = sportscode.load(asset_paths()["smart_tagging"]).to_df()  # load tag instances
    start_s = pd.to_timedelta(df["timestamp"]).dt.total_seconds()  # start timestamps in seconds
    end_s = pd.to_timedelta(df["end_timestamp"]).dt.total_seconds()  # end timestamps in seconds
    df = df.copy()  # own the frame before adding columns
    df["start_frame"] = (start_s * FPS).round().astype(np.int64)  # seconds → frame index
    df["end_frame"] = (end_s * FPS).round().astype(np.int64)  # seconds → frame index
    _write_cache("smart_tagging.pkl", df)  # cache result
    return df  # tags with frame columns


def graded_mask(n: int = N_FRAMES) -> np.ndarray:
    """True on the 158,857 graded video frames (both halves)."""
    mask = np.zeros(n, dtype=bool)  # start all ungraded
    for _half, first, last in half_video_blocks(GAME):
        mask[first : last + 1] = True  # mark half's graded block (inclusive)
    return mask  # graded-frame mask


def half_labels(n: int = N_FRAMES) -> np.ndarray:
    """int8 per frame: 0 = ungraded, 1 = first half, 2 = second half."""
    out = np.zeros(n, dtype=np.int8)  # 0 = ungraded by default
    for half, first, last in half_video_blocks(GAME):
        out[first : last + 1] = half  # stamp half number on its block
    return out  # half number per frame


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
    kickoff_starts: np.ndarray   # (K,) int64, start frames of `Kick Off (Start)` tags (K1)

    @property
    def n_frames(self) -> int:
        return len(self.frame_id)  # total frame count


def set_piece_starts() -> np.ndarray:
    """Start frames of the smart-tagging SET PIECES instances (E7 timing input)."""
    tags = load_smart_tagging()  # all smart tags
    sel = tags[tags["code"] == "SET PIECES"]  # keep set-piece instances
    return np.sort(sel["start_frame"].to_numpy(dtype=np.int64))  # sorted start frames


KICKOFF_START_TYPE = "kick off (start)"  # tag Type value marking kickoffs


def kickoff_starts() -> np.ndarray:
    """Start frames of the SET PIECES instances typed `Kick Off (Start)` (K1, dossier §7 P7).

    Matched case-insensitively (DATA.md: codes are upper case, tag values title case). An
    export without a `Type` tag group yields an empty array — the rule is then a no-op.
    """
    tags = load_smart_tagging()  # all smart tags
    type_col = next((c for c in tags.columns if str(c).strip().lower() == "type"), None)  # find Type column
    if type_col is None:
        return np.empty(0, dtype=np.int64)  # no Type column → no kickoffs to find
    is_set_piece = tags["code"].astype(str).str.strip().str.upper() == "SET PIECES"  # set-piece rows
    is_kickoff = tags[type_col].astype(str).str.strip().str.lower() == KICKOFF_START_TYPE  # typed kickoff
    return np.sort(tags.loc[is_set_piece & is_kickoff, "start_frame"].to_numpy(dtype=np.int64))  # sorted starts


def load_arrays(force: bool = False) -> MatchArrays:
    df = load_tracking(force=force)  # one row per frame
    if len(df) != N_FRAMES:
        raise ValueError(f"expected {N_FRAMES} rows, got {len(df)}")  # contract check

    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=np.float64)  # (N, 2) ball coords
    xs = df[[f"{i}_x" for i in range(1, N_SLOTS + 1)]].to_numpy(dtype=np.float64)  # per-slot x
    ys = df[[f"{i}_y" for i in range(1, N_SLOTS + 1)]].to_numpy(dtype=np.float64)  # per-slot y
    ss = df[[f"{i}_s" for i in range(1, N_SLOTS + 1)]].to_numpy(dtype=np.float64)  # per-slot speeds
    players = np.stack([xs, ys], axis=-1)  # (N, 46, 2) xy pairs

    return MatchArrays(  # assemble the match bundle
        ball=ball,
        players=players,
        players_s=ss,
        frame_id=df["frame_id"].to_numpy(dtype=np.int64),
        slot_team=load_slot_teams(),
        graded=graded_mask(len(df)),
        half=half_labels(len(df)),
        setpiece_starts=set_piece_starts(),
        kickoff_starts=kickoff_starts(),
    )
