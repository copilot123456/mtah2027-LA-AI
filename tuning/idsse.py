"""IDSSE external test bed: load, cache, adapt to the pipeline's arrays, derive labels.

License: CC BY 4.0 (DFL via Bassek et al., Scientific Data 2025). Files are never
committed: set IDSSE_DIR to wherever they live, or leave the default `data/idsse/`
(gitignored with the rest of data/; a symlink there works). Missing files are fetched
from the HF mirror on first use.
Parsed results are cached under the repo's gitignored data/.cache/ (pickle, local only —
the plan said parquet; pickle is the same-env cache actually available and keeps the
dependency list at numpy+pandas).

Labels are derived ALGORITHMICALLY from the provider's automatic fields — no humans:
    ball_state == 'dead'              -> D
    alive + ball_owning_team_id       -> H/A by the owning club (home/away from metadata)
    alive + unknown owner             -> forward/back-fill within the period, else D
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MATCH_IDS = ["J03WPY", "J03WN1", "J03WMX", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
_COM = {"J03WN1": "DFL-COM-000001", "J03WMX": "DFL-COM-000001"}  # rest are DFL-COM-000002
_HF = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
IDSSE_DIR = Path(os.environ.get("IDSSE_DIR") or (_SRC / "data" / "idsse"))
CACHE_DIR = _SRC / "data" / ".cache"


@dataclass(frozen=True)
class IdsseMatch:
    """Pipeline-ready arrays for one IDSSE match (coordinates normalized 0-1)."""

    match_id: str
    ball: np.ndarray       # (N, 2)
    players: np.ndarray    # (N, K, 2)
    players_s: np.ndarray  # (N, K) m/s
    slot_team: np.ndarray  # (K,) 0 home / 1 away
    half: np.ndarray       # (N,) int8 0/1/2 (from period_id)
    letters: np.ndarray    # (N,) object: 'H'/'A'/'D'/'X' derived labels
    info: dict             # parse log (units, unmapped columns, label counts...)


def _files(match_id: str) -> tuple[Path, Path]:
    com = _COM.get(match_id, "DFL-COM-000002")
    meta = IDSSE_DIR / f"DFL_02_01_matchinformation_{com}_DFL-MAT-{match_id}.xml"
    raw = IDSSE_DIR / f"DFL_04_03_positions_raw_observed_{com}_DFL-MAT-{match_id}.xml"
    return meta, raw


def ensure_downloaded(match_id: str) -> tuple[Path, Path]:
    meta, raw = _files(match_id)
    for path in (meta, raw):
        if not path.exists():
            IDSSE_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_HF + path.name, path)
    return meta, raw


def derive_letters(period: np.ndarray, state: np.ndarray, owner,
                   home_id: str, away_id: str) -> tuple[np.ndarray, dict]:
    """Algorithmic H/A/D/X labels from provider fields. state: 'alive'/'dead'."""
    n = len(period)
    owner_arr = owner.to_numpy() if hasattr(owner, "to_numpy") else np.asarray(owner)
    letters = np.full(n, "X", dtype=object)
    in_match = period >= 1
    dead = state == "dead"

    side = np.full(n, "", dtype=object)
    side[owner_arr == home_id] = "H"
    side[owner_arr == away_id] = "A"

    letters[in_match & dead] = "D"
    alive_known = in_match & ~dead & (side != "")
    letters[alive_known] = side[alive_known]

    # fill unknown alive owners within each period (algorithmic, no humans)
    alive_unknown = in_match & ~dead & (side == "")
    filled = int(alive_unknown.sum())
    series = pd.Series(list(letters), dtype=object)
    series[alive_unknown] = np.nan
    grouped = series.groupby(pd.Series(period)).ffill().groupby(pd.Series(period)).bfill()
    letters = np.array(list(grouped), dtype=object)   # writable copy
    letters[pd.isna(letters)] = "D"       # still unknown after fill
    letters[~in_match] = "X"
    counts = {k: int((letters == k).sum()) for k in ("H", "A", "D", "X")}
    counts["owner_filled"] = filled
    return letters, counts


def derive_letters_controlled(letters: np.ndarray, fps: int = 25,
                              hold_s: float = 3.0) -> np.ndarray:
    """Challenge-definition labels (E1) — pre-declared: dossier §7 P6 (2026-11-09).

    Rule (plan §0 "literal rule, sequential, chaining merges"; T a priori from
    CHALLENGE.md "a player keeps the ball for a few seconds"):

      * maximal H/A runs are alive runs; D or X is a stoppage between them;
      * the first alive run after a stoppage starts a fresh possession (never absorbed);
      * an alive run shorter than `hold_s`, directly preceded by another alive run with no
        stoppage between, is relabelled to the preceding team ("loose or contested ball
        stays with the last team"); the absorbing team persists, so consecutive short runs
        chain into the same possession;
      * an alive run >= `hold_s` is a genuine possession change;
      * D/X frames never change (dead share invariant by construction).

    Pure: the input array is not modified. `hold_s` is pinned a priori (primary 3 s,
    sensitivity {2, 5} reported — never selected on the public window).
    """
    out = np.array(letters, dtype=object)          # writable copy
    need = max(1, int(round(hold_s * fps)))
    current = None                                  # team of the possession in progress
    n, i = len(out), 0
    while i < n:
        lab = out[i]
        if lab not in ("H", "A"):
            current = None                          # stoppage/outside: next run is fresh
            i += 1
            continue
        j = i
        while j < n and out[j] == lab:
            j += 1                                 # maximal alive run [i, j)
        if current is None or current == lab:
            current = lab                          # fresh possession or continuation
        elif (j - i) < need:
            out[i:j] = current                     # absorb short control into last team
        else:
            current = lab                          # held >= hold_s: genuine change
        i = j
    return out


def _parse(match_id: str) -> IdsseMatch:
    from kloppy import sportec

    meta_path, raw_path = ensure_downloaded(match_id)
    ds = sportec.load_tracking(meta_data=meta_path, raw_data=raw_path, only_alive=False)
    df = ds.to_df()
    info: dict = {"rows": len(df)}

    home = next((t for t in ds.metadata.teams if t.ground == "home"), None)
    away = next((t for t in ds.metadata.teams if t.ground == "away"), None)
    if home is None or away is None:
        teams = list(ds.metadata.teams)
        if len(teams) < 2:
            raise ValueError(f"{match_id}: need 2 teams, got {len(teams)}")
        home, away = teams[0], teams[1]      # consistent sides even without ground flags
        info["side_source"] = "metadata-order-fallback"
    info["teams"] = (home.name, away.name)

    prefix_team: dict[str, int] = {}
    for side_idx, team in ((0, home), (1, away)):
        for player in team.players:
            prefix_team[player.player_id] = side_idx
    groups = sorted({c.rsplit("_", 1)[0] for c in df.columns
                     if c.rsplit("_", 1)[-1] in ("x", "y")} - {"ball"})
    used = [g for g in groups if g in prefix_team]
    unmapped = [g for g in groups if g not in prefix_team]
    info["slots"] = len(used)
    info["unmapped_columns"] = unmapped

    xs = df[[f"{g}_x" for g in used]].to_numpy(dtype=np.float64)
    ys = df[[f"{g}_y" for g in used]].to_numpy(dtype=np.float64)
    players = np.stack([xs, ys], axis=-1)
    if all(f"{g}_s" in df.columns for g in used):
        players_s = df[[f"{g}_s" for g in used]].to_numpy(dtype=np.float64)
    else:
        steps = np.gradient(players, axis=0) * 25.0
        players_s = np.hypot(steps[..., 0], steps[..., 1])
    slot_team = np.array([prefix_team[g] for g in used], dtype=np.int8)

    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=np.float64)
    coord_max = float(np.nanmax(np.abs(ball)))
    if coord_max > 1.5:
        raise ValueError(f"{match_id}: ball coords look like metres (max {coord_max})")

    periods = df["period_id"].to_numpy()
    uniq = sorted(int(p) for p in np.unique(periods))
    if not set(uniq) <= {1, 2}:
        raise ValueError(f"{match_id}: unexpected period ids {uniq}")
    half = np.zeros(len(df), dtype=np.int8)
    for p in uniq:
        sel = np.flatnonzero(periods == p)
        if len(sel) and not (np.diff(sel) == 1).all():
            raise ValueError(f"{match_id}: period {p} is not a contiguous frame block")
        half[sel] = p
    info["periods"] = uniq

    letters, label_counts = derive_letters(
        periods.astype(np.int8),
        df["ball_state"].to_numpy(dtype=object),
        df["ball_owning_team_id"],
        home.team_id, away.team_id)
    info["labels"] = label_counts

    sample = players_s[np.isfinite(players_s)]
    info["player_speed_p50"] = float(np.median(sample)) if sample.size else float("nan")

    return IdsseMatch(match_id=match_id, ball=ball, players=players,
                      players_s=players_s, slot_team=slot_team, half=half,
                      letters=letters, info=info)


def load_match(match_id: str, force: bool = False) -> IdsseMatch:
    """Parse once (minutes), then load from the local cache (seconds)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"idsse_{match_id}.pkl"
    if cache.exists() and not force:
        return IdsseMatch(**pd.read_pickle(cache))
    case = _parse(match_id)
    pd.to_pickle(case.__dict__, cache)
    return case


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse IDSSE matches into the local cache")
    parser.add_argument("--matches", nargs="*", default=MATCH_IDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for mid in args.matches:
        print(f"parsing {mid} ...", flush=True)
        case = load_match(mid, force=args.force)
        print(f"{mid}: rows={case.info['rows']} slots={case.info['slots']} "
              f"periods={case.info['periods']} labels={case.info['labels']} "
              f"speed_p50={case.info['player_speed_p50']:.2f} "
              f"unmapped={case.info['unmapped_columns']}", flush=True)
