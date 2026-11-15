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

IDSSE is the ONLY tuning bed (7 matches) for the sweep.
"""

from __future__ import annotations  # postponed annotation evaluation

# stdlib: env var, sys.path bootstrap, downloads, frozen record, paths
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# arrays and table plumbing
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]  # repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # allow top-level imports (config, src)

MATCH_IDS = ["J03WPY", "J03WN1", "J03WMX", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]  # the 7 tuning matches
_COM = {"J03WN1": "DFL-COM-000001", "J03WMX": "DFL-COM-000001"}  # rest are DFL-COM-000002
_HF = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"  # download mirror
IDSSE_DIR = Path(os.environ.get("IDSSE_DIR") or (_SRC / "data" / "idsse"))  # env-overridable dir
CACHE_DIR = _SRC / "data" / ".cache"  # local parse cache


@dataclass(frozen=True)
class IdsseMatch:
    """Pipeline-ready arrays for one IDSSE match (coordinates normalized 0-1)."""

    match_id: str             # match identifier
    ball: np.ndarray       # (N, 2)
    players: np.ndarray    # (N, K, 2)
    players_s: np.ndarray  # (N, K) m/s
    slot_team: np.ndarray  # (K,) 0 home / 1 away
    half: np.ndarray       # (N,) int8 0/1/2 (from period_id)
    letters: np.ndarray    # (N,) object: 'H'/'A'/'D'/'X' derived labels
    info: dict             # parse log (units, unmapped columns, label counts...)


KMH_PER_MS = 3.6   # divisor turning km/h into m/s
_PITCH_M = (105.3, 68.0)   # same convention config.CFG applies to normalized coordinates


def speed_ratio(players: np.ndarray, players_s: np.ndarray, *, fps: int = 25,
                lag: int = 25, stride: int = 5) -> float:
    """Reported speed over position-derived speed — a unit self-check (dossier §7 U0).

    Per frame: the median over slots of the reported speed, and of the chord speed over
    `lag` frames (1 s: long enough that position jitter and slot swaps do not dominate).
    Returns the ratio of the two medians over frames with at least 10 usable slots.
    ≈ 1 when the reported column is m/s, ≈ 3.6 when it is km/h.
    """
    a, b = players[:-lag:stride], players[lag::stride]  # position pairs 1 s apart
    chord = np.hypot((b[..., 0] - a[..., 0]) * _PITCH_M[0],
                     (b[..., 1] - a[..., 1]) * _PITCH_M[1]) * (fps / lag)  # speed from positions
    rep = players_s[:-lag:stride]  # reported speed, same frames
    ok = np.isfinite(chord) & np.isfinite(rep)  # usable slots only
    enough = ok.sum(axis=1) >= 10  # frames with >=10 good slots
    if not enough.any():  # nothing measurable
        raise ValueError("speed_ratio: no frame with 10 slots carrying position and speed")  # cannot self-check
    chord = np.where(ok, chord, np.nan)[enough]  # mask bad slots, keep good frames
    rep = np.where(ok, rep, np.nan)[enough]  # same masking for reported
    denom = float(np.median(np.nanmedian(chord, axis=1)))  # typical position speed
    if denom <= 0:  # zero position speed
        raise ValueError("speed_ratio: position-derived speed is zero")  # ratio undefined
    return float(np.median(np.nanmedian(rep, axis=1))) / denom  # ~1 m/s, ~3.6 km/h


def speed_unit_factor(players: np.ndarray, players_s: np.ndarray, **kwargs) -> float:
    """Divisor that turns the reported speed column into m/s: 1.0 (already m/s) or 3.6
    (km/h). Anything in between is an error — never guess a unit."""
    ratio = speed_ratio(players, players_s, **kwargs)  # measure the unit
    if 0.7 <= ratio <= 1.4:
        return 1.0  # already m/s
    if 2.9 <= ratio <= 4.5:
        return KMH_PER_MS  # km/h
    raise ValueError(f"speed column is neither m/s nor km/h (ratio {ratio:.2f})")  # never guess


def _speeds_to_ms(players: np.ndarray, players_s: np.ndarray, info: dict) -> np.ndarray:
    """m/s speeds + the conversion logged in `info` (idempotent on m/s input)."""
    factor = speed_unit_factor(players, players_s)  # detect unit via self-check
    info["speed_units"] = "m/s"  # record outcome in the parse log
    info["speed_source_factor"] = factor  # log the divisor used
    return players_s / factor if factor != 1.0 else players_s  # divide only if km/h


def _files(match_id: str) -> tuple[Path, Path]:
    com = _COM.get(match_id, "DFL-COM-000002")  # provider match code
    meta = IDSSE_DIR / f"DFL_02_01_matchinformation_{com}_DFL-MAT-{match_id}.xml"  # metadata path
    raw = IDSSE_DIR / f"DFL_04_03_positions_raw_observed_{com}_DFL-MAT-{match_id}.xml"  # raw positions path
    return meta, raw  # metadata XML and raw positions XML


def ensure_downloaded(match_id: str) -> tuple[Path, Path]:
    meta, raw = _files(match_id)  # resolve both file paths
    for path in (meta, raw):
        if not path.exists():
            IDSSE_DIR.mkdir(parents=True, exist_ok=True)  # data dir may not exist yet
            urllib.request.urlretrieve(_HF + path.name, path)  # fetch from HF mirror
    return meta, raw  # both files now on disk


def derive_letters(period: np.ndarray, state: np.ndarray, owner,
                   home_id: str, away_id: str) -> tuple[np.ndarray, dict]:
    """Algorithmic H/A/D/X labels from provider fields. state: 'alive'/'dead'."""
    n = len(period)  # frame count
    owner_arr = owner.to_numpy() if hasattr(owner, "to_numpy") else np.asarray(owner)  # plain array
    letters = np.full(n, "X", dtype=object)  # default X (unknown/outside)
    in_match = period >= 1  # frames inside a period
    dead = state == "dead"  # provider dead-ball flag

    side = np.full(n, "", dtype=object)  # H/A per frame, blank if unknown
    side[owner_arr == home_id] = "H"  # owning club is home
    side[owner_arr == away_id] = "A"  # owning club is away

    letters[in_match & dead] = "D"  # dead ball -> D
    alive_known = in_match & ~dead & (side != "")  # alive with known owner
    letters[alive_known] = side[alive_known]  # apply the H/A labels

    # fill unknown alive owners within each period (algorithmic, no humans)
    alive_unknown = in_match & ~dead & (side == "")  # alive, owner missing
    filled = int(alive_unknown.sum())  # how many get filled (logged)
    series = pd.Series(list(letters), dtype=object)  # pandas view for filling
    series[alive_unknown] = np.nan  # blank the unknowns
    grouped = series.groupby(pd.Series(period)).ffill().groupby(pd.Series(period)).bfill()  # fill inside period only
    letters = np.array(list(grouped), dtype=object)   # writable copy
    letters[pd.isna(letters)] = "D"       # still unknown after fill
    letters[~in_match] = "X"  # outside the match stays X
    counts = {k: int((letters == k).sum()) for k in ("H", "A", "D", "X")}  # label histogram
    counts["owner_filled"] = filled  # log the fill count
    return letters, counts  # labels plus counts


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
    need = max(1, int(round(hold_s * fps)))  # hold rule in frames
    current = None                                  # team of the possession in progress
    n, i = len(out), 0  # scan length and cursor
    while i < n:  # walk runs left to right
        lab = out[i]  # label at the cursor
        if lab not in ("H", "A"):
            current = None                          # stoppage/outside: next run is fresh
            i += 1  # advance one frame
            continue  # step one frame through stoppages
        j = i  # run starts here
        while j < n and out[j] == lab:
            j += 1                                 # maximal alive run [i, j)
        if current is None or current == lab:
            current = lab                          # fresh possession or continuation
        elif (j - i) < need:
            out[i:j] = current                     # absorb short control into last team
        else:
            current = lab                          # held >= hold_s: genuine change
        i = j  # jump past the run just handled
    return out  # relabelled copy; input untouched


def _parse(match_id: str) -> IdsseMatch:
    from kloppy import sportec  # provider's tracking parser (lazy import)

    meta_path, raw_path = ensure_downloaded(match_id)  # files on disk
    ds = sportec.load_tracking(meta_data=meta_path, raw_data=raw_path, only_alive=False)  # all frames
    df = ds.to_df()  # tracking as a table
    info: dict = {"rows": len(df)}  # parse log starts with row count

    home = next((t for t in ds.metadata.teams if t.ground == "home"), None)  # home by ground flag
    away = next((t for t in ds.metadata.teams if t.ground == "away"), None)  # away by ground flag
    if home is None or away is None:  # ground flags missing
        teams = list(ds.metadata.teams)  # all listed teams
        if len(teams) < 2:  # fewer than two teams
            raise ValueError(f"{match_id}: need 2 teams, got {len(teams)}")  # cannot determine sides
        home, away = teams[0], teams[1]      # consistent sides even without ground flags
        info["side_source"] = "metadata-order-fallback"  # log the fallback
    info["teams"] = (home.name, away.name)  # team names for the log

    prefix_team: dict[str, int] = {}  # player_id -> side index
    for side_idx, team in ((0, home), (1, away)):  # 0 home, 1 away
        for player in team.players:  # every player on this side
            prefix_team[player.player_id] = side_idx  # map player to side
    groups = sorted({c.rsplit("_", 1)[0] for c in df.columns  # entity prefixes with x/y
                     if c.rsplit("_", 1)[-1] in ("x", "y")} - {"ball"})  # ball handled separately
    used = [g for g in groups if g in prefix_team]  # known players become slots
    unmapped = [g for g in groups if g not in prefix_team]  # skipped columns
    info["slots"] = len(used)  # log slot count
    info["unmapped_columns"] = unmapped  # log skipped columns

    xs = df[[f"{g}_x" for g in used]].to_numpy(dtype=np.float64)  # x per slot
    ys = df[[f"{g}_y" for g in used]].to_numpy(dtype=np.float64)  # y per slot
    players = np.stack([xs, ys], axis=-1)  # (N, K, 2) coordinates
    if all(f"{g}_s" in df.columns for g in used):
        players_s = df[[f"{g}_s" for g in used]].to_numpy(dtype=np.float64)  # provider speeds
    else:
        steps = np.gradient(players, axis=0) * 25.0  # per-frame delta x25 fps -> per second
        players_s = np.hypot(steps[..., 0], steps[..., 1])  # speed magnitude fallback
    slot_team = np.array([prefix_team[g] for g in used], dtype=np.int8)  # side per slot
    # The provider column is km/h; the pipeline's thresholds are m/s (dossier §7 U0).
    players_s = _speeds_to_ms(players, players_s, info)  # convert with unit self-check

    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=np.float64)  # (N, 2) ball coords
    coord_max = float(np.nanmax(np.abs(ball)))  # largest coordinate seen
    if coord_max > 1.5:
        raise ValueError(f"{match_id}: ball coords look like metres (max {coord_max})")  # expect 0-1

    periods = df["period_id"].to_numpy()  # period id per frame
    uniq = sorted(int(p) for p in np.unique(periods))  # periods present
    if not set(uniq) <= {1, 2}:
        raise ValueError(f"{match_id}: unexpected period ids {uniq}")  # only halves expected
    half = np.zeros(len(df), dtype=np.int8)  # default 0 = outside match
    for p in uniq:
        sel = np.flatnonzero(periods == p)  # frames of this period
        if len(sel) and not (np.diff(sel) == 1).all():
            raise ValueError(f"{match_id}: period {p} is not a contiguous frame block")  # sanity
        half[sel] = p  # tag frames with their period
    info["periods"] = uniq  # log periods

    letters, label_counts = derive_letters(  # algorithmic H/A/D/X labels
        periods.astype(np.int8),
        df["ball_state"].to_numpy(dtype=object),
        df["ball_owning_team_id"],
        home.team_id, away.team_id)
    info["labels"] = label_counts  # log label counts

    sample = players_s[np.isfinite(players_s)]  # finite speeds only
    info["player_speed_p50"] = float(np.median(sample)) if sample.size else float("nan")  # sanity stat

    return IdsseMatch(match_id=match_id, ball=ball, players=players,
                      players_s=players_s, slot_team=slot_team, half=half,
                      letters=letters, info=info)  # assemble the result


def load_match(match_id: str, force: bool = False) -> IdsseMatch:
    """Parse once (minutes), then load from the local cache (seconds)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # cache dir may not exist yet
    cache = CACHE_DIR / f"idsse_{match_id}.pkl"  # per-match pickle
    if cache.exists() and not force:
        payload = pd.read_pickle(cache)  # load cached parse
        if payload["info"].get("speed_units") != "m/s":
            # Cache written before the 2026-11-14 units fix: convert once, re-save.
            info = dict(payload["info"])
            payload["players_s"] = _speeds_to_ms(payload["players"], payload["players_s"], info)  # convert once
            sample = payload["players_s"][np.isfinite(payload["players_s"])]  # finite speeds only
            info["player_speed_p50"] = float(np.median(sample)) if sample.size else float("nan")  # refresh median
            payload["info"] = info  # store updated log
            pd.to_pickle(payload, cache)  # persist the converted cache
        return IdsseMatch(**payload)  # rebuild the frozen record
    case = _parse(match_id)  # full parse (minutes)
    pd.to_pickle(case.__dict__, cache)  # cache for next time
    return case  # freshly parsed match


if __name__ == "__main__":
    import argparse  # CLI to warm the cache

    parser = argparse.ArgumentParser(description="Parse IDSSE matches into the local cache")  # CLI parser
    parser.add_argument("--matches", nargs="*", default=MATCH_IDS)  # default: all 7
    parser.add_argument("--force", action="store_true")  # reparse even if cached
    args = parser.parse_args()  # parsed flags
    for mid in args.matches:
        print(f"parsing {mid} ...", flush=True)  # announce current match
        case = load_match(mid, force=args.force)  # parse or reuse cache
        print(f"{mid}: rows={case.info['rows']} slots={case.info['slots']} "  # per-match summary
              f"periods={case.info['periods']} labels={case.info['labels']} "
              f"speed_p50={case.info['player_speed_p50']:.2f} "
              f"unmapped={case.info['unmapped_columns']}", flush=True)
