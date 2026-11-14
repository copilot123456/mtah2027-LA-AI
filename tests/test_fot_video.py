"""Adapter tests: the FoT v2 notebook functions expressed as framework stages.

Design rule: no gated data, no video, no model checkpoints. The I/O boundary
(video file, homography CSV, RT-DETR/FoT weights, seeded clip pick) is monkeypatched
with deterministic synthetics; everything downstream of it — the ATD-merge rules,
the BallTrack contract, and the unchanged emission/decode/x_policy/summary stages —
runs for real, which is exactly the compatibility claim being tested.

Heavy dependencies are optional: torch/cv2 tests skip themselves where those
packages are absent (requirements-dev.txt does not pin them).
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from config import CFG
from src.ball_fill import (SOURCE_ATD, SOURCE_DETECTOR, SOURCE_KINEMATIC, SOURCE_NONE,
                           BallTrack)
from src.pipeline import Ctx, run, validate_spec

ROOT = Path(__file__).resolve().parents[1]

N_SYNTH = 600          # frames in the synthetic match used by the handler tests
FPS = 25
# Pitch occupies a sub-rectangle of the frame, so margin masks are strictly larger.
H_SYNTH = np.array([[1600.0, 0.0, 160.0],
                    [0.0, 900.0, 90.0],
                    [0.0, 0.0, 1.0]])


def _homography(n):
    """Identity-style homography (normalised pitch -> pixels) for n frames."""
    matrices = np.tile(H_SYNTH, (n, 1, 1))
    reproj = np.zeros(n)
    valid = np.ones(n, dtype=bool)
    return matrices, reproj, valid


def _synthetic_arrays(n=N_SYNTH):
    """MatchArrays look-alike: two halves, sparse ATD ball, visible players."""
    half = np.zeros(n, dtype=np.int8)
    half[100:400] = 1
    half[400:600] = 2

    ball = np.full((n, 2), np.nan)
    atd_frames = [100, 150, 200, 250, 300, 350]
    for f in atd_frames:
        ball[f] = (0.4 + f / 10000.0, 0.5)

    rng = np.random.default_rng(0)
    players = rng.uniform(0.05, 0.95, size=(n, 46, 2))
    players_s = rng.uniform(0.0, 6.0, size=(n, 46))
    return SimpleNamespace(
        ball=ball,
        players=players,
        players_s=players_s,
        frame_id=np.arange(1, n + 1, dtype=np.int64),
        slot_team=np.array([0] * 23 + [1] * 23, dtype=np.int8),
        graded=half > 0,
        half=half,
        n_frames=n,
    )


def _fake_detect(video, matrices, reproj, valid, start, end, cfg):
    """Every frame detected except a periodic hole; constant position."""
    history = {}
    for f in range(start, end + 1):
        if f % 7 == 0:
            history[f] = (f, np.nan, np.nan, 0.0)
        else:
            history[f] = (f, 960.0, 540.0, 0.9)
    return SimpleNamespace(history=history, frame_w=1920, frame_h=1080, fps=float(FPS))


def _fake_fuse(fot_model, det_pass, matrices, reproj, valid, cfg, device):
    """Deterministic mix of the four notebook states over the window."""
    preds = {}
    for f in sorted(det_pass.history):
        if f % 11 == 0:
            preds[f] = (np.nan, np.nan, 0.0, "NONE", 999)
        elif f % 3 == 0:
            preds[f] = (960.0, 540.0, 0.7, "FOT_BRIDGED", 2)
        else:
            preds[f] = (960.0 + f, 540.0, 0.9, "DETECTED", 0)
    return preds


def _patch_video_boundary(monkeypatch, n=N_SYNTH):
    """Replace only the I/O boundary of src.fot_video; the rest runs for real."""
    import src.fot_video as F

    matrices, reproj, valid = _homography(n)
    monkeypatch.setattr(F, "set_global_seed", lambda seed: None)
    monkeypatch.setattr(F, "require",
                        lambda names, **kw: ["/fake/tactical.mp4", "/fake/homography.csv"])
    monkeypatch.setattr(F, "load_homography", lambda **kw: (matrices, reproj, valid))
    monkeypatch.setattr(F, "video_geometry", lambda video: (float(FPS), n))
    monkeypatch.setattr(F, "choose_window",
                        lambda valid, cfg, fps, total: (0, n - 1, 1))
    monkeypatch.setattr(F, "load_models", lambda cfg: (object(), object(), "cpu"))
    monkeypatch.setattr(F, "detect_window",
                        lambda rt, video, m, r, v, start, end, cfg:
                        _fake_detect(video, m, r, v, start, end, cfg))
    monkeypatch.setattr(F, "fuse_predictions",
                        lambda fot, det, m, r, v, cfg, dev: _fake_fuse(fot, det, m, r, v, cfg, dev))
    return F


# --------------------------------------------------------------------------- #
# Spec contract
# --------------------------------------------------------------------------- #
def test_fot_video_spec_validates_clean():
    from pipeline_spec_fot_video import FOT_VIDEO_SPEC

    assert validate_spec(FOT_VIDEO_SPEC) == []


def test_fot_video_spec_swaps_only_the_ball_fill_handler():
    """Adoption, not modification: identical stage names, handlers and data-flow
    everywhere except ball_fill, which binds the notebook's stage instead."""
    from pipeline_spec import SPEC
    from pipeline_spec_fot_video import FOT_VIDEO_SPEC
    import src.fot_video as F

    assert [s.name for s in FOT_VIDEO_SPEC] == [s.name for s in SPEC]
    for ours, theirs in zip(FOT_VIDEO_SPEC, SPEC):
        if ours.name == "ball_fill":
            assert ours.fn is F.fot_video_track
            assert ours.requires == theirs.requires
            assert ours.provides == theirs.provides
        else:
            assert ours.fn is theirs.fn, f"{ours.name} must reuse the framework handler"
            assert ours.requires == theirs.requires
            assert ours.provides == theirs.provides


def test_default_spec_is_untouched_by_this_branch():
    from pipeline_spec import SPEC

    assert [s.name for s in SPEC] == [
        "load", "ball_fill", "emission", "decode", "x_policy", "summary",
    ]


def test_module_import_does_not_pull_heavy_dependencies():
    """Framework runtime is numpy+pandas: torch/cv2/ultralytics only on the video path."""
    code = (
        "import sys; import src.fot_video; "
        "heavy = {'torch', 'cv2', 'ultralytics'} & set(sys.modules); "
        "assert not heavy, f'imported too early: {heavy}'"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Pure notebook functions
# --------------------------------------------------------------------------- #
def test_build_window_features_shapes_and_interpolation():
    from src.fot_video import build_window_features

    x = np.array([0.1, np.nan, 0.3, np.nan], dtype=np.float32)
    y = np.array([0.2, np.nan, 0.4, np.nan], dtype=np.float32)
    sc = np.array([0.9, 0.0, 0.8, 0.0], dtype=np.float32)
    valid = np.array([True, False, True, False])

    feats, baseline = build_window_features(x, y, sc, valid)
    assert feats.shape == (4, 12)
    assert baseline.shape == (4, 2)
    np.testing.assert_allclose(feats[valid, 0], x[valid], rtol=1e-6)
    np.testing.assert_allclose(feats[:, 2], valid.astype(np.float32))  # is_obs
    assert np.isfinite(feats).all()


def test_build_window_features_all_missing_degrades_to_centre():
    from src.fot_video import build_window_features

    w = 8
    feats, baseline = build_window_features(
        np.full(w, np.nan, np.float32), np.full(w, np.nan, np.float32),
        np.zeros(w, np.float32), np.zeros(w, dtype=bool))

    assert np.isfinite(feats).all()
    np.testing.assert_allclose(baseline, 0.5)
    assert (feats[:, 2] == 0).all()          # nothing observed
    assert (feats[:, 11] == 0).all()         # no pair of anchors


def test_compute_tile_coords_covers_the_frame_flush_with_edges():
    from src.fot_video import compute_tile_coords

    tiles = compute_tile_coords(1920, 1080, tile_size=320, stride=260)
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})

    assert xs[-1] == 1920 - 320 and ys[-1] == 1080 - 320
    assert all(0 <= x <= 1920 - 320 for x in xs)
    assert all(0 <= y <= 1080 - 320 for y in ys)

    covered = np.zeros(1920, dtype=bool)
    for x in xs:
        covered[x:x + 320] = True
    assert covered.all(), "tiles must cover the frame without holes"


def test_nms_candidates_keeps_the_confident_one_per_cluster():
    from src.fot_video import nms_candidates

    cands = [(100.0, 100.0, 0.9), (105.0, 100.0, 0.5), (300.0, 300.0, 0.7)]
    kept = nms_candidates(cands, min_dist_px=14.0)

    assert (100.0, 100.0, 0.9) in kept
    assert (300.0, 300.0, 0.7) in kept
    assert len(kept) == 2, "the 5 px-distant duplicate must be dropped"
    assert nms_candidates([]) == []


def test_get_valid_H_gates_on_valid_error_and_nan():
    from src.fot_video import get_valid_H

    matrices, reproj, valid = _homography(3)
    reproj[1] = 9.0
    valid[2] = False

    assert get_valid_H(0, matrices, reproj, valid) is not None
    assert get_valid_H(1, matrices, reproj, valid) is None, "reproj > 5 must be rejected"
    assert get_valid_H(2, matrices, reproj, valid) is None, "invalid frame must be rejected"
    assert get_valid_H(99, matrices, reproj, valid) is None, "out of range must be rejected"

    broken = matrices.copy()
    broken[0] = np.nan
    assert get_valid_H(0, broken, reproj, valid) is None


def test_kinematic_gate_accepts_hits_and_confirms_far_jumps():
    from src.fot_video import PitchKinematicGate

    # An unequivocal hit (> 0.68) is accepted on the spot, even with no prior track.
    gate = PitchKinematicGate(fps=FPS, pitch_length_m=105.3, pitch_width_m=68.0)
    history = {}
    gx, gy, sc = gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, history)
    assert (gx, gy, sc) == (960.0, 540.0, 0.80)
    assert gate.last_mx is not None, "an accepted hit must seed the track"

    # A weak far candidate needs JUMP_CONFIRM_FRAMES consecutive frames.
    fresh = PitchKinematicGate(fps=FPS, pitch_length_m=105.3, pitch_width_m=68.0)
    seen = []
    for fr in range(3):
        gx, gy, sc = fresh.step(fr, [(200.0, 200.0, 0.30)], H_SYNTH, history)
        seen.append(bool(np.isfinite(gx)))
    assert seen == [False, False, True], "far jumps must need 3 confirming frames"


def test_kinematic_gate_rejects_weak_far_candidate_without_confirmation():
    from src.fot_video import PitchKinematicGate

    gate = PitchKinematicGate(fps=FPS, pitch_length_m=105.3, pitch_width_m=68.0)
    history = {}
    gx, _, _ = gate.step(0, [(200.0, 200.0, 0.30)], H_SYNTH, history)
    assert np.isnan(gx), "a lone weak far candidate must not become the track"


# --------------------------------------------------------------------------- #
# Pitch-space fusion stages
# --------------------------------------------------------------------------- #
def _track(n, anchors, conf=0.9, state="DETECTED"):
    """A full frame->point dict: None everywhere except the given anchors."""
    pts = {f: None for f in range(n)}
    for f, (x, y) in anchors.items():
        pts[f] = [x, y, conf, state, 0]
    return pts


def test_prune_spikes_removes_the_isolated_anchor_only():
    from src.fot_video import prune_spikes

    # frame 1: 50 m/s in, 50 m/s out, ends agree -> spike; frame 3 is a plain hole.
    pts = _track(5, {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (0.0, 0.0), 4: (0.1, 0.0)})
    prune_spikes(list(range(5)), pts, fps=FPS)

    assert pts[1] is None, "the isolated impossible anchor must be pruned"
    assert pts[0] is not None and pts[2] is not None


def test_prune_spikes_keeps_a_fast_but_consistent_run():
    from src.fot_video import prune_spikes

    pts = _track(4, {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)})
    prune_spikes(list(range(4)), pts, fps=FPS)

    assert all(pts[f] is not None for f in range(4)), "a real fast run must survive"


def test_bridge_kinematic_gaps_fills_a_short_gap_within_the_lateral_limit():
    from src.fot_video import bridge_kinematic_gaps

    frames = list(range(11))
    pts = _track(11, {0: (0.0, 0.0), 10: (10.0, 0.0)})
    bridge_kinematic_gaps(frames, pts, fps=FPS)

    assert pts[5] is not None, "a 9-frame gap at 25 m/s must be bridged"
    assert pts[5][3] == "KINEMATIC_BRIDGED"
    assert pts[5][2] == pytest.approx(0.9 * 0.85)
    for f in range(1, 10):
        mx, my = pts[f][0], pts[f][1]
        assert -0.65 <= my <= 0.65, "lateral bow must stay within MAX_LATERAL_DEV_M"
        assert -0.65 <= mx <= 10.65, "the bridge must stay near the chord"


def test_bridge_kinematic_gaps_respects_both_time_and_speed_limits():
    from src.fot_video import bridge_kinematic_gaps

    # Too long in time: 99 frames > round(1.4 s * 25).
    long_gap = _track(101, {0: (0.0, 0.0), 100: (10.0, 0.0)})
    bridge_kinematic_gaps(list(range(101)), long_gap, fps=FPS)
    assert long_gap[50] is None, "a gap beyond max_gap_s must stay unfilled"

    # Too fast: 500 m in 0.16 s is not a ball.
    fast = _track(5, {0: (0.0, 0.0), 4: (500.0, 0.0)})
    bridge_kinematic_gaps(list(range(5)), fast, fps=FPS)
    assert fast[2] is None, "an implausible chord speed must block the bridge"


def test_rts_smoothing_damps_zigzag_and_leaves_short_segments_alone():
    from src.fot_video import rts_smooth_segment

    frames = list(range(12))
    pts = _track(12, {f: (f * 0.4, 0.3 if f % 2 else -0.3) for f in frames})
    before = np.array([pts[f][1] for f in frames])

    rts_smooth_segment(frames, pts, fps=FPS)
    after = np.array([pts[f][1] for f in frames])

    assert np.isfinite(after).all()
    assert np.abs(after).mean() <= np.abs(before).mean(), "RTS must damp the zig-zag"

    short = _track(2, {0: (0.0, 0.0), 1: (0.4, 0.0)})
    snapshot = {f: list(v) for f, v in short.items()}
    rts_smooth_segment([0, 1], short, fps=FPS)
    assert short == snapshot, "segments below 3 frames must be left untouched"


# --------------------------------------------------------------------------- #
# BallTrack contract: the compatibility seam with the rest of the framework
# --------------------------------------------------------------------------- #
def test_build_fot_ball_track_maps_states_to_framework_tiers():
    from src.fot_video import build_fot_ball_track

    matrices, reproj, valid = _homography(6)
    preds = {
        0: (960.0, 540.0, 0.9, "DETECTED", 0),
        1: (960.0, 540.0, 0.7, "FOT_BRIDGED", 2),
        2: (960.0, 540.0, 0.6, "KINEMATIC_BRIDGED", 3),
        3: (np.nan, np.nan, 0.0, "NONE", 999),
        4: (960.0, 540.0, 0.9, "DETECTED", 0),
        5: (np.nan, np.nan, 0.0, "NONE", 999),
    }
    track = build_fot_ball_track(6, preds, matrices, reproj, valid, CFG)

    assert isinstance(track, BallTrack)
    assert track.source[0] == SOURCE_DETECTOR
    assert track.source[1] == SOURCE_KINEMATIC
    assert track.source[2] == SOURCE_KINEMATIC
    assert track.source[3] == SOURCE_NONE
    assert track.source[5] == SOURCE_NONE
    assert np.isnan(track.xy[3]).all()

    # Normalised coordinates round-trip through the homography: 960/1920-ish.
    assert track.xy[0, 0] == pytest.approx((960.0 - 160.0) / 1600.0)
    assert track.xy[0, 1] == pytest.approx((540.0 - 90.0) / 900.0)

    # age_s counts to the nearest raw observation (frames 0 and 4), not to fills.
    assert track.age_s[0] == 0.0 and track.age_s[4] == 0.0
    assert track.age_s[2] == pytest.approx(2 / FPS)
    assert track.age_s[5] == pytest.approx(1 / FPS)


def test_merge_video_never_overwrites_the_atd_and_only_fills_gaps():
    from src.fot_video import merge_video_into_track

    matrices, reproj, valid = _homography(6)
    base_xy = np.full((6, 2), np.nan)
    base_xy[0] = (0.11, 0.22)          # the ATD says something different here
    base = BallTrack(xy=base_xy,
                     source=np.array([SOURCE_ATD] + [SOURCE_NONE] * 5, dtype=np.int8),
                     age_s=np.zeros(6),
                     out_of_pitch=np.zeros(6, dtype=bool))

    preds = {
        0: (960.0, 540.0, 0.9, "DETECTED", 0),   # must lose to the ATD
        1: (960.0, 540.0, 0.9, "DETECTED", 0),   # fills a gap
        2: (960.0, 540.0, 0.7, "FOT_BRIDGED", 2),
        3: (np.nan, np.nan, 0.0, "NONE", 999),   # stays empty
        4: (960.0, 540.0, 0.9, "DETECTED", 0),
        5: (960.0, 540.0, 0.9, "DETECTED", 0),
    }
    merged = merge_video_into_track(base, preds, matrices, reproj, valid, CFG)

    assert merged.source[0] == SOURCE_ATD, "raw ATD must never be overwritten"
    np.testing.assert_allclose(merged.xy[0], (0.11, 0.22))
    assert merged.source[1] == SOURCE_DETECTOR
    assert merged.source[2] == SOURCE_KINEMATIC
    assert merged.source[3] == SOURCE_NONE
    assert np.isnan(merged.xy[3]).all()

    # The input track is untouched (pure function).
    assert base.source[1] == SOURCE_NONE and np.isnan(base.xy[1]).all()


def test_out_of_pitch_estimates_are_dropped_but_observations_are_only_flagged():
    from src.fot_video import build_fot_ball_track

    matrices, reproj, valid = _homography(4)
    far_px = (-100.0, 540.0)          # outside the pitch -> negative normalised x
    preds = {
        0: (far_px[0], far_px[1], 0.9, "DETECTED", 0),
        1: (far_px[0], far_px[1], 0.7, "FOT_BRIDGED", 2),
        2: (960.0, 540.0, 0.9, "DETECTED", 0),
        3: (np.nan, np.nan, 0.0, "NONE", 999),
    }
    track = build_fot_ball_track(4, preds, matrices, reproj, valid, CFG)

    assert track.out_of_pitch[0], "an out-of-pitch observation must be flagged"
    assert track.source[0] == SOURCE_DETECTOR, "observations are flagged, never dropped"
    assert np.isfinite(track.xy[0]).all()

    assert track.out_of_pitch[1]
    assert track.source[1] == SOURCE_NONE, "estimated out-of-pitch positions are dropped"
    assert np.isnan(track.xy[1]).all()
    assert not track.out_of_pitch[2]


def test_predictions_frame_table_contract():
    from src.fot_video import predictions_frame_table

    preds = {
        0: (10.0, 20.0, 0.9, "DETECTED", 0),
        5: (np.nan, np.nan, 0.0, "NONE", 999),
        9: (30.0, 40.0, 0.7, "FOT_BRIDGED", 2),
    }
    table = predictions_frame_table(preds, FPS)

    assert list(table.columns) == [
        "frame", "time_s", "pred_x", "pred_y", "score", "state", "gap_steps",
    ]
    assert table["frame"].tolist() == [0, 5, 9]
    assert table["time_s"].iloc[2] == pytest.approx(9 / FPS)
    assert set(table["state"]) == {"DETECTED", "NONE", "FOT_BRIDGED"}


def test_export_frame_table_matches_the_notebook_csv_contract():
    """The 12 play_*.csv columns, pitch geometry from CFG, speed resetting on NONE."""
    from src.fot_video import export_frame_table

    matrices, reproj, valid = _homography(5)
    preds = {
        0: (100.0, 200.0, 0.9, "DETECTED", 0),
        1: (130.0, 200.0, 0.9, "DETECTED", 0),      # 30 px -> 750 px/s at 25 fps
        2: (np.nan, np.nan, 0.0, "NONE", 999),      # speed resets here
        3: (100.0, 200.0, 0.7, "FOT_BRIDGED", 2),   # first frame after NONE: no speed
        4: (100.0, 200.0, 0.7, "KINEMATIC_BRIDGED", 3),
    }
    table = export_frame_table(preds, matrices, reproj, valid, FPS, CFG)

    assert list(table.columns) == [
        "frame", "time_s", "pred_x", "pred_y",
        "pitch_x_norm", "pitch_y_norm", "pitch_x_m", "pitch_y_m",
        "score", "state", "gap_steps", "speed_px_s",
    ]
    assert table["speed_px_s"].tolist()[0] == 0.0
    assert table["speed_px_s"].tolist()[1] == pytest.approx(30 * FPS)
    assert table["speed_px_s"].tolist()[2] == 0.0, "NONE rows carry no speed"
    assert table["speed_px_s"].tolist()[3] == 0.0, "speed resets across a NONE gap"

    # Pitch geometry: pixel -> normalised -> metres via config.CFG (105.3 x 68.0).
    row0 = table.iloc[0]
    assert row0["pitch_x_norm"] == pytest.approx((100.0 - 160.0) / 1600.0)
    assert row0["pitch_x_m"] == pytest.approx(row0["pitch_x_norm"] * CFG["pitch_length_m"])
    assert row0["pitch_y_m"] == pytest.approx(row0["pitch_y_norm"] * CFG["pitch_width_m"])
    none_row = table.iloc[2]
    assert np.isnan(none_row["pitch_x_norm"]) and np.isnan(none_row["pitch_x_m"]), (
        "a NONE row carries no pitch coordinate")


def test_cli_csv_only_exports_the_notebook_csv(monkeypatch, tmp_path):
    """`python -m src.fot_video --csv-only` writes play_*.csv without touching cv2/video."""
    F = _patch_video_boundary(monkeypatch)
    from src.fot_video import main

    rc = main(["--csv-only", "--out-dir", str(tmp_path), "--seed", "0",
               "--half", "1st_half", "--seconds", "1"])
    assert rc == 0

    csvs = list(tmp_path.glob("play_1st_half_seed0_*.csv"))
    assert len(csvs) == 1, f"expected exactly one play_*.csv, got {csvs}"

    import pandas as pd
    table = pd.read_csv(csvs[0])
    assert len(table) == N_SYNTH
    assert "pitch_x_m" in table.columns and "speed_px_s" in table.columns
    assert set(table["state"]) <= {"DETECTED", "FOT_BRIDGED", "KINEMATIC_BRIDGED", "NONE"}
    assert not list(tmp_path.glob("*.mp4")), "--csv-only must not render a video"


def test_cli_rejects_an_unknown_half(monkeypatch, tmp_path):
    _patch_video_boundary(monkeypatch)
    from src.fot_video import main

    # argparse's --half choices reject it before any work runs.
    with pytest.raises(SystemExit) as excinfo:
        main(["--csv-only", "--half", "3rd_half", "--out-dir", str(tmp_path)])
    assert excinfo.value.code == 2


def test_choose_window_is_seeded_deterministic_and_inside_the_half():
    from src.fot_video import choose_window

    valid = np.ones(199_319, dtype=bool)
    cfg = {**CFG, "fot_seed": 0, "fot_half": "1st_half", "fot_clip_seconds": 30.0}

    first = choose_window(valid, cfg, FPS, 199_319)
    second = choose_window(valid, cfg, FPS, 199_319)
    assert first == second, "the same seed must pick the same clip"

    start, end, half = first
    assert half == 1
    assert 11_185 <= start <= 90_651 - 750, "start must sit inside half 1"
    assert end - start + 1 == 750, "30 s at 25 fps"

    _, _, half2 = choose_window(valid, {**cfg, "fot_half": "2nd_half"}, FPS, 199_319)
    assert half2 == 2

    with pytest.raises(ValueError, match="fot_half"):
        choose_window(valid, {**cfg, "fot_half": "3rd_half"}, FPS, 199_319)


def test_choose_window_prefers_homography_valid_stretches():
    from src.fot_video import choose_window

    # Half 1 is only valid in its back half; a valid candidate must be preferred.
    valid = np.zeros(199_319, dtype=bool)
    valid[50_000:90_652] = True
    cfg = {**CFG, "fot_seed": 0, "fot_half": "1st_half", "fot_clip_seconds": 30.0}

    start, end, _ = choose_window(valid, cfg, FPS, 199_319)
    window_valid = valid[start:end + 1].mean()
    assert start <= end
    assert window_valid >= 0.75, "a >=75% valid candidate must be preferred"

    # With no valid stretch at all the seeded fallback still yields a window.
    start2, end2, _ = choose_window(np.zeros(199_319, dtype=bool), cfg, FPS, 199_319)
    assert start2 <= end2


# --------------------------------------------------------------------------- #
# The handler inside the unchanged framework
# --------------------------------------------------------------------------- #
def test_fot_video_handler_produces_a_ball_track_and_diagnostics(monkeypatch):
    F = _patch_video_boundary(monkeypatch)

    ctx = Ctx()
    ctx.arrays = _synthetic_arrays()
    cfg = {**CFG, "fot_verbose": False}
    F.fot_video_track(ctx, cfg)

    assert isinstance(ctx.track, BallTrack)
    assert len(ctx.track.source) == N_SYNTH
    assert set(np.unique(ctx.track.source)) <= {
        SOURCE_ATD, SOURCE_DETECTOR, SOURCE_KINEMATIC, SOURCE_NONE,
    }

    # The ATD frames of the synthetic match survive the video overlay.
    atd_frames = [100, 150, 200, 250, 300, 350]
    assert (ctx.track.source[atd_frames] == SOURCE_ATD).all()

    extra = ctx.extra["fot_video"]
    assert extra["start_frame"] == 0 and extra["end_frame"] == N_SYNTH - 1
    assert len(extra["predictions"]) == N_SYNTH
    assert sum(extra["state_counts"].values()) == N_SYNTH
    assert len(extra["table"]) == N_SYNTH


def test_full_fot_video_spec_runs_end_to_end_on_the_framework(monkeypatch):
    """The definitive compatibility check: the notebook stage as a Step, driving the
    untouched emission -> decode -> x_policy -> summary handlers through `run()`."""
    from pipeline_spec_fot_video import FOT_VIDEO_SPEC
    import src.data as D

    arrays = _synthetic_arrays()
    monkeypatch.setattr(D, "load_arrays", lambda force=False: arrays)
    _patch_video_boundary(monkeypatch)

    labels, summary = run({**CFG, "fot_verbose": False}, spec=FOT_VIDEO_SPEC)

    assert len(labels) == N_SYNTH
    assert set(np.unique(labels)) <= {-1, 0, 1, 2}
    assert (labels[:100] == -1).all(), "outside the halves stays X"
    assert (labels[100:600] != -1).all(), "graded frames are always decoded"

    for key in ("n_segments", "median_len_s", "subsecond_share", "d_share",
                "n_possessions", "reverted_switches", "coverage", "graded_frames"):
        assert key in summary, f"the framework summary must carry {key}"
    assert summary["graded_frames"] == 500
    assert "graded_detector" in summary["coverage"]


# --------------------------------------------------------------------------- #
# Optional heavy paths: skipped where torch/cv2 are not installed
# --------------------------------------------------------------------------- #
def test_fuse_predictions_runs_the_real_torch_path():
    pytest.importorskip("torch")
    from src.fot_video import DetectionPass, build_fot_model, fuse_predictions

    n = 100
    matrices, reproj, valid = _homography(n)
    model = build_fot_model(weights=None)      # random init: exercises the graph only

    # (a) Fully observed window: every frame is DETECTED and the pitch round-trip
    #     (unproject -> metres -> RTS -> project) must reproduce the input exactly,
    #     because identical measurements leave zero innovation at every filter step.
    dense = {f: (f, 960.0, 540.0, 0.9) for f in range(n)}
    det_pass = DetectionPass(history=dense, frame_w=1920, frame_h=1080, fps=float(FPS))
    preds = fuse_predictions(model, det_pass, matrices, reproj, valid, CFG, "cpu")

    assert set(preds) == set(dense), "every window frame must come back with a verdict"
    for f, (px, py, conf, state, gap) in preds.items():
        assert state == "DETECTED" and gap == 0, f"frame {f} must pass through as DETECTED"
        assert px == pytest.approx(960.0, abs=0.01), "pitch round-trip is lossless here"
        assert py == pytest.approx(540.0, abs=0.01)

    # (b) Window with holes: structure only — bridged coordinates come from a
    #     random-init network here, so their values are not assertable.
    holed = {f: (f, np.nan, np.nan, 0.0) if f % 7 == 0 else (f, 960.0, 540.0, 0.9)
             for f in range(n)}
    det_pass = DetectionPass(history=holed, frame_w=1920, frame_h=1080, fps=float(FPS))
    preds = fuse_predictions(model, det_pass, matrices, reproj, valid, CFG, "cpu")

    assert set(preds) == set(holed)
    allowed = {"DETECTED", "FOT_BRIDGED", "KINEMATIC_BRIDGED", "NONE"}
    for f, (px, py, conf, state, gap) in preds.items():
        assert state in allowed, f"frame {f} has an unknown state {state!r}"
        if state == "NONE":
            assert np.isnan(px) and np.isnan(py), "NONE must carry no position"
        else:
            assert np.isfinite(px) and np.isfinite(py)


def test_build_fot_model_forward_shapes():
    pytest.importorskip("torch")
    import torch

    from src.fot_video import build_fot_model

    model = build_fot_model(window_size=16, weights=None)
    feats = torch.zeros(1, 16, 12)
    baseline = torch.zeros(1, 16, 2)
    pred_xy, hit5 = model(feats, baseline)

    assert pred_xy.shape == (1, 16, 2)
    assert hit5.shape == (1, 16)


def test_run_fot_v2_sequence_smoke():
    pytest.importorskip("torch")
    from src.fot_video import build_fot_model, run_fot_v2_sequence

    model = build_fot_model(weights=None)
    dets = [(f, 960.0, 540.0, 0.9) if f % 5 else (f, np.nan, np.nan, 0.0)
            for f in range(80)]

    out = run_fot_v2_sequence(model, dets, 1920, 1080, "cpu", profile="max_coverage")

    assert set(out) == set(range(80))
    assert set(v[3] for v in out.values()) <= {"DETECTED", "FOT_BRIDGED", "NONE"}
    for f, (px, py, conf, state, gap) in out.items():
        if state == "DETECTED":
            assert px == pytest.approx(960.0, abs=0.01)


def test_pitch_masks_and_overlay_render():
    cv2 = pytest.importorskip("cv2")
    from src.fot_video import build_pitch_masks, draw_pitch_overlay

    mask_margin, mask_exact = build_pitch_masks(H_SYNTH, 1920, 1080, margin=0.05)

    assert mask_margin.shape == (1080, 1920) and mask_margin.max() == 255
    assert mask_exact.sum() < mask_margin.sum(), "the margin mask must be strictly larger"

    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    out = draw_pitch_overlay(frame, H_SYNTH, mask_margin, mask_exact)
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert not np.array_equal(out, frame), "the overlay must actually draw something"
