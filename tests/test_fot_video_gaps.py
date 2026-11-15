"""Gap tests for src/fot_video.py — the parts tests/test_fot_video.py never executes.

Written during the 2026-11-14 review of the RT-DETR + FoT v2 adapter. The committed
suite monkeypatches the detector, the video and the fusion away, so the gate, the FoT
windowing, the RTS smoother, the Hermite bridge and ``detect_window`` ran zero lines
under test. Here they execute for real: torch, cv2 and ultralytics are replaced by
small duck-typed fakes injected through sys.modules, so everything runs with numpy +
pandas + pytest only. No match data, no video, no checkpoints.

Tests marked xfail(strict=True) pin a defect found in the review that was deliberately
NOT changed (it alters tracking behaviour, which needs external-game evidence first):
they fail today and flip to XPASS — an error under strict — the day the defect is
fixed, forcing the marker to be removed. They are documentation that runs.
"""

import contextlib
import sys
import types

import numpy as np
import pytest

from config import CFG
from src.ball_fill import (SOURCE_ATD, SOURCE_DETECTOR, SOURCE_KINEMATIC, SOURCE_NONE,
                           build_ball_track)

FPS = 25
H_SYNTH = np.array([[1600.0, 0.0, 160.0],
                    [0.0, 900.0, 90.0],
                    [0.0, 0.0, 1.0]])
PX_PER_M_X = 1600.0 / 105.3


def _homography(n):
    return np.tile(H_SYNTH, (n, 1, 1)), np.zeros(n), np.ones(n, dtype=bool)


def _track(n, anchors, conf=0.9, state="DETECTED"):
    pts = {f: None for f in range(n)}
    for f, (x, y) in anchors.items():
        pts[f] = [x, y, conf, state, 0]
    return pts


def _gate():
    from src.fot_video import PitchKinematicGate
    return PitchKinematicGate(fps=FPS, pitch_length_m=105.3, pitch_width_m=68.0)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _T:
    """The five tensor methods run_fot_v2_sequence touches."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def unsqueeze(self, d):
        return _T(np.expand_dims(self.a, d))

    def squeeze(self, d):
        return _T(np.squeeze(self.a, d))

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


@pytest.fixture
def fake_torch(monkeypatch):
    mod = types.ModuleType("torch")
    mod.from_numpy = _T
    mod.sigmoid = lambda t: _T(1.0 / (1.0 + np.exp(-t.a)))
    mod.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class _IdentityFoT:
    """Zero residual, hit5 logit +10: each window returns its own interpolation baseline."""

    def __init__(self):
        self.calls = 0

    def __call__(self, feats, base):
        assert feats.a.shape == (1, 64, 12) and base.a.shape == (1, 64, 2)
        self.calls += 1
        return _T(base.a.copy()), _T(np.full(base.a.shape[:2], 10.0, dtype=np.float32))


W_PX, H_PX, N_VIDEO = 1920, 1080, 12
BALL = {f: (300.0 + 37.0 * f, 250.0 + 11.0 * f) for f in range(N_VIDEO)}
BALL[5] = None


@pytest.fixture
def fake_cv2(monkeypatch):
    class Cap:
        def __init__(self, path):
            self.pos = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {1: 25.0, 2: float(W_PX), 3: float(H_PX), 4: float(N_VIDEO)}[prop]

        def set(self, prop, val):
            self.pos = int(val)

        def read(self):
            if self.pos >= N_VIDEO:
                return False, None
            img = np.zeros((H_PX, W_PX, 3), dtype=np.uint8)
            if BALL[self.pos] is not None:
                x, y = (int(round(v)) for v in BALL[self.pos])
                img[y - 3:y + 4, x - 3:x + 4] = 255
            self.pos += 1
            return True, img

        def release(self):
            pass

    def fill_poly(mask, polys, val):          # axis-aligned homography -> bbox fill is exact
        p = polys[0]
        x0, y0 = np.clip(p.min(axis=0), 0, None)
        x1, y1 = p.max(axis=0)
        mask[y0:y1 + 1, x0:x1 + 1] = val

    mod = types.ModuleType("cv2")
    mod.CAP_PROP_FPS, mod.CAP_PROP_FRAME_WIDTH = 1, 2
    mod.CAP_PROP_FRAME_HEIGHT, mod.CAP_PROP_FRAME_COUNT, mod.CAP_PROP_POS_FRAMES = 3, 4, 5
    mod.VideoCapture = Cap
    mod.fillPoly = fill_poly
    mod.bitwise_and = lambda a, b, mask=None: a * (mask[..., None] > 0).astype(a.dtype)
    monkeypatch.setitem(sys.modules, "cv2", mod)
    return mod


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, conf, xywhn):
        self.conf, self.xywhn = _Arr(conf), _Arr(xywhn)

    def __len__(self):
        return len(self.conf.a)


class _FakeRTDETR:
    """Mimics the ultralytics surface detect_window uses: predict(source=[crops]) ->
    one result per crop, in order; boxes.conf / boxes.xywhn (normalised by the CROP)."""

    def __init__(self):
        self.batches, self.kwargs = [], None

    def predict(self, source, **kwargs):
        self.batches.append(len(source))
        self.kwargs = kwargs
        out = []
        for crop in source:
            h, w = crop.shape[:2]
            ys, xs = np.nonzero(crop[..., 0] > 200)
            if len(xs) == 49:                 # whole 7x7 ball inside the tile
                boxes = _Boxes([0.9], [[xs.mean() / w, ys.mean() / h, 7 / w, 7 / h]])
            elif len(xs):                     # ball cut by the tile edge: weaker, off-centre
                boxes = _Boxes([0.4], [[xs.mean() / w, ys.mean() / h, 0.01, 0.01]])
            else:
                boxes = _Boxes([], np.zeros((0, 4)))
            out.append(types.SimpleNamespace(boxes=boxes))
        return out


# --------------------------------------------------------------------------- #
# 1. detect_window: the RT-DETR / cv2 integration seam
# --------------------------------------------------------------------------- #
def test_detect_window_maps_tile_detections_to_frame_pixels(fake_cv2):
    from src.fot_video import DetectionPass, detect_window

    matrices, reproj, valid = _homography(N_VIDEO)
    rt = _FakeRTDETR()
    det = detect_window(rt, "/fake.mp4", matrices, reproj, valid, 0, N_VIDEO - 1,
                        {**CFG, "fot_device": "cpu", "fot_verbose": False})

    assert isinstance(det, DetectionPass)
    assert (det.frame_w, det.frame_h, det.fps) == (W_PX, H_PX, 25.0)
    assert sorted(det.history) == list(range(N_VIDEO)), "every frame gets a history entry"
    assert set(rt.batches) == {32}, "1920x1080 at 320/260 is an 8 x 4 tile grid"
    assert rt.kwargs["imgsz"] == 320 and rt.kwargs["conf"] == pytest.approx(0.07)
    assert rt.kwargs["max_det"] == 4

    for f in range(N_VIDEO):
        fr, gx, gy, sc = det.history[f]
        assert fr == f
        if BALL[f] is None:
            assert np.isnan(gx) and np.isnan(gy) and sc == 0.0
        else:
            # overlapping tiles see the ball up to 4 times: NMS must leave the full-ball hit
            assert (gx, gy) == pytest.approx(BALL[f], abs=1e-3)
            assert sc == pytest.approx(0.9)


def test_detect_window_start_offset_keeps_frame_index_alignment(fake_cv2):
    from src.fot_video import detect_window

    matrices, reproj, valid = _homography(N_VIDEO)
    det = detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 4, 8,
                        {**CFG, "fot_device": "cpu", "fot_verbose": False})

    assert sorted(det.history) == [4, 5, 6, 7, 8]
    assert det.history[4][1:3] == pytest.approx(BALL[4], abs=1e-3)
    assert det.history[8][1:3] == pytest.approx(BALL[8], abs=1e-3)


# --------------------------------------------------------------------------- #
# 2. run_fot_v2_sequence / fuse_predictions without torch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n, calls", [(10, 1), (64, 1), (65, 2), (80, 2), (81, 3), (200, 10)])
def test_fot_window_schedule_is_a_16_step_hop_with_a_flush_tail(fake_torch, n, calls):
    from src.fot_video import run_fot_v2_sequence

    model = _IdentityFoT()
    dets = [(f, 960.0, 540.0, 0.9) for f in range(n)]
    out = run_fot_v2_sequence(model, dets, W_PX, H_PX, "cpu")

    assert model.calls == calls
    assert sorted(out) == list(range(n))
    assert all(v[3] == "DETECTED" and v[0] == pytest.approx(960.0, abs=1e-3)
               and v[1] == pytest.approx(540.0, abs=1e-3) for v in out.values())


@pytest.mark.parametrize("gap, state", [(1, "FOT_BRIDGED"), (4, "FOT_BRIDGED"), (5, "NONE")])
def test_fot_overlap_add_bridges_short_gaps_on_the_interpolation(fake_torch, gap, state):
    from src.fot_video import run_fot_v2_sequence

    n = 300
    truth = np.column_stack([200.0 + 4.0 * np.arange(n), 300.0 + 1.5 * np.arange(n)])
    holes = set()
    for s in range(7, n - 7, 11):             # every phase relative to the 16-step hop
        holes |= set(range(s, s + gap))
    dets = [(f, np.nan, np.nan, 0.0) if f in holes else (f, truth[f, 0], truth[f, 1], 0.9)
            for f in range(n)]
    out = run_fot_v2_sequence(_IdentityFoT(), dets, W_PX, H_PX, "cpu", profile="max_coverage")

    for f in sorted(holes):
        px, py, conf, st, gap_steps = out[f]
        assert st == state
        if state == "FOT_BRIDGED":
            assert gap_steps == gap
            assert (px, py) == pytest.approx(tuple(truth[f]), abs=0.2), (
                "an identity model must reproduce the linear interpolation")
        else:
            assert np.isnan(px) and np.isnan(py)


def test_fot_drops_detections_below_the_profile_input_threshold(fake_torch):
    """Characterisation: what the gate keeps between 0.07 and 0.30 never reaches the output."""
    from src.fot_video import run_fot_v2_sequence

    dets = [(f, 960.0, 540.0, 0.25) for f in range(40)]
    out = run_fot_v2_sequence(_IdentityFoT(), dets, W_PX, H_PX, "cpu", profile="max_coverage")
    assert {v[3] for v in out.values()} == {"NONE"}


def test_fuse_predictions_even_odd_passes_cover_every_frame_and_bridge_in_place(fake_torch):
    from src.fot_video import DetectionPass, fuse_predictions

    n = 60
    matrices, reproj, valid = _homography(n)
    history = {f: (f, 400.0 + 3.0 * f, 500.0, 0.9) for f in range(n)}
    history[30] = (30, np.nan, np.nan, 0.0)
    det = DetectionPass(history=history, frame_w=W_PX, frame_h=H_PX, fps=float(FPS))
    preds = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid, CFG, "cpu")

    assert sorted(preds) == list(range(n)), "both parities must come back"
    assert preds[30][3] == "FOT_BRIDGED" and preds[30][4] == 1
    assert preds[30][0] == pytest.approx(490.0, abs=0.5)
    assert preds[30][1] == pytest.approx(500.0, abs=0.5)
    for f in range(n):
        assert preds[f][0] == pytest.approx(400.0 + 3.0 * f, abs=0.5), (
            "a constant-velocity ball must survive metres -> RTS -> pixels")


def test_fuse_predictions_survives_a_one_frame_window(fake_torch):
    """Fixed in review: the empty odd pass used to die in np.pad(mode='edge')."""
    from src.fot_video import DetectionPass, fuse_predictions

    matrices, reproj, valid = _homography(3)
    det = DetectionPass(history={0: (0, 960.0, 540.0, 0.9)}, frame_w=W_PX, frame_h=H_PX,
                        fps=float(FPS))
    preds = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid, CFG, "cpu")
    assert sorted(preds) == [0]


# --------------------------------------------------------------------------- #
# 3. PitchKinematicGate: the branches the committed suite never executes
# --------------------------------------------------------------------------- #
def test_gate_in_gate_scoring_blends_confidence_with_closeness():
    gate = _gate()
    gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, {})
    # sorted by confidence, as nms_candidates delivers them
    kept = gate.step(1, [(1010.0, 540.0, 0.55), (965.0, 540.0, 0.50)], H_SYNTH, {})
    assert kept == (965.0, 540.0, 0.50), "0.33 m away at 0.50 must beat 3.3 m away at 0.55"


def test_gate_reach_is_measured_in_metres_per_axis():
    # one frame after the seed the reach is 3.2 + 36 * 0.04 = 4.64 m
    gate = _gate()
    gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, {})
    assert np.isnan(gate.step(1, [(1035.0, 540.0, 0.10)], H_SYNTH, {})[0]), "75 px in x = 4.94 m"
    gate = _gate()
    gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, {})
    assert gate.step(1, [(960.0, 600.0, 0.10)], H_SYNTH, {}) == (960.0, 600.0, 0.10), (
        "60 px in y = 4.53 m")


def test_gate_velocity_is_an_ema_of_the_clipped_step_speed():
    gate = _gate()
    gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, {})
    gate.step(1, [(975.0, 540.0, 0.80)], H_SYNTH, {})
    step_mps = (15.0 / PX_PER_M_X) / (1.0 / FPS)
    assert gate.vx == pytest.approx(0.35 * step_mps, rel=1e-6)
    assert gate.vy == pytest.approx(0.0, abs=1e-9)


def test_gate_confirmed_far_jump_rescues_the_buffered_frames():
    gate, history = _gate(), {}
    for fr in range(3):
        gx, gy, sc = gate.step(fr, [(200.0, 200.0, 0.30)], H_SYNTH, history)
        history[fr] = (fr, gx, gy, sc)          # exactly what detect_window does
    assert history == {0: (0, 200.0, 200.0, 0.30), 1: (1, 200.0, 200.0, 0.30),
                       2: (2, 200.0, 200.0, 0.30)}


def test_gate_far_candidates_need_the_far_floor_a_cluster_and_a_fresh_buffer():
    # below FAR_CONF_FLOOR: never, however persistent
    gate = _gate()
    assert all(np.isnan(gate.step(fr, [(200.0, 200.0, 0.10)], H_SYNTH, {})[0])
               for fr in range(6))
    # three weak hits in three different places are not a confirmation
    gate = _gate()
    spots = [(200.0, 200.0), (900.0, 500.0), (1500.0, 800.0)]
    assert all(np.isnan(gate.step(fr, [(x, y, 0.30)], H_SYNTH, {})[0])
               for fr, (x, y) in enumerate(spots))
    # the buffer forgets: two hits, a long pause, one more hit
    gate = _gate()
    seen = [bool(np.isfinite(gate.step(fr, [(200.0, 200.0, 0.30)], H_SYNTH, {})[0]))
            for fr in (0, 1, 10)]
    assert seen == [False, False, False]


@pytest.mark.parametrize("conf, accepted", [(0.68, True), (0.67, False)])
def test_gate_unequivocal_hit_threshold(conf, accepted):
    gx, _, _ = _gate().step(0, [(960.0, 540.0, conf)], H_SYNTH, {})
    assert bool(np.isfinite(gx)) is accepted


def test_gate_track_memory_expires_after_one_and_a_half_seconds():
    gate = _gate()
    gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, {})
    assert np.isfinite(gate.step(10, [(962.0, 541.0, 0.10)], H_SYNTH, {})[0]), (
        "within 1.5 s a weak candidate next to the track is kept")
    gate = _gate()
    gate.step(0, [(960.0, 540.0, 0.80)], H_SYNTH, {})
    assert np.isnan(gate.step(100, [(962.0, 541.0, 0.10)], H_SYNTH, {})[0]), (
        "after 4 s the track is gone: a weak candidate needs the far rules again")


def test_gate_without_homography_passes_the_top_candidate_ungated():
    gate = _gate()
    assert gate.step(0, [(960.0, 540.0, 0.50)], None, {}) == (960.0, 540.0, 0.50)
    assert np.isnan(gate.step(1, [(960.0, 540.0, 0.10)], None, {})[0])
    assert gate.last_fr is None, "frames without a homography never move the track"


@pytest.mark.xfail(strict=True,
                   reason="track capture: any in-gate candidate >= 0.07 hides a >= 0.68 far hit")
def test_gate_strong_far_ball_wins_over_a_weak_in_gate_false_positive():
    gate, history = _gate(), {}
    fp = (800.0 + 10.0 * PX_PER_M_X, 500.0, 0.10)          # static, 10 m from the ball
    kept = []
    for fr in range(60):
        cands = [fp]
        if fr < 20:
            cands.append((800.0 + 0.5 * fr, 500.0, 0.90))   # ball visible
        elif fr >= 30:                                      # back after a 0.4 s occlusion
            cands.append((fp[0] + 25.0 * PX_PER_M_X, 500.0, 0.95))
        cands.sort(key=lambda c: c[2], reverse=True)
        kept.append(gate.step(fr, cands, H_SYNTH, history)[2])
    assert max(kept[30:]) >= 0.90, "the 0.95 ball must be re-acquired"


# --------------------------------------------------------------------------- #
# 4. FoT feature contract (the only definition of the training features)
# --------------------------------------------------------------------------- #
def test_build_window_features_golden_values():
    from src.fot_video import build_window_features

    x = np.array([0.1, np.nan, 0.3, np.nan], dtype=np.float32)
    y = np.array([0.2, np.nan, 0.4, np.nan], dtype=np.float32)
    sc = np.array([0.9, 0.5, 0.8, 0.5], dtype=np.float32)   # scores on holes must be zeroed
    valid = np.array([True, False, True, False])
    feats, baseline = build_window_features(x, y, sc, valid)

    expected = np.array([
        # x    y    obs  dt_prev dt_next vx   vy   ax     ay     score gap  both
        [0.1, 0.2, 1.0, 0.00,   0.00,   1.0, 1.0,  0.00,  0.00, 0.9, 0.0, 1.0],
        [0.2, 0.3, 0.0, 0.25,   0.25,   1.0, 1.0, -0.25, -0.25, 0.0, 0.5, 1.0],
        [0.3, 0.4, 1.0, 0.00,   0.00,   0.5, 0.5, -0.50, -0.50, 0.8, 0.0, 1.0],
        [0.3, 0.4, 0.0, 0.25,   1.00,   0.0, 0.0, -0.50, -0.50, 0.0, 1.0, 0.0],
    ])
    np.testing.assert_allclose(feats, expected, atol=1e-5)
    np.testing.assert_allclose(baseline, expected[:, :2], atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. RTS smoother against an independent reference
# --------------------------------------------------------------------------- #
def _batch_map_reference(z, r_std, fps=25.0, q=18.0):
    """One-shot weighted least squares of the same model: unknowns are x0 and the
    per-step accelerations (no Kalman recursion involved)."""
    n, dt = len(z), 1.0 / fps
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
    nu = 4 + 2 * (n - 1)
    A = np.zeros((n, 4, nu))
    A[0, :, :4] = np.eye(4)
    for k in range(1, n):
        A[k] = F @ A[k - 1]
        A[k][:, 4 + 2 * (k - 1): 4 + 2 * k] += G
    rows, rhs = [], []
    m0, s0 = np.array([z[0, 0], z[0, 1], 0.0, 0.0]), np.sqrt([0.5, 0.5, 25.0, 25.0])
    for i in range(4):
        row = np.zeros(nu)
        row[i] = 1.0 / s0[i]
        rows.append(row)
        rhs.append(m0[i] / s0[i])
    for j in range(4, nu):
        row = np.zeros(nu)
        row[j] = 1.0 / np.sqrt(q)
        rows.append(row)
        rhs.append(0.0)
    for k in range(n):
        for j in range(2):
            rows.append(A[k][j] / r_std[k])
            rhs.append(z[k, j] / r_std[k])
    u = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return np.array([(A[k] @ u)[:2] for k in range(n)])


def test_rts_smooth_segment_matches_the_batch_map_solution():
    from src.fot_video import rts_smooth_segment

    rng = np.random.default_rng(7)
    n = 40
    t = np.arange(n) / FPS
    z = np.column_stack([10 + 12 * t + 3 * np.sin(2 * t), 30 - 6 * t]) + rng.normal(0, 0.3, (n, 2))
    states = ["DETECTED" if i % 4 else "FOT_BRIDGED" for i in range(n)]
    pts = {f: [float(z[f, 0]), float(z[f, 1]), 0.9, states[f], 0] for f in range(n)}

    rts_smooth_segment(list(range(n)), pts, fps=FPS)
    got = np.array([[pts[f][0], pts[f][1]] for f in range(n)])
    ref = _batch_map_reference(z, np.array([0.22 if s == "DETECTED" else 0.85 for s in states]))

    np.testing.assert_allclose(got, ref, atol=1e-8)
    assert np.abs(got - z).max() > 0.05, "the smoother must actually move noisy points"


def test_rts_leaves_a_long_constant_velocity_run_almost_untouched():
    from src.fot_video import rts_smooth_segment

    n = 25
    pts = _track(n, {f: (30.0 / FPS * f, 0.0) for f in range(n)})   # 30 m/s, noise free
    rts_smooth_segment(list(range(n)), pts, fps=FPS)
    moved = max(abs(pts[f][0] - 30.0 / FPS * f) for f in range(n))
    assert moved < 0.05


# --------------------------------------------------------------------------- #
# 6. Hermite bridging: exact curve, the clamp, the limits
# --------------------------------------------------------------------------- #
def test_bridge_single_anchor_gap_follows_the_damped_hermite_exactly():
    from src.fot_video import bridge_kinematic_gaps

    pts = _track(11, {0: (0.0, 0.0), 10: (10.0, 0.0)})
    bridge_kinematic_gaps(list(range(11)), pts, fps=FPS)
    # cubic Hermite with both tangents 0.65 * chord: x(t) = -7 t^3 + 10.5 t^2 + 6.5 t
    for f in range(1, 10):
        t = f / 10.0
        assert pts[f][0] == pytest.approx(-7 * t ** 3 + 10.5 * t ** 2 + 6.5 * t, abs=1e-9)
        assert pts[f][1] == 0.0
        assert pts[f][3] == "KINEMATIC_BRIDGED" and pts[f][4] == 9


def test_bridge_uses_window_velocities_and_clamps_the_bow():
    from src.fot_video import MAX_LATERAL_DEV_M, bridge_kinematic_gaps

    # ball at rest at x=4 (frames 0-4), then at rest at x=14 (frames 14-18): zero boundary
    # velocities -> smoothstep, whose 0.96 m lag behind the chord must be scaled to 0.65 m
    anchors = {f: (4.0, 0.0) for f in range(5)}
    anchors.update({f: (14.0, 0.0) for f in range(14, 19)})
    pts = _track(19, anchors)
    bridge_kinematic_gaps(list(range(19)), pts, fps=FPS)

    assert pts[6][0] == pytest.approx(6.0 - MAX_LATERAL_DEV_M, abs=1e-9)
    assert pts[9][0] == pytest.approx(9.0, abs=1e-9)
    assert pts[12][0] == pytest.approx(12.0 + MAX_LATERAL_DEV_M, abs=1e-9)

    # perpendicular arrival / departure at 50 m/s: the bow is real and still clamped
    anchors = {f: (0.0, -8.0 + 2.0 * f) for f in range(5)}
    anchors.update({f: (10.0, -2.0 * (f - 14)) for f in range(14, 19)})
    pts = _track(19, anchors)
    bridge_kinematic_gaps(list(range(19)), pts, fps=FPS)
    dev = [np.hypot(pts[f][0] - (f - 4), pts[f][1]) for f in range(5, 14)]
    assert max(dev) == pytest.approx(MAX_LATERAL_DEV_M, abs=1e-9)
    assert max(abs(pts[f][1]) for f in range(5, 14)) > 0.3, "the curve must bow sideways"


def test_bridge_limits_and_confidence():
    from src.fot_video import bridge_kinematic_gaps

    for gap, bridged in ((35, True), (36, False)):        # round(1.4 s * 25) = 35 frames
        pts = _track(gap + 2, {0: (0.0, 0.0), gap + 1: (5.0, 0.0)})
        bridge_kinematic_gaps(list(range(gap + 2)), pts, fps=FPS)
        assert (pts[1] is not None) is bridged

    pts = _track(11, {0: (0.0, 0.0), 10: (1.0, 0.0)})
    bridge_kinematic_gaps(list(range(11)), pts, fps=FPS, max_gap_s=0.2)   # 5 frames
    assert pts[5] is None, "max_gap_s must be honoured"

    pts = _track(4, {0: (0.0, 0.0), 3: (1.0, 0.0)})
    pts[3][2] = 0.5
    bridge_kinematic_gaps(list(range(4)), pts, fps=FPS)
    assert pts[1][2] == pytest.approx(0.5 * 0.85), "confidence comes from the weaker anchor"


# --------------------------------------------------------------------------- #
# 7. prune_spikes: every clause of the rule
# --------------------------------------------------------------------------- #
def test_prune_spikes_clauses():
    from src.fot_video import prune_spikes

    # a consistent 35 m/s run: fast in, fast out, but the ends do NOT agree
    pts = _track(4, {0: (0.0, 0.0), 1: (1.4, 0.0), 2: (2.8, 0.0), 3: (4.2, 0.0)})
    prune_spikes(list(range(4)), pts, fps=FPS)
    assert all(pts[f] is not None for f in range(4))

    # fast in only (37.5 m/s in, 12.5 m/s out): not a spike
    pts = _track(3, {0: (0.0, 0.0), 1: (1.5, 0.0), 2: (1.0, 0.0)})
    prune_spikes(list(range(3)), pts, fps=FPS)
    assert pts[1] is not None

    # the same jump seen through neighbours 0.4 s away is outside the 0.25 s window
    pts = _track(21, {0: (0.0, 0.0), 10: (40.0, 0.0), 20: (0.0, 0.0)})
    prune_spikes(list(range(21)), pts, fps=FPS)
    assert pts[10] is not None


@pytest.mark.xfail(strict=True, reason="docstring promises 1-2 frame anchors; only 1 is handled")
def test_prune_spikes_removes_a_two_frame_spike():
    from src.fot_video import prune_spikes

    pts = _track(5, {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (2.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0)})
    prune_spikes(list(range(5)), pts, fps=FPS)
    assert pts[1] is None and pts[2] is None


# --------------------------------------------------------------------------- #
# 8. Tile grid: both axes, several frame sizes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w, h", [(1920, 1080), (1280, 720), (3840, 2160), (321, 320), (320, 320)])
def test_tile_grid_covers_both_axes_without_duplicates(w, h):
    from src.fot_video import compute_tile_coords

    tiles = compute_tile_coords(w, h)
    covered = np.zeros((h, w), dtype=bool)
    for x, y in tiles:
        assert 0 <= x <= w - 320 and 0 <= y <= h - 320
        covered[y:y + 320, x:x + 320] = True
    assert covered.all()
    assert len(set(tiles)) == len(tiles)


def test_tile_grid_has_no_duplicates_on_small_frames():
    """Fixed in review: a frame smaller than a tile used to get the same tile 2-4 times."""
    from src.fot_video import compute_tile_coords

    assert compute_tile_coords(200, 150) == [(0, 0)]


# --------------------------------------------------------------------------- #
# 9. The framework seam: what the video is allowed to change
# --------------------------------------------------------------------------- #
def _base_track(n=12, cfg=CFG):
    ball = np.full((n, 2), np.nan)
    ball[0] = (1.20, 0.5)
    ball[4] = (1.20, 0.5)
    ball[8] = (0.5, 0.5)
    return build_ball_track(ball, np.ones(n, dtype=np.int8), cfg)


def test_merge_without_video_evidence_is_the_identity_on_the_default_track():
    from src.fot_video import merge_video_into_track

    base = _base_track()
    matrices, reproj, valid = _homography(12)
    none_preds = {f: (np.nan, np.nan, 0.0, "NONE", 999) for f in range(12)}
    for preds in ({}, none_preds):
        merged = merge_video_into_track(base, preds, matrices, reproj, valid, CFG)
        for field in ("xy", "source", "age_s", "out_of_pitch"):
            np.testing.assert_array_equal(getattr(merged, field), getattr(base, field))


@pytest.mark.xfail(strict=True, reason="_finalize_track recomputes out_of_pitch after the drop")
def test_merge_keeps_the_pre_drop_out_of_pitch_flags_of_the_kinematic_tier():
    from src.fot_video import merge_video_into_track

    cfg = {**CFG, "fill_long_gap_s": 10.0}
    base = _base_track(cfg=cfg)
    matrices, reproj, valid = _homography(12)
    merged = merge_video_into_track(base, {}, matrices, reproj, valid, cfg)
    np.testing.assert_array_equal(merged.out_of_pitch, base.out_of_pitch)


def test_overlay_ignores_frames_without_a_homography_and_out_of_range_keys():
    from src.fot_video import build_fot_ball_track

    matrices, reproj, valid = _homography(3)
    valid[1] = False
    matrices[1] = np.nan
    hit = (960.0, 540.0, 0.9, "DETECTED", 0)
    track = build_fot_ball_track(3, {0: hit, 1: hit, 2: hit, 7: hit, -1: hit},
                                 matrices, reproj, valid, CFG)
    assert track.source.tolist() == [SOURCE_DETECTOR, SOURCE_NONE, SOURCE_DETECTOR]


def test_emission_confidence_per_video_tier_is_pinned():
    """Characterisation of the documented adoption gap, so any change is deliberate."""
    from src.emission import team_confidence
    from src.fot_video import build_fot_ball_track

    matrices, reproj, valid = _homography(4)
    preds = {0: (960.0, 540.0, 0.9, "DETECTED", 0), 1: (960.0, 540.0, 0.7, "FOT_BRIDGED", 2),
             2: (960.0, 540.0, 0.6, "KINEMATIC_BRIDGED", 3), 3: (960.0, 540.0, 0.9, "DETECTED", 0)}
    conf = team_confidence(build_fot_ball_track(4, preds, matrices, reproj, valid, CFG), CFG)

    bridged = CFG["c0_kinematic"] * np.exp(-(1 / FPS) / CFG["tau_kinematic_s"])
    np.testing.assert_allclose(conf, [0.0, bridged, bridged, 0.0])


def _patched_handler(monkeypatch, fuse, n=600):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "committed_fot_tests", Path(__file__).with_name("test_fot_video.py"))
    committed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(committed)
    F = committed._patch_video_boundary(monkeypatch, n)
    if fuse is not None:
        monkeypatch.setattr(F, "fuse_predictions", fuse)
    return F, committed


def test_handler_writes_the_video_evidence_into_ctx_track(monkeypatch):
    from src.pipeline import Ctx

    F, committed = _patched_handler(monkeypatch, None)
    ctx = Ctx()
    ctx.arrays = committed._synthetic_arrays()
    F.fot_video_track(ctx, {**CFG, "fot_verbose": False})

    src = ctx.track.source
    expected_det = sum(1 for f in range(600)
                       if f % 11 and f % 3 and f not in (100, 150, 200, 250, 300, 350))
    assert int((src == SOURCE_DETECTOR).sum()) == expected_det
    assert int((src == SOURCE_ATD).sum()) == 6
    f = 101                                   # DETECTED by the fake fuse at (960 + f, 540)
    assert src[f] == SOURCE_DETECTOR
    assert ctx.track.xy[f, 0] == pytest.approx((960.0 + f - 160.0) / 1600.0)
    assert ctx.track.xy[f, 1] == pytest.approx((540.0 - 90.0) / 900.0)


def test_spec_without_video_evidence_reproduces_the_default_labels(monkeypatch):
    """No video evidence -> the experiment spec must decode exactly like pipeline_spec.SPEC."""
    import src.data as D
    from pipeline_spec import SPEC
    from pipeline_spec_fot_video import FOT_VIDEO_SPEC
    from src.pipeline import run

    def blind_fuse(fot, det, m, r, v, cfg, dev):
        return {f: (np.nan, np.nan, 0.0, "NONE", 999) for f in det.history}

    F, committed = _patched_handler(monkeypatch, blind_fuse)
    arrays = committed._synthetic_arrays()
    monkeypatch.setattr(D, "load_arrays", lambda force=False: arrays)
    cfg = {**CFG, "fot_verbose": False}

    default_labels, _ = run(cfg, spec=SPEC)
    video_labels, _ = run(cfg, spec=FOT_VIDEO_SPEC)
    np.testing.assert_array_equal(video_labels, default_labels)


# --------------------------------------------------------------------------- #
# 10. choose_window, export and CLI plumbing
# --------------------------------------------------------------------------- #
def test_choose_window_seed_half_validity_and_video_length():
    from src.fot_video import choose_window

    full = np.ones(199_319, dtype=bool)
    cfg = {**CFG, "fot_half": "1st_half", "fot_clip_seconds": 30.0}
    starts = {choose_window(full, {**cfg, "fot_seed": s}, FPS, 199_319)[0] for s in range(8)}
    assert len(starts) == 8, "the seed must move the clip"

    halves = {choose_window(full, {**cfg, "fot_half": "any_half", "fot_seed": s}, FPS, 199_319)[2]
              for s in range(8)}
    assert halves == {1, 2}

    # only 2% of half 1 has a usable homography: the preference loop must find it
    sparse = np.zeros(199_319, dtype=bool)
    sparse[60_000:61_600] = True
    hits = 0
    for s in range(20):
        start, end, _ = choose_window(sparse, {**cfg, "fot_seed": s}, FPS, 199_319)
        hits += sparse[start:end + 1].mean() >= 0.75
    assert hits >= 8, "40 draws at ~2% each should succeed in more than half the seeds"

    for s in range(10):
        start, end, _ = choose_window(full, {**cfg, "fot_seed": s}, FPS, 50_000)
        assert 11_185 <= start <= end <= 49_999, "the clip must stay inside a short video"


def test_export_frame_table_time_column():
    from src.fot_video import export_frame_table

    matrices, reproj, valid = _homography(10)
    table = export_frame_table({9: (10.0, 20.0, 0.9, "DETECTED", 0)}, matrices, reproj, valid,
                               FPS, CFG)
    assert table["time_s"].tolist() == [pytest.approx(9 / FPS)]


def test_cli_arguments_reach_the_run(monkeypatch, tmp_path):
    F, _ = _patched_handler(monkeypatch, None)
    seen = {}

    def spy_window(valid, cfg, fps, total):
        seen["cfg"] = dict(cfg)
        return 0, 49, 2

    def spy_require(names, **kw):
        seen["require"] = kw
        return ["/fake/tactical.mp4", "/fake/homography.csv"]

    def spy_load_h(**kw):
        seen["load_homography"] = kw
        return _homography(600)

    def spy_detect(rt, video, m, r, v, start, end, cfg):
        seen["detect_cfg"], seen["window"] = dict(cfg), (start, end)
        return F.DetectionPass(history={f: (f, 960.0, 540.0, 0.9) for f in range(start, end + 1)},
                               frame_w=1920, frame_h=1080, fps=float(FPS))

    monkeypatch.setattr(F, "choose_window", spy_window)
    monkeypatch.setattr(F, "require", spy_require)
    monkeypatch.setattr(F, "load_homography", spy_load_h)
    monkeypatch.setattr(F, "detect_window", spy_detect)

    rc = F.main(["--csv-only", "--out-dir", str(tmp_path), "--seed", "7", "--half", "any_half",
                 "--seconds", "2", "--profile", "high_precision", "--device", "cpu",
                 "--data-dir", "/somewhere/else", "--prefix", "clip"])
    assert rc == 0
    assert seen["cfg"]["fot_seed"] == 7 and seen["cfg"]["fot_half"] == "any_half"
    assert seen["cfg"]["fot_clip_seconds"] == 2.0
    assert seen["cfg"]["fot_profile"] == "high_precision"
    assert seen["detect_cfg"]["fot_device"] == "cpu"
    assert seen["require"] == {"data_dir": "/somewhere/else"}
    assert seen["load_homography"] == {"data_dir": "/somewhere/else"}
    assert seen["window"] == (0, 49)
    assert [p.name for p in tmp_path.iterdir()] == ["clip.csv"]


# --------------------------------------------------------------------------- #
# 11. Review fixes: behaviour that changed on purpose (2026-11-14)
# --------------------------------------------------------------------------- #
def test_unknown_profile_is_an_error_not_a_silent_fallback(fake_torch):
    """A typo used to select max_coverage — the LESS conservative profile — silently."""
    from src.fot_video import PROFILES, profile_params, run_fot_v2_sequence

    assert profile_params("high_precision") == PROFILES["high_precision"]
    with pytest.raises(ValueError, match="high-precision"):
        profile_params("high-precision")
    with pytest.raises(ValueError):
        run_fot_v2_sequence(_IdentityFoT(), [(0, 960.0, 540.0, 0.9)], W_PX, H_PX, "cpu",
                            profile="max-coverage")


def test_empty_sequence_returns_no_predictions(fake_torch):
    from src.fot_video import run_fot_v2_sequence

    assert run_fot_v2_sequence(_IdentityFoT(), [], W_PX, H_PX, "cpu") == {}


@pytest.mark.parametrize("device, expected", [
    ("cuda:1", 1), ("cuda:0", 0), ("cuda", 0), ("cpu", "cpu"), (None, "cpu"), ("mps", "cpu"),
])
def test_yolo_device_keeps_the_gpu_index(device, expected):
    """The index used to be dropped: every CUDA request ran on GPU 0."""
    from src.fot_video import yolo_device

    assert yolo_device(device) == expected


def test_weights_default_to_the_committed_files_and_can_be_overridden(tmp_path):
    """A retrained detector is selected by config, not by renaming files in models/."""
    from src.fot_video import weight_paths

    rt, fot = weight_paths(CFG)
    assert (rt.name, fot.name) == ("rtdetr_fold0_best.pt", "fot_v2_fold0.pt")
    rt, fot = weight_paths({**CFG, "fot_rtdetr_weights": str(tmp_path / "new.pt")})
    assert rt == tmp_path / "new.pt" and fot.name == "fot_v2_fold0.pt"
    rt, fot = weight_paths({**CFG, "fot_rtdetr_weights": "a.pt", "fot_fot_weights": "b.pt"})
    assert (str(rt), str(fot)) == ("a.pt", "b.pt")


def test_detect_window_refuses_a_video_that_ends_early(fake_cv2):
    """A failed read used to truncate the window silently (or crash later in FoT)."""
    from src.fot_video import detect_window

    matrices, reproj, valid = _homography(30)
    cfg = {**CFG, "fot_device": "cpu", "fot_verbose": False}
    with pytest.raises(IOError, match=r"got 12 of 21 frames"):
        detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 0, 20, cfg)

    det = detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 0, 20,
                        {**cfg, "fot_allow_short_window": True})
    assert sorted(det.history) == list(range(N_VIDEO))


def test_detect_window_runs_the_detector_at_the_configured_input_size(fake_cv2):
    """imgsz follows the checkpoint; the tile -> frame mapping does not depend on it."""
    from src.fot_video import detect_window

    matrices, reproj, valid = _homography(N_VIDEO)
    rt = _FakeRTDETR()
    det = detect_window(rt, "/fake.mp4", matrices, reproj, valid, 0, 3,
                        {**CFG, "fot_device": "cuda:1", "fot_verbose": False,
                         "fot_imgsz": 640})
    assert rt.kwargs["imgsz"] == 640 and rt.kwargs["device"] == 1
    assert det.history[2][1:3] == pytest.approx(BALL[2], abs=1e-3)


def test_frame_candidates_map_back_with_the_real_crop_size():
    """On a frame smaller than a tile the crop is short; xywhn is relative to the crop."""
    from src.fot_video import compute_tile_coords, detect_frame_candidates

    frame = np.zeros((150, 200, 3), dtype=np.uint8)
    frame[97:104, 147:154] = 255                       # 7x7 ball centred on (150, 100)
    tiles = compute_tile_coords(200, 150)
    assert tiles == [(0, 0)]
    cands = detect_frame_candidates(_FakeRTDETR(), frame, tiles, None, "cpu")
    assert len(cands) == 1
    assert cands[0][:2] == pytest.approx((150.0, 100.0), abs=1e-3)
    assert detect_frame_candidates(_FakeRTDETR(), frame, [], None, "cpu") == []


def test_cli_exports_to_the_ignored_runs_folder_and_does_not_render_by_default(
        monkeypatch, tmp_path):
    """Default run: CSV only, under runs/fot_video/ — never an overlay, never the repo root."""
    F, _ = _patched_handler(monkeypatch, None)
    monkeypatch.chdir(tmp_path)

    def no_render(*args, **kwargs):
        raise AssertionError("the overlay video must be opt-in (--render)")

    monkeypatch.setattr(F, "render_clip", no_render)
    assert F.main(["--seed", "1", "--seconds", "2"]) == 0
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")
                     if p.is_file())
    assert len(written) == 1
    assert written[0].startswith("runs/fot_video/play_") and written[0].endswith(".csv")


def test_cli_renders_only_on_request(monkeypatch, tmp_path):
    import pandas as pd

    F, _ = _patched_handler(monkeypatch, None)
    calls = []

    def spy_render(video, preds, m, r, v, out_video, out_csv, fps, title="", cfg=None):
        calls.append((out_video, out_csv))
        return pd.DataFrame({"state": ["DETECTED"]})

    monkeypatch.setattr(F, "render_clip", spy_render)
    assert F.main(["--render", "--out-dir", str(tmp_path), "--prefix", "clip"]) == 0
    assert calls == [(str(tmp_path / "clip.mp4"), str(tmp_path / "clip.csv"))]
    with pytest.raises(SystemExit):
        F.main(["--render", "--csv-only", "--out-dir", str(tmp_path)])


def test_cli_weight_and_size_flags_reach_the_models(monkeypatch, tmp_path):
    F, _ = _patched_handler(monkeypatch, None)
    seen = {}

    def spy_models(cfg):
        seen.update({k: cfg.get(k) for k in ("fot_rtdetr_weights", "fot_fot_weights",
                                             "fot_imgsz")})
        return object(), object(), "cpu"

    monkeypatch.setattr(F, "load_models", spy_models)
    assert F.main(["--out-dir", str(tmp_path), "--rtdetr-weights", "new.pt",
                   "--fot-weights", "fot.pt", "--imgsz", "640"]) == 0
    assert seen == {"fot_rtdetr_weights": "new.pt", "fot_fot_weights": "fot.pt",
                    "fot_imgsz": 640}


def test_fot_state_dict_is_loaded_with_the_restricted_unpickler(monkeypatch, tmp_path):
    """weights_only=True: a tampered FoT file cannot execute code at load time."""
    torch = pytest.importorskip("torch")
    from src.fot_video import build_fot_model

    path = tmp_path / "fot.pt"
    torch.save(build_fot_model(window_size=16).state_dict(), path)
    real_load, seen = torch.load, {}

    def spy_load(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    build_fot_model(window_size=16, weights=path, device="cpu")
    assert seen.get("weights_only") is True


# --------------------------------------------------------------------------- #
# 12. Opt-in switches: defaults reproduce the committed behaviour, options work
# --------------------------------------------------------------------------- #
def _straight_pass(n=40, hole=20):
    from src.fot_video import DetectionPass

    history = {f: (f, 400.0 + 3.0 * f, 500.0, 0.9) for f in range(n)}
    history[hole] = (hole, np.nan, np.nan, 0.0)
    return DetectionPass(history=history, frame_w=W_PX, frame_h=H_PX, fps=float(FPS))


def test_fot_ablation_switch_removes_fot_bridging_only(fake_torch):
    from src.fot_video import fuse_predictions

    matrices, reproj, valid = _homography(40)
    det = _straight_pass()
    with_fot = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid, CFG, "cpu")
    without = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid,
                               {**CFG, "fot_use_fot": False}, "cpu")
    assert with_fot[20][3] == "FOT_BRIDGED"
    assert without[20][3] == "KINEMATIC_BRIDGED", "the Hermite bridge still fills the hole"
    assert without[21][2] == pytest.approx(0.9), "DETECTED keeps its raw score"
    # no model at all is accepted when the switch is off: torch is not needed
    assert fuse_predictions(None, det, matrices, reproj, valid,
                            {**CFG, "fot_use_fot": False}, "cpu")[20][3] == "KINEMATIC_BRIDGED"


def test_rts_switch_keeps_detections_on_their_pixel(fake_torch):
    from src.fot_video import fuse_predictions

    matrices, reproj, valid = _homography(30)
    history = {f: (f, 400.0 + (0.0 if f < 15 else 40.0 * (f - 15)), 500.0, 0.9)
               for f in range(30)}                       # a kick at frame 15
    from src.fot_video import DetectionPass
    det = DetectionPass(history=history, frame_w=W_PX, frame_h=H_PX, fps=float(FPS))
    smoothed = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid, CFG, "cpu")
    raw = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid,
                           {**CFG, "fot_rts_enabled": False}, "cpu")
    assert max(abs(raw[f][0] - history[f][1]) for f in range(30)) < 1e-6
    assert max(abs(smoothed[f][0] - history[f][1]) for f in range(30)) > 1.0
    same = fuse_predictions(_IdentityFoT(), det, matrices, reproj, valid,
                            {**CFG, "fot_rts_q": 18.0, "fot_rts_r_det": 0.22,
                             "fot_rts_r_bridge": 0.85}, "cpu")
    assert same == smoothed, "explicit default noises must equal the default run"


def test_bridged_tier_none_keeps_estimates_out_of_the_track():
    from src.fot_video import build_fot_ball_track

    matrices, reproj, valid = _homography(3)
    preds = {0: (960.0, 540.0, 0.9, "DETECTED", 0), 1: (960.0, 540.0, 0.7, "FOT_BRIDGED", 1),
             2: (960.0, 540.0, 0.6, "KINEMATIC_BRIDGED", 1)}
    default = build_fot_ball_track(3, preds, matrices, reproj, valid, CFG)
    assert default.source.tolist() == [SOURCE_DETECTOR, SOURCE_KINEMATIC, SOURCE_KINEMATIC]
    off = build_fot_ball_track(3, preds, matrices, reproj, valid,
                               {**CFG, "fot_bridged_tier": "none"})
    assert off.source.tolist() == [SOURCE_DETECTOR, SOURCE_NONE, SOURCE_NONE]
    with pytest.raises(ValueError):
        build_fot_ball_track(3, preds, matrices, reproj, valid,
                             {**CFG, "fot_bridged_tier": "detector"})


def test_age_clock_atd_only_matches_the_ball_fill_definition():
    from src.fot_video import merge_video_into_track

    base = _base_track()                                  # ATD at frames 0, 4, 8
    matrices, reproj, valid = _homography(12)
    preds = {11: (960.0, 540.0, 0.9, "DETECTED", 0)}
    merged = merge_video_into_track(base, preds, matrices, reproj, valid, CFG)
    atd_only = merge_video_into_track(base, preds, matrices, reproj, valid,
                                      {**CFG, "fot_age_clock": "atd"})
    assert merged.age_s[11] == 0.0, "default: a video detection resets the clock"
    np.testing.assert_array_equal(atd_only.age_s, base.age_s)
    with pytest.raises(ValueError):
        merge_video_into_track(base, preds, matrices, reproj, valid,
                               {**CFG, "fot_age_clock": "detector"})


def test_detect_window_keeps_every_candidate_for_offline_study(fake_cv2):
    from src.fot_video import detect_window

    matrices, reproj, valid = _homography(N_VIDEO)
    det = detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 0, 3,
                        {**CFG, "fot_device": "cpu", "fot_verbose": False})
    assert sorted(det.candidates) == [0, 1, 2, 3]
    assert det.candidates[2][0][:2] == pytest.approx(BALL[2], abs=1e-3)
    assert det.history[2][1:3] == pytest.approx(det.candidates[2][0][:2])


def test_detect_window_sequential_decode_matches_the_seek(fake_cv2):
    from src.fot_video import detect_window

    grabs = []

    def grab(self):
        grabs.append(self.pos)
        self.pos += 1
        return self.pos <= N_VIDEO

    fake_cv2.VideoCapture.grab = grab
    matrices, reproj, valid = _homography(N_VIDEO)
    cfg = {**CFG, "fot_device": "cpu", "fot_verbose": False}
    seek = detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 4, 8, cfg)
    seq = detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 4, 8,
                        {**cfg, "fot_decode": "sequential"})
    assert grabs == [0, 1, 2, 3]
    assert seq.history == seek.history
    with pytest.raises(ValueError):
        detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 4, 8,
                      {**cfg, "fot_decode": "fast"})
    with pytest.raises(ValueError, match="empty window"):
        detect_window(_FakeRTDETR(), "/fake.mp4", matrices, reproj, valid, 8, 4, cfg)


def test_choose_window_explains_a_video_shorter_than_the_half():
    from src.fot_video import choose_window

    with pytest.raises(ValueError, match="truncated"):
        choose_window(np.ones(199_319, dtype=bool), {**CFG, "fot_half": "2nd_half"},
                      FPS, 50_000)


def test_reprojection_never_reports_a_nan_position_as_detected():
    from src.fot_video import STATE_NONE, reproject_predictions

    matrices, reproj, valid = _homography(2)
    matrices[1] = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0.5]])   # x=0.5 -> w=0
    pts = {0: [10.0, 10.0, 0.9, "DETECTED", 0], 1: [52.65, 10.0, 0.9, "DETECTED", 0]}
    preds = reproject_predictions([0, 1], pts, matrices, reproj, valid, 105.3, 68.0)
    assert preds[0][3] == "DETECTED"
    assert preds[1][3] == STATE_NONE and np.isnan(preds[1][0])


def test_every_module_tunable_has_a_provenance_entry():
    """New thresholds must be declared with their origin (competition rule b)."""
    import src.fot_video as F

    imported = {"CFG", "SOURCE_ATD", "SOURCE_DETECTOR", "SOURCE_KINEMATIC", "SOURCE_NONE"}
    names = [n for n, v in vars(F).items()
             if n.isupper() and not n.startswith("_") and isinstance(v, (int, float, dict))
             and not n.startswith("STATE_") and n not in imported
             and n not in ("PROGRESS_EVERY", "TUNABLE_PROVENANCE")]
    missing = sorted(set(names) - set(F.TUNABLE_PROVENANCE))
    assert not missing, f"add these to TUNABLE_PROVENANCE: {missing}"
    for name, entry in F.TUNABLE_PROVENANCE.items():
        assert hasattr(F, name), name
        assert entry["origin"] in ("checkpoint", "run", "notebook-unrecorded")
        assert isinstance(entry["score_scale"], bool)
    assert F.TUNABLE_PROVENANCE["DET_CONF_FLOOR"]["score_scale"] is True
