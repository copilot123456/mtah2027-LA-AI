"""FoT v2 + RT-DETR ball tracking from the tactical video — framework adapter.

Extracted from ``notebooks/FotV2_videotest.ipynb`` (left untouched as the experiment
record) and reshaped to the team contract (PIPELINE_STANDARD_AI.md):

* pure stage functions with docstrings; no framework file is modified — this module
  is *added*, and ``pipeline_spec_fot_video.py`` declares the experiment spec that
  swaps it in (standard recipe 4.1);
* homography IO/projection is reused from ``starter/load_homography.py`` and asset
  paths from ``games.py`` — the notebook's local copies of those helpers are gone;
* geometry comes from ``config.CFG`` (fps, pitch dimensions, margin), so the notebook's
  literal ``105.0`` metres becomes the measured ``CFG['pitch_length_m']`` (105.3);
  the notebook's own knobs stay module constants or ``cfg.get("fot_*")`` overrides,
  which leaves ``config.py`` and its provenance table untouched;
* the output is a ``src.ball_fill.BallTrack`` carrying the framework's provenance
  tiers, so the existing emission / decode / x_policy / summary stages consume it
  unchanged;
* torch, ultralytics and cv2 are imported inside the functions that need them:
  importing this module requires only numpy + pandas (the framework's runtime).

Adoption gap, documented rather than worked around: ``src/emission.py`` gives
``SOURCE_DETECTOR`` zero team confidence ("confidence TBD when that tier is
implemented"), so detector frames steer the dead-ball features and the kinematic
confidence decay but contribute no H/A evidence. Giving the tier a confidence is a
framework change (a new CFG key plus an emission line) and follows PIPELINE_STANDARD
§7 — it is not done here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from games import GAME, half_video_blocks, require
from src.ball_fill import (SOURCE_ATD, SOURCE_DETECTOR, SOURCE_KINEMATIC, SOURCE_NONE,
                           BallTrack, build_ball_track)
from starter.load_homography import load as load_homography
from starter.load_homography import project, unproject

# --------------------------------------------------------------------------- #
# Parameters. Stage-local, NOT config.CFG keys: config.py is frozen framework
# territory (tests/test_config.py owns its provenance table). Defaults are the
# notebook's values; override per run via cfg["fot_*"].
# --------------------------------------------------------------------------- #
TILE_SIZE = 320                 # RT-DETR tile edge, pixels
TILE_STRIDE = 260               # 60 px overlap between tiles
WINDOW_SIZE = 64                # FoT v2 sequence window
PROGRESS_EVERY = 125            # frames between progress lines

MASK_MARGIN = 0.05              # notebook MARGIN: +5% pitch margin for the video mask
DET_CONF_FLOOR = 0.07           # low floor: rescue partly occluded balls near the track
FAR_CONF_FLOOR = 0.18           # stricter floor away from the track (filters distant boots)
MAX_BALL_SPEED_MPS = 36.0       # physical ceiling (~130 km/h)
BASE_GATE_RADIUS_M = 3.2        # dribble/rebound radius, metres
JUMP_CONFIRM_FRAMES = 3         # consecutive frames required to accept a far jump
MAX_KINEMATIC_GAP_S = 1.4       # memory for reconstructing occlusions, seconds
MAX_REPROJ_ERROR = 5.0          # homography fit error tolerated for a frame
SPIKE_SPEED_MPS = 32.0          # isolated 1-2 frame anchors above this are pruned
MAX_LATERAL_DEV_M = 0.65        # allowed curve bow versus the straight chord, metres
BRIDGE_SPEED_MPS = 35.0         # chord speed ceiling for a kinematic bridge

# profile -> (detector input threshold, max bridgeable gap, FoT confidence threshold)
PROFILES = {
    "max_coverage": (0.30, 4, 0.45),
    "high_precision": (0.40, 2, 0.72),
}

DEFAULT_SEED = 0                # reproducible clip selection
DEFAULT_HALF = "1st_half"       # "1st_half", "2nd_half" or "any_half"
DEFAULT_CLIP_SECONDS = 30.0

STATE_DETECTED = "DETECTED"
STATE_FOT = "FOT_BRIDGED"
STATE_KINEMATIC = "KINEMATIC_BRIDGED"
STATE_NONE = "NONE"


@dataclass(frozen=True)
class DetectionPass:
    """One tiled RT-DETR pass: per-frame candidates in full-frame pixels + geometry."""

    history: dict        # frame -> (frame, gx, gy, conf); NaN gx/gy where nothing was kept
    frame_w: int
    frame_h: int
    fps: float


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch for a reproducible clip and pass."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Geometry and pitch masks
# --------------------------------------------------------------------------- #
def get_valid_H(fr_idx: int, matrices: np.ndarray, reproj_err: np.ndarray,
                valid: np.ndarray, max_reproj: float = MAX_REPROJ_ERROR):
    """The frame's homography, or None where it is missing, invalid or too distorted."""
    if 0 <= fr_idx < len(valid) and valid[fr_idx] and reproj_err[fr_idx] <= max_reproj:
        H = matrices[fr_idx]
        if not np.isnan(H).any():
            return H
    return None


def densify_rect_norm(x0: float, y0: float, x1: float, y1: float, steps: int = 32):
    """Dense point ring around a normalised-pitch rectangle (for fillPoly)."""
    xs = np.concatenate([
        np.linspace(x0, x1, steps, endpoint=False), np.full(steps, x1),
        np.linspace(x1, x0, steps, endpoint=False), np.full(steps, x0),
    ])
    ys = np.concatenate([
        np.full(steps, y0), np.linspace(y0, y1, steps, endpoint=False),
        np.full(steps, y1), np.linspace(y1, y0, steps, endpoint=False),
    ])
    return xs, ys


def project_polygon_px(H: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Polygon of normalised points in pixels, keeping only the visible side; None if degenerate."""
    denom = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    center_denom = H[2, 0] * 0.5 + H[2, 1] * 0.5 + H[2, 2]
    same_side = (denom * center_denom) > 1e-8
    if same_side.sum() < 3:
        return None
    px, py = project(H, xs[same_side], ys[same_side])
    return np.round(np.column_stack([px, py])).astype(np.int32)


def build_pitch_masks(H: np.ndarray, frame_w: int, frame_h: int,
                      margin: float = MASK_MARGIN):
    """(mask_margin, mask_exact): uint8 images of the pitch with and without the margin."""
    mask_margin = np.zeros((frame_h, frame_w), dtype=np.uint8)
    mask_exact = np.zeros((frame_h, frame_w), dtype=np.uint8)

    xm, ym = densify_rect_norm(-margin, -margin, 1.0 + margin, 1.0 + margin, steps=32)
    poly_margin = project_polygon_px(H, xm, ym)
    if poly_margin is not None:
        cv2_fill_poly(mask_margin, poly_margin)

    xe, ye = densify_rect_norm(0.0, 0.0, 1.0, 1.0, steps=32)
    poly_exact = project_polygon_px(H, xe, ye)
    if poly_exact is not None:
        cv2_fill_poly(mask_exact, poly_exact)

    return mask_margin, mask_exact


def cv2_fill_poly(mask: np.ndarray, poly: np.ndarray) -> None:
    """cv2.fillPoly with the import kept local so this module stays torch/cv2-free."""
    import cv2

    cv2.fillPoly(mask, [poly], 255)


# --------------------------------------------------------------------------- #
# Detection helpers: tiling, NMS, kinematic gate
# --------------------------------------------------------------------------- #
def compute_tile_coords(frame_w: int, frame_h: int, tile_size: int = TILE_SIZE,
                        stride: int = TILE_STRIDE):
    """Top-left corners of the tile grid covering the frame, last tile flush with each edge."""
    xs = list(range(0, max(1, frame_w - tile_size + 1), stride))
    if xs[-1] != frame_w - tile_size:
        xs.append(max(0, frame_w - tile_size))
    ys = list(range(0, max(1, frame_h - tile_size + 1), stride))
    if ys[-1] != frame_h - tile_size:
        ys.append(max(0, frame_h - tile_size))
    return [(x, y) for y in ys for x in xs]


def nms_candidates(cands: list, min_dist_px: float = 14.0) -> list:
    """Drop duplicate detections from overlapping tiles, keeping the most confident."""
    if not cands:
        return []
    cands_sorted = sorted(cands, key=lambda c: c[2], reverse=True)
    kept = []
    for gx, gy, sc in cands_sorted:
        if all(np.hypot(gx - kx, gy - ky) >= min_dist_px for kx, ky, _ in kept):
            kept.append((gx, gy, sc))
    return kept


class PitchKinematicGate:
    """Filter boot false positives with kinematics in metres on the pitch.

    * A candidate inside the physically reachable radius of the last seen ball blends
      visual confidence with spatial closeness.
    * A distant candidate (a boot elsewhere) needs JUMP_CONFIRM_FRAMES coherent frames
      before the track moves, unless it is an unequivocal hit (>= 0.68).
    * Confirmed far jumps rescue their buffered frames retroactively into
      ``det_history``, the notebook's anti-ghosting behaviour.
    """

    def __init__(self, fps: float, pitch_length_m: float, pitch_width_m: float):
        self.fps = fps
        self.dt = 1.0 / fps
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.last_fr = None
        self.last_mx = None
        self.last_my = None
        self.vx = 0.0  # m/s
        self.vy = 0.0  # m/s
        self.jump_buffer = []  # [(fr, gx, gy, sc, mx, my)]

    def _to_meters(self, H, gx, gy):
        if H is None:
            return None, None
        nx, ny = unproject(H, gx, gy)
        return float(nx) * self.pitch_length_m, float(ny) * self.pitch_width_m

    def _update_track(self, fr: int, mx: float, my: float):
        if self.last_fr is not None and mx is not None:
            dt_sec = max(1, fr - self.last_fr) * self.dt
            if dt_sec <= 0.4:
                new_vx = np.clip((mx - self.last_mx) / dt_sec,
                                 -MAX_BALL_SPEED_MPS, MAX_BALL_SPEED_MPS)
                new_vy = np.clip((my - self.last_my) / dt_sec,
                                 -MAX_BALL_SPEED_MPS, MAX_BALL_SPEED_MPS)
                self.vx = 0.65 * self.vx + 0.35 * new_vx
                self.vy = 0.65 * self.vy + 0.35 * new_vy
            else:
                self.vx, self.vy = 0.0, 0.0
        self.last_fr = fr
        self.last_mx = mx
        self.last_my = my
        self.jump_buffer.clear()

    def step(self, fr: int, candidates: list, H: np.ndarray, det_history: dict):
        """(gx, gy, conf) kept for this frame, or (NaN, NaN, 0.0) when nothing qualifies."""
        if not candidates or H is None:
            if not candidates:
                return np.nan, np.nan, 0.0
            gx, gy, sc = candidates[0]
            return (gx, gy, sc) if sc >= FAR_CONF_FLOOR else (np.nan, np.nan, 0.0)

        enriched = []
        for gx, gy, sc in candidates:
            mx, my = self._to_meters(H, gx, gy)
            enriched.append((gx, gy, sc, mx, my))

        # No recent track (> 1.5 s without the ball): demand confidence or confirmation.
        has_active_track = (
            self.last_fr is not None
            and self.last_mx is not None
            and (fr - self.last_fr) <= int(round(1.5 * self.fps))
        )

        if has_active_track:
            gap_frames = max(1, fr - self.last_fr)
            dt_sec = gap_frames * self.dt
            damp = 0.90 ** gap_frames
            pred_mx = self.last_mx + self.vx * dt_sec * damp
            pred_my = self.last_my + self.vy * dt_sec * damp
            max_reach_m = BASE_GATE_RADIUS_M + MAX_BALL_SPEED_MPS * dt_sec
            sigma_m = 1.6 + 0.45 * gap_frames

            in_gate = []
            for gx, gy, sc, mx, my in enriched:
                d_m = np.hypot(mx - pred_mx, my - pred_my)
                if d_m <= max_reach_m:
                    spatial_w = float(np.exp(-0.5 * (d_m / sigma_m) ** 2))
                    eff_score = sc * (0.35 + 0.65 * spatial_w)
                    in_gate.append((eff_score, sc, gx, gy, mx, my))

            if in_gate:
                in_gate.sort(key=lambda item: item[0], reverse=True)
                _, best_sc, best_gx, best_gy, best_mx, best_my = in_gate[0]
                self._update_track(fr, best_mx, best_my)
                return best_gx, best_gy, best_sc

        # Outside the kinematic radius: a distant boot, or a real ball transition.
        far_cands = [c for c in enriched if c[2] >= FAR_CONF_FLOOR]
        if not far_cands:
            return np.nan, np.nan, 0.0

        # An unequivocal hit (e.g. a high ball > 0.68) is accepted immediately.
        top_far = far_cands[0]
        if top_far[2] >= 0.68:
            self._update_track(fr, top_far[3], top_far[4])
            return top_far[0], top_far[1], top_far[2]

        # Temporal confirmation, to reject intermittent boots.
        self.jump_buffer = [b for b in self.jump_buffer
                            if (fr - b[0]) <= JUMP_CONFIRM_FRAMES + 1]
        best_cluster = None
        for gx, gy, sc, mx, my in far_cands:
            matches = [b for b in self.jump_buffer if np.hypot(mx - b[4], my - b[5]) <= 3.5]
            if best_cluster is None or len(matches) > len(best_cluster[1]):
                best_cluster = ((fr, gx, gy, sc, mx, my), matches)

        cur_item, matches = best_cluster
        self.jump_buffer.append(cur_item)

        if len(matches) + 1 >= JUMP_CONFIRM_FRAMES:
            # Confirmed: rescue the buffered frames into det_history retroactively.
            for b_fr, b_gx, b_gy, b_sc, _, _ in matches:
                det_history[b_fr] = (b_fr, b_gx, b_gy, b_sc)
            self._update_track(fr, cur_item[4], cur_item[5])
            return cur_item[1], cur_item[2], cur_item[3]

        return np.nan, np.nan, 0.0


# --------------------------------------------------------------------------- #
# FoT v2 model
# --------------------------------------------------------------------------- #
def build_window_features(x_raw, y_raw, score_raw, valid_mask):
    """(feats, baseline_xy) for one window: 12 interpolated features + the interpolation itself.

    Pure numpy; the all-missing window degrades to a stationary ball at the centre
    instead of producing NaNs.
    """
    W = len(x_raw)
    idx = np.arange(W)
    obs_idx = idx[valid_mask]

    if len(obs_idx) == 0:
        x_interp = np.full(W, 0.5, dtype=np.float32)
        y_interp = np.full(W, 0.5, dtype=np.float32)
        dt_prev = np.ones(W, dtype=np.float32)
        dt_next = np.ones(W, dtype=np.float32)
        has_both_anchors = np.zeros(W, dtype=np.float32)
    else:
        x_interp = np.interp(idx, obs_idx, x_raw[obs_idx]).astype(np.float32)
        y_interp = np.interp(idx, obs_idx, y_raw[obs_idx]).astype(np.float32)

        last_seen = np.full(W, -W, dtype=np.float32)
        next_seen = np.full(W, 2 * W, dtype=np.float32)
        cur = -W
        for t in range(W):
            if valid_mask[t]:
                cur = t
            last_seen[t] = cur
        cur = 2 * W
        for t in range(W - 1, -1, -1):
            if valid_mask[t]:
                cur = t
            next_seen[t] = cur

        dt_prev = np.clip((idx - last_seen) / float(W), 0.0, 1.0).astype(np.float32)
        dt_next = np.clip((next_seen - idx) / float(W), 0.0, 1.0).astype(np.float32)
        has_both_anchors = ((last_seen >= 0) & (next_seen < W)).astype(np.float32)

    vx = np.gradient(x_interp) * 10.0
    vy = np.gradient(y_interp) * 10.0
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    is_obs = valid_mask.astype(np.float32)
    sc = np.where(valid_mask, score_raw, 0.0).astype(np.float32)
    gap_len = np.where(valid_mask, 0.0,
                       np.clip(dt_prev + dt_next, 0.0, 1.0)).astype(np.float32)

    feats = np.stack([
        x_interp, y_interp, is_obs, dt_prev, dt_next,
        vx, vy, ax, ay, sc, gap_len, has_both_anchors
    ], axis=-1)
    baseline_xy = np.stack([x_interp, y_interp], axis=-1)
    return feats, baseline_xy


def build_fot_model(window_size: int = WINDOW_SIZE, weights=None, device="cpu"):
    """Instantiate FootballTransformerV2 and optionally load the fold-0 checkpoint.

    torch is imported here, not at module scope, so the framework's dependency-light
    runtime never needs it unless this stage actually runs.
    """
    import torch
    import torch.nn as nn

    class FootballTransformerV2(nn.Module):
        """Transformer over one detection window: refined (x, y) + hit-within-5 logit."""

        def __init__(self, in_features=12, d_model=160, nhead=8, num_layers=4,
                     max_len=128, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.coord_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2)
            )
            self.hit5_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )

        def forward(self, feats, baseline_xy):
            W = feats.shape[1]
            h = self.input_proj(feats) + self.pos_embed[:, :W, :]
            h = self.encoder(h)
            pred_xy = baseline_xy + self.coord_head(h)
            hit5_logits = self.hit5_head(h).squeeze(-1)
            return pred_xy, hit5_logits

    model = FootballTransformerV2(in_features=12, d_model=160, nhead=8,
                                  num_layers=4, max_len=window_size)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_fot_v2_sequence(fot_model, raw_dets, frame_w, frame_h, device,
                        profile: str = "max_coverage"):
    """Hann-windowed FoT v2 inference over one (sub-sampled) detection sequence.

    ``raw_dets`` is a list of ``(frame, gx, gy, conf)``. Returns ``frame -> (px, py,
    conf, state, gap)``: kept detections ride through as DETECTED with boosted
    confidence, bridged frames as FOT_BRIDGED, everything else NONE.
    """
    import torch

    det_input_thr, max_gap, conf_thr = PROFILES.get(profile, PROFILES["max_coverage"])

    N = len(raw_dets)
    obs_x = np.zeros(N, dtype=np.float32)
    obs_y = np.zeros(N, dtype=np.float32)
    scores = np.zeros(N, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    for i, (_, gx, gy, sc) in enumerate(raw_dets):
        if sc >= det_input_thr and not np.isnan(gx):
            obs_x[i], obs_y[i], scores[i], valid[i] = gx / frame_w, gy / frame_h, sc, True

    # Drop isolated mid-gap anchors: fast in and fast out while the ends agree.
    max_jump_px = 160.0 * (frame_w / 1920.0)
    v_idx = np.where(valid)[0]
    for k in range(1, len(v_idx) - 1):
        i_p, i_c, i_n = v_idx[k - 1], v_idx[k], v_idx[k + 1]
        s1, s2 = max(1, i_c - i_p), max(1, i_n - i_c)
        v1 = np.hypot((obs_x[i_c] - obs_x[i_p]) * frame_w,
                      (obs_y[i_c] - obs_y[i_p]) * frame_h) / s1
        v2 = np.hypot((obs_x[i_n] - obs_x[i_c]) * frame_w,
                      (obs_y[i_n] - obs_y[i_c]) * frame_h) / s2
        if v1 > max_jump_px and v2 > max_jump_px and scores[i_c] < 0.72:
            valid[i_c] = False

    win_weights = np.hanning(WINDOW_SIZE + 2)[1:-1].astype(np.float32)
    acc_xy = np.zeros((N, 2), dtype=np.float32)
    acc_p5 = np.zeros(N, dtype=np.float32)
    acc_w = np.zeros(N, dtype=np.float32)

    starts = list(range(0, max(1, N - WINDOW_SIZE + 1), 16))
    if N > WINDOW_SIZE and starts[-1] != N - WINDOW_SIZE:
        starts.append(N - WINDOW_SIZE)

    with torch.no_grad():
        for s0 in starts:
            s1 = min(N, s0 + WINDOW_SIZE)
            cur_len = s1 - s0
            if cur_len < WINDOW_SIZE:
                pad = WINDOW_SIZE - cur_len
                fx = np.pad(obs_x[s0:s1], (0, pad), mode="edge")
                fy = np.pad(obs_y[s0:s1], (0, pad), mode="edge")
                fsc = np.pad(scores[s0:s1], (0, pad), mode="constant")
                fv = np.pad(valid[s0:s1], (0, pad), mode="constant")
            else:
                fx, fy, fsc, fv = obs_x[s0:s1], obs_y[s0:s1], scores[s0:s1], valid[s0:s1]

            feats, base_xy = build_window_features(fx, fy, fsc, fv)
            p_xy, h5_log = fot_model(
                torch.from_numpy(feats).unsqueeze(0).to(device),
                torch.from_numpy(base_xy).unsqueeze(0).to(device)
            )
            p_xy = p_xy.squeeze(0).cpu().numpy()[:cur_len]
            p_h5 = torch.sigmoid(h5_log).squeeze(0).cpu().numpy()[:cur_len]

            w_sl = win_weights[:cur_len]
            acc_xy[s0:s1] += p_xy * w_sl[:, None]
            acc_p5[s0:s1] += p_h5 * w_sl
            acc_w[s0:s1] += w_sl

    final_xy = acc_xy / np.maximum(acc_w[:, None], 1e-6)
    final_p5 = acc_p5 / np.maximum(acc_w, 1e-6)

    det_idx = np.where(valid)[0]
    results = {}
    for i, (fr_idx, _, _, _) in enumerate(raw_dets):
        if valid[i]:
            px, py = float(obs_x[i] * frame_w), float(obs_y[i] * frame_h)
            conf = float(max(scores[i], 0.5 * scores[i] + 0.5 * final_p5[i]))
            results[fr_idx] = (px, py, conf, STATE_DETECTED, 0)
        else:
            pos = np.searchsorted(det_idx, i)
            if 0 < pos < len(det_idx):
                gap_steps = int(det_idx[pos] - det_idx[pos - 1] - 1)
                if gap_steps <= max_gap and final_p5[i] >= conf_thr:
                    px, py = float(final_xy[i, 0] * frame_w), float(final_xy[i, 1] * frame_h)
                    results[fr_idx] = (px, py, float(final_p5[i]), STATE_FOT, gap_steps)
                else:
                    results[fr_idx] = (np.nan, np.nan, 0.0, STATE_NONE, gap_steps)
            else:
                results[fr_idx] = (np.nan, np.nan, 0.0, STATE_NONE, 999)
    return results


# --------------------------------------------------------------------------- #
# Stage 1: tiled RT-DETR pass over a frame window
# --------------------------------------------------------------------------- #
def models_dir() -> Path:
    """First existing models/ directory (repo root, then cwd, then cwd's parent)."""
    root = Path(__file__).resolve().parent.parent
    candidates = [root / "models", Path.cwd() / "models", Path.cwd().parent / "models"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "no models/ directory found; checked: " + ", ".join(str(c) for c in candidates))


def load_models(cfg: dict = CFG):
    """(rt_model, fot_model, device) — RT-DETR and FoT v2 weights from models/."""
    import torch
    from ultralytics import RTDETR

    device = cfg.get("fot_device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mdir = models_dir()
    rt_weights = mdir / "rtdetr_fold0_best.pt"
    fot_weights = mdir / "fot_v2_fold0.pt"
    missing = [str(p) for p in (rt_weights, fot_weights) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing FoT v2 weights:\n  " + "\n  ".join(missing)
            + "\n(the .pt checkpoints are not committed; place them in " + str(mdir) + ")")

    rt_model = RTDETR(str(rt_weights))
    fot_model = build_fot_model(WINDOW_SIZE, weights=fot_weights, device=device)
    return rt_model, fot_model, device


def video_geometry(video: str):
    """(fps, total_frames) from the container, without decoding any frame."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def choose_window(valid: np.ndarray, cfg: dict, fps: float,
                  n_video_frames: int) -> tuple[int, int, int]:
    """Seeded clip ``(start, end, half)`` inside the chosen half.

    Notebook behaviour: draw a seeded start, then keep the first of 40 candidates whose
    homography is valid over >= 75% of the clip. The half spans come from
    ``games.half_video_blocks`` instead of the notebook's hand-typed ROIs.
    """
    seed = int(cfg.get("fot_seed", DEFAULT_SEED))
    half_choice = str(cfg.get("fot_half", DEFAULT_HALF))
    clip_seconds = float(cfg.get("fot_clip_seconds", DEFAULT_CLIP_SECONDS))

    rng = random.Random(seed)
    blocks = {h: (first, last) for h, first, last in half_video_blocks(GAME)}
    if half_choice == "any_half":
        half = rng.choice([1, 2])
    elif half_choice in ("1st_half", "2nd_half"):
        half = 1 if half_choice.startswith("1") else 2
    else:
        raise ValueError(f"fot_half must be 1st_half, 2nd_half or any_half, got {half_choice!r}")

    roi_start, roi_end = blocks[half]
    roi_end = min(roi_end, n_video_frames - 1)

    clip_len_frames = int(round(clip_seconds * fps))
    max_start = max(roi_start, roi_end - clip_len_frames)

    best_start = rng.randint(roi_start, max_start)
    for _ in range(40):
        cand_start = rng.randint(roi_start, max_start)
        sl = slice(cand_start, min(len(valid), cand_start + clip_len_frames))
        if valid[sl].mean() >= 0.75:
            best_start = cand_start
            break

    start = best_start
    end = min(roi_end, start + clip_len_frames - 1)
    return start, end, half


def detect_window(rt_model, video: str, matrices: np.ndarray, reproj_err: np.ndarray,
                  valid: np.ndarray, start: int, end: int,
                  cfg: dict = CFG) -> DetectionPass:
    """Run tiled RT-DETR + the kinematic gate over video frames ``[start, end]``.

    Every frame of the window gets a history entry (NaN where nothing was kept), so
    downstream stages can bridge the gaps. Heavy cv2 use stays inside this function.
    """
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    device = cfg.get("fot_device")
    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    yolo_device = 0 if str(device).startswith("cuda") else "cpu"

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    all_tiles = compute_tile_coords(frame_w, frame_h, TILE_SIZE, TILE_STRIDE)
    gate = PitchKinematicGate(fps=fps,
                              pitch_length_m=float(cfg["pitch_length_m"]),
                              pitch_width_m=float(cfg["pitch_width_m"]))

    history: dict = {}
    verbose = bool(cfg.get("fot_verbose", True))
    t0 = time.perf_counter()

    for fr_idx in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        H_cur = get_valid_H(fr_idx, matrices, reproj_err, valid)
        if H_cur is not None:
            mask_margin, _ = build_pitch_masks(H_cur, frame_w, frame_h, margin=MASK_MARGIN)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask_margin)
            active_tiles = [
                (tx, ty) for (tx, ty) in all_tiles
                if mask_margin[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE].any()
            ]
        else:
            mask_margin = None
            masked_frame = frame
            active_tiles = all_tiles

        if len(active_tiles) == 0:
            history[fr_idx] = (fr_idx, np.nan, np.nan, 0.0)
            continue

        crops = [masked_frame[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                 for (tx, ty) in active_tiles]
        results = rt_model.predict(
            source=crops, imgsz=TILE_SIZE, conf=DET_CONF_FLOOR, max_det=4,
            device=yolo_device, verbose=False
        )

        frame_cands = []
        for (tx, ty), r in zip(active_tiles, results):
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                xywhn = r.boxes.xywhn.cpu().numpy()
                for i in range(len(confs)):
                    sc = float(confs[i])
                    gx = tx + float(xywhn[i, 0]) * TILE_SIZE
                    gy = ty + float(xywhn[i, 1]) * TILE_SIZE
                    if mask_margin is not None:
                        ix = min(max(int(round(gx)), 0), frame_w - 1)
                        iy = min(max(int(round(gy)), 0), frame_h - 1)
                        if mask_margin[iy, ix] == 0:
                            continue
                    frame_cands.append((gx, gy, sc))

        frame_cands = nms_candidates(frame_cands, min_dist_px=14.0)
        gx, gy, sc = gate.step(fr_idx, frame_cands, H_cur, history)
        history[fr_idx] = (fr_idx, gx, gy, sc)

        if verbose and (fr_idx - start + 1) % PROGRESS_EVERY == 0:
            print(f"  Procesados {fr_idx - start + 1}/{end - start + 1} frames"
                  f" | {time.perf_counter() - t0:.1f}s")

    cap.release()
    return DetectionPass(history=history, frame_w=frame_w, frame_h=frame_h, fps=fps)


# --------------------------------------------------------------------------- #
# Stage 2: FoT v2 fusion + pitch-space smoothing and bridging
# --------------------------------------------------------------------------- #
def prune_spikes(all_f: list, pts: dict, fps: float) -> None:
    """Null 1-2 frame anchors that are fast in and fast out but whose ends agree (boots).

    ``pts`` maps frame -> [mx, my, conf, state, gap] in metres; modified in place.
    """
    valid_f = [f for f in all_f if pts[f] is not None]
    for i in range(1, len(valid_f) - 1):
        f_prev, f_cur, f_next = valid_f[i - 1], valid_f[i], valid_f[i + 1]
        # A frame pruned by an earlier iteration leaves a hole in valid_f; the
        # notebook's one-shot list would dereference None here. Skipping is
        # behaviour-identical on every input where the original did not crash.
        if pts[f_prev] is None or pts[f_cur] is None or pts[f_next] is None:
            continue
        dt1 = (f_cur - f_prev) / fps
        dt2 = (f_next - f_cur) / fps
        dt_tot = (f_next - f_prev) / fps
        if dt1 <= 0.25 and dt2 <= 0.25:
            p0, p1, p2 = pts[f_prev], pts[f_cur], pts[f_next]
            v_in = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(dt1, 1e-3)
            v_out = np.hypot(p2[0] - p1[0], p2[1] - p1[1]) / max(dt2, 1e-3)
            v_direct = np.hypot(p2[0] - p0[0], p2[1] - p0[1]) / max(dt_tot, 1e-3)
            if v_in > SPIKE_SPEED_MPS and v_out > SPIKE_SPEED_MPS and v_direct < 22.0:
                pts[f_cur] = None


def rts_smooth_segment(seg_frames: list, pts: dict, fps: float) -> None:
    """Bidirectional Kalman (RTS) smoothing of (mx, my) with a constant-velocity model.

    Measurement noise differs by provenance: 0.22 m for DETECTED anchors, 0.85 m for
    bridged ones. ``pts`` is modified in place; segments shorter than 3 are left alone.
    """
    n = len(seg_frames)
    if n < 3:
        return
    dt = 1.0 / fps
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=np.float64)
    H_obs = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
    q = 18.0  # process acceleration noise: allows real pass-speed changes without zig-zag
    G = np.array([[0.5 * dt ** 2, 0],
                  [0, 0.5 * dt ** 2],
                  [dt, 0],
                  [0, dt]], dtype=np.float64)
    Q = (G @ G.T) * q

    xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
    p0 = pts[seg_frames[0]]
    x = np.array([p0[0], p0[1], 0.0, 0.0], dtype=np.float64)
    P = np.diag([0.5, 0.5, 25.0, 25.0])

    for k, fr in enumerate(seg_frames):
        if k > 0:
            step_k = max(1, fr - seg_frames[k - 1])
            Fk = np.eye(4)
            Fk[0, 2] = step_k * dt
            Fk[1, 3] = step_k * dt
            x = Fk @ x
            P = Fk @ P @ Fk.T + Q * step_k
        xs_p.append(x.copy())
        Ps_p.append(P.copy())

        obs = pts[fr]
        r_std = 0.22 if obs[3] == STATE_DETECTED else 0.85
        R = np.eye(2) * (r_std ** 2)
        z = np.array([obs[0], obs[1]], dtype=np.float64)
        y_res = z - H_obs @ x
        S = H_obs @ P @ H_obs.T + R
        K = P @ H_obs.T @ np.linalg.inv(S)
        x = x + K @ y_res
        P = (np.eye(4) - K @ H_obs) @ P
        xs_f.append(x.copy())
        Ps_f.append(P.copy())

    xs_s = [None] * n
    xs_s[-1] = xs_f[-1]
    for k in range(n - 2, -1, -1):
        step_k = max(1, seg_frames[k + 1] - seg_frames[k])
        Fk = np.eye(4)
        Fk[0, 2] = step_k * dt
        Fk[1, 3] = step_k * dt
        C = Ps_f[k] @ Fk.T @ np.linalg.inv(Ps_p[k + 1])
        xs_s[k] = xs_f[k] + C @ (xs_s[k + 1] - xs_p[k + 1])

    for k, fr in enumerate(seg_frames):
        pts[fr][0] = float(xs_s[k][0])
        pts[fr][1] = float(xs_s[k][1])


def smooth_rts(all_f: list, pts: dict, fps: float) -> None:
    """Apply :func:`rts_smooth_segment` to every contiguous block of tracked frames."""
    cur_seg = []
    for f in all_f:
        if pts[f] is not None:
            cur_seg.append(f)
        else:
            if len(cur_seg) >= 3:
                rts_smooth_segment(cur_seg, pts, fps)
            cur_seg = []
    if len(cur_seg) >= 3:
        rts_smooth_segment(cur_seg, pts, fps)


def bridge_kinematic_gaps(all_f: list, pts: dict, fps: float,
                          max_gap_s: float = MAX_KINEMATIC_GAP_S) -> None:
    """Fill tracked gaps with a damped Hermite curve, bowed no more than
    MAX_LATERAL_DEV_M off the straight chord and only when the chord speed is
    physically plausible. ``pts`` is modified in place; bridged frames get
    KINEMATIC_BRIDGED and the neighbouring confidences * 0.85.
    """
    max_gap_frames = int(round(max_gap_s * fps))
    idx_ptr = 0

    while idx_ptr < len(all_f):
        if pts[all_f[idx_ptr]] is not None:
            idx_ptr += 1
            continue

        gap_start = idx_ptr
        while idx_ptr < len(all_f) and pts[all_f[idx_ptr]] is None:
            idx_ptr += 1
        gap_end = idx_ptr - 1

        if gap_start > 0 and idx_ptr < len(all_f):
            f_prev = all_f[gap_start - 1]
            f_next = all_f[idx_ptr]
            gap_len = f_next - f_prev - 1
            dt_span_s = (f_next - f_prev) / fps

            if 1 <= gap_len <= max_gap_frames:
                p_prev = pts[f_prev]
                p_next = pts[f_next]
                dx_chord = p_next[0] - p_prev[0]
                dy_chord = p_next[1] - p_prev[1]
                chord_m = np.hypot(dx_chord, dy_chord)

                if (chord_m / max(dt_span_s, 1e-3)) <= BRIDGE_SPEED_MPS:
                    # Boundary velocities from a robust window of up to 4 frames each side.
                    w_in = [all_f[j] for j in range(max(0, gap_start - 5), gap_start)
                            if pts[all_f[j]] is not None]
                    w_out = [all_f[j] for j in range(idx_ptr, min(len(all_f), idx_ptr + 5))
                             if pts[all_f[j]] is not None]

                    if len(w_in) >= 2:
                        dt_in = (w_in[-1] - w_in[0]) / fps
                        vx0 = (pts[w_in[-1]][0] - pts[w_in[0]][0]) / max(dt_in, 1e-3)
                        vy0 = (pts[w_in[-1]][1] - pts[w_in[0]][1]) / max(dt_in, 1e-3)
                    else:
                        vx0, vy0 = dx_chord / dt_span_s, dy_chord / dt_span_s

                    if len(w_out) >= 2:
                        dt_out = (w_out[-1] - w_out[0]) / fps
                        vx1 = (pts[w_out[-1]][0] - pts[w_out[0]][0]) / max(dt_out, 1e-3)
                        vy1 = (pts[w_out[-1]][1] - pts[w_out[0]][1]) / max(dt_out, 1e-3)
                    else:
                        vx1, vy1 = dx_chord / dt_span_s, dy_chord / dt_span_s

                    # Damped tangents so passes and rebounds do not loop.
                    m0x, m0y = vx0 * dt_span_s * 0.65, vy0 * dt_span_s * 0.65
                    m1x, m1y = vx1 * dt_span_s * 0.65, vy1 * dt_span_s * 0.65

                    ts = np.array([(all_f[k] - f_prev) / float(f_next - f_prev)
                                   for k in range(gap_start, gap_end + 1)])
                    h00 = 2 * ts ** 3 - 3 * ts ** 2 + 1
                    h10 = ts ** 3 - 2 * ts ** 2 + ts
                    h01 = -2 * ts ** 3 + 3 * ts ** 2
                    h11 = ts ** 3 - ts ** 2

                    hx = h00 * p_prev[0] + h10 * m0x + h01 * p_next[0] + h11 * m1x
                    hy = h00 * p_prev[1] + h10 * m0y + h01 * p_next[1] + h11 * m1y

                    # Straight chord reference; scale the curve back if it bows too far.
                    lx = p_prev[0] + ts * dx_chord
                    ly = p_prev[1] + ts * dy_chord

                    dev_m = np.hypot(hx - lx, hy - ly)
                    max_dev = float(dev_m.max()) if len(dev_m) > 0 else 0.0
                    blend_curve = min(1.0, MAX_LATERAL_DEV_M / max(max_dev, 1e-6))

                    final_mx = lx + blend_curve * (hx - lx)
                    final_my = ly + blend_curve * (hy - ly)
                    c_base = min(p_prev[2], p_next[2]) * 0.85

                    for idx_local, k in enumerate(range(gap_start, gap_end + 1)):
                        fk = all_f[k]
                        pts[fk] = [float(final_mx[idx_local]), float(final_my[idx_local]),
                                   c_base, STATE_KINEMATIC, gap_len]


def reproject_predictions(all_f: list, pts: dict, matrices: np.ndarray,
                          reproj_err: np.ndarray, valid: np.ndarray,
                          pitch_length_m: float, pitch_width_m: float) -> dict:
    """Smooth pitch metres back to per-frame pixel predictions (D of the notebook)."""
    preds = {}
    for f in all_f:
        p = pts[f]
        H_f = get_valid_H(f, matrices, reproj_err, valid)
        if p is not None and H_f is not None:
            px_f, py_f = project(H_f, p[0] / pitch_length_m, p[1] / pitch_width_m)
            preds[f] = (float(px_f), float(py_f), p[2], p[3], p[4])
        else:
            preds[f] = (np.nan, np.nan, 0.0, STATE_NONE, 999)
    return preds


def fuse_predictions(fot_model, det_pass: DetectionPass, matrices: np.ndarray,
                     reproj_err: np.ndarray, valid: np.ndarray, cfg: dict,
                     device) -> dict:
    """Full fusion of one detection pass (notebook section 5).

    Even/odd FoT v2 passes -> pitch metres -> spike pruning -> RTS smoothing ->
    Hermite gap bridging -> reprojection. Returns ``frame -> (px, py, conf, state,
    gap)`` for every frame of the window.
    """
    fps = det_pass.fps
    length = float(cfg["pitch_length_m"])
    width = float(cfg["pitch_width_m"])
    profile = str(cfg.get("fot_profile", "max_coverage"))

    ordered = [det_pass.history[f] for f in sorted(det_pass.history)]
    fot_even = run_fot_v2_sequence(fot_model, ordered[0::2], det_pass.frame_w,
                                   det_pass.frame_h, device, profile=profile)
    fot_odd = run_fot_v2_sequence(fot_model, ordered[1::2], det_pass.frame_w,
                                  det_pass.frame_h, device, profile=profile)

    raw_merged = {}
    raw_merged.update(fot_even)
    raw_merged.update(fot_odd)
    all_f = sorted(raw_merged.keys())

    # A: pitch metres, dropping anchors with no valid homography.
    pts = {}
    for f in all_f:
        px, py, conf, st, gap = raw_merged[f]
        H_f = get_valid_H(f, matrices, reproj_err, valid)
        if st != STATE_NONE and not np.isnan(px) and H_f is not None:
            nx, ny = unproject(H_f, px, py)
            pts[f] = [float(nx) * length, float(ny) * width, float(conf), st, int(gap)]
        else:
            pts[f] = None

    # B: prune boots, then smooth each tracked block.
    prune_spikes(all_f, pts, fps)
    smooth_rts(all_f, pts, fps)

    # C: bounded Hermite reconstruction of what is still missing.
    bridge_kinematic_gaps(all_f, pts, fps,
                          max_gap_s=float(cfg.get("fot_max_kinematic_gap_s",
                                                   MAX_KINEMATIC_GAP_S)))

    # D: back to pixels.
    return reproject_predictions(all_f, pts, matrices, reproj_err, valid, length, width)


# --------------------------------------------------------------------------- #
# Stage 3: the notebook's output as a framework BallTrack
# --------------------------------------------------------------------------- #
def _overlay_video(xy: np.ndarray, source: np.ndarray, preds: dict,
                   matrices: np.ndarray, reproj_err: np.ndarray,
                   valid: np.ndarray) -> None:
    """Write video evidence into empty frames only; ATD and existing fills are never touched."""
    for f, (px, py, conf, state, gap) in preds.items():
        if not (0 <= f < len(source)):
            continue
        if state == STATE_NONE or np.isnan(px) or np.isnan(py):
            continue
        if source[f] != SOURCE_NONE:
            continue  # raw ATD or an existing fill wins over the video
        H_f = get_valid_H(f, matrices, reproj_err, valid)
        if H_f is None:
            continue
        nx, ny = unproject(H_f, px, py)
        if not (np.isfinite(nx) and np.isfinite(ny)):
            continue
        xy[f] = (float(nx), float(ny))
        source[f] = SOURCE_DETECTOR if state == STATE_DETECTED else SOURCE_KINEMATIC


def _finalize_track(xy: np.ndarray, source: np.ndarray, cfg: dict) -> BallTrack:
    """age_s to the nearest observation, the out-of-pitch flag, and the drop rule.

    Mirrors ``src.ball_fill``: positions are flagged before nulling, and only
    *estimated* (kinematic) out-of-pitch positions are dropped — observations are
    flagged, never removed. ``age_s`` counts to the nearest *raw* observation (ATD
    or video detection); fills never reset the clock, so a bridge's confidence decays
    exactly as it does for the default track.
    """
    n = len(source)
    fps = int(cfg["fps"])

    margin = float(cfg["pitch_margin"])
    has_pos = ~np.isnan(xy).any(axis=1)
    outside = ((xy[:, 0] < -margin) | (xy[:, 0] > 1 + margin)
               | (xy[:, 1] < -margin) | (xy[:, 1] > 1 + margin)) & has_pos

    drop = (source == SOURCE_KINEMATIC) & outside
    xy[drop] = np.nan
    source[drop] = SOURCE_NONE

    idx = np.arange(n)
    raw = (source == SOURCE_ATD) | (source == SOURCE_DETECTOR)
    obs = np.flatnonzero(raw)
    if obs.size:
        pos = np.searchsorted(obs, idx)
        right = obs[np.clip(pos, 0, obs.size - 1)]
        left = obs[np.clip(pos - 1, 0, obs.size - 1)]
        age_s = np.minimum(np.abs(idx - left), np.abs(idx - right)) / float(fps)
    else:
        age_s = np.full(n, np.inf)

    return BallTrack(xy=xy, source=source, age_s=age_s, out_of_pitch=outside)


def build_fot_ball_track(n_frames: int, preds: dict, matrices: np.ndarray,
                         reproj_err: np.ndarray, valid: np.ndarray,
                         cfg: dict = CFG) -> BallTrack:
    """Video-only BallTrack: every frame starts as `none`, video evidence fills it.

    Useful on its own and for tests; :func:`merge_video_into_track` is what the
    pipeline step uses, because it keeps the ATD as the base.
    """
    xy = np.full((n_frames, 2), np.nan, dtype=np.float64)
    source = np.full(n_frames, SOURCE_NONE, dtype=np.int8)
    _overlay_video(xy, source, preds, matrices, reproj_err, valid)
    return _finalize_track(xy, source, cfg)


def merge_video_into_track(base: BallTrack, preds: dict, matrices: np.ndarray,
                           reproj_err: np.ndarray, valid: np.ndarray,
                           cfg: dict = CFG) -> BallTrack:
    """Fill the gaps of an ATD-based track with FoT v2 video evidence.

    Rules adopted from ``src.ball_fill``: the raw ATD is never overwritten, video
    evidence only lands where the base track has none, and ``age_s`` counts to the
    nearest raw observation (ATD or video detection), so fills decay exactly as they
    do on the default track.
    """
    xy = base.xy.copy()
    source = base.source.copy()
    _overlay_video(xy, source, preds, matrices, reproj_err, valid)
    return _finalize_track(xy, source, cfg)


def predictions_frame_table(preds: dict, fps: float) -> pd.DataFrame:
    """One row per frame: pixel prediction, confidence, provenance state, gap length."""
    frames = sorted(preds)
    rows = [(f, round(f / fps, 4), *preds[f]) for f in frames]
    return pd.DataFrame(rows, columns=[
        "frame", "time_s", "pred_x", "pred_y", "score", "state", "gap_steps",
    ])


# --------------------------------------------------------------------------- #
# Optional visualisation (kept from the notebook, still not a pipeline stage)
# --------------------------------------------------------------------------- #
def draw_pitch_overlay(frame: np.ndarray, H: np.ndarray, mask_margin: np.ndarray,
                       mask_exact: np.ndarray, black_outside: bool = True,
                       plane_color=(220, 50, 220), line_color=(255, 240, 255),
                       alpha: float = 0.24) -> np.ndarray:
    """Pitch plane, markings and mask contour drawn onto a copy of the frame."""
    import cv2

    out = cv2.bitwise_and(frame, frame, mask=mask_margin) if black_outside else frame.copy()

    idx_plane = mask_exact > 0
    if np.any(idx_plane):
        tint = np.full_like(out[idx_plane], plane_color, dtype=np.uint8)
        out[idx_plane] = cv2.addWeighted(out[idx_plane], 1.0 - alpha, tint, alpha, 0)

    def seg(x0, y0, x1, y1, steps=24):
        return np.linspace(x0, x1, steps), np.linspace(y0, y1, steps)

    pa_x = 16.5 / 105.0
    pa_y0, pa_y1 = 0.5 - 20.16 / 68.0, 0.5 + 20.16 / 68.0
    ga_x = 5.5 / 105.0
    ga_y0, ga_y1 = 0.5 - 9.16 / 68.0, 0.5 + 9.16 / 68.0
    theta = np.linspace(0, 2 * np.pi, 72)

    lines_norm = [
        seg(0.0, 0.0, 1.0, 0.0), seg(1.0, 0.0, 1.0, 1.0),
        seg(1.0, 1.0, 0.0, 1.0), seg(0.0, 1.0, 0.0, 0.0),
        seg(0.5, 0.0, 0.5, 1.0),
        (0.5 + (9.15 / 105.0) * np.cos(theta), 0.5 + (9.15 / 68.0) * np.sin(theta)),
        seg(0.0, pa_y0, pa_x, pa_y0), seg(pa_x, pa_y0, pa_x, pa_y1), seg(pa_x, pa_y1, 0.0, pa_y1),
        seg(1.0, pa_y0, 1.0 - pa_x, pa_y0), seg(1.0 - pa_x, pa_y0, 1.0 - pa_x, pa_y1),
        seg(1.0 - pa_x, pa_y1, 1.0, pa_y1),
        seg(0.0, ga_y0, ga_x, ga_y0), seg(ga_x, ga_y0, ga_x, ga_y1), seg(ga_x, ga_y1, 0.0, ga_y1),
        seg(1.0, ga_y0, 1.0 - ga_x, ga_y0), seg(1.0 - ga_x, ga_y0, 1.0 - ga_x, ga_y1),
        seg(1.0 - ga_x, ga_y1, 1.0, ga_y1),
    ]

    center_denom = H[2, 0] * 0.5 + H[2, 1] * 0.5 + H[2, 2]
    for lx, ly in lines_norm:
        denom = H[2, 0] * lx + H[2, 1] * ly + H[2, 2]
        same_side = (denom * center_denom) > 1e-8
        if same_side.sum() >= 2:
            px_l, py_l = project(H, lx[same_side], ly[same_side])
            pts = np.round(np.column_stack([px_l, py_l])).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=line_color, thickness=2,
                          lineType=cv2.LINE_AA)

    for sx, sy in [(0.5, 0.5), (11.0 / 105.0, 0.5), (1.0 - 11.0 / 105.0, 0.5)]:
        denom_s = H[2, 0] * sx + H[2, 1] * sy + H[2, 2]
        if (denom_s * center_denom) > 1e-8:
            cx_s, cy_s = project(H, sx, sy)
            if 0 <= cx_s < out.shape[1] and 0 <= cy_s < out.shape[0]:
                cv2.circle(out, (int(round(cx_s)), int(round(cy_s))), 4, line_color, -1,
                           lineType=cv2.LINE_AA)

    contours, _ = cv2.findContours(mask_margin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (160, 240, 240), 1, lineType=cv2.LINE_AA)
    return out


STATE_COLORS = {
    STATE_DETECTED: (0, 230, 118),     # green: RT-DETR detection kept by the gate
    STATE_FOT: (255, 215, 0),          # gold: FoT v2 temporal reconstruction
    STATE_KINEMATIC: (0, 140, 255),    # orange: pitch-space memory (Hermite)
}


def render_clip(video: str, preds: dict, matrices: np.ndarray, reproj_err: np.ndarray,
                valid: np.ndarray, out_video: str, out_csv: str, fps: float,
                title: str = "FoT v2") -> pd.DataFrame:
    """Write the annotated clip and its prediction CSV.

    ``preds`` must be one contiguous frame run (what :func:`detect_window` produces).
    Returns the frame table that was written to ``out_csv``.
    """
    import cv2

    frames = sorted(preds)
    if not frames:
        raise ValueError("no predictions to render")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame_w, frame_h))

    trail = []
    prev_xy = None
    for fr_idx in frames:
        ret, frame = cap.read()
        if not ret:
            break

        H_cur = get_valid_H(fr_idx, matrices, reproj_err, valid)
        if H_cur is not None:
            mask_margin, mask_exact = build_pitch_masks(H_cur, frame_w, frame_h,
                                                        margin=MASK_MARGIN)
            frame = draw_pitch_overlay(frame, H_cur, mask_margin, mask_exact)

        px, py, conf, state, gap = preds[fr_idx]
        speed_px_s = 0.0
        if state != STATE_NONE and not np.isnan(px):
            if prev_xy is not None:
                speed_px_s = float(np.hypot(px - prev_xy[0], py - prev_xy[1]) * fps)
            prev_xy = (px, py)
            trail.append((int(round(px)), int(round(py)), state))
            if len(trail) > 22:
                trail.pop(0)
        else:
            prev_xy = None
            if trail:
                trail.pop(0)

        for t_i in range(1, len(trail)):
            pt1 = (trail[t_i - 1][0], trail[t_i - 1][1])
            pt2 = (trail[t_i][0], trail[t_i][1])
            if np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < 150:
                col = STATE_COLORS.get(trail[t_i][2], STATE_COLORS[STATE_DETECTED])
                cv2.line(frame, pt1, pt2, col,
                         max(1, int(round(3 * (t_i / len(trail))))), lineType=cv2.LINE_AA)

        if state != STATE_NONE and not np.isnan(px):
            cx_i, cy_i = int(round(px)), int(round(py))
            color = STATE_COLORS.get(state, STATE_COLORS[STATE_DETECTED])
            if state == STATE_DETECTED:
                tag = f"BALL {conf:.2f}"
            elif state == STATE_FOT:
                tag = f"FoT {conf:.2f} (g{gap})"
            else:
                tag = f"MEM {conf:.2f} (g{gap})"
            cv2.circle(frame, (cx_i, cy_i), 11, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx_i, cy_i), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, tag, (cx_i + 14, cy_i + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, color, 2, cv2.LINE_AA)

        hud = f"[{title}] Fr {fr_idx} | {fr_idx / fps:06.2f}s | {state:17s} | {speed_px_s:5.0f} px/s"
        cv2.rectangle(frame, (12, 12), (820, 68), (15, 15, 15), -1)
        cv2.putText(frame, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    table = predictions_frame_table(preds, fps)
    table.to_csv(out_csv, index=False)
    return table


# --------------------------------------------------------------------------- #
# Step handler: fn(ctx, cfg) -> None, with a docstring (validate_spec enforces it)
# --------------------------------------------------------------------------- #
def fot_video_track(ctx, cfg: dict) -> None:
    """Ball stage from the video: tiled RT-DETR + FoT v2 over a seeded clip, merged
    onto the ATD track (raw ATD never overwritten).

    The `ball_fill` swap of pipeline_spec_fot_video.py. Stores the BallTrack in
    ``ctx.track`` and the clip window, frame table and state counts in
    ``ctx.extra["fot_video"]`` for inspection. Requires the gated video and the
    models/ checkpoints; fails with a readable error when either is missing.
    """
    arrays = ctx.arrays
    set_global_seed(int(cfg.get("fot_seed", DEFAULT_SEED)))

    video, _ = require(["video", "homography"])
    matrices, reproj_err, valid = load_homography()

    fps, n_video_frames = video_geometry(video)
    start, end, half = choose_window(valid, cfg, fps, n_video_frames)

    rt_model, fot_model, device = load_models(cfg)
    det_pass = detect_window(rt_model, video, matrices, reproj_err, valid,
                             start, end, cfg)
    preds = fuse_predictions(fot_model, det_pass, matrices, reproj_err, valid,
                             cfg, device)

    base = build_ball_track(arrays.ball, arrays.half, cfg)
    ctx.track = merge_video_into_track(base, preds, matrices, reproj_err, valid, cfg)

    table = predictions_frame_table(preds, det_pass.fps)
    counts = table["state"].value_counts().to_dict()
    ctx.extra["fot_video"] = {
        "half": half,
        "start_frame": start,
        "end_frame": end,
        "fps": det_pass.fps,
        "predictions": preds,
        "table": table,
        "state_counts": counts,
    }

    if bool(cfg.get("fot_verbose", True)):
        n_total = len(preds)
        n_none = int(counts.get(STATE_NONE, 0))
        print(f"FoT v2 clip [{half}] frames [{start} .. {end}] @ {det_pass.fps:.1f} fps")
        print(f"  coverage {100.0 * (n_total - n_none) / max(1, n_total):.1f}%"
              f" ({n_total - n_none}/{n_total} frames)")
        for state in (STATE_DETECTED, STATE_FOT, STATE_KINEMATIC, STATE_NONE):
            print(f"  {state:18s}: {int(counts.get(state, 0))}")
