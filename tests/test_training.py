"""Tests for the training package (``training/``).

Same design rule as ``tests/test_fot_video.py``: no gated data, no real video, no
GPU. The pure planning / scoring / validation functions run for real on synthetic
inputs; tests that need torch or cv2 skip themselves where those are absent
(``requirements-dev.txt`` does not pin them).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.fot_video as F
from training import compliance, eval_detector, fot_data, inspect_checkpoint, manifest
from training import tiles, train_rtdetr

ROOT = Path(__file__).resolve().parents[1]
RTDETR_PT = ROOT / "models" / "rtdetr_fold0_best.pt"
FOT_PT = ROOT / "models" / "fot_v2_fold0.pt"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_manifest(folder: Path, n_games: int = 3, **top) -> Path:
    """A valid manifest with empty stand-in videos and a tiny annotation per game."""
    games = []
    for index in range(1, n_games + 1):
        game_dir = folder / f"game{index}"
        game_dir.mkdir(parents=True, exist_ok=True)
        (game_dir / "video.mp4").write_bytes(b"not a real video")
        (game_dir / "ball.csv").write_text(
            "frame,x,y,w,h\n0,100,100,8,8\n2,110,104,8,8\n4,120,108,8,8\n")
        games.append({"id": str(index), "video": f"game{index}/video.mp4",
                      "annotations": f"game{index}/ball.csv"})
    content = {"dataset": "unit-test set", "license": "CC0-1.0",
               "source_url": "https://example.org/dataset", "games": games}
    content.update(top)
    target = folder / "manifest.json"
    target.write_text(json.dumps(content))
    return target


# --------------------------------------------------------------------------- #
# compliance: the provided match must be unreachable from training code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [
    "data/tactical/BAR-ATH_tactical.mp4",
    "data/broadcast/anything.mp4",
    "data/tactical",
    "/somewhere/else/BAR-ATH_tactical.mp4",
    "/copies/bar-ath_broadcast_atd.txt",
    "/tmp/bar-ath/clip.mp4",
    "evaluation/gt/BAR-ATH_possession_gt_public.csv",
    "evaluation/gt/whatever.csv",
])
def test_provided_material_is_refused(path):
    assert compliance.is_provided(ROOT / path if not path.startswith("/") else path)
    with pytest.raises(compliance.ProvidedDataError):
        compliance.assert_external(ROOT / path if not path.startswith("/") else path)


@pytest.mark.parametrize("path", [
    "data/external/game4/video.mp4",
    "data/idsse/match.xml",
    "runs/rtdetr/fold_0",
    "/datasets/public/game9/ball.csv",
])
def test_external_paths_are_accepted(path):
    target = ROOT / path if not path.startswith("/") else path
    assert not compliance.is_provided(target)
    compliance.assert_external(target, None)   # None entries are ignored


def test_guard_follows_the_live_data_dir_env(monkeypatch, tmp_path):
    """A bundle moved with MTAH_DATA_DIR is protected even after import."""
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path))
    assert compliance.is_provided(tmp_path / "tactical" / "renamed.mp4")
    assert not compliance.is_provided(tmp_path / "external" / "g1.mp4")


def test_guard_sees_through_a_symlink(tmp_path):
    """A harmless-looking link that points into the bundle is still refused."""
    link = tmp_path / "public_game.mp4"
    try:
        link.symlink_to(ROOT / "data" / "tactical" / "x.mp4")
    except OSError:
        pytest.skip("symlinks not available")
    assert compliance.is_provided(link)


# --------------------------------------------------------------------------- #
# manifest: provenance is mandatory, the split is by game
# --------------------------------------------------------------------------- #
def test_manifest_loads_and_resolves_relative_paths(tmp_path):
    loaded = manifest.load_manifest(_write_manifest(tmp_path))
    assert loaded.ids() == ["1", "2", "3"]
    assert loaded.game(2).video == tmp_path / "game2" / "video.mp4"
    assert loaded.license == "CC0-1.0"
    with pytest.raises(manifest.ManifestError):
        loaded.game("99")


@pytest.mark.parametrize("field,value", [
    ("dataset", "<public dataset name>"), ("license", ""), ("license", "TBD"),
    ("source_url", "unknown"),
])
def test_manifest_rejects_undocumented_data(tmp_path, field, value):
    """No name / licence / source, no training: the report could not defend it."""
    target = _write_manifest(tmp_path, **{field: value})
    with pytest.raises(manifest.ManifestError, match=field):
        manifest.load_manifest(target)


def test_manifest_reports_every_problem_at_once(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps({
        "dataset": "x", "license": "y",
        "games": [{"id": "1", "annotations": "a.csv"},
                  {"id": "1", "video": "v.mp4", "frames_dir": "d", "annotations": "a"}]}))
    with pytest.raises(manifest.ManifestError) as error:
        manifest.load_manifest(target)
    text = str(error.value)
    assert "source_url" in text and "exactly one of" in text and "duplicate" in text


def test_manifest_refuses_the_provided_match_as_a_game(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps({
        "dataset": "x", "license": "y", "source_url": "z",
        "games": [{"id": "1", "video": str(ROOT / "data/tactical/BAR-ATH_tactical.mp4"),
                   "annotations": "a.csv"}]}))
    with pytest.raises(compliance.ProvidedDataError):
        manifest.load_manifest(target, check_files=False)


def test_annotations_are_canonicalised(tmp_path):
    game_dir = tmp_path / "g"
    game_dir.mkdir()
    (game_dir / "ball.csv").write_text(
        "Frame, X ,y,visible\n5,10.5,20,1\n3,1,2,1\n7,9,9,0\n")
    game = manifest.Game(id="1", annotations=game_dir / "ball.csv",
                         video=game_dir / "v.mp4")
    table = manifest.load_annotations(game)
    assert table["frame"].tolist() == [3, 5]            # sorted, invisible row dropped
    assert list(table.columns) == ["frame", "x", "y", "w", "h"]
    assert table["w"].isna().all()                      # no size column -> NaN


def test_annotations_reject_two_balls_in_one_frame(tmp_path):
    (tmp_path / "ball.csv").write_text("frame,x,y\n1,1,1\n1,2,2\n")
    game = manifest.Game(id="1", annotations=tmp_path / "ball.csv",
                         video=tmp_path / "v.mp4")
    with pytest.raises(manifest.ManifestError, match="more than once"):
        manifest.load_annotations(game)


def test_split_is_by_game_and_never_overlaps(tmp_path):
    loaded = manifest.load_manifest(_write_manifest(tmp_path))
    folds = manifest.leave_one_game_out(loaded)
    assert [f["val_games"] for f in folds] == [["1"], ["2"], ["3"]]
    for fold in folds:
        assert not set(fold["val_games"]) & set(fold["train_games"])
    assert manifest.resolve_split(loaded, fold=1) == (["1", "3"], ["2"])
    with pytest.raises(manifest.ManifestError, match="both train and validation"):
        manifest.resolve_split(loaded, ["1", "2"], ["2"])
    with pytest.raises(manifest.ManifestError):
        manifest.resolve_split(loaded, ["1"], [])


def test_fingerprint_changes_with_the_labels(tmp_path):
    target = _write_manifest(tmp_path)
    before = manifest.fingerprint(manifest.load_manifest(target))
    assert before == manifest.fingerprint(manifest.load_manifest(target))
    (tmp_path / "game1" / "ball.csv").write_text("frame,x,y\n0,1,1\n")
    assert before != manifest.fingerprint(manifest.load_manifest(target))


def test_convert_mot_centres_boxes_and_keeps_one_ball(tmp_path):
    source = tmp_path / "gt.txt"
    # frame,id,left,top,w,h,conf,class  (class 1 = ball here; id 7 is a player)
    source.write_text("1,3,100,200,10,12,1,1\n"
                      "1,7,300,300,40,90,1,2\n"
                      "2,3,110,205,10,12,1,1\n"
                      "2,4,500,500,20,20,1,1\n")
    out = manifest.convert_mot(source, tmp_path / "ball.csv", class_ids=[1], frame_add=-1)
    assert out["frame"].tolist() == [0, 1]
    assert out.loc[0, ["x", "y", "w", "h"]].tolist() == [105.0, 206.0, 10.0, 12.0]
    assert out.loc[1, "x"] == 510.0                     # the larger box wins the frame
    with pytest.raises(manifest.ManifestError):
        manifest.convert_mot(source, tmp_path / "b.csv")   # must say what the ball is


def test_convert_csv_handles_top_left_and_normalised(tmp_path):
    source = tmp_path / "src.csv"
    source.write_text("f,left,top,bw,bh\n10,0.5,0.25,0.01,0.02\n")
    out = manifest.convert_csv(source, tmp_path / "ball.csv", frame="f", x="left",
                               y="top", w="bw", h="bh", top_left=True,
                               normalized_size=(1000, 500))
    row = out.iloc[0]
    assert (row["frame"], row["w"], row["h"]) == (10, 10.0, 10.0)
    assert (row["x"], row["y"]) == (505.0, 130.0)


# --------------------------------------------------------------------------- #
# tiles: the training grid is the inference grid
# --------------------------------------------------------------------------- #
def test_training_tiles_use_the_inference_grid_and_size():
    assert tiles.TILE_SIZE is F.TILE_SIZE and tiles.TILE_STRIDE is F.TILE_STRIDE
    rng = np.random.default_rng(0)
    plans = tiles.plan_frame(1920, 1080, (700.0, 400.0, 8.0, 8.0), rng,
                             neg_per_pos=3.0)
    grid = set(F.compute_tile_coords(1920, 1080, F.TILE_SIZE, F.TILE_STRIDE))
    assert plans and all((p.tx, p.ty) in grid for p in plans)


def test_every_tile_containing_the_ball_is_a_positive():
    """A ball in an overlap zone is seen by several tiles at inference; train on all."""
    rng = np.random.default_rng(0)
    ball = (290.0, 290.0, 8.0, 8.0)          # inside the 60 px overlap of 4 tiles
    plans = tiles.plan_frame(1920, 1080, ball, rng, neg_per_pos=0.0)
    positives = [p for p in plans if p.kind == tiles.KIND_POS]
    assert {(p.tx, p.ty) for p in positives} == {(0, 0), (260, 0), (0, 260), (260, 260)}
    for plan in positives:
        cx, cy, w, h = plan.label
        assert plan.tx + cx * F.TILE_SIZE == pytest.approx(290.0)
        assert plan.ty + cy * F.TILE_SIZE == pytest.approx(290.0)
        assert w * F.TILE_SIZE == pytest.approx(8.0) and 0 < h <= 1


def test_label_is_clipped_at_the_tile_border():
    label = tiles.label_for_tile(0, 0, (2.0, 100.0, 10.0, 10.0))
    cx, cy, w, h = label
    assert w * F.TILE_SIZE == pytest.approx(7.0)        # 3 px of the box are outside
    assert cx * F.TILE_SIZE == pytest.approx(3.5)
    assert tiles.label_for_tile(0, 0, (320.0, 100.0, 10.0, 10.0)) is None  # right edge


def test_background_tiles_never_touch_the_ball():
    ball = (700.0, 400.0, 12.0, 12.0)
    box = tiles.ball_box(*ball)
    for seed in range(20):
        plans = tiles.plan_frame(1920, 1080, ball, np.random.default_rng(seed),
                                 neg_per_pos=4.0)
        negatives = [p for p in plans if p.kind == tiles.KIND_NEG]
        assert negatives
        assert all(p.label is None for p in negatives)
        assert not any(tiles.tile_touches_box(p.tx, p.ty, box) for p in negatives)
        assert len({(p.tx, p.ty) for p in plans}) == len(plans)     # no duplicates


def test_empty_frames_yield_only_background():
    plans = tiles.plan_frame(1920, 1080, None, np.random.default_rng(1),
                             neg_per_empty=2.0)
    assert len(plans) == 2 and all(p.kind == tiles.KIND_NEG for p in plans)


def test_missing_box_size_falls_back_to_the_declared_default():
    assert tiles.ball_box(5.0, 6.0, np.nan, np.nan, box_px=9.0) == (5.0, 6.0, 9.0, 9.0)
    assert tiles.ball_box(5.0, 6.0, 7.0, 3.0, box_px=9.0) == (5.0, 6.0, 7.0, 3.0)


def test_frame_smaller_than_a_tile_is_skipped():
    assert tiles.plan_frame(300, 200, (10.0, 10.0, 8.0, 8.0),
                            np.random.default_rng(0)) == []


def test_planning_is_deterministic_per_seed():
    ball = (900.0, 500.0, 8.0, 8.0)
    first = tiles.plan_frame(1920, 1080, ball, np.random.default_rng([0, 1, 250]), 2.0)
    again = tiles.plan_frame(1920, 1080, ball, np.random.default_rng([0, 1, 250]), 2.0)
    assert first == again


def test_hard_negative_adds_the_tile_where_the_false_alarm_is_most_central():
    ball = (1500.0, 800.0, 8.0, 8.0)
    plans = tiles.plan_frame(1920, 1080, ball, np.random.default_rng(0),
                             neg_per_pos=0.0, hard_points=[(290.0, 290.0)])
    hard = [p for p in plans if p.kind == tiles.KIND_HARD_NEG]
    assert len(hard) == 1 and hard[0].label is None
    assert (hard[0].tx, hard[0].ty) in {(0, 0), (260, 0), (0, 260), (260, 260)}
    # a "false alarm" sitting on the real ball is not a negative
    plans = tiles.plan_frame(1920, 1080, ball, np.random.default_rng(0),
                             neg_per_pos=0.0, hard_points=[(1501.0, 801.0)])
    assert not [p for p in plans if p.kind == tiles.KIND_HARD_NEG]


def test_hard_negative_mining_ignores_true_detections(tmp_path):
    truth = {"1": pd.DataFrame({"frame": [0, 2], "x": [100.0, 200.0],
                                "y": [100.0, 200.0], "w": np.nan, "h": np.nan})}
    cands = tmp_path / "cands.csv"
    pd.DataFrame({"game": ["1", "1", "1", "1"], "frame": [0, 0, 2, 4],
                  "gx": [101.0, 900.0, 205.0, 50.0], "gy": [99.0, 900.0, 200.0, 50.0],
                  "conf": [0.9, 0.5, 0.1, 0.4]}).to_csv(cands, index=False)
    mined = tiles.load_hard_negatives(cands, min_conf=0.18, radius_px=15.0,
                                      annotations=truth)
    # frame 0: the far candidate only; frame 2: below the floor; frame 4: no ball at all
    assert mined == {("1", 0): [(900.0, 900.0)], ("1", 4): [(50.0, 50.0)]}


def test_tile_names_and_dataset_yaml(tmp_path):
    assert tiles.tile_stem("4", 12, 260, 0) == "g4_f0000012_x0260_y0000"
    text = tiles.write_data_yaml(tmp_path).read_text()
    assert "train: images/train" in text and "0: ball" in text
    assert str(tmp_path.resolve()) in text


# --------------------------------------------------------------------------- #
# train_rtdetr: the baseline IS the committed checkpoint's recipe
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RTDETR_PT.is_file(), reason="detector checkpoint not present")
def test_baseline_profile_matches_the_committed_checkpoint():
    """Every baseline value equals what the checkpoint recorded — no silent drift."""
    recorded = inspect_checkpoint.read_train_args(RTDETR_PT)
    assert recorded is not None
    for key, value in train_rtdetr.BASELINE.items():
        assert key in recorded, key
        assert recorded[key] == pytest.approx(value) if isinstance(value, float) \
            else recorded[key] == value, (key, recorded[key], value)
    assert recorded["imgsz"] == F.TILE_SIZE          # trained on the inference tile size
    assert recorded["device"] == "0,1"               # the one intended difference


def test_profiles_are_baseline_plus_named_overrides():
    assert train_rtdetr.resolve_args("baseline") == train_rtdetr.BASELINE
    long = train_rtdetr.resolve_args("long")
    changed = {k for k in long if long[k] != train_rtdetr.BASELINE.get(k)}
    assert changed == set(train_rtdetr.PROFILES["long"]["overrides"])
    for profile in train_rtdetr.PROFILES.values():
        assert profile["why"].strip()
        assert set(profile["overrides"]) <= set(train_rtdetr.BASELINE)


def test_overrides_parse_types_and_reserved_keys_are_refused():
    assert train_rtdetr.parse_override("epochs=36") == ("epochs", 36)
    assert train_rtdetr.parse_override("lr0=3e-5") == ("lr0", 3e-5)
    assert train_rtdetr.parse_override("cos_lr=false") == ("cos_lr", False)
    assert train_rtdetr.parse_override("optimizer=AdamW") == ("optimizer", "AdamW")
    with pytest.raises(ValueError):
        train_rtdetr.parse_override("epochs")
    with pytest.raises(ValueError):
        train_rtdetr.resolve_args("baseline", {"data": "elsewhere.yaml"})
    with pytest.raises(ValueError):
        train_rtdetr.resolve_args("finetune")            # needs --weights
    with pytest.raises(KeyError):
        train_rtdetr.resolve_args("no-such-profile")


def test_plan_run_is_torch_free_and_records_the_why(tmp_path):
    data_yaml = tiles.write_data_yaml(tmp_path)
    plan = train_rtdetr.plan_run(data_yaml, "fold_0", "long", {"batch": 8})
    args = plan["train_args"]
    assert (args["epochs"], args["batch"], args["name"]) == (36, 8, "fold_0")
    assert args["data"] == str(data_yaml.resolve())
    assert plan["overrides"] == {"batch": 8} and plan["profile_why"]
    assert "device" not in args                         # decided when the run starts


def test_training_refuses_a_dataset_inside_the_provided_bundle(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"path: {ROOT / 'data' / 'tactical'}\n"
                         "train: images/train\nval: images/val\nnames:\n  0: ball\n")
    with pytest.raises(compliance.ProvidedDataError):
        train_rtdetr.plan_run(data_yaml, "x")


def test_train_meta_carries_the_tile_provenance(tmp_path):
    data_yaml = tiles.write_data_yaml(tmp_path)
    (tmp_path / "provenance.json").write_text(json.dumps({"dataset": "unit-test set"}))
    plan = train_rtdetr.plan_run(data_yaml, "run", project=tmp_path / "runs")
    meta_path = train_rtdetr.write_meta(plan, "t0", None, "running")
    meta = json.loads(meta_path.read_text())
    assert meta["tile_dataset_provenance"] == {"dataset": "unit-test set"}
    assert meta["status"] == "running" and meta["profile"] == "baseline"


# --------------------------------------------------------------------------- #
# eval_detector: scoring
# --------------------------------------------------------------------------- #
def _cands(rows):
    return pd.DataFrame(rows, columns=eval_detector.CANDIDATE_COLUMNS)


def _truth(rows):
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def test_scoring_counts_hits_misses_and_false_alarms():
    truth = _truth([(0, 100.0, 100.0), (1, 110.0, 100.0), (2, 120.0, 100.0)])
    cands = _cands([
        ("9", 0, 102.0, 100.0, 0.9),     # hit
        ("9", 1, 400.0, 400.0, 0.8),     # confident miss ...
        ("9", 1, 111.0, 100.0, 0.3),     # ... although a weaker candidate was right
        ("9", 3, 50.0, 50.0, 0.6),       # frame 3 has no ball: false alarm
    ])                                   # frame 2: nothing detected
    hits = eval_detector.frame_hits(cands, truth, [0, 1, 2, 3, 4], 0.18, 10.0)
    assert hits["hit"].tolist() == [True, False, False, False, False]
    assert hits["any_hit"].tolist() == [True, True, False, False, False]
    summary = eval_detector.summarise(hits)
    assert summary["frames"] == 5 and summary["frames_with_ball"] == 3
    assert summary["recall"] == pytest.approx(1 / 3)
    assert summary["precision"] == pytest.approx(1 / 3)          # 1 hit of 3 detections
    assert summary["any_candidate_recall"] == pytest.approx(2 / 3)
    assert summary["false_alarm_rate_empty"] == pytest.approx(1 / 2)
    assert summary["median_err_px"] == pytest.approx(2.0)


def test_confidence_threshold_changes_which_candidate_counts():
    truth = _truth([(0, 100.0, 100.0)])
    cands = _cands([("9", 0, 100.0, 100.0, 0.25), ("9", 0, 900.0, 900.0, 0.20)])
    low = eval_detector.summarise(eval_detector.frame_hits(cands, truth, [0], 0.18, 5))
    high = eval_detector.summarise(eval_detector.frame_hits(cands, truth, [0], 0.30, 5))
    assert low["recall"] == 1.0 and high["recall"] == 0.0
    assert np.isnan(high["precision"])                   # nothing detected: undefined


def test_score_table_shape_and_empty_candidates():
    truth = _truth([(0, 1.0, 1.0)])
    table = eval_detector.score_table(_cands([]), truth, [0, 1], radii=(5.0, 10.0),
                                      conf_grid=(0.1, 0.5))
    assert len(table) == 4 and (table["recall"] == 0.0).all()
    assert (table["frames"] == 2).all()


def test_paired_delta_detects_a_real_difference_and_not_noise():
    has_ball = np.ones(2000, dtype=bool)
    rng = np.random.default_rng(0)
    base = rng.random(2000) < 0.6
    better = base | (rng.random(2000) < 0.5)
    result = eval_detector.paired_delta(base, better, has_ball)
    assert result["delta_recall"] > 0.15 and result["ci95"][0] > 0
    assert result["verdict"] == "B better" and result["n_eff"] > 100
    same = eval_detector.paired_delta(base, base, has_ball)
    assert same["delta_recall"] == 0.0 and same["ci95"] == [0.0, 0.0]
    assert same["verdict"] == "no evidence"
    with pytest.raises(ValueError):
        eval_detector.paired_delta(base[:10], better, has_ball)
    empty = eval_detector.paired_delta(base, better, np.zeros(2000, dtype=bool))
    assert np.isnan(empty["delta_recall"]) and empty["verdict"] == "insufficient data"


def _sticky(n, p, stay, rng):
    """Hits that come in runs (a lost ball stays lost for a while), as on real footage."""
    out = np.empty(n, dtype=bool)
    current = rng.random() < p
    for i in range(n):
        if rng.random() > stay:
            current = rng.random() < p
        out[i] = current
    return out


def test_paired_delta_keeps_false_alarms_near_nominal_on_correlated_frames():
    """Equal detectors, errors in runs: the earlier block bootstrap said 'better' in
    15-45 % of such runs. Corrected for autocorrelation, it must stay near 5 %."""
    calls, wins = 0, 0
    for trial in range(120):
        rng = np.random.default_rng(2000 + trial)
        a, b = _sticky(3000, 0.6, 0.8, rng), _sticky(3000, 0.6, 0.8, rng)
        result = eval_detector.paired_delta(a, b, np.ones(3000, dtype=bool))
        calls += result["verdict"] != "insufficient data"
        wins += result["verdict"] in ("A better", "B better")
    assert calls >= 60
    assert wins / calls <= 0.12


def test_paired_delta_refuses_a_verdict_on_too_few_effective_frames():
    # five long runs: B looks better on 2 of them, A on 1 — far too little to decide
    a = np.repeat([True, False, True, False, True], 60)
    b = np.repeat([True, True, False, True, True], 60)
    result = eval_detector.paired_delta(a, b, np.ones(300, dtype=bool))
    assert result["n_eff"] < eval_detector.MIN_EFFECTIVE_FRAMES
    assert result["verdict"] == "insufficient data"


def test_cache_identity_changes_with_the_checkpoint_and_settings(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"one")
    first = eval_detector.cache_identity(weights, 5, None, 320)
    assert first == eval_detector.cache_identity(weights, 5, None, 320)
    assert first != eval_detector.cache_identity(weights, 1, None, 320)
    assert first != eval_detector.cache_identity(weights, 5, 100, 320)
    assert first != eval_detector.cache_identity(weights, 5, None, 640)
    other = tmp_path / "run2" / "best.pt"            # same file name, another run
    other.parent.mkdir()
    other.write_bytes(b"two!")
    assert first != eval_detector.cache_identity(other, 5, None, 320)
    assert eval_detector.read_cache_identity(tmp_path, "best", "9") is None


def test_markdown_report_lists_every_checkpoint():
    truth = _truth([(0, 1.0, 1.0)])
    table = eval_detector.score_table(_cands([("9", 0, 1.0, 1.0, 0.9)]), truth, [0],
                                      radii=(5.0,), conf_grid=(0.18,))
    text = eval_detector.markdown_report(
        {"kaggle": table, "local": table},
        [{"a": "kaggle", "b": "local", "min_conf": 0.18, "radius_px": 5.0,
          "frames_with_ball": 1, "delta_recall": 0.0, "ci95": [0.0, 0.0], "n_eff": 1.0,
          "verdict": "insufficient data"}], {"dataset": "unit-test set"})
    assert "## kaggle" in text and "## local" in text and "Paired comparison" in text


# --------------------------------------------------------------------------- #
# fot_data: training windows are built by the inference functions
# --------------------------------------------------------------------------- #
def _straight_game(n=200, miss_every=9):
    """Ball moving right at 4 px/frame; the detector misses every ``miss_every``-th frame."""
    frames = np.arange(n)
    truth = pd.DataFrame({"frame": frames, "x": 100.0 + 4.0 * frames, "y": 500.0})
    rows = [("1", int(f), 100.0 + 4.0 * f, 500.0, 0.9)
            for f in frames if f % miss_every]
    return _cands(rows), frames, truth


def test_fot_detections_mirror_the_no_homography_gate():
    cands = _cands([("1", 0, 10.0, 10.0, 0.5), ("1", 0, 99.0, 99.0, 0.9),
                    ("1", 1, 20.0, 20.0, 0.10)])
    dets = fot_data.detections_from_candidates(cands, [0, 1, 2])
    assert dets[0] == (0, 99.0, 99.0, 0.9)                 # the most confident
    assert np.isnan(dets[1][1]) and dets[1][3] == 0.0      # below FAR_CONF_FLOOR
    assert np.isnan(dets[2][1])                            # no candidate at all
    # same rule as PitchKinematicGate.step with H=None
    gate = F.PitchKinematicGate(25.0, 105.0, 68.0)
    assert gate.step(0, [(99.0, 99.0, 0.9), (10.0, 10.0, 0.5)], None, {}) == (99.0, 99.0, 0.9)
    assert np.isnan(gate.step(1, [(20.0, 20.0, 0.10)], None, {})[0])


def test_fot_windows_have_the_model_shapes_and_normalised_targets():
    cands, frames, truth = _straight_game()
    data = fot_data.windows_for_game(cands, frames, truth, 1920, 1080, frame_step=2)
    assert len(data) > 0
    assert data.feats.shape[1:] == (F.WINDOW_SIZE, 12)
    assert data.baseline.shape[1:] == (F.WINDOW_SIZE, 2)
    assert data.feats.dtype == np.float32 and np.isfinite(data.feats).all()
    assert data.target_mask.all()                          # every frame is annotated
    assert not data.observed.all()                         # the misses are in there
    assert data.target[..., 0].max() <= 1.0 and data.target[..., 1].max() <= 1.0
    # on a straight line the interpolation baseline is already (nearly) exact
    err = fot_data.pixel_error(data.baseline, data.target, 1920, 1080)
    assert np.median(err[data.target_mask]) < 1.0


def test_fot_windows_match_what_inference_builds():
    """Feature-for-feature equality with the inference path on the same sequence."""
    cands, frames, truth = _straight_game(n=128, miss_every=5)
    dets = fot_data.detections_from_candidates(cands, frames)[0::2]
    piece = fot_data.windows_from_sequence(dets, truth, 1920, 1080)
    obs_x, obs_y, scores, valid = F.prepare_fot_observations(dets, 1920, 1080,
                                                            "max_coverage")
    feats, base = F.build_window_features(obs_x[:64], obs_y[:64], scores[:64], valid[:64])
    np.testing.assert_array_equal(piece["feats"][0], feats.astype(np.float32))
    np.testing.assert_array_equal(piece["baseline"][0], base.astype(np.float32))


def test_occlusion_augmentation_keeps_the_anchors():
    valid = np.zeros(64, dtype=bool)
    valid[5:60] = True
    rng = np.random.default_rng(0)
    removed_any = False
    for _ in range(200):
        out = fot_data.drop_observations(valid, rng, gap_prob=1.0, max_gap=6)
        assert out[5] and out[59]                          # first / last anchor kept
        assert not (out & ~valid).any()                    # never invents observations
        assert valid.sum() - out.sum() <= 6
        removed_any |= bool(valid.sum() - out.sum())
    assert removed_any


def test_short_sequences_and_unannotated_windows_produce_nothing():
    cands, frames, truth = _straight_game(n=40)
    assert len(fot_data.windows_for_game(cands, frames, truth, 1920, 1080)) == 0
    cands, frames, truth = _straight_game(n=200)
    assert len(fot_data.windows_for_game(cands, frames, truth.iloc[0:0], 1920, 1080)) == 0
    with pytest.raises(ValueError):
        fot_data.concat([])


# --------------------------------------------------------------------------- #
# inspect_checkpoint: the committed files, read without unpickling
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (RTDETR_PT.is_file() and FOT_PT.is_file()),
                    reason="checkpoints not present")
def test_committed_checkpoints_pass_the_static_import_audit():
    for path in (RTDETR_PT, FOT_PT):
        audit = inspect_checkpoint.audit_imports(path)
        assert audit["safe"], audit["outside_allowlist"]
        assert audit["imports"]
    header = inspect_checkpoint.read_header(RTDETR_PT)
    assert header["version"] == "8.4.212"
    results = inspect_checkpoint.read_train_results(RTDETR_PT)
    assert len(results["metrics/mAP50(B)"]) == 12
    assert inspect_checkpoint.read_train_args(FOT_PT) is None       # bare state_dict
    assert "input_proj.0.weight" in inspect_checkpoint.state_dict_keys(FOT_PT)


def test_audit_flags_a_pickle_that_imports_os(tmp_path):
    """The reason the audit exists: a checkpoint that would run a shell command."""
    import pickle
    import zipfile

    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo never-executed",))

    target = tmp_path / "evil.pt"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(Evil()))
    audit = inspect_checkpoint.audit_imports(target)
    assert not audit["safe"]
    assert any("system" in name for name in audit["outside_allowlist"])
    assert inspect_checkpoint.main([str(target), "--json"]) == 1


def test_audit_rejects_a_file_that_is_not_a_checkpoint(tmp_path):
    import zipfile

    target = tmp_path / "empty.pt"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("readme.txt", "nothing here")
    with pytest.raises(ValueError):
        inspect_checkpoint.audit_imports(target)


@pytest.mark.skipif(not FOT_PT.is_file(), reason="FoT checkpoint not present")
def test_fot_checkpoint_loads_strictly_into_the_code_architecture():
    """Binds the committed weights to build_fot_model: a renamed layer fails here."""
    pytest.importorskip("torch")
    model = F.build_fot_model(F.WINDOW_SIZE, weights=FOT_PT, device="cpu")
    assert sum(p.numel() for p in model.parameters()) == 1_314_883


# --------------------------------------------------------------------------- #
# cv2-backed pieces (skipped without OpenCV)
# --------------------------------------------------------------------------- #
def test_synthetic_dataset_round_trips_through_the_tile_builder(tmp_path):
    pytest.importorskip("cv2")
    from training import frames as frames_mod
    from training import synth

    manifest_path = synth.write_dataset(tmp_path / "synth", n_games=2, n_frames=12,
                                        width=640, height=480, seed=3)
    loaded = manifest.load_manifest(manifest_path)
    game = loaded.game("1")
    assert frames_mod.frame_size(game) == (640, 480)
    assert frames_mod.frame_count(game) == 12
    numbers = [n for n, _ in frames_mod.iter_frames(game, step=5)]
    assert numbers == [0, 5, 10]

    provenance = tiles.build(loaded, tmp_path / "tiles", ["1"], ["2"], frame_step=2,
                             progress_every=0)
    out = tmp_path / "tiles"
    index = pd.read_csv(out / "tiles_index.csv", dtype={"game": str})
    assert set(index["split"]) == {"train", "val"}
    assert set(index.loc[index["split"] == "val", "game"]) == {"2"}       # by game
    for row in index.itertuples():
        assert (out / row.file).is_file()
        label = out / row.file.replace("images/", "labels/").replace(".jpg", ".txt")
        assert label.is_file() == (row.kind == tiles.KIND_POS)
    assert provenance["dataset"] == synth.DATASET_NAME
    assert provenance["pitch_masked"] is False
    assert json.loads((out / "provenance.json").read_text())["train_games"] == ["1"]
    with pytest.raises(FileExistsError):
        tiles.build(loaded, out, ["1"], ["2"])             # no silent overwrite


def test_video_frames_are_read_sequentially_and_numbered_by_offset(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from training import frames as frames_mod

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no mp4v writer in this OpenCV build")
    for value in range(0, 250, 25):                        # 10 frames, distinct grey
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    (tmp_path / "ball.csv").write_text("frame,x,y\n1,1,1\n")
    game = manifest.Game(id="1", annotations=tmp_path / "ball.csv", video=video,
                         frame_offset=1)
    got = [(n, int(img.mean())) for n, img in frames_mod.iter_frames(game, step=3)]
    assert [n for n, _ in got] == [1, 4, 7, 10]            # offset + position
    greys = [g for _, g in got]
    assert greys == sorted(greys) and greys[1] - greys[0] > 50   # really every 3rd
    assert len(list(frames_mod.iter_frames(game, step=1, start=8))) == 2
    assert len(list(frames_mod.iter_frames(game, max_frames=4))) == 4


# --------------------------------------------------------------------------- #
# Regression tests for the second review (2026-11-14)
# --------------------------------------------------------------------------- #
def _zip_pickle(path, payload: bytes):
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
    return path


def _short_str(text):
    raw = text.encode()
    return b"\x8c" + bytes([len(raw)]) + raw


def test_audit_resolves_memoized_stack_global_operands(tmp_path):
    """Protocol-4 operands read back from the memo, with harmless-looking decoys after."""
    memo = b"\x94"
    payload = (b"\x80\x04" + _short_str("zzmalicious") + memo + _short_str("run") + memo
               + _short_str("torch") + memo + _short_str("Tensor") + memo
               + b"h\x00h\x01" + b"\x93" + b"0.")
    audit = inspect_checkpoint.audit_imports(_zip_pickle(tmp_path / "a.pt", payload))
    assert "zzmalicious.run" in audit["imports"] and not audit["safe"]


def test_audit_fails_closed_on_unresolvable_or_legacy_imports(tmp_path):
    inst = _zip_pickle(tmp_path / "inst.pt", b"(izzmalicious\nBad\n0.")
    assert not inspect_checkpoint.audit_imports(inst)["safe"]
    # STACK_GLOBAL whose operands are not the two most recent plain strings
    popped = (b"\x80\x04" + _short_str("os") + _short_str("system") + _short_str("x")
              + b"0" + b"\x93" + b"0.")
    audit = inspect_checkpoint.audit_imports(_zip_pickle(tmp_path / "pop.pt", popped))
    assert audit["unresolved"] and not audit["safe"]


@pytest.mark.parametrize("name, allowed", [
    ("torch._utils._rebuild_tensor_v2", True), ("torch.FloatStorage", True),
    ("torch.nn.modules.conv.Conv2d", True), ("ultralytics.nn.tasks.RTDETRDetectionModel", True),
    ("collections.OrderedDict", True), ("builtins.set", True),
    ("torch.load", False), ("torch.hub.load", False), ("torch_evil.x", False),
    ("builtins.eval", False), ("builtins.object", False), ("os.system", False),
    ("ultralytics.utils.downloads.attempt_download_asset", False),
])
def test_audit_allowlist_is_by_name_not_by_package(name, allowed):
    assert inspect_checkpoint.is_allowed(name) is allowed


def test_dataset_paths_reads_comments_quotes_and_lists(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f'path: "{tmp_path}"  # root\n'
                         "train:\n  - images/a\n  - images/b   # second\n"
                         "val: images/val  # inline comment\n"
                         "names:\n  0: ball\n")
    paths = [str(p) for p in train_rtdetr.dataset_paths(data_yaml)]
    assert str(tmp_path / "images" / "a") in paths and str(tmp_path / "images" / "b") in paths
    assert str(tmp_path / "images" / "val") in paths


def test_guard_sees_provided_folders_behind_yaml_comments_and_lists(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"path: {ROOT}\ntrain:\n  - data/external/x\n"
                         "  - data/broadcast   # note\nval: data/external/y\n")
    with pytest.raises(compliance.ProvidedDataError):
        train_rtdetr.plan_run(data_yaml, "x")
    data_yaml.write_text(f"path: {ROOT}\ntrain:\n  images: weird\n")
    with pytest.raises(ValueError, match="unsupported structure"):
        train_rtdetr.dataset_paths(data_yaml)


def test_guard_resolves_symlinks_before_dot_dot(monkeypatch, tmp_path):
    bundle, store = tmp_path / "bundle", tmp_path / "store"
    (bundle / "tactical").mkdir(parents=True)
    (store / "cam1").mkdir(parents=True)
    ext = tmp_path / "ext"
    ext.mkdir()
    try:
        (ext / "ds").symlink_to(bundle / "tactical")
    except OSError:
        pytest.skip("symlinks not available")
    monkeypatch.setenv("MTAH_DATA_DIR", str(bundle))
    assert compliance.is_provided(ext / "ds" / ".." / ".." / "bundle" / "tactical" / "v.mp4")
    assert compliance.is_provided(ext / "ds" / "v.mp4")
    # a bundle folder that is itself a link: its target is protected too
    (bundle / "tactical").rmdir()
    (bundle / "tactical").symlink_to(store / "cam1")
    assert compliance.is_provided(store / "cam1" / "renamed.mp4")


@pytest.mark.parametrize("value, placeholder", [
    ("Todos los derechos reservados; uso academico permitido", False),
    ("https://github.com/todorov-lab/soccer-ball-dataset", False),
    ("CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>", False),
    ("MIT (author unknown to us; see LICENSE file)", False),
    ("SoccerNetBD-2025", False),
    ("<name>", True), ("TBD", True), ("n/a", True), ("N/A", True), ("none", True),
    ("?", True), ("xxx", True), ("", True), (None, True), (0, True), ([], True),
])
def test_placeholder_detection_is_whole_value(value, placeholder):
    assert manifest._is_placeholder(value) is placeholder


def test_frames_dir_needs_an_explicit_offset_and_ids_are_file_safe(tmp_path):
    (tmp_path / "img1").mkdir()
    base = {"dataset": "x", "license": "CC0-1.0", "source_url": "https://example.org"}
    cases = [
        ({"id": "1", "frames_dir": "img1", "annotations": "a.csv"}, "frame_offset"),
        ({"id": "1", "frames_dir": "img1", "annotations": "a.csv", "frame_offset": -1},
         "frame_offset"),
        ({"id": "1", "frames_dir": "img1", "annotations": "a.csv", "frame_offset": "one"},
         "frame_offset"),
        ({"id": "1", "video": "v.mp4", "annotations": "a.csv", "fps": "25fps"}, "fps"),
        ({"id": "SNMOT/060", "video": "v.mp4", "annotations": "a.csv"}, "game id"),
    ]
    for entry, message in cases:
        (tmp_path / "m.json").write_text(json.dumps({**base, "games": [entry]}))
        with pytest.raises(manifest.ManifestError, match=message):
            manifest.load_manifest(tmp_path / "m.json", check_files=False)


def test_annotation_frames_must_be_whole_non_negative_numbers(tmp_path):
    game = manifest.Game(id="1", annotations=tmp_path / "ball.csv", video=tmp_path / "v.mp4")
    (tmp_path / "ball.csv").write_text("frame,x,y\n1,5,5\n,,\n2,6,6\n")
    assert manifest.load_annotations(game)["frame"].tolist() == [1, 2]   # blank row dropped
    (tmp_path / "ball.csv").write_text("frame,x,y\n1.5,5,5\n")
    with pytest.raises(manifest.ManifestError, match="non-integer"):
        manifest.load_annotations(game)
    (tmp_path / "ball.csv").write_text("frame,x,y\n-1,5,5\n")
    with pytest.raises(manifest.ManifestError, match="negative frame"):
        manifest.load_annotations(game)


def test_frame_files_use_natural_order_and_skip_hidden_files(tmp_path):
    for name in ("1.jpg", "2.jpg", "10.jpg", "100.jpg", "9.jpg", "._1.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in manifest.list_frame_files(tmp_path)] == [
        "1.jpg", "2.jpg", "9.jpg", "10.jpg", "100.jpg"]


def test_convert_mot_refuses_headers_and_empty_selections(tmp_path):
    src = tmp_path / "gt.txt"
    src.write_text("frame,id,left,top,w,h,conf,class\n1,3,100,200,10,12,1,1\n")
    with pytest.raises(manifest.ManifestError, match="header"):
        manifest.convert_mot(src, tmp_path / "b.csv", class_ids=[1])
    src.write_text("1,3,100,200,10,12,1,1\n")
    with pytest.raises(manifest.ManifestError, match="no row matched"):
        manifest.convert_mot(src, tmp_path / "b.csv", class_ids=[5])


def test_annotation_range_check_catches_offset_and_resolution_mistakes(tmp_path):
    game = manifest.Game(id="1", annotations=tmp_path / "a.csv", frames_dir=tmp_path,
                         frame_offset=0)
    truth = pd.DataFrame({"frame": [1, 40], "x": [10.0, 10.0], "y": [10.0, 10.0]}).set_index("frame")
    with pytest.raises(ValueError, match="frame_offset"):
        tiles.check_annotation_range(game, truth, 640, 480, n_frames=40)   # frames 0..39
    ok = tiles.check_annotation_range(game, truth.iloc[:1], 640, 480, n_frames=40)
    assert ok == {"frames": [0, 39], "annotated": [1, 1]}
    wide = pd.DataFrame({"frame": [1], "x": [1500.0], "y": [10.0]}).set_index("frame")
    with pytest.raises(ValueError, match="resolution"):
        tiles.check_annotation_range(game, wide, 640, 480, n_frames=40)


def test_data_yaml_path_survives_hash_and_colon(tmp_path):
    for name in ("tiles #2", "set: v2", "My Tiles"):
        folder = tmp_path / name
        folder.mkdir()
        tiles.write_data_yaml(folder)
        paths = train_rtdetr.dataset_paths(folder / "data.yaml")
        assert paths[1] == folder.resolve()


def test_force_rebuilds_instead_of_merging_and_hard_negatives_must_be_used(tmp_path):
    pytest.importorskip("cv2")
    from training import synth

    loaded = manifest.load_manifest(synth.write_dataset(
        tmp_path / "synth", n_games=2, n_frames=12, width=640, height=480, seed=5))
    out = tmp_path / "tiles"
    tiles.build(loaded, out, ["1"], ["2"], frame_step=2, progress_every=0)
    tiles.build(loaded, out, ["2"], ["1"], frame_step=2, progress_every=0, force=True)
    index = pd.read_csv(out / "tiles_index.csv", dtype={"game": str})
    on_disk = sorted(p.relative_to(out).as_posix() for p in (out / "images").rglob("*.jpg"))
    assert on_disk == sorted(index["file"])                     # nothing stale left behind
    assert set(index.loc[index["split"] == "train", "game"]) == {"2"}
    labels = {p.stem for p in (out / "labels").rglob("*.txt")}
    assert labels == {Path(f).stem for f in index.loc[index["kind"] == "pos", "file"]}

    stranger = tmp_path / "not_tiles"
    (stranger / "images").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="not written by training.tiles"):
        tiles.build(loaded, stranger, ["1"], ["2"], force=True)

    cands = tmp_path / "cands.csv"                              # a false alarm on frame 1
    pd.DataFrame({"game": ["1"], "frame": [1], "gx": [600.0], "gy": [20.0],
                  "conf": [0.9]}).to_csv(cands, index=False)
    with pytest.raises(ValueError, match="never visited"):
        tiles.build(loaded, tmp_path / "hn", ["1"], ["2"], frame_step=2,
                    hard_negatives=cands, progress_every=0)


def test_env_check_reports_instead_of_crashing(tmp_path):
    from training import env_check

    row = env_check.check_disk(tmp_path / "does-not-exist")
    assert row["status"] == env_check.FAIL and "cannot read" in row["detail"]


def test_bridged_steps_are_the_ones_inference_can_bridge():
    observed = np.array([[False, True, False, False, True, False, False, False,
                          False, False, True, False]])
    mask = fot_data.bridgeable_mask(observed, max_gap=4)
    # leading gap, the 5-step gap and the trailing gap are never bridged by FoT
    assert np.flatnonzero(mask[0]).tolist() == [2, 3]
    assert fot_data.bridgeable_mask(observed[0], max_gap=5)[5:10].all()
