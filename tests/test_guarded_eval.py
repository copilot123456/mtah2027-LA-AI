"""Guarded-eval wrapper: band mapping and the four categorical flags of plan §1.

Pure-function tests — the wrapper must never need the evaluator to be exercised.
"""

from tuning.guarded_eval import band_and_flags, count_bucket


def _report(mid, poss=17, f1d=0.5, f1h=0.6, f1a=0.55, mean_lat=0.0):
    return {
        "spell_bootstrap_low": mid - 0.01,
        "spell_bootstrap_high": mid + 0.01,
        "n_pred_spells": poss,
        "taus": [12, 25],
        "transitions": {"12": {"mean_latency": mean_lat},
                        "25": {"mean_latency": mean_lat}},
        "per_class": {"H": {"f1": f1h}, "A": {"f1": f1a}, "D": {"f1": f1d}},
    }


def test_band_boundaries_plan_1():
    assert band_and_flags(_report(0.43))[0] == "A"
    assert band_and_flags(_report(0.4299))[0] == "B"
    assert band_and_flags(_report(0.30))[0] == "B"
    assert band_and_flags(_report(0.2999))[0] == "C"
    assert band_and_flags(_report(0.15))[0] == "C"
    assert band_and_flags(_report(0.1499))[0] == "D"


def test_count_bucket_edges():
    assert count_bucket(10) == "<=10"
    assert count_bucket(11) == "11-13"
    assert count_bucket(13) == "11-13"
    assert count_bucket(14) == "14-22"
    assert count_bucket(22) == "14-22"
    assert count_bucket(23) == "23-30"
    assert count_bucket(30) == "23-30"
    assert count_bucket(31) == ">30"


def test_f1_d_and_balance_flags():
    band, flags = band_and_flags(_report(0.35, f1d=0.3))
    assert flags["f1_D"] == "normal"
    _, flags = band_and_flags(_report(0.35, f1d=0.299))
    assert flags["f1_D"] == "low"
    _, flags = band_and_flags(_report(0.35, f1a=0.3, f1h=0.3))
    assert flags["balance"] == "normal"
    _, flags = band_and_flags(_report(0.35, f1h=0.299))
    assert flags["balance"] == "collapsed"
    assert band == "B"


def test_latency_flag_uses_tight_tau():
    _, flags = band_and_flags(_report(0.35, mean_lat=12.0))
    assert flags["latency"] == "none"
    _, flags = band_and_flags(_report(0.35, mean_lat=-12.0))
    assert flags["latency"] == "none"
    _, flags = band_and_flags(_report(0.35, mean_lat=13.0))
    assert flags["latency"] == "late"
    _, flags = band_and_flags(_report(0.35, mean_lat=-13.0))
    assert flags["latency"] == "early"


def test_missing_latency_reads_as_none():
    report = _report(0.35)
    report["transitions"] = {"25": {}}
    _, flags = band_and_flags(report)
    assert flags["latency"] == "none"
