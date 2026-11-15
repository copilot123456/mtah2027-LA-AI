"""Guarded-eval wrapper: band mapping and the four categorical flags of plan §1.

Pure-function tests — the wrapper must never need the evaluator to be exercised.
"""

from tuning.guarded_eval import band_and_flags, count_bucket


def _report(mid, poss=17, f1d=0.5, f1h=0.6, f1a=0.55, mean_lat=0.0):
    """Minimal evaluator report for the wrapper."""
    return {                                      # report skeleton
        "spell_bootstrap_low": mid - 0.01,
        "spell_bootstrap_high": mid + 0.01,
        "n_pred_spells": poss,
        "taus": [12, 25],
        "transitions": {"12": {"mean_latency": mean_lat},
                        "25": {"mean_latency": mean_lat}},
        "per_class": {"H": {"f1": f1h}, "A": {"f1": f1a}, "D": {"f1": f1d}},
    }


def test_band_boundaries_plan_1():
    """Bands: A >= 0.43, B >= 0.30, C >= 0.15, else D."""
    assert band_and_flags(_report(0.43))[0] == "A" # 0.43 opens A
    assert band_and_flags(_report(0.4299))[0] == "B" # just below A
    assert band_and_flags(_report(0.30))[0] == "B" # 0.30 opens B
    assert band_and_flags(_report(0.2999))[0] == "C" # just below B
    assert band_and_flags(_report(0.15))[0] == "C" # 0.15 opens C
    assert band_and_flags(_report(0.1499))[0] == "D" # below C is D


def test_count_bucket_edges():
    """Count bucket edges are inclusive."""
    assert count_bucket(10) == "<=10"             # 10 in first bucket
    assert count_bucket(11) == "11-13"            # 11 opens second
    assert count_bucket(13) == "11-13"            # 13 still second
    assert count_bucket(14) == "14-22"            # 14 opens third
    assert count_bucket(22) == "14-22"            # 22 still third
    assert count_bucket(23) == "23-30"            # 23 opens fourth
    assert count_bucket(30) == "23-30"            # 30 still fourth
    assert count_bucket(31) == ">30"              # 31 opens last bucket


def test_f1_d_and_balance_flags():
    """F1-D and balance flags trip below 0.30."""
    band, flags = band_and_flags(_report(0.35, f1d=0.3)) # D f1 at floor
    assert flags["f1_D"] == "normal"              # floor counts normal
    _, flags = band_and_flags(_report(0.35, f1d=0.299)) # D f1 under floor
    assert flags["f1_D"] == "low"                 # flagged low
    _, flags = band_and_flags(_report(0.35, f1a=0.3, f1h=0.3)) # both teams at floor
    assert flags["balance"] == "normal"           # both floors met
    _, flags = band_and_flags(_report(0.35, f1h=0.299)) # one team under floor
    assert flags["balance"] == "collapsed"        # flagged collapsed
    # 0.35 sits in band B
    assert band == "B"


def test_latency_flag_uses_tight_tau():
    """Latency flag reads the mean against the tight tau."""
    _, flags = band_and_flags(_report(0.35, mean_lat=12.0)) # latency plus 12
    # within +/-12 counts as on time
    assert flags["latency"] == "none"
    _, flags = band_and_flags(_report(0.35, mean_lat=-12.0)) # latency minus 12
    assert flags["latency"] == "none"             # still on time
    _, flags = band_and_flags(_report(0.35, mean_lat=13.0)) # latency plus 13
    assert flags["latency"] == "late"             # flagged late
    _, flags = band_and_flags(_report(0.35, mean_lat=-13.0)) # latency minus 13
    assert flags["latency"] == "early"            # flagged early


def test_missing_latency_reads_as_none():
    """Missing latency data reads as no flag."""
    report = _report(0.35)                        # base report
    report["transitions"] = {"25": {}}            # latency block missing
    _, flags = band_and_flags(report)             # run the wrapper
    assert flags["latency"] == "none"             # reads as none
