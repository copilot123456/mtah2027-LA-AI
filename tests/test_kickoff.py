"""K1 kick-off tag rule (pre-declared: dossier §7 P7, 2026-11-14).

Unit tests of the pure post-pass on synthetic halves, plus the loader contract. The
provided-match verification is categorical and lives in tuning/kickoff_check.py.
"""

import numpy as np

from config import CFG
from src.decoder import CODE_A, CODE_D, CODE_H, CODE_X, apply_kickoff_rule

# P7 K1 kick-off tag rule enabled
ON = {**CFG, "kickoff_rule": True}


def _match():
    """Two halves: frames 100..499 and 700..1099; X elsewhere."""
    half = np.zeros(1200, dtype=np.int8)          # 1200-frame canvas
    half[100:500] = 1                             # half one span
    half[700:1100] = 2                            # half two span
    labels = np.full(1200, CODE_X, dtype=np.int8) # X everywhere first
    labels[100:500] = CODE_H                      # half one graded H
    labels[700:1100] = CODE_A                     # half two graded A
    return labels, half                           # hand back the pair


def test_frames_before_the_kickoff_tag_become_dead():
    """Frames before each kick-off tag become dead (D)."""
    labels, half = _match()                       # reuse synthetic match
    out = apply_kickoff_rule(labels, half, np.array([100, 760]), ON) # tags at 100 and 760
    assert (out[700:760] == CODE_D).all()         # before tag is dead
    assert (out[760:1100] == CODE_A).all(), "the kick-off frame itself is live" # kick-off frame stays live
    assert (out[100:500] == CODE_H).all(), "a tag on the half's first frame changes nothing" # first-frame tag changes nothing
    assert (out[half == 0] == CODE_X).all(), "ungraded frames are never touched" # X frames untouched
    assert (labels[700:760] == CODE_A).all(), "input must not be mutated" # input left unchanged


def test_earliest_tag_inside_the_half_wins():
    """The earliest tag inside a half wins."""
    labels, half = _match()                       # reuse synthetic match
    out = apply_kickoff_rule(labels, half, np.array([900, 730, 760]), ON) # three competing tags
    assert (out[700:730] == CODE_D).all() and (out[730:1100] == CODE_A).all() # earliest tag wins


def test_tags_outside_the_graded_spans_are_ignored():
    """Tags outside graded spans are ignored."""
    labels, half = _match()                       # reuse synthetic match
    out = apply_kickoff_rule(labels, half, np.array([50, 600, 1150]), ON) # all tags fall outside
    np.testing.assert_array_equal(out, labels)    # labels unchanged


def test_no_tags_is_a_noop():
    """No tags: output equals input."""
    labels, half = _match()                       # reuse synthetic match
    out = apply_kickoff_rule(labels, half, np.empty(0, dtype=np.int64), ON) # no tags passed
    np.testing.assert_array_equal(out, labels)    # labels unchanged


def test_flag_off_returns_the_input_unchanged():
    """Flag off returns the input array itself."""
    labels, half = _match()                       # reuse synthetic match
    out = apply_kickoff_rule(labels, half, np.array([100, 760]), {**CFG, "kickoff_rule": False}) # rule turned off
    assert out is labels                          # same object returned


def test_arrays_carry_kickoff_starts(arrays):
    """Loaded arrays carry sorted integer kick-off starts."""
    starts = arrays.kickoff_starts                # loaded kick-off starts
    assert starts.dtype.kind == "i"               # integer dtype
    assert np.all(np.diff(starts) >= 0)           # sorted ascending
    assert len(starts) == 2, "DATA.md: two `Kick Off (Start)` instances" # exactly two tags
    # kick-offs are a subset of set pieces
    assert set(starts.tolist()) <= set(arrays.setpiece_starts.tolist())


def test_check_verdicts_and_buckets():
    """Verdicts, diagnosis and buckets of the kick-off check."""
    from tuning.kickoff_check import amended_verdicts, bucket, diagnosis, spans, verdicts

    half = np.zeros(20_000, dtype=np.int8)        # 20k-frame canvas
    half[1_000:8_000] = 1                         # half one span
    half[10_000:18_000] = 2                       # half two span
    # a tag on each half's kick-off: 100 frames into half 1, 2,934 frames into half 2
    good = verdicts(spans(half, np.array([1_100, 12_934])))
    assert all(good.values())                     # good tags pass
    # first-half tag just BEFORE the graded span: the pinned check fails, the diagnosis says why
    starts = np.array([950, 12_934])
    pinned = verdicts(spans(half, starts))        # check the pinned rule
    assert not pinned["one tag inside each graded half"] # pinned check fails here
    assert pinned["second-half span within +/-250 frames of the FAQ's 2,934"] # second half still passes
    assert all(diagnosis(half, starts).values())  # diagnosis explains the miss
    assert all(amended_verdicts(spans(half, starts)).values()), "amended clause accepts it" # amended clause accepts it
    assert not all(amended_verdicts(spans(half, np.array([1_600, 12_934]))).values()), \
        "a first-half tag deep inside the half is still rejected" # deep tag still rejected
    # a second-half tag far from the documented kick-off is rejected
    assert not verdicts(spans(half, np.array([1_100, 14_000])))[
        "second-half span within +/-250 frames of the FAQ's 2,934"]
    # percentage bucket edges
    assert [bucket(x) for x in (0.0, 0.019, 0.03, 0.07, 0.15, 0.4)] == [
        "< 2 %", "< 2 %", "2-5 %", "5-10 %", "10-20 %", "> 20 %"]


def test_rule_is_on_in_the_shipped_config():
    """The shipped config enables the kick-off rule."""
    assert CFG["kickoff_rule"] is True            # rule ships enabled
