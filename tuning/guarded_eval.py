"""Guarded evaluator run — redacting wrapper (plan §1, dossier §3 protocol).

Operator-only: the *operator* runs this. It copies the frozen root ``LA+AI.csv`` into
``evaluation/results/``, executes ``evaluation/run_evaluation.py --bootstrap 1000`` with
stdout/stderr captured into an archive under ``evaluation/reports/`` (gitignored, operator
eyes only), and prints ONLY the band letter and the four categorical flags of plan §1.
No score ever reaches an AI channel.

Band: plan §1 table on the bootstrap mid = (spell_bootstrap_low + spell_bootstrap_high) / 2
of the public-window spellF1. Flags: the plan §1 buckets — count vs 17 true, f1_D normal
(≥ 0.3), H/A balance normal (min f1_H, f1_A ≥ 0.3), latency early/late/none with the tight
transition tau (12 frames) as the "no systematic shift" cut. Flags diagnose; they never tune.
"""

from __future__ import annotations   # postponed annotation evaluation

import datetime as dt   # timestamped archive file names
import hashlib          # md5 fingerprint of the submission
import json             # reading the evaluator's report
import os               # paths, directory and file checks
import shutil           # copying the submission into results/
import subprocess       # launching the official evaluator
import sys              # python executable and exit codes

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
_EVAL = os.path.join(_ROOT, "evaluation")        # evaluator lives here
_RESULTS = os.path.join(_EVAL, "results")        # submission is staged here
_REPORTS = os.path.join(_EVAL, "reports")        # operator-only archives land here
_SUBMISSION = os.path.join(_ROOT, "LA+AI.csv")   # the frozen submission
_NAME = "LA+AI"                                  # base name used in both dirs
TIGHT_TAU = 12  # pm.TAUS[0], frames: |mean latency| within it reads as "none"


def count_bucket(poss: int) -> str:
    """Plan §1 count bucket: ≤10 / 11–13 / 14–22 / 23–30 / >30 (17 true on the window)."""
    # walk the upper bounds lowest first; the first match wins
    if poss <= 10:
        return "<=10"   # bucket for ten or fewer
    if poss <= 13:   # bound at thirteen
        return "11-13"   # bucket for eleven to thirteen
    if poss <= 22:   # bound at twenty-two
        return "14-22"   # bucket for fourteen to twenty-two
    if poss <= 30:   # bound at thirty
        return "23-30"   # bucket for twenty-three to thirty
    return ">30"   # bucket above thirty


def band_and_flags(report: dict) -> tuple[str, dict[str, str]]:
    """Band letter + the four categorical flags from one per-submission JSON (plan §1)."""
    mid = (report["spell_bootstrap_low"] + report["spell_bootstrap_high"]) / 2.0   # bootstrap mid
    band = "A" if mid >= 0.43 else "B" if mid >= 0.30 else "C" if mid >= 0.15 else "D"   # plan §1 cuts

    transitions = report.get("transitions") or {}   # latency block, may be absent
    taus = report.get("taus") or [TIGHT_TAU]        # tau grid, else just the tight cut
    mean_lat = (transitions.get(str(max(taus))) or {}).get("mean_latency")   # latency at largest tau
    if mean_lat is None or abs(mean_lat) <= TIGHT_TAU:
        latency = "none"                            # inside the tight window
    else:
        latency = "late" if mean_lat > 0 else "early"   # sign of the shift

    per_class = report["per_class"]   # per-class scores from the report
    flags = {   # the four categorical flags of plan §1
        "count": count_bucket(report["n_pred_spells"]),   # predicted spell count
        "f1_D": "normal" if per_class["D"]["f1"] >= 0.3 else "low",   # dead-class floor
        "balance": ("normal"
                    if min(per_class["H"]["f1"], per_class["A"]["f1"]) >= 0.3
                    else "collapsed"),               # both team floors must hold
        "latency": latency,
    }
    return band, flags   # band letter plus four flags


def main() -> int:
    if not os.path.isfile(_SUBMISSION):
        print("BAND: D (frozen LA+AI.csv missing)")   # missing file reported as band D
        return 2   # distinct code: submission missing
    os.makedirs(_RESULTS, exist_ok=True)   # both output dirs must exist
    os.makedirs(_REPORTS, exist_ok=True)   # same for the reports directory
    shutil.copy2(_SUBMISSION, os.path.join(_RESULTS, _NAME + ".csv"))   # stage submission
    report_path = os.path.join(_REPORTS, _NAME + ".json")   # path of the JSON report
    if os.path.exists(report_path):
        # never let a stale report masquerade as this run's result
        os.remove(report_path)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")   # unique per run
    archive = os.path.join(_REPORTS, "guarded_board_{}.txt".format(stamp))   # unique archive file path
    proc = subprocess.run(   # official evaluator, full output captured
        [sys.executable, os.path.join(_EVAL, "run_evaluation.py"), "--bootstrap", "1000"],
        cwd=_ROOT, capture_output=True, text=True)
    with open(archive, "w", encoding="utf-8") as handle:
        handle.write(proc.stdout)          # evaluator stdout goes to the archive,
        handle.write("\n--- stderr ---\n")  # never to the console
        handle.write(proc.stderr)   # append captured stderr

    with open(_SUBMISSION, "rb") as handle:
        md5 = hashlib.md5(handle.read()).hexdigest()   # log fingerprint of the file

    if not os.path.isfile(report_path):
        # skipped submission or a crashed run: band D either way; the reason is in the archive
        print("BAND: D (validator failure or evaluator error — reason in archive)")
        print("submission md5: {}".format(md5))   # fingerprint for the failure log
        print("archive (operator eyes only): {}".format(archive))   # archive location for the operator
        return 1   # failure code

    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)   # per-submission JSON just written
    band, flags = band_and_flags(report)   # reduce to letter + flags only
    print("BAND: {}".format(band))   # the only performance output: a letter
    print("flags: count={}  f1_D={}  balance={}  latency={}".format(   # four categorical flags
        flags["count"], flags["f1_D"], flags["balance"], flags["latency"]))
    print("submission md5: {}".format(md5))   # fingerprint for the run log
    print("archive (operator eyes only): {}".format(archive))   # full output stays there
    return 0   # success exit code


if __name__ == "__main__":   # run only as a script
    sys.exit(main())   # exit with main's code
