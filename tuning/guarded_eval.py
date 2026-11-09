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

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL = os.path.join(_ROOT, "evaluation")
_RESULTS = os.path.join(_EVAL, "results")
_REPORTS = os.path.join(_EVAL, "reports")
_SUBMISSION = os.path.join(_ROOT, "LA+AI.csv")
_NAME = "LA+AI"
TIGHT_TAU = 12  # pm.TAUS[0], frames: |mean latency| within it reads as "none"


def count_bucket(poss: int) -> str:
    """Plan §1 count bucket: ≤10 / 11–13 / 14–22 / 23–30 / >30 (17 true on the window)."""
    if poss <= 10:
        return "<=10"
    if poss <= 13:
        return "11-13"
    if poss <= 22:
        return "14-22"
    if poss <= 30:
        return "23-30"
    return ">30"


def band_and_flags(report: dict) -> tuple[str, dict[str, str]]:
    """Band letter + the four categorical flags from one per-submission JSON (plan §1)."""
    mid = (report["spell_bootstrap_low"] + report["spell_bootstrap_high"]) / 2.0
    band = "A" if mid >= 0.43 else "B" if mid >= 0.30 else "C" if mid >= 0.15 else "D"

    transitions = report.get("transitions") or {}
    taus = report.get("taus") or [TIGHT_TAU]
    mean_lat = (transitions.get(str(max(taus))) or {}).get("mean_latency")
    if mean_lat is None or abs(mean_lat) <= TIGHT_TAU:
        latency = "none"
    else:
        latency = "late" if mean_lat > 0 else "early"

    per_class = report["per_class"]
    flags = {
        "count": count_bucket(report["n_pred_spells"]),
        "f1_D": "normal" if per_class["D"]["f1"] >= 0.3 else "low",
        "balance": ("normal"
                    if min(per_class["H"]["f1"], per_class["A"]["f1"]) >= 0.3
                    else "collapsed"),
        "latency": latency,
    }
    return band, flags


def main() -> int:
    if not os.path.isfile(_SUBMISSION):
        print("BAND: D (frozen LA+AI.csv missing)")
        return 2
    os.makedirs(_RESULTS, exist_ok=True)
    os.makedirs(_REPORTS, exist_ok=True)
    shutil.copy2(_SUBMISSION, os.path.join(_RESULTS, _NAME + ".csv"))
    report_path = os.path.join(_REPORTS, _NAME + ".json")
    if os.path.exists(report_path):
        # never let a stale report masquerade as this run's result
        os.remove(report_path)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(_REPORTS, "guarded_board_{}.txt".format(stamp))
    proc = subprocess.run(
        [sys.executable, os.path.join(_EVAL, "run_evaluation.py"), "--bootstrap", "1000"],
        cwd=_ROOT, capture_output=True, text=True)
    with open(archive, "w", encoding="utf-8") as handle:
        handle.write(proc.stdout)
        handle.write("\n--- stderr ---\n")
        handle.write(proc.stderr)

    with open(_SUBMISSION, "rb") as handle:
        md5 = hashlib.md5(handle.read()).hexdigest()

    if not os.path.isfile(report_path):
        # skipped submission or a crashed run: band D either way; the reason is in the archive
        print("BAND: D (validator failure or evaluator error — reason in archive)")
        print("submission md5: {}".format(md5))
        print("archive (operator eyes only): {}".format(archive))
        return 1

    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    band, flags = band_and_flags(report)
    print("BAND: {}".format(band))
    print("flags: count={}  f1_D={}  balance={}  latency={}".format(
        flags["count"], flags["f1_D"], flags["balance"], flags["latency"]))
    print("submission md5: {}".format(md5))
    print("archive (operator eyes only): {}".format(archive))
    return 0


if __name__ == "__main__":
    sys.exit(main())
