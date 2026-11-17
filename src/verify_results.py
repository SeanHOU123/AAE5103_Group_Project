"""Compare reproduced results with the reference metrics in expected_results/.

Tolerances are defined per metric family rather than as one global number,
because the two families behave differently across platforms:

* Continuous metrics (PR-AUC, ROC-AUC, Brier score) move only by floating-point
  noise when the CPU, BLAS build or dependency wheel changes, so a tight
  absolute tolerance is appropriate.
* Metrics derived from a fitted decision threshold (precision, recall, F1,
  flagged share) are step functions of that threshold. One report crossing the
  boundary shifts recall by 1 / n_positives, which is 0.025 on the M?-excluded
  test set where only 40 reports are positive. A single 0.01 bound would reject
  a correct reproduction over a one-report difference.
* Sample sizes (train/validation/test rows, airport-month counts) never depend
  on a fitted model and must therefore match exactly.

The reference files in expected_results/ were regenerated on 64-bit Windows
with the pinned environment listed in requirements.txt.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Continuous ranking / probability metrics: agreement expected within 0.01.
TOLERANCE = 0.01

# Threshold-derived proportions: 0.05 covers the 1 / n_positives step (0.025)
# with margin, while still catching a model that failed to fit.
PROPORTION_TOLERANCE = 0.05

# Flagged share, expressed in percentage points. One report is about 0.015 pp
# on the 6,593-report test set, so 0.25 pp tolerates a few reports crossing.
FLAGGED_TOLERANCE = 0.25

# Confusion-matrix entries, in reports. A slightly different forest can move a
# handful of test reports across the fitted threshold without being wrong.
COUNT_TOLERANCE = 10

# Sample sizes and selection counts must match exactly.
EXACT = 0

PER_COLUMN_TOLERANCE = {
    "precision": PROPORTION_TOLERANCE,
    "recall": PROPORTION_TOLERANCE,
    "f1": PROPORTION_TOLERANCE,
    "validation_best_f1": PROPORTION_TOLERANCE,
    "flagged_pct": FLAGGED_TOLERANCE,
    "true_negative": COUNT_TOLERANCE,
    "false_positive": COUNT_TOLERANCE,
    "false_negative": COUNT_TOLERANCE,
    "true_positive": COUNT_TOLERANCE,
    "train_rows": EXACT,
    "validation_rows": EXACT,
    "test_rows": EXACT,
    "airport_months_total": EXACT,
    "airport_months_selected_top20pct": EXACT,
    "actual_damaging_reports_total": EXACT,
    # Runway transfer test: counts are fixed, rounded metrics allow one step.
    "runway_cells": EXACT,
    "airports": EXACT,
    "damaging_total": EXACT,
    "within_airport_spearman": TOLERANCE,
    "p_value": TOLERANCE,
    "capture_model_pct": 0.5,
    "capture_volume_pct": 0.5,
    "top1_runway_agreement_pct": 5,
}


def compare_csv(relative_path: str, keys: list[str]) -> None:
    actual = pd.read_csv(ROOT / "outputs" / relative_path).sort_values(keys).reset_index(drop=True)
    expected = pd.read_csv(ROOT / "expected_results" / relative_path).sort_values(keys).reset_index(drop=True)

    if list(actual.columns) != list(expected.columns):
        raise AssertionError(f"{relative_path}: column set differs from reference")
    if len(actual) != len(expected):
        raise AssertionError(f"{relative_path}: {len(actual)} rows, expected {len(expected)}")

    for column in actual.columns:
        if column in keys:
            continue
        if actual[column].dtype.kind in "biufc":
            tolerance = PER_COLUMN_TOLERANCE.get(column, TOLERANCE)
            diff = (actual[column].astype(float) - expected[column].astype(float)).abs()
            worst = float(np.nanmax(diff))
            assert worst <= tolerance, (
                f"{relative_path}: column {column} differs by {worst:.6g} "
                f"(tolerance {tolerance:g})"
            )
            if worst > 0:
                print(f"  within tolerance ({worst:.2g} <= {tolerance:g}): {relative_path} :: {column}")
        else:
            mismatch = (actual[column].fillna("<NA>") != expected[column].fillna("<NA>")).sum()
            assert mismatch == 0, f"{relative_path}: column {column} has {mismatch} differing values"

    print(f"Matched reference results: {relative_path}")


def main() -> None:
    print(f"Environment: Python {platform.python_version()} on {platform.system()}")

    compare_csv("model/model_metrics.csv", ["model", "sensitivity_exclude_M?"])
    compare_csv("decision/backtest_metrics.csv", ["method"])
    compare_csv("decision/runway_transfer_test.csv", ["model"])

    with (ROOT / "outputs/model/analysis_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    expected_counts = {
        "selected_airports": 20,
        "airport_month_rows": 2640,
        "all_selected_airport_reports": 51032,
        "labeled_model_events": 25318,
        "positive_events": 1377,
    }
    for key, expected in expected_counts.items():
        assert summary[key] == expected, f"{key}: got {summary[key]}, expected {expected}"
    assert len(summary["primary_models"]) == 4, "expected the four-method comparison"
    calendar = pd.read_csv(ROOT / "outputs/decision/seasonal_priority_calendar.csv")
    tiers = calendar["priority_tier"].value_counts().to_dict()
    assert len(calendar) == 240, f"calendar rows: got {len(calendar)}, expected 240"
    assert tiers == {"Highest": 7, "Elevated": 117, "Routine": 116}, f"tier counts: {tiers}"
    print("Matched the 240-cell calendar and its 7 / 117 / 116 tiers.")
    print("Matched all core sample counts.")
    print("Matched four-method comparison configuration.")
    print("Reproduction verification passed.")


if __name__ == "__main__":
    main()
