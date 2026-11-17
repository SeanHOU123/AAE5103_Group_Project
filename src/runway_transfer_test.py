"""Runway-level transfer test of the four-method event score (report Section 5.3).

Question: the score ranks airport-months and individual reports. Does it also
rank runways *within* an airport, and does it beat the simpler rule of putting
the busiest runway first at equal budget?

Method: take the 2024-2025 holdout predictions already produced by the
four-method pipeline, attach the runway recorded in the report, aggregate the
predicted probability to airport-runway cells, and compare with observed
outcomes. Runway is used only for grouping, never as a feature, so no
information leaks back into the models. Cells with fewer than 20 reports are
dropped, which leaves 72 airport-runway cells across 20 airports.

Usage:
    python src/runway_transfer_test.py
"""

import os
import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/Public.xlsx"
RUNWAY_CACHE = ROOT / "data/processed/runway_index.pkl"
PRED = ROOT / "outputs/model/test_predictions.csv"
OUT = ROOT / "outputs/decision/runway_transfer_test.csv"

MIN_EVENTS = 20  # ignore runway cells too thin to say anything


def normalise(runway):
    """Fold '17R/35L' to a single physical strip; numbers >18 wrap around."""
    if pd.isna(runway):
        return None
    first = [p.strip() for p in str(runway).strip().upper().split("/") if p.strip()][0]
    m = re.match(r"^(\d{1,2})([LRC]?)$", first)
    if not m:
        return None
    number, suffix = int(m.group(1)), m.group(2)
    if number > 18:
        number -= 18
        suffix = {"L": "R", "R": "L"}.get(suffix, suffix)
    return "%02d%s" % (number, suffix)


def load_runway():
    if RUNWAY_CACHE.exists():
        frame = pd.read_pickle(RUNWAY_CACHE)
    else:
        frame = pd.read_excel(RAW, usecols=["INDEX_NR", "RUNWAY"], engine="openpyxl")
        RUNWAY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(RUNWAY_CACHE)
    frame["rwy"] = frame["RUNWAY"].map(normalise)
    return frame[["INDEX_NR", "rwy"]]


def cells(predictions, model):
    subset = predictions[
        (predictions["model"] == model)
        & (~predictions["sensitivity_exclude_M?"])
    ].dropna(subset=["rwy"])
    grouped = (subset.groupby(["AIRPORT_ID", "rwy"])
               .agg(events=("damaging", "size"),
                    damaging=("damaging", "sum"),
                    mean_probability=("damage_probability", "mean"))
               .reset_index())
    grouped = grouped[grouped["events"] >= MIN_EVENTS].copy()
    grouped["observed_rate"] = grouped["damaging"] / grouped["events"]
    return subset, grouped


def main():
    predictions = pd.read_csv(PRED).merge(load_runway(), on="INDEX_NR", how="left")
    print("holdout rows %d, runway joined %.1f%%"
          % (len(predictions), 100 * predictions["rwy"].notna().mean()))

    rows = []
    for model in ["Logistic regression", "Random forest",
                  "MLP neural network", "Gradient boosting"]:
        subset, grouped = cells(predictions, model)
        if grouped.empty:
            continue

        # Airport-centred values isolate the within-airport ranking.
        centred_prob = grouped["mean_probability"] - grouped.groupby(
            "AIRPORT_ID")["mean_probability"].transform("mean")
        centred_rate = grouped["observed_rate"] - grouped.groupby(
            "AIRPORT_ID")["observed_rate"].transform("mean")
        rho, p_value = spearmanr(centred_prob, centred_rate)

        total = captured_model = captured_volume = 0
        top1 = 0
        airports = 0
        for _, group in grouped.groupby("AIRPORT_ID"):
            if len(group) < 2:
                continue
            airports += 1
            total += int(group["damaging"].sum())
            captured_model += int(group.sort_values(
                "mean_probability", ascending=False)["damaging"].iloc[0])
            captured_volume += int(group.sort_values(
                "events", ascending=False)["damaging"].iloc[0])
            if (group.sort_values("mean_probability", ascending=False)["rwy"].iloc[0]
                    == group.sort_values("damaging", ascending=False)["rwy"].iloc[0]):
                top1 += 1

        rows.append({
            "model": model,
            "runway_cells": len(grouped),
            "airports": airports,
            "within_airport_spearman": round(rho, 3),
            "p_value": round(p_value, 4),
            "top1_runway_agreement_pct": round(100 * top1 / airports),
            "damaging_total": total,
            "capture_model_pct": round(100 * captured_model / total, 1),
            "capture_volume_pct": round(100 * captured_volume / total, 1),
        })

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    os.makedirs(OUT.parent, exist_ok=True)
    table.to_csv(OUT, index=False)
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
