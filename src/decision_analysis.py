from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
AUDIT = ROOT / "outputs/audit"
MODEL = ROOT / "outputs/model"
OUT = ROOT / "outputs/decision"
FIG = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 220})


def capture_metrics(frame: pd.DataFrame, score_column: str, label: str) -> dict:
    ranked = frame.sort_values([score_column, "AIRPORT_ID", "year", "month_number"], ascending=[False, True, True, True]).copy()
    selected_count = int(np.ceil(len(ranked) * 0.20))
    selected = ranked.head(selected_count)
    total_damage = int(frame["damaging_reports"].sum())
    captured = int(selected["damaging_reports"].sum())
    return {
        "method": label,
        "airport_months_total": int(len(frame)),
        "airport_months_selected_top20pct": selected_count,
        "actual_damaging_reports_total": total_damage,
        "actual_damaging_reports_captured": captured,
        "damage_capture_pct": captured / total_damage * 100 if total_damage else np.nan,
        "selected_reported_strikes": int(selected["reported_strikes"].sum()),
        "selected_damaging_per_100k_operations": selected["damaging_reports"].sum() / selected["total_operations"].sum() * 100000,
    }


def main() -> None:
    panel = pd.read_csv(DATA / "airport_month_panel.csv", parse_dates=["month"])
    airports = pd.read_csv(AUDIT / "selected_airports.csv")["AIRPORT_ID"]
    events = pd.read_pickle(DATA / "faa_2015_2025_audit.pkl")
    events = events[events["AIRPORT_ID"].isin(airports)].copy()
    predictions = pd.read_csv(MODEL / "test_predictions.csv")
    predictions = predictions[(predictions["model"] == "Random forest") & (~predictions["sensitivity_exclude_M?"])].copy()

    history = panel[panel["year"] <= 2023].copy()
    test = panel[panel["year"] >= 2024].copy()
    global_month = history.groupby("month_number").agg(
        strikes=("reported_strikes", "sum"), operations=("total_operations", "sum")
    ).reset_index()
    global_month["global_rate"] = global_month["strikes"] / global_month["operations"] * 100000

    frequency = history.groupby(["AIRPORT_ID", "month_number"], as_index=False).agg(
        historical_strikes=("reported_strikes", "sum"),
        historical_operations=("total_operations", "sum"),
        historical_years=("year", "nunique"),
    ).merge(global_month[["month_number", "global_rate"]], on="month_number", how="left")
    frequency["prior_operations"] = frequency["historical_operations"] / frequency["historical_years"]
    frequency["smoothed_reported_rate"] = (
        (frequency["historical_strikes"] + frequency["global_rate"] * frequency["prior_operations"] / 100000)
        / (frequency["historical_operations"] + frequency["prior_operations"]) * 100000
    )

    test_severity = predictions.groupby(["AIRPORT_ID", "INCIDENT_YEAR", "INCIDENT_MONTH"], as_index=False).agg(
        mean_predicted_damage_probability=("damage_probability", "mean"),
        labeled_prediction_records=("INDEX_NR", "size"),
    ).rename(columns={"INCIDENT_YEAR": "year", "INCIDENT_MONTH": "month_number"})

    backtest = test.merge(frequency[["AIRPORT_ID", "month_number", "smoothed_reported_rate"]], on=["AIRPORT_ID", "month_number"], how="left", validate="many_to_one")
    backtest = backtest.merge(test_severity, on=["AIRPORT_ID", "year", "month_number"], how="left", validate="one_to_one")
    airport_fallback = predictions.groupby("AIRPORT_ID")["damage_probability"].mean()
    global_fallback = predictions["damage_probability"].mean()
    backtest["mean_predicted_damage_probability"] = backtest.apply(
        lambda row: row["mean_predicted_damage_probability"] if pd.notna(row["mean_predicted_damage_probability"])
        else airport_fallback.get(row["AIRPORT_ID"], global_fallback), axis=1,
    )
    backtest["frequency_rank"] = backtest["smoothed_reported_rate"].rank(pct=True)
    backtest["severity_rank"] = backtest["mean_predicted_damage_probability"].rank(pct=True)
    backtest["frequency_only_score"] = backtest["frequency_rank"]
    backtest["combined_risk_score"] = 0.5 * backtest["frequency_rank"] + 0.5 * backtest["severity_rank"]
    backtest.to_csv(OUT / "backtest_airport_month_scores.csv", index=False)

    metrics = [
        capture_metrics(backtest, "frequency_only_score", "Historical reported-strike rate only"),
        capture_metrics(backtest, "combined_risk_score", "Historical rate + predicted damage severity"),
    ]
    metrics.append({
        "method": "Equal 20% selection benchmark",
        "airport_months_total": int(len(backtest)),
        "airport_months_selected_top20pct": int(np.ceil(len(backtest) * 0.20)),
        "actual_damaging_reports_total": int(backtest["damaging_reports"].sum()),
        "actual_damaging_reports_captured": np.nan,
        "damage_capture_pct": 20.0,
        "selected_reported_strikes": np.nan,
        "selected_damaging_per_100k_operations": np.nan,
    })
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(OUT / "backtest_metrics.csv", index=False)

    severity_calendar = predictions.groupby(["AIRPORT_ID", "INCIDENT_MONTH"], as_index=False).agg(
        mean_predicted_damage_probability=("damage_probability", "mean"),
        labeled_prediction_records=("INDEX_NR", "size"),
    ).rename(columns={"INCIDENT_MONTH": "month_number"})
    calendar = frequency.merge(severity_calendar, on=["AIRPORT_ID", "month_number"], how="left")
    calendar["mean_predicted_damage_probability"] = calendar.apply(
        lambda row: row["mean_predicted_damage_probability"] if pd.notna(row["mean_predicted_damage_probability"])
        else airport_fallback.get(row["AIRPORT_ID"], global_fallback), axis=1,
    )
    calendar["frequency_percentile"] = calendar["smoothed_reported_rate"].rank(pct=True)
    calendar["severity_percentile"] = calendar["mean_predicted_damage_probability"].rank(pct=True)
    calendar["priority_score"] = 0.5 * calendar["frequency_percentile"] + 0.5 * calendar["severity_percentile"]
    calendar["priority_tier"] = pd.cut(
        calendar["priority_score"], bins=[-np.inf, 0.5, 0.8, np.inf], labels=["Routine", "Elevated", "Highest"]
    )
    calendar["frequency_level"] = np.where(calendar["frequency_percentile"] >= 0.8, "Higher reported frequency", "Lower reported frequency")
    calendar["severity_level"] = np.where(calendar["severity_percentile"] >= 0.8, "Higher predicted severity", "Lower predicted severity")
    calendar["recommended_use"] = np.select(
        [
            (calendar["frequency_percentile"] >= 0.8) & (calendar["severity_percentile"] >= 0.8),
            (calendar["frequency_percentile"] >= 0.8) | (calendar["severity_percentile"] >= 0.8),
        ],
        [
            "Priority review: seasonal patrol planning, species monitoring and attractant audit",
            "Targeted review: investigate the dominant frequency or severity driver",
        ],
        default="Routine monitoring; retain airport-specific wildlife biologist oversight",
    )
    calendar.sort_values("priority_score", ascending=False).to_csv(OUT / "seasonal_priority_calendar.csv", index=False)

    heatmap = calendar.pivot(index="AIRPORT_ID", columns="month_number", values="priority_score")
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    sns.heatmap(heatmap, cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.15, ax=ax, cbar_kws={"label": "Relative priority score (0–1)"})
    ax.set(xlabel="Calendar month", ylabel="Airport", title="Seasonal wildlife-strike management priority")
    save_name = FIG / "fig06_seasonal_priority_heatmap.png"
    fig.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    plot_metrics = metrics_frame[metrics_frame["method"] != "Equal 20% selection benchmark"]
    sns.barplot(data=plot_metrics, x="damage_capture_pct", y="method", hue="method", legend=False, palette=["#2878B5", "#C82423"], ax=ax)
    ax.axvline(20, color="grey", linestyle="--", label="Equal-selection benchmark")
    ax.set(xlabel="Actual damaging reports captured (%)", ylabel="", title="Retrospective top-20% airport-month prioritization, 2024–2025", xlim=(0, max(100, plot_metrics["damage_capture_pct"].max() * 1.2)))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig07_prioritization_backtest.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "backtest_period": "2024-2025",
        "frequency_training_period": "2015-2023",
        "selection_fraction": 0.20,
        "combined_score": "50% within-test percentile of smoothed historical rate + 50% within-test percentile of mean predicted damage probability",
        "interpretation": "Retrospective prioritization evidence only; not a causal estimate of mitigation effectiveness",
    }
    with (OUT / "decision_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(metrics_frame.to_string(index=False))
    print("\nHighest-priority calendar entries:\n", calendar.nlargest(15, "priority_score")[["AIRPORT_ID", "month_number", "smoothed_reported_rate", "mean_predicted_damage_probability", "priority_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
