from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAA = ROOT / "data/processed/faa_2015_2025_audit.pkl"
OPSNET = ROOT / "data/processed/opsnet_monthly_operations.csv"
AIRPORTS = ROOT / "outputs/audit/selected_airports.csv"
OUT = ROOT / "data/processed/airport_month_panel.csv"


def main() -> None:
    faa = pd.read_pickle(FAA)
    operations = pd.read_csv(OPSNET, parse_dates=["month"])
    selected = pd.read_csv(AIRPORTS)["AIRPORT_ID"].tolist()
    available = operations["AIRPORT_ID"].unique().tolist()

    faa = faa[faa["AIRPORT_ID"].isin(selected) & faa["AIRPORT_ID"].isin(available)].copy()
    faa["month"] = pd.to_datetime(
        faa["INCIDENT_YEAR"].astype(int).astype(str) + "-" + faa["INCIDENT_MONTH"].astype(int).astype(str) + "-01"
    )
    strikes = faa.groupby(["AIRPORT_ID", "month"]).agg(
        reported_strikes=("INDEX_NR", "size"),
        labeled_damage_reports=("damage_known", "sum"),
        damaging_reports=("damaging", "sum"),
        undetermined_damage_level=("damage_undetermined", "sum"),
    ).reset_index()

    panel = operations.merge(strikes, on=["AIRPORT_ID", "month"], how="left", validate="one_to_one")
    for col in ["reported_strikes", "labeled_damage_reports", "damaging_reports", "undetermined_damage_level"]:
        panel[col] = panel[col].fillna(0).astype(int)
    panel["reported_strikes_per_100k_operations"] = (
        panel["reported_strikes"] / panel["total_operations"] * 100000
    )
    panel["labeled_damaging_reports_per_100k_operations"] = (
        panel["damaging_reports"] / panel["total_operations"] * 100000
    )
    panel["year"] = panel["month"].dt.year
    panel["month_number"] = panel["month"].dt.month
    panel = panel.sort_values(["AIRPORT_ID", "month"])
    panel.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(panel):,} rows for {panel['AIRPORT_ID'].nunique()} airport(s) to {OUT}")
    print(panel.groupby("AIRPORT_ID").agg(
        months=("month", "size"),
        operations=("total_operations", "sum"),
        reported_strikes=("reported_strikes", "sum"),
        damaging_reports=("damaging_reports", "sum"),
    ).to_string())


if __name__ == "__main__":
    main()
