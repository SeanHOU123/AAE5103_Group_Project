from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "data/raw/Public.xlsx"
OUT = ROOT / "outputs/audit"
DATA = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

wanted = [
    "INDEX_NR", "INCIDENT_DATE", "INCIDENT_MONTH", "INCIDENT_YEAR",
    "TIME_OF_DAY", "AIRPORT_ID", "AIRPORT", "STATE", "AC_CLASS",
    "AC_MASS", "TYPE_ENG", "NUM_ENGS", "PHASE_OF_FLIGHT", "HEIGHT",
    "SPEED", "SKY", "PRECIPITATION", "INDICATED_DAMAGE", "DAMAGE_LEVEL",
    "EFFECT", "SPECIES_ID", "SPECIES", "SIZE", "NUM_SEEN", "NUM_STRUCK",
    "WARNED", "SOURCE", "PERSON", "LUPDATE"
]

def display_path(path: Path) -> str:
    """Path relative to the package root, so the audit summary stays portable."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


print("Reading source workbook...", flush=True)
df = pd.read_excel(
    SOURCE,
    sheet_name="Sheet1",
    engine="openpyxl",
    usecols=lambda name: name in wanted,
)
print(f"Loaded {len(df):,} rows and {len(df.columns)} selected columns", flush=True)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    df[col] = df[col].replace("", pd.NA)

df["INCIDENT_YEAR"] = pd.to_numeric(df["INCIDENT_YEAR"], errors="coerce")
df["INCIDENT_MONTH"] = pd.to_numeric(df["INCIDENT_MONTH"], errors="coerce")
df["INCIDENT_DATE_PARSED"] = pd.to_datetime(df["INCIDENT_DATE"], errors="coerce")

scope = df[df["INCIDENT_YEAR"].between(2015, 2025, inclusive="both")].copy()
scope["DAMAGE_LEVEL"] = scope["DAMAGE_LEVEL"].str.upper()
scope["damaging"] = scope["DAMAGE_LEVEL"].isin(["M", "M?", "S", "D"])
scope["damage_known"] = scope["DAMAGE_LEVEL"].isin(["N", "M", "M?", "S", "D"])
scope["damage_undetermined"] = scope["DAMAGE_LEVEL"].eq("M?")

summary = {
    "source": display_path(SOURCE),
    "source_rows": int(len(df)),
    "source_columns_total": 102,
    "audit_columns": list(df.columns),
    "scope_years": [2015, 2025],
    "scope_rows": int(len(scope)),
    "scope_airports": int(scope["AIRPORT_ID"].nunique(dropna=True)),
    "scope_min_date": str(scope["INCIDENT_DATE_PARSED"].min().date()),
    "scope_max_date": str(scope["INCIDENT_DATE_PARSED"].max().date()),
    "missing_incident_date": int(scope["INCIDENT_DATE_PARSED"].isna().sum()),
    "duplicate_index_nr": int(scope["INDEX_NR"].duplicated().sum()),
    "duplicate_full_audit_rows": int(scope[wanted].duplicated().sum()),
    "damage_level_counts": {
        str(key): int(value)
        for key, value in scope["DAMAGE_LEVEL"].fillna("(missing)").value_counts(dropna=False).items()
    },
    "primary_target": "N=no reported damage; M/M?/S/D=damaging strike; blank=excluded from supervised model",
    "modeling_rows_with_labels": int(scope["damage_known"].sum()),
    "modeling_damaging_events": int(scope["damaging"].sum()),
    "modeling_positive_pct": round(float(scope.loc[scope["damage_known"], "damaging"].mean() * 100), 3),
}

candidate_features = [
    "INCIDENT_MONTH", "TIME_OF_DAY", "AIRPORT_ID", "AC_CLASS", "AC_MASS",
    "TYPE_ENG", "NUM_ENGS", "PHASE_OF_FLIGHT", "HEIGHT", "SPEED", "SKY",
    "PRECIPITATION", "SPECIES_ID", "SPECIES", "SIZE", "NUM_SEEN",
    "NUM_STRUCK", "WARNED"
]
completeness = pd.DataFrame({
    "field": candidate_features,
    "non_missing": [int(scope[col].notna().sum()) for col in candidate_features],
    "missing": [int(scope[col].isna().sum()) for col in candidate_features],
    "missing_pct": [round(float(scope[col].isna().mean() * 100), 2) for col in candidate_features],
    "unique_values": [int(scope[col].nunique(dropna=True)) for col in candidate_features],
}).sort_values("missing_pct")
completeness.to_csv(OUT / "candidate_completeness.csv", index=False)

damage_distribution = (
    scope["DAMAGE_LEVEL"].fillna("(missing)").value_counts(dropna=False)
    .rename_axis("damage_level").reset_index(name="count")
)
damage_distribution["pct"] = (damage_distribution["count"] / len(scope) * 100).round(3)
damage_distribution.to_csv(OUT / "damage_distribution.csv", index=False)

airport_stats = scope.groupby(["AIRPORT_ID", "AIRPORT"], dropna=False).agg(
    reports=("INDEX_NR", "size"),
    known_damage_labels=("damage_known", "sum"),
    damaging_events=("damaging", "sum"),
    undetermined_damage=("damage_undetermined", "sum"),
    phase_completeness=("PHASE_OF_FLIGHT", lambda x: x.notna().mean()),
    species_completeness=("SPECIES_ID", lambda x: x.notna().mean()),
    aircraft_class_completeness=("AC_CLASS", lambda x: x.notna().mean()),
).reset_index()
airport_stats["damaging_pct_all_reports"] = (
    airport_stats["damaging_events"] / airport_stats["reports"] * 100
).round(2)
airport_stats["phase_completeness"] = (airport_stats["phase_completeness"] * 100).round(2)
airport_stats["species_completeness"] = (airport_stats["species_completeness"] * 100).round(2)
airport_stats["aircraft_class_completeness"] = (airport_stats["aircraft_class_completeness"] * 100).round(2)
airport_stats = airport_stats.sort_values("reports", ascending=False)
airport_stats.to_csv(OUT / "airport_statistics.csv", index=False)

eligible = airport_stats[
    (airport_stats["AIRPORT_ID"].notna())
    & (airport_stats["AIRPORT_ID"] != "ZZZZ")
    & (airport_stats["reports"] >= 500)
    & (airport_stats["known_damage_labels"] >= 400)
    & (airport_stats["damaging_events"] >= 30)
].copy()
eligible["selection_rule"] = "named airport; reports>=500; labeled_damage>=400; damaging>=30"
selected = eligible.head(20).copy()
selected.to_csv(OUT / "selected_airports.csv", index=False)
summary["eligible_airports"] = int(len(eligible))
summary["selected_airports"] = selected["AIRPORT_ID"].tolist()
summary["selection_rule"] = "Top 20 by reports among named airports with reports>=500, labeled damage records>=400, and damaging events>=30; M/M?/S/D are positive"

scope.to_pickle(DATA / "faa_2015_2025_audit.pkl")
with (OUT / "audit_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
print("\nCandidate completeness:\n", completeness.to_string(index=False), flush=True)
print("\nSelected airports:\n", selected[["AIRPORT_ID", "AIRPORT", "reports", "damaging_events", "phase_completeness"]].to_string(index=False), flush=True)
