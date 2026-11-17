from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/opsnet_monthly_operations.csv"


def read_opsnet_report(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    table = pd.read_html(StringIO(text))[0]
    operations_columns = [
        idx for idx, col in enumerate(table.columns)
        if "Total Operations" in str(col)
    ]
    if len(operations_columns) != 1:
        raise ValueError(f"Expected one Total Operations column in {path}, found {len(operations_columns)}")

    result = pd.DataFrame({
        "facility": table.iloc[:, 0].astype("string").str.strip(),
        "report_month": table.iloc[:, 1].astype("string").str.strip(),
        "total_operations": pd.to_numeric(table.iloc[:, operations_columns[0]], errors="coerce"),
        "source_file": str(path),
    })
    result = result[
        result["report_month"].str.match(r"^\d{2}/\d{4}$", na=False)
        & result["facility"].ne("Total:")
    ].copy()
    result["month"] = pd.to_datetime(result["report_month"], format="%m/%Y").dt.to_period("M").dt.to_timestamp()
    result["AIRPORT_ID"] = result["facility"].map(
        lambda code: code if len(code) == 4 else "K" + code
    )
    result = result[["AIRPORT_ID", "facility", "month", "total_operations", "source_file"]]
    return result


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    frames = [read_opsnet_report(path) for path in args.files]
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["AIRPORT_ID", "month"]).any():
        duplicates = data[data.duplicated(["AIRPORT_ID", "month"], keep=False)]
        raise ValueError(f"Duplicate airport-month rows found:\n{duplicates}")
    if data["total_operations"].isna().any() or (data["total_operations"] < 0).any():
        raise ValueError("Missing or negative Total Operations values found")

    data = data.sort_values(["AIRPORT_ID", "month"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(data):,} airport-month rows for {data['AIRPORT_ID'].nunique()} airport(s) to {OUT}")
    print(data.groupby("AIRPORT_ID")["month"].agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()
