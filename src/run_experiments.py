from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

COMMANDS = [
    [sys.executable, str(SRC / "audit_faa.py"), str(ROOT / "data/raw/Public.xlsx")],
    [sys.executable, str(SRC / "prepare_opsnet.py"), str(ROOT / "data/raw/WEB-Report-41661.xls")],
    [sys.executable, str(SRC / "build_monthly_panel.py")],
    [sys.executable, str(SRC / "analyze_and_model.py")],
    [sys.executable, str(SRC / "decision_analysis.py")],
    [sys.executable, str(SRC / "runway_transfer_test.py")],
]


def main() -> None:
    for step, command in enumerate(COMMANDS, start=1):
        print(f"\n===== Experiment step {step}/{len(COMMANDS)} =====", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nAll experiments completed. Results are in the outputs directory.", flush=True)


if __name__ == "__main__":
    main()
