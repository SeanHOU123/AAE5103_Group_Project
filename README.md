# AAE5103 Group 1 — Wildlife-Strike Damage Prediction: Experiment Reproduction

This package contains the complete experiment code for our group project,
*Explainable Wildlife-Strike Damage Prediction for Airport Decision Support*.
Running it end to end regenerates every number, table and figure that the
report uses. 

`data/raw/Public.xlsx` (159 MB) is stored with **Git LFS**. Install Git LFS
before cloning (`git lfs install`); if the file arrives as a small pointer file,
run `git lfs pull`. Cloning without Git LFS still works for the code, and the
workbook can also be downloaded directly from the repository's web page.

The pipeline reads the two original FAA source files shipped in `data/raw/`,
builds the airport-month panel, trains and evaluates the four classifiers on a
strictly chronological split, constructs the seasonal priority calendar, runs
the retrospective prioritisation backtest, and runs the runway-level transfer
test. A verification script compares the regenerated metrics with the reference
values in `expected_results/`.

---

## 1. Package layout

```
AAE5103_Gp1_Experiment_Reproduction/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── Public.xlsx                 # FAA NWSD workbook, 2015-2025 extract
│       └── WEB-Report-41661.xls        # FAA OPSNET monthly operations report
├── src/
│   ├── audit_faa.py                    # audit NWSD, define labels, select airports
│   ├── prepare_opsnet.py               # parse monthly operations from OPSNET
│   ├── build_monthly_panel.py          # airport-month panel and reported rates
│   ├── analyze_and_model.py            # four classifiers, validation, importance
│   ├── decision_analysis.py            # seasonal calendar, tiers, backtest
│   ├── runway_transfer_test.py         # runway-level transfer test (Section 5.3)
│   ├── run_experiments.py              # runs steps 1-6 in order
│   └── verify_results.py               # compares outputs with the references
└── expected_results/
    ├── model/model_metrics.csv         # reference metrics, both label definitions
    └── decision/
        ├── backtest_metrics.csv        # 26.78% vs 36.27% capture
        └── runway_transfer_test.csv    # Spearman rho and equal-budget capture
```

`data/processed/` and `outputs/` are created at run time; they are not shipped,
so that every reported number is produced by the code in front of you.

---

## 2. Environment

* Python 3.13 (tested with 3.13.14, 64-bit, Windows).
* About 4 GB of free RAM (the NWSD workbook is large and permutation
  importance refits the models).
* Disk space: about 400 MB free for the intermediate pickles, result tables and
  figures written during a run.

Install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the reported results:
pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2,
seaborn 0.13.2, openpyxl 3.1.5, lxml 6.1.3.

No network access is required: both source files are included.

---

## 3. Quick start

From the package root:

```bash
python src/run_experiments.py     # steps 1-6, about 5 minutes
python src/verify_results.py      # compares outputs with expected_results/
```

`run_experiments.py` stops on the first error and prints the step it is running.
`verify_results.py` prints each comparison it makes and exits non-zero if a
value falls outside tolerance.

### Step-by-step equivalent

Run these from the package root if you prefer to execute the stages yourself:

```bash
python src/audit_faa.py data/raw/Public.xlsx
python src/prepare_opsnet.py data/raw/WEB-Report-41661.xls
python src/build_monthly_panel.py
python src/analyze_and_model.py
python src/decision_analysis.py
python src/runway_transfer_test.py
```

| Step | Script | Main output |
|---|---|---|
| 1 | `audit_faa.py` | `outputs/audit/*`, `data/processed/faa_2015_2025_audit.pkl` |
| 2 | `prepare_opsnet.py` | `data/processed/opsnet_monthly_operations.csv` |
| 3 | `build_monthly_panel.py` | `data/processed/airport_month_panel.csv` |
| 4 | `analyze_and_model.py` | `outputs/model/*`, `outputs/figures/fig01-fig05b` |
| 5 | `decision_analysis.py` | `outputs/decision/*`, `outputs/figures/fig06-fig07` |
| 6 | `runway_transfer_test.py` | `outputs/decision/runway_transfer_test.csv` |

---

## 4. What the code does

**Step 1 — Audit and label (`audit_faa.py`).** Reads the NWSD workbook, keeps
2015-2025 records, and selects the 20 airports with the highest report counts
among those with at least 500 reports, 400 labelled damage records and 30
damaging events. Damage follows the FAA database definition (an event that
adversely affects structural strength, performance or flight characteristics
and would normally require major repair or replacement; bent fairings, small
skin dents or puncture holes, wing tips, antennae, tyres, brakes and engine
blades not requiring replacement are excluded). `N` is negative; `M`, `M?`, `S`
and `D` are positive in the primary analysis. `M?` is the FAA *undetermined
level*: the aircraft was damaged but the extent of the damage is unknown, which
is why it is counted positive in the primary analysis and removed in the
sensitivity analysis.

**Step 2 — Operations (`prepare_opsnet.py`).** Parses the OPSNET monthly report
into airport-month totals, counting take-offs and landings alike.

**Step 3 — Panel (`build_monthly_panel.py`).** Joins reports to operations and
computes the reported-strike rate per 100,000 operations for every complete
airport-month.

**Step 4 — Models (`analyze_and_model.py`).** One preprocessing pipeline, one
split and one threshold policy shared by four classifiers: logistic regression,
random forest (450 class-balanced trees), a two-layer MLP and histogram gradient
boosting. Inputs are strike context only; outcome fields (repair cost, damage
level, effect on flight) are excluded. Split: 2015-2022 train, 2023 validation
(hyperparameters for the two added learners, F1-maximising thresholds), 2024-2025
test evaluated once with frozen thresholds. Reports PR-AUC, ROC-AUC, Brier
score, precision, recall, F1, specificity and flagged share, plus permutation
importance on the validation year and cross-method rank agreement.

**Step 5 — Decision layer (`decision_analysis.py`).** Smooths historical
airport-calendar-month rates toward the global month rate,
`(S + r·N̄/100,000)/(O + N̄)·100,000`, where `N̄` is mean annual operations (one
airport-year of prior exposure borrowed at the global rate). For 2024-2025 the
severity term is the mean random-forest probability over labelled reports in
each airport-month, falling back to the airport mean and then the global mean.
Frequency and severity percentiles are averaged with equal weight over the
480-cell test panel; the top 20% (96 cells) is compared with frequency alone.
The same score is cut into three tiers at 0.80 and 0.50 for the 20 × 12
calendar.

**Step 6 — Runway transfer test (`runway_transfer_test.py`).** Runway is never a
feature. Holdout predictions are joined to the runway recorded in the report,
aggregated to airport-runway cells with at least 20 reports, and scored with an
airport-centred Spearman correlation; at equal budget (one runway per airport)
the score is compared with simply taking the busiest runway.

All random algorithms use `random_state=42`.

---

## 5. Reference results

Core sample counts (printed by `verify_results.py`):

| Quantity | Value |
|---|---|
| Source rows in NWSD workbook | 360,384 |
| Selected airports | 20 |
| Complete airport-months | 2,640 |
| Total operations | 84,078,507 |
| Reported strikes | 51,032 |
| Labelled modelling events | 25,318 |
| Damaging reports | 1,377 (5.44%) |
| Test-period labelled reports (2024-2025) | 6,593 |

Held-out 2024-2025 performance (M? counted positive, thresholds frozen on 2023):

| Model | PR-AUC | Brier | Precision | Recall | Flagged |
|---|---|---|---|---|---|
| Logistic regression | 0.375 | 0.118 | 43.61% | 39.32% | 4.03% |
| Random forest | 0.358 | 0.087 | 32.65% | 48.47% | 6.64% |
| MLP neural network | 0.356 | 0.129 | 36.90% | 44.41% | 5.38% |
| Gradient boosting | 0.354 | 0.095 | 32.05% | 47.80% | 6.67% |

Seasonal prioritisation backtest (480 airport-months, 295 damaging reports,
top-20% budget of 96 cells):

| Ranking rule | Damaging reports captured | Capture |
|---|---|---|
| Smoothed historical rate only | 79 / 295 | 26.78% |
| Historical rate + predicted severity | 107 / 295 | 36.27% |

Three-tier calendar (20 airports × 12 calendar months): 7 Highest (score ≥ 0.80),
117 Elevated (0.50-0.80), 116 Routine (< 0.50).

Runway transfer test (72 airport-runway cells, 20 airports):

| Model | Within-airport Spearman ρ | Capture, score (1 runway/airport) | Capture, report volume |
|---|---|---|---|
| Logistic regression | 0.41 | 28.1% | 40.2% |
| Random forest | 0.44 | 29.3% | 40.2% |
| MLP neural network | 0.39 | 27.7% | 40.2% |
| Gradient boosting | 0.41 | 30.1% | 40.2% |

The score orders runways better than chance but loses to raw report volume at
equal budget; the report therefore keeps the framework at airport-month level.

---

## 6. Verification and tolerances

`verify_results.py` compares the regenerated CSV files with
`expected_results/` column by column. Tolerances differ by metric family,
because threshold-derived metrics are step functions of the fitted threshold:

| Family | Tolerance |
|---|---|
| PR-AUC, ROC-AUC, Brier, Spearman ρ | absolute difference ≤ 0.01 |
| Precision, recall, F1, validation F1 | ≤ 0.05 |
| Flagged share | ≤ 0.25 percentage points |
| Confusion-matrix cells | ≤ 10 reports |
| Row counts, airport-months, runway cells, damaging totals | exact |

Small differences can still appear across operating systems, BLAS builds or CPU
types. If a value falls just outside tolerance, re-run with the pinned
environment before assuming a coding error.

---

## 7. Runtime

On a 64-bit Windows machine with the pinned environment, the full pipeline took
**4 minutes 15 seconds** in our own run (Python 3.13.14 on 64-bit Windows). Most
of the time is spent reading the 159 MB workbook (step 1), fitting the random
forest and gradient boosting models, computing validation permutation importance
(step 4) and re-reading the workbook for the runway column (step 6, cached in
`data/processed/runway_index.pkl` after the first run). On a slower machine,
expect 10-20 minutes; the run needs about 4 GB of RAM.

To re-run only the last stage, `python src/runway_transfer_test.py` reuses
`outputs/model/test_predictions.csv` and the runway cache.

---

## 8. Scope and interpretation

* The model estimates **damage conditional on an already reported strike**. It
  does not estimate the probability that a flight strikes wildlife.
* Reported rates are **reporting** rates: reporting is voluntary and uneven, so
  airport differences partly reflect reporting practice.
* The 2024-2025 backtest is a **retrospective** ranking exercise. Its severity
  term uses same-month reports, so it is a within-month instrument; a
  month-start version would need severity estimated from historical months only
  and is not validated here.
* The prioritisation gains are not causal estimates of mitigation
  effectiveness, and the runway-level test is reported as a negative result
  rather than a recommendation.

---

## 9. Troubleshooting

* **`Missing optional dependency 'openpyxl'`** — install the pinned
  requirements; `audit_faa.py` and `runway_transfer_test.py` read `.xlsx`.
* **Memory error while reading the workbook** — close other applications, or
  read `Public.xlsx` once and reuse the cached pickle in `data/processed/`.
* **A metric sits just outside tolerance** — check the Python version and the
  package versions first; the reference values were produced with the pinned
  environment.
* **Figures look different** — the pipeline sets a fixed style and DPI, but
  matplotlib versions can shift font metrics slightly; the CSVs are the
  authoritative results.

---

