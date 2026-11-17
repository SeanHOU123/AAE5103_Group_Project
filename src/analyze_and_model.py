from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
AUDIT = ROOT / "outputs/audit"
OUT = ROOT / "outputs/model"
FIG = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CATEGORICAL = [
    "AIRPORT_ID", "TIME_OF_DAY", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "SKY", "PRECIPITATION", "SPECIES_ID", "SIZE", "NUM_SEEN", "NUM_STRUCK", "WARNED",
]
NUMERIC = ["AC_MASS", "NUM_ENGS", "HEIGHT", "SPEED", "month_sin", "month_cos"]
FEATURES = CATEGORICAL + NUMERIC
SPLIT_YEARS = {"train": (2015, 2022), "validation": (2023, 2023), "test": (2024, 2025)}
MODELS = ["Logistic regression", "Random forest", "MLP neural network", "Gradient boosting"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 220})


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def descriptive_analysis(panel: pd.DataFrame, events: pd.DataFrame) -> None:
    monthly = panel.groupby("month_number", as_index=False).agg(
        operations=("total_operations", "sum"),
        reported_strikes=("reported_strikes", "sum"),
        damaging_reports=("damaging_reports", "sum"),
    )
    monthly["reported_strikes_per_100k_operations"] = monthly["reported_strikes"] / monthly["operations"] * 100000
    monthly.to_csv(OUT / "seasonal_monthly_summary.csv", index=False)

    yearly = panel.groupby("year", as_index=False).agg(
        operations=("total_operations", "sum"),
        reported_strikes=("reported_strikes", "sum"),
        damaging_reports=("damaging_reports", "sum"),
    )
    yearly["reported_strikes_per_100k_operations"] = yearly["reported_strikes"] / yearly["operations"] * 100000
    yearly.to_csv(OUT / "yearly_summary.csv", index=False)

    airports = panel.groupby("AIRPORT_ID", as_index=False).agg(
        operations=("total_operations", "sum"),
        reported_strikes=("reported_strikes", "sum"),
        damaging_reports=("damaging_reports", "sum"),
    )
    airports["reported_strikes_per_100k_operations"] = airports["reported_strikes"] / airports["operations"] * 100000
    airports.sort_values("reported_strikes_per_100k_operations", ascending=False).to_csv(OUT / "airport_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    sns.lineplot(data=monthly, x="month_number", y="reported_strikes_per_100k_operations", marker="o", linewidth=2.2, ax=ax)
    ax.set(xticks=range(1, 13), xlabel="Calendar month", ylabel="Reported strikes per 100,000 operations", title="Seasonality of reported wildlife strikes, 2015–2025")
    save_figure("fig01_monthly_reported_strike_rate.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    sns.lineplot(data=yearly, x="year", y="reported_strikes_per_100k_operations", marker="o", linewidth=2.2, ax=ax)
    ax.set(xticks=yearly["year"], xlabel="Year", ylabel="Reported strikes per 100,000 operations", title="Annual reported strike rate across 20 airports")
    ax.tick_params(axis="x", rotation=45)
    save_figure("fig02_annual_reported_strike_rate.png")

    plot_airports = airports.sort_values("reported_strikes_per_100k_operations", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    sns.barplot(data=plot_airports, x="reported_strikes_per_100k_operations", y="AIRPORT_ID", color="#2878B5", ax=ax)
    ax.set(xlabel="Reported strikes per 100,000 operations", ylabel="Airport", title="Airport-level reported strike rates, 2015–2025")
    save_figure("fig03_airport_reported_strike_rates.png")

    phase = events.groupby("PHASE_OF_FLIGHT", dropna=False).agg(
        labeled_reports=("damaging", "size"), damaging_reports=("damaging", "sum")
    ).reset_index()
    phase["PHASE_OF_FLIGHT"] = phase["PHASE_OF_FLIGHT"].fillna("Missing")
    phase["damaging_pct"] = phase["damaging_reports"] / phase["labeled_reports"] * 100
    phase.sort_values("labeled_reports", ascending=False).to_csv(OUT / "phase_damage_summary.csv", index=False)


def make_preprocessor(dense: bool = False) -> ColumnTransformer:
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical_pipe, CATEGORICAL),
        ("numeric", numeric_pipe, NUMERIC),
    ], sparse_threshold=0.0 if dense else 0.3)


def make_model(name: str, params: dict | None = None) -> Pipeline:
    params = dict(params or {})
    dense = name in {"MLP neural network", "Gradient boosting"}
    if name == "Logistic regression":
        classifier = LogisticRegression(class_weight="balanced", C=0.5, max_iter=1500, solver="lbfgs", random_state=42)
    elif name == "Random forest":
        classifier = RandomForestClassifier(
            n_estimators=450, max_depth=16, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
    elif name == "MLP neural network":
        # Dense fully connected network. Class imbalance is handled through
        # balanced sample weights supplied at fit time, mirroring the
        # class_weight setting used by the linear and forest baselines.
        classifier = MLPClassifier(
            activation="relu", solver="adam", batch_size=256, max_iter=300,
            early_stopping=True, n_iter_no_change=15, validation_fraction=0.12,
            shuffle=True, random_state=42,
            **{"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 5e-4, **params},
        )
    elif name == "Gradient boosting":
        # Histogram-based gradient-boosted trees (LightGBM-style algorithm
        # available in scikit-learn), a modern non-linear tabular benchmark.
        classifier = HistGradientBoostingClassifier(
            l2_regularization=1.0, class_weight="balanced", early_stopping=False,
            random_state=42,
            **{"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 40, **params},
        )
    else:
        raise ValueError(name)
    return Pipeline([("preprocessor", make_preprocessor(dense=dense)), ("classifier", classifier)])



def balanced_sample_weight(target: pd.Series) -> np.ndarray:
    """'balanced' class weights expressed as per-row sample weights."""
    counts = target.value_counts()
    weights = target.map(lambda value: len(target) / (2 * counts[value]))
    return weights.to_numpy(dtype=float)


# Small, pre-declared grids for the two additional models. Hyperparameters are
# selected on the 2023 validation year only; the 2024-2025 test period is never
# used for selection, and the two original baselines keep their frozen settings
# so that the previously reported numbers stay comparable.
SEARCH_GRIDS = {
    "MLP neural network": [
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 5e-4},
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 5e-4},
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (32,), "alpha": 1e-4, "learning_rate_init": 1e-3},
    ],
    "Gradient boosting": [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "max_iter": 400},
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40, "max_iter": 300},
        {"learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 20, "max_iter": 250},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 40, "max_iter": 300},
    ],
}


def select_hyperparameters(name: str, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, list[dict]]:
    records, best_params, best_score = [], None, -np.inf
    for index, params in enumerate(SEARCH_GRIDS[name], start=1):
        model = make_model(name, params)
        fit_kwargs = {}
        if name == "MLP neural network":
            fit_kwargs["classifier__sample_weight"] = balanced_sample_weight(train["damaging"])
        model.fit(train[FEATURES], train["damaging"], **fit_kwargs)
        score = float(average_precision_score(validation["damaging"], model.predict_proba(validation[FEATURES])[:, 1]))
        print(f"  [{name}] config {index}: validation PR-AUC={score:.4f} {params}", flush=True)
        records.append({"model": name, "config": index, "params": json.dumps(params), "validation_pr_auc": score})
        if score > best_score:
            best_params, best_score = params, score
    for record in records:
        record["selected"] = record["params"] == json.dumps(best_params)
    return best_params, records



def best_validation_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def evaluate(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    prevalence = float(y_true.mean())
    return {
        "pr_auc": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
        "brier_score": brier_score_loss(y_true, probability),
        "threshold": threshold,
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "flagged_pct": float(predicted.mean() * 100),
        "prevalence_pct": prevalence * 100,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def main() -> None:
    panel = pd.read_csv(DATA / "airport_month_panel.csv", parse_dates=["month"])
    airports = pd.read_csv(AUDIT / "selected_airports.csv")["AIRPORT_ID"]
    events = pd.read_pickle(DATA / "faa_2015_2025_audit.pkl")
    events = events[events["AIRPORT_ID"].isin(airports)].copy()
    descriptive_analysis(panel, events[events["damage_known"]])

    model_events = events[events["damage_known"]].copy()
    model_events["month_sin"] = np.sin(2 * np.pi * model_events["INCIDENT_MONTH"] / 12)
    model_events["month_cos"] = np.cos(2 * np.pi * model_events["INCIDENT_MONTH"] / 12)
    for col in CATEGORICAL:
        model_events[col] = model_events[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    model_events["damaging"] = model_events["damaging"].astype(int)
    model_events.to_csv(DATA / "modeling_events.csv", index=False)

    masks = {
        split: model_events["INCIDENT_YEAR"].between(start, end)
        for split, (start, end) in SPLIT_YEARS.items()
    }
    train = model_events[masks["train"]]
    validation = model_events[masks["validation"]]
    test = model_events[masks["test"]]

    split_summary = []
    for split, frame in [("train", train), ("validation", validation), ("test", test)]:
        split_summary.append({
            "split": split, "years": str(SPLIT_YEARS[split]), "rows": len(frame),
            "damaging": int(frame["damaging"].sum()), "damaging_pct": frame["damaging"].mean() * 100,
        })
    pd.DataFrame(split_summary).to_csv(OUT / "split_summary.csv", index=False)

    all_metrics = []
    test_prediction_frames = []
    fitted_models = {}

    tuning_records = []
    tuned_params = {}
    for name in SEARCH_GRIDS:
        print(f"Selecting hyperparameters for {name} on 2023 validation data", flush=True)
        params, records = select_hyperparameters(name, train, validation)
        tuned_params[name] = params
        tuning_records.extend(records)
    pd.DataFrame(tuning_records).to_csv(OUT / "hyperparameter_search.csv", index=False)

    for sensitivity in [False, True]:
        train_fit = train[~train["damage_undetermined"]] if sensitivity else train
        val_fit = validation[~validation["damage_undetermined"]] if sensitivity else validation
        test_fit = test[~test["damage_undetermined"]] if sensitivity else test
        for name in MODELS:
            print(f"Fitting {name}; exclude_M?={sensitivity}", flush=True)
            model = make_model(name, tuned_params.get(name))
            fit_kwargs = {}
            if name == "MLP neural network":
                fit_kwargs["classifier__sample_weight"] = balanced_sample_weight(train_fit["damaging"])
            model.fit(train_fit[FEATURES], train_fit["damaging"], **fit_kwargs)
            val_probability = model.predict_proba(val_fit[FEATURES])[:, 1]
            threshold, val_f1 = best_validation_threshold(val_fit["damaging"], val_probability)
            test_probability = model.predict_proba(test_fit[FEATURES])[:, 1]
            row = {
                "model": name, "sensitivity_exclude_M?": sensitivity,
                "train_rows": len(train_fit), "validation_rows": len(val_fit), "test_rows": len(test_fit),
                "validation_best_f1": val_f1, **evaluate(test_fit["damaging"], test_probability, threshold),
            }
            all_metrics.append(row)
            predictions = test_fit[["INDEX_NR", "INCIDENT_YEAR", "INCIDENT_MONTH", "AIRPORT_ID", "PHASE_OF_FLIGHT", "DAMAGE_LEVEL", "damaging"]].copy()
            predictions["model"] = name
            predictions["sensitivity_exclude_M?"] = sensitivity
            predictions["damage_probability"] = test_probability
            predictions["predicted_high_risk"] = (test_probability >= threshold).astype(int)
            test_prediction_frames.append(predictions)
            if not sensitivity:
                fitted_models[name] = model

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    pd.concat(test_prediction_frames, ignore_index=True).to_csv(OUT / "test_predictions.csv", index=False)

    importance_frames = []
    importance_sample = validation.sample(n=min(2200, len(validation)), random_state=42)
    for name, model in fitted_models.items():
        print(f"Computing permutation importance: {name}", flush=True)
        result = permutation_importance(
            model, importance_sample[FEATURES], importance_sample["damaging"],
            scoring="average_precision", n_repeats=5, random_state=42, n_jobs=-1,
        )
        importance_frames.append(pd.DataFrame({
            "model": name, "feature": FEATURES,
            "importance_mean": result.importances_mean, "importance_std": result.importances_std,
        }))
    importance = pd.concat(importance_frames, ignore_index=True).sort_values(["model", "importance_mean"], ascending=[True, False])
    importance.to_csv(OUT / "permutation_importance.csv", index=False)

    # Agreement of the learned risk-factor ranking across the four methods:
    # a rank correlation close to 1 means the methods explain damage in the
    # same way even when their discrimination differs.
    wide = importance.pivot(index="feature", columns="model", values="importance_mean")[MODELS]
    agreement = wide.corr(method="spearman").round(4)
    agreement.to_csv(OUT / "importance_agreement.csv")
    print("\nSpearman agreement of permutation importance across models:\n", agreement.to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    prediction_frame = pd.concat(test_prediction_frames, ignore_index=True)
    for name in MODELS:
        subset = metrics[(metrics["model"] == name) & (~metrics["sensitivity_exclude_M?"])].iloc[0]
        frame = prediction_frame[(prediction_frame["model"] == name) & (~prediction_frame["sensitivity_exclude_M?"])]
        precision, recall, _ = precision_recall_curve(frame["damaging"], frame["damage_probability"])
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={subset['pr_auc']:.3f})")
    baseline = test["damaging"].mean()
    ax.axhline(baseline, color="grey", linestyle="--", label=f"Prevalence={baseline:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Damage prediction on held-out 2024–2025 data", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save_figure("fig04_precision_recall_curves.png")

    primary = metrics[~metrics["sensitivity_exclude_M?"]].set_index("model").loc[MODELS]
    comparison = primary[["pr_auc", "roc_auc", "recall", "precision", "f1", "brier_score"]].reset_index()
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    plot_frame = comparison.melt(id_vars="model", value_vars=["pr_auc", "recall", "precision", "f1"], var_name="metric", value_name="score")
    metric_labels = {"pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision", "f1": "F1"}
    plot_frame["metric"] = plot_frame["metric"].map(metric_labels)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    palette = ["#2878B5", "#C82423", "#F8AC8C", "#4DAF4A"]
    sns.barplot(data=plot_frame, x="metric", y="score", hue="model", palette=palette, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2, rotation=90)
    ax.set(xlabel="Evaluation metric on 2024–2025 test set", ylabel="Score", ylim=(0, 0.75),
           title="Four-method comparison on the held-out test period")
    ax.legend(title="Model", fontsize=8, title_fontsize=9, loc="upper right")
    save_figure("fig04b_model_comparison.png")

    top = importance[importance["model"] == "Random forest"].nlargest(12, "importance_mean").sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#C82423", alpha=0.85)
    ax.set(xlabel="Decrease in average precision after permutation", ylabel="Original input feature", title="Random forest permutation importance (validation data)")
    save_figure("fig05_random_forest_permutation_importance.png")

    run_summary = {
        "selected_airports": int(len(airports)),
        "airport_month_rows": int(len(panel)),
        "all_selected_airport_reports": int(len(events)),
        "labeled_model_events": int(len(model_events)),
        "positive_events": int(model_events["damaging"].sum()),
        "split_years": SPLIT_YEARS,
        "primary_models": MODELS,
        "hyperparameter_selection": {
            "tuned_models": list(SEARCH_GRIDS),
            "selection_data": "2023 validation year, highest validation PR-AUC",
            "fixed_models": ["Logistic regression", "Random forest"],
            "test_set_used_for_selection": False,
            "selected_params": {name: params for name, params in tuned_params.items()},
        },
        "sensitivity_analysis": "Repeat both models after excluding M? records",
        "threshold_policy": "Independently maximize F1 on 2023 validation data; freeze for 2024-2025 test",
    }
    with (OUT / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    print("\nMetrics:\n", metrics.to_string(index=False), flush=True)
    print("\nTop random forest importance:\n", importance[importance["model"] == "Random forest"].head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
