from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    _mpl_config_dir = os.path.join(os.getcwd(), ".mplconfig")
    os.makedirs(_mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_config_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import shap
except ImportError:
    shap = None

try:
    import streamlit as st
except ImportError:
    st = None


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

DATASET_NAME = "imnim/multiclass-email-classification"
POSITIVE_LABEL = "Job Application"
RANDOM_STATE = 42

KEYWORD_GROUPS = {
    "job": ["job", "position", "role", "career", "opportunity", "opening", "vacancy", "hiring", "apply", "application"],
    "candidate": ["candidate", "resume", "cv", "portfolio", "experience", "skills", "qualification", "background"],
    "recruiting": ["recruiter", "talent", "interview", "screening", "manager", "hr", "human resources"],
    "compensation": ["salary", "compensation", "benefits", "bonus", "equity", "$", "per year", "per hour"],
    "business": ["meeting", "agenda", "conference", "quarterly", "team", "deadline", "report"],
    "support": ["invoice", "booking", "support", "ticket", "delivery", "order", "account"],
}


@dataclass
class PipelineArtifacts:
    dataset: pd.DataFrame
    cleaned_dataset: pd.DataFrame
    feature_frame: pd.DataFrame
    selected_features: List[str]
    feature_weights: pd.DataFrame
    model_metrics: pd.DataFrame
    figures: Dict[str, plt.Figure]
    shap_figures: Dict[str, plt.Figure]
    preprocessing_summary: Dict[str, Any]


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _count_matches(text: str, patterns: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(token.lower()) for token in patterns)


def _contains_attachment_hint(text: str) -> int:
    return int(any(hint in text.lower() for hint in ("attached", "attachment", "resume enclosed", "cv attached")))


def _extract_numeric_features(subject: str, body: str) -> Dict[str, float]:
    subject = subject or ""
    body = body or ""
    combined = f"{subject} {body}".strip()
    lowered = combined.lower()
    words = re.findall(r"\b\w+\b", combined)
    unique_words = set(word.lower() for word in words)
    char_count = len(combined)
    alpha_chars = sum(1 for char in combined if char.isalpha())
    uppercase_chars = sum(1 for char in combined if char.isupper())
    digit_chars = sum(1 for char in combined if char.isdigit())
    punctuation_chars = sum(1 for char in combined if re.match(r"[^\w\s]", char))

    features = {
        "subject_length": float(len(subject)),
        "body_length": float(len(body)),
        "total_length": float(char_count),
        "word_count": float(len(words)),
        "avg_word_length": _safe_divide(sum(len(word) for word in words), len(words)),
        "unique_word_ratio": _safe_divide(len(unique_words), len(words)),
        "uppercase_ratio": _safe_divide(uppercase_chars, alpha_chars),
        "digit_ratio": _safe_divide(digit_chars, max(char_count, 1)),
        "punctuation_ratio": _safe_divide(punctuation_chars, max(char_count, 1)),
        "exclamation_count": float(combined.count("!")),
        "question_count": float(combined.count("?")),
        "colon_count": float(combined.count(":")),
        "url_count": float(len(re.findall(r"https?://|www\.", lowered))),
        "email_count": float(len(re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", combined))),
        "phone_count": float(len(re.findall(r"(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}", combined))),
        "currency_symbol_count": float(sum(combined.count(symbol) for symbol in ("$", "\u00A3", "\u20AC"))),
        "attachment_hint": float(_contains_attachment_hint(combined)),
        "dear_greeting": float(int(lowered.startswith("dear"))),
        "contains_date_like": float(
            int(
                bool(
                    re.search(
                        r"\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}/\d{1,2}/\d{2,4})\b",
                        lowered,
                    )
                )
            )
        ),
    }

    for prefix, keywords in KEYWORD_GROUPS.items():
        features[f"{prefix}_keyword_count"] = float(_count_matches(combined, keywords))

    features["job_signal_ratio"] = _safe_divide(
        features["job_keyword_count"]
        + features["candidate_keyword_count"]
        + features["recruiting_keyword_count"]
        + features["compensation_keyword_count"],
        max(features["word_count"], 1.0),
    )
    features["non_job_signal_ratio"] = _safe_divide(
        features["business_keyword_count"] + features["support_keyword_count"],
        max(features["word_count"], 1.0),
    )
    return features


def _normalize_labels(value: Any) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    return [str(value)]


def load_email_dataframe(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    if load_dataset is None:
        raise ImportError("The `datasets` package is required to load the Hugging Face dataset.") from DATASETS_IMPORT_ERROR

    raw = load_dataset(dataset_name, split="train")
    df = raw.to_pandas()
    df["subject"] = df["subject"].astype("string")
    df["body"] = df["body"].astype("string")
    df["labels"] = df["labels"].apply(_normalize_labels)
    df["is_job"] = df["labels"].apply(lambda labels: int(POSITIVE_LABEL in labels))
    return df

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    working = df.copy()
    summary: Dict[str, Any] = {
        "input_rows": int(len(working)),
        "missing_subject_before": int(working["subject"].isna().sum()),
        "missing_body_before": int(working["body"].isna().sum()),
        "missing_labels_before": int(working["labels"].isna().sum()),
    }

    working["subject"] = working["subject"].fillna("")
    working["body"] = working["body"].fillna("")
    working["labels"] = working["labels"].apply(_normalize_labels)
    working["text"] = (working["subject"].astype(str) + " " + working["body"].astype(str)).str.strip()

    duplicate_mask = working.duplicated(subset=["subject", "body"], keep="first")
    summary["duplicate_rows_removed"] = int(duplicate_mask.sum())
    working = working.loc[~duplicate_mask].copy()

    whitespace_mask = working["text"].str.len() == 0
    summary["empty_rows_removed"] = int(whitespace_mask.sum())
    working = working.loc[~whitespace_mask].copy()

    working.reset_index(drop=True, inplace=True)
    summary["rows_after_basic_cleaning"] = int(len(working))
    return working, summary


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = [_extract_numeric_features(subject, body) for subject, body in zip(df["subject"].astype(str), df["body"].astype(str))]
    features = pd.DataFrame(feature_rows, index=df.index)
    features["is_job"] = df["is_job"].astype(int).values
    return features


def filter_outliers(features: pd.DataFrame, target_column: str = "is_job") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    numeric_columns = [column for column in features.columns if column != target_column]
    target = features[target_column].astype(int)
    majority_class = int(target.mode().iloc[0])
    majority_mask = target == majority_class
    reference_frame = features.loc[majority_mask, numeric_columns]
    q1 = reference_frame.quantile(0.25)
    q3 = reference_frame.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)
    lower = q1 - (3.0 * iqr)
    upper = q3 + (3.0 * iqr)

    extreme_mask = features[numeric_columns].lt(lower) | features[numeric_columns].gt(upper)
    extreme_feature_count = extreme_mask.sum(axis=1)
    outlier_rows = (extreme_feature_count >= 2) & majority_mask

    removed_rows = features.loc[outlier_rows, target_column].astype(int)
    summary = {
        "outlier_rows_removed": int(outlier_rows.sum()),
        "majority_outlier_rows_removed": int((removed_rows == majority_class).sum()),
        "minority_rows_preserved": int((target != majority_class).sum()),
        "rows_after_outlier_filter": int((~outlier_rows).sum()),
    }
    filtered = features.loc[~outlier_rows].copy()
    return filtered, summary

def apply_correlation_filter(
    features: pd.DataFrame,
    target_column: str = "is_job",
    threshold: float = 0.9,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = features.drop(columns=[target_column])
    constant_columns = [column for column in predictors.columns if predictors[column].nunique(dropna=False) <= 1]
    if constant_columns:
        predictors = predictors.drop(columns=constant_columns)
    corr_matrix = predictors.corr().fillna(0.0)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    dropped = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]
    filtered_predictors = predictors.drop(columns=dropped)
    feature_target_corr = filtered_predictors.corrwith(features[target_column]).abs().sort_values(ascending=False)
    ranking = feature_target_corr.reset_index()
    ranking.columns = ["feature", "abs_target_correlation"]
    return filtered_predictors, corr_matrix, ranking

def compute_feature_weights(
    predictors: pd.DataFrame,
    target: pd.Series,
    alpha_lasso: float = 0.001,
    alpha_ridge: float = 1.0,
) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(predictors)
    scaled_frame = pd.DataFrame(scaled, columns=predictors.columns, index=predictors.index)

    linear_model = LinearRegression()
    lasso_model = Lasso(alpha=alpha_lasso, random_state=RANDOM_STATE, max_iter=10000)
    ridge_model = Ridge(alpha=alpha_ridge, random_state=RANDOM_STATE)

    linear_model.fit(scaled_frame, target)
    lasso_model.fit(scaled_frame, target)
    ridge_model.fit(scaled_frame, target)

    weights = pd.DataFrame(
        {
            "feature": predictors.columns,
            "linear_weight": np.abs(linear_model.coef_),
            "lasso_weight": np.abs(lasso_model.coef_),
            "ridge_weight": np.abs(ridge_model.coef_),
        }
    )
    weights["ensemble_weight"] = weights[["linear_weight", "lasso_weight", "ridge_weight"]].mean(axis=1)
    weights.sort_values("ensemble_weight", ascending=False, inplace=True)
    weights.reset_index(drop=True, inplace=True)
    return weights, scaler


def get_model_registry() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "LASSO": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "CART": DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return models


def _predict_scores(model: Any, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
        if probabilities.ndim == 1:
            return np.clip(probabilities, 0.0, 1.0)
        if probabilities.shape[1] == 1:
            classes = np.asarray(getattr(model, "classes_", [0]))
            only_class = int(classes[0]) if classes.size else 0
            fill_value = 1.0 if only_class == 1 else 0.0
            return np.full(len(x_test), fill_value, dtype=float)
        classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
        positive_index = classes.index(1) if 1 in classes else min(1, probabilities.shape[1] - 1)
        return probabilities[:, positive_index]
    raw_predictions = np.asarray(model.predict(x_test), dtype=float)
    return np.clip(raw_predictions, 0.0, 1.0)

def evaluate_models(
    predictors: pd.DataFrame,
    target: pd.Series,
    top_features: List[str],
    folds: int = 5,
) -> pd.DataFrame:
    x = predictors[top_features]
    y = target.astype(int)
    class_counts = y.value_counts()
    if y.nunique() < 2:
        raise ValueError("The derived target contains only one class after preprocessing. Check label normalization and filtering.")
    minority_class_count = int(class_counts.min())
    if minority_class_count < 2:
        raise ValueError("At least two samples are required in each class after preprocessing to run stratified cross-validation.")
    effective_folds = min(folds, minority_class_count)
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_STATE)
    model_registry = get_model_registry()
    rows: List[Dict[str, Any]] = []
    scale_models = {"Linear Regression", "LASSO", "Ridge"}

    for model_name, estimator in model_registry.items():
        fold_metrics: List[Dict[str, float]] = []
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            x_train = x.iloc[train_idx].copy()
            x_test = x.iloc[test_idx].copy()
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if model_name in scale_models:
                scaler = StandardScaler()
                x_train_values = scaler.fit_transform(x_train)
                x_test_values = scaler.transform(x_test)
                x_train_fold = pd.DataFrame(x_train_values, columns=x.columns, index=x_train.index)
                x_test_fold = pd.DataFrame(x_test_values, columns=x.columns, index=x_test.index)
            else:
                x_train_fold = x_train
                x_test_fold = x_test

            model = clone(estimator)
            model.fit(x_train_fold, y_train)
            scores = _predict_scores(model, x_test_fold)
            predictions = (scores >= 0.5).astype(int)

            fold_metrics.append(
                {
                    "fold": fold_index,
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions, zero_division=0),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1_score": f1_score(y_test, predictions, zero_division=0),
                    "auc_score": 0.5 if y_test.nunique() < 2 else roc_auc_score(y_test, scores),
                }
            )

        fold_frame = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "model": model_name,
                "accuracy": fold_frame["accuracy"].mean(),
                "precision": fold_frame["precision"].mean(),
                "recall": fold_frame["recall"].mean(),
                "f1_score": fold_frame["f1_score"].mean(),
                "auc_score": fold_frame["auc_score"].mean(),
                "accuracy_std": fold_frame["accuracy"].std(ddof=0),
                "precision_std": fold_frame["precision"].std(ddof=0),
                "recall_std": fold_frame["recall"].std(ddof=0),
                "f1_score_std": fold_frame["f1_score"].std(ddof=0),
                "auc_score_std": fold_frame["auc_score"].std(ddof=0),
            }
        )

    return pd.DataFrame(rows).sort_values(["f1_score", "auc_score"], ascending=False).reset_index(drop=True)

def fit_best_model(
    predictors: pd.DataFrame,
    target: pd.Series,
    model_metrics: pd.DataFrame,
    top_features: List[str],
) -> Tuple[str, Any]:
    best_model_name = model_metrics.iloc[0]["model"]
    estimator = clone(get_model_registry()[best_model_name])
    x = predictors[top_features].copy()

    if best_model_name in {"Linear Regression", "LASSO", "Ridge"}:
        scaler = StandardScaler()
        x_values = scaler.fit_transform(x)
        x_fit = pd.DataFrame(x_values, columns=x.columns, index=x.index)
        estimator.fit(x_fit, target)
        return best_model_name, (estimator, scaler)

    estimator.fit(x, target)
    return best_model_name, estimator


def _make_class_distribution_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts = df["is_job"].map({0: "Not Job", 1: "Job"}).value_counts().sort_index()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#d97706", "#2563eb"], ax=ax)
    ax.set_title("Distribution of Job vs. Non-Job Emails")
    ax.set_xlabel("Class")
    ax.set_ylabel("Email Count")
    return fig


def _make_feature_weight_figure(feature_weights: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = feature_weights.head(top_n).sort_values("ensemble_weight")
    sns.barplot(data=subset, x="ensemble_weight", y="feature", color="#0f766e", ax=ax)
    ax.set_title("Top Weighted Features")
    ax.set_xlabel("Average Absolute Weight")
    ax.set_ylabel("Feature")
    return fig


def _make_correlation_heatmap(correlation_matrix: pd.DataFrame, features: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    subset = correlation_matrix.loc[features, features]
    sns.heatmap(subset, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix of Selected Features")
    return fig


def _make_feature_distribution_figure(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    chosen = features[: min(3, len(features))]
    melted = df[chosen + ["is_job"]].melt(id_vars="is_job", var_name="feature", value_name="value")
    melted["class"] = melted["is_job"].map({0: "Not Job", 1: "Job"})
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="class", ax=ax)
    ax.set_title("Feature Distributions by Class")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    return fig


def _make_model_comparison_figure(model_metrics: pd.DataFrame) -> plt.Figure:
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    chart_frame = model_metrics[["model"] + metric_columns].melt(id_vars="model", var_name="metric", value_name="score")
    min_score = float(chart_frame["score"].min())
    max_score = float(chart_frame["score"].max())
    score_span = max(max_score - min_score, 0.0025)
    lower_bound = max(0.0, min_score - (score_span * 0.35))
    upper_bound = min(1.0, max_score + (score_span * 0.2))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.barplot(data=chart_frame, x="model", y="score", hue="metric", ax=ax, palette="viridis")
    ax.set_title("Five-Fold Cross-Validation Metric Comparison (Precision Zoom)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(lower_bound, upper_bound)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower left", ncol=2)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.6f}",
            (patch.get_x() + (patch.get_width() / 2.0), height),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            xytext=(0, 3),
            textcoords="offset points",
        )

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    return fig
def create_shap_figures(
    fitted_model: Any,
    predictors: pd.DataFrame,
    top_features: List[str],
    model_name: str,
) -> Dict[str, plt.Figure]:
    if shap is None:
        return {}

    model_object = fitted_model[0] if isinstance(fitted_model, tuple) else fitted_model
    if model_name in {"Linear Regression", "LASSO", "Ridge"}:
        scaler = fitted_model[1]
        sample = pd.DataFrame(scaler.transform(predictors[top_features]), columns=top_features)
    else:
        sample = predictors[top_features].copy()

    sample = sample.sample(n=min(len(sample), 300), random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(model_object) if model_name in {"Random Forest", "LightGBM", "CART"} else shap.LinearExplainer(model_object, sample)
        shap_values = explainer.shap_values(sample)
    except Exception:
        try:
            explainer = shap.Explainer(model_object, sample)
            explained = explainer(sample)
            shap_values = explained.values
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        except Exception:
            return {}

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, -1]

    figures: Dict[str, plt.Figure] = {}

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False, plot_type="bar")
    figures["shap_summary_bar"] = plt.gcf()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False)
    figures["shap_summary_beeswarm"] = plt.gcf()
    return figures


def generate_figures(
    filtered_features: pd.DataFrame,
    feature_weights: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    top_features: List[str],
    model_metrics: pd.DataFrame,
) -> Dict[str, plt.Figure]:
    return {
        "class_distribution": _make_class_distribution_figure(filtered_features),
        "feature_weights": _make_feature_weight_figure(feature_weights),
        "selected_feature_correlation": _make_correlation_heatmap(correlation_matrix, top_features),
        "feature_distribution": _make_feature_distribution_figure(filtered_features, top_features),
        "model_comparison": _make_model_comparison_figure(model_metrics),
    }


def persist_artifacts(artifacts: PipelineArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    artifacts.feature_weights.to_csv(output_dir / "feature_weights.csv", index=False)
    artifacts.cleaned_dataset.to_csv(output_dir / "cleaned_dataset.csv", index=False)
    artifacts.feature_frame.to_csv(output_dir / "engineered_features.csv", index=False)

    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.preprocessing_summary, handle, indent=2)

    for name, figure in artifacts.figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    for name, figure in artifacts.shap_figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")


def run_pipeline(
    dataset_name: str = DATASET_NAME,
    top_k_features: int = 8,
    persist_dir: Optional[Path] = None,
) -> PipelineArtifacts:
    raw_df = load_email_dataframe(dataset_name=dataset_name)
    cleaned_df, cleaning_summary = preprocess_dataframe(raw_df)

    full_feature_frame = build_feature_frame(cleaned_df)
    filtered_feature_frame, outlier_summary = filter_outliers(full_feature_frame)
    filtered_df = cleaned_df.loc[filtered_feature_frame.index].reset_index(drop=True)
    filtered_feature_frame = filtered_feature_frame.reset_index(drop=True)

    predictors, correlation_matrix, correlation_ranking = apply_correlation_filter(filtered_feature_frame)
    target = filtered_feature_frame["is_job"].astype(int)

    feature_weights, _ = compute_feature_weights(predictors, target)
    feature_weights = feature_weights.merge(correlation_ranking, on="feature", how="left")
    selected_features = feature_weights["feature"].head(min(top_k_features, len(feature_weights))).tolist()

    model_metrics = evaluate_models(predictors, target, selected_features)
    best_model_name, fitted_model = fit_best_model(predictors, target, model_metrics, selected_features)
    figures = generate_figures(filtered_feature_frame, feature_weights, correlation_matrix, selected_features, model_metrics)
    shap_figures = create_shap_figures(fitted_model, predictors, selected_features, best_model_name)

    preprocessing_summary = {
        **cleaning_summary,
        **outlier_summary,
        "positive_class_label": POSITIVE_LABEL,
        "positive_class_count": int(filtered_feature_frame["is_job"].sum()),
        "negative_class_count": int((1 - filtered_feature_frame["is_job"]).sum()),
        "selected_features": selected_features,
        "highest_weight_feature": feature_weights.iloc[0]["feature"],
        "best_model": best_model_name,
    }

    artifacts = PipelineArtifacts(
        dataset=raw_df,
        cleaned_dataset=filtered_df,
        feature_frame=filtered_feature_frame,
        selected_features=selected_features,
        feature_weights=feature_weights,
        model_metrics=model_metrics,
        figures=figures,
        shap_figures=shap_figures,
        preprocessing_summary=preprocessing_summary,
    )

    if persist_dir is not None:
        persist_artifacts(artifacts, persist_dir)
    return artifacts


def _format_metric(value: float) -> str:
    return f"{value:.8f}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.6f}%"


def _build_precise_model_table(model_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    best_values = model_metrics[metric_columns].max()
    table = model_metrics.copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    for column in metric_columns:
        table[f"{column}_gap_from_best"] = best_values[column] - table[column]
    ordered_columns = [
        "rank",
        "model",
        "accuracy",
        "accuracy_gap_from_best",
        "accuracy_std",
        "precision",
        "precision_gap_from_best",
        "precision_std",
        "recall",
        "recall_gap_from_best",
        "recall_std",
        "f1_score",
        "f1_score_gap_from_best",
        "f1_score_std",
        "auc_score",
        "auc_score_gap_from_best",
        "auc_score_std",
    ]
    table = table[ordered_columns]
    for column in [name for name in table.columns if name not in {"rank", "model"}]:
        table[column] = table[column].map(_format_metric)
    return table


def _inject_streamlit_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top right, #fde68a 0%, #fff7ed 32%, #ecfeff 68%, #eff6ff 100%);
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.25rem;
        }
        .hero-shell {
            background: linear-gradient(135deg, rgba(15,118,110,0.96), rgba(37,99,235,0.92));
            border-radius: 24px;
            padding: 1.3rem 1.5rem;
            color: white;
            box-shadow: 0 18px 40px rgba(30, 41, 59, 0.16);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
        }
        .hero-copy {
            font-size: 0.98rem;
            line-height: 1.5;
            opacity: 0.94;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.06);
        }
        .stat-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #475569;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            font-size: 1.4rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
        }
        .stat-subtext {
            margin-top: 0.25rem;
            font-size: 0.8rem;
            color: #64748b;
        }
        .section-chip {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #0f766e;
            background: rgba(20, 184, 166, 0.12);
            margin-bottom: 0.5rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_stat_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_chip(label: str) -> None:
    st.markdown(f"<div class=\"section-chip\">{label}</div>", unsafe_allow_html=True)

# FIX 1: Moved @st.cache_resource to module level so Streamlit can key the
# cache on a stable function object. The original nested definition created a
# new function object on every call, causing the cache to miss on every rerun.
if st is not None:
    @st.cache_resource(show_spinner=False)
    def _streamlit_pipeline_cache() -> PipelineArtifacts:
        return run_pipeline(persist_dir=Path("artifacts"))
else:
    def _streamlit_pipeline_cache() -> PipelineArtifacts:  # type: ignore[misc]
        raise ImportError("streamlit is not installed.")


def streamlit_main() -> None:
    if st is None:
        raise ImportError("streamlit is required to launch the dashboard.")

    st.set_page_config(
        page_title="Email Job Classification Pipeline",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_streamlit_theme()

    with st.spinner("Running preprocessing, feature engineering, and model evaluation..."):
        artifacts = _streamlit_pipeline_cache()

    summary = artifacts.preprocessing_summary
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    all_models = artifacts.model_metrics["model"].tolist()
    default_models = all_models[: min(4, len(all_models))]

    with st.sidebar:
        st.markdown("### Run Snapshot")
        st.caption("Exact preprocessing numbers for quick auditability.")
        st.metric("Rows After Cleaning", f"{summary['rows_after_basic_cleaning']:,}")
        st.metric("Rows After Filter", f"{summary['rows_after_outlier_filter']:,}")
        st.metric("Positive Class", f"{summary['positive_class_count']:,}")
        st.metric("Negative Class", f"{summary['negative_class_count']:,}")
        st.markdown("### Model Playground")
        selected_models = st.multiselect(
            "Choose models to compare",
            options=all_models,
            default=default_models,
        )
        if not selected_models:
            selected_models = all_models
        ranking_metric = st.selectbox(
            "Primary ranking metric",
            options=metric_columns,
            index=metric_columns.index("f1_score"),
        )
        st.markdown("### Selected Features")
        for feature_name in summary["selected_features"]:
            st.write(f"- `{feature_name}`")

    filtered_metrics = artifacts.model_metrics.loc[
        artifacts.model_metrics["model"].isin(selected_models)
    ].copy()
    filtered_metrics.sort_values([ranking_metric, "auc_score"], ascending=False, inplace=True)
    filtered_metrics.reset_index(drop=True, inplace=True)
    precise_model_table = _build_precise_model_table(filtered_metrics)
    best_row = filtered_metrics.iloc[0]
    positive_ratio = summary["positive_class_count"] / max(summary["rows_after_outlier_filter"], 1)
    scorecard = best_row[metric_columns].to_frame(name="value")
    scorecard["exact"] = scorecard["value"].map(_format_metric)
    scorecard["percent"] = scorecard["value"].map(_format_percent)
    model_metrics_csv = filtered_metrics.to_csv(index=False).encode("utf-8")

    feature_export = artifacts.feature_weights.copy()
    feature_export_csv = feature_export.to_csv(index=False).encode("utf-8")
    cleaned_export = artifacts.cleaned_dataset.copy()
    cleaned_export["labels"] = cleaned_export["labels"].apply(lambda value: ", ".join(value))
    cleaned_export_csv = cleaned_export.to_csv(index=False).encode("utf-8")

    filtered_model_figure = _make_model_comparison_figure(filtered_metrics)
    plot_catalog: Dict[str, plt.Figure] = {
        "Class Distribution Dashboard": artifacts.figures["class_distribution"],
        "Weighted Feature Importance Dashboard": artifacts.figures["feature_weights"],
        "Selected Feature Correlation Dashboard": artifacts.figures["selected_feature_correlation"],
        "Feature Distribution by Class Dashboard": artifacts.figures["feature_distribution"],
        "Precise Model Metric Comparison Dashboard": filtered_model_figure,
    }
    for shap_name, shap_figure in artifacts.shap_figures.items():
        pretty_name = shap_name.replace("_", " ").title()
        plot_catalog[f"{pretty_name} SHAP Plot"] = shap_figure
    plot_options = list(plot_catalog.keys())

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-title">Email Classifier Control Room</div>
            <div class="hero-copy">
                A brighter view of the full pipeline: cleaning, feature selection, cross-validation, exact score inspection, and explanation plots.
                Built on <strong>{DATASET_NAME}</strong> with <strong>{summary['rows_after_outlier_filter']}</strong> retained rows and
                a preserved positive class rate of <strong>{_format_percent(positive_ratio)}</strong>. The live comparison table is filtered to
                <strong>{len(selected_models)}</strong> model(s) and ranked by <strong>{ranking_metric}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_row = st.columns(5)
    with stat_row[0]:
        _render_stat_card("Best Model", str(best_row["model"]), f"Leader on {ranking_metric}")
    with stat_row[1]:
        _render_stat_card("Best F1", _format_metric(float(best_row["f1_score"])), _format_percent(float(best_row["f1_score"])))
    with stat_row[2]:
        _render_stat_card("Best AUC", _format_metric(float(best_row["auc_score"])), _format_percent(float(best_row["auc_score"])))
    with stat_row[3]:
        _render_stat_card("Lead Feature", str(summary["highest_weight_feature"]), "Highest ensemble coefficient weight")
    with stat_row[4]:
        _render_stat_card("Positive Rate", _format_percent(positive_ratio), f"{summary['positive_class_count']} job emails after preprocessing")

    model_tab, data_tab = st.tabs(["Model", "Data"])

    with model_tab:
        _render_section_chip("Hyper-Precise Comparison")
        controls_left, controls_right = st.columns([1.35, 1.0])
        with controls_left:
            st.markdown("#### Full model leaderboard")
            st.dataframe(precise_model_table, use_container_width=True, hide_index=True)
        with controls_right:
            st.markdown("#### Export current comparison")
            st.download_button(
                label="Download filtered metrics CSV",
                data=model_metrics_csv,
                file_name="filtered_model_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("The export respects the active model filter and ranking control from the sidebar.")
            st.markdown("#### Best-model scorecard")
            st.dataframe(scorecard[["exact", "percent"]], use_container_width=True)
        st.markdown("#### Metric spotlight")
        spotlight_columns = st.columns(5)
        for idx, metric_name in enumerate(metric_columns):
            top_model = filtered_metrics.iloc[filtered_metrics[metric_name].idxmax()]
            with spotlight_columns[idx]:
                st.metric(
                    metric_name.replace("_", " ").title(),
                    _format_metric(float(top_model[metric_name])),
                    delta=f"Leader: {top_model['model']}",
                )
        st.caption("Each `*_gap_from_best` field in the table is the exact difference from the best score for that metric.")
        st.markdown("#### Visual metric comparison")
        st.pyplot(filtered_model_figure, clear_figure=False)
        st.caption("The y-axis is zoomed to the observed score range and every bar is labeled to six decimal places for tighter visual comparison.")

    with data_tab:
        _render_section_chip("Searchable Visual Explorer")
        st.markdown("#### Jump directly to a dashboard or SHAP view")
        selected_plot_name = st.selectbox(
            "Search and choose a plot",
            options=plot_options,
            index=0,
            help="Type in the dropdown to quickly find a specific dashboard or SHAP plot.",
        )
        st.pyplot(plot_catalog[selected_plot_name], clear_figure=False)

        details_left, details_right = st.columns([1.05, 1.25])
        with details_left:
            st.markdown("#### Weighted feature ranking")
            feature_table = artifacts.feature_weights.copy()
            numeric_feature_columns = [column for column in feature_table.columns if column != "feature"]
            for column in numeric_feature_columns:
                feature_table[column] = feature_table[column].map(_format_metric)
            st.dataframe(feature_table, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download feature weights CSV",
                data=feature_export_csv,
                file_name="feature_weights.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with details_right:
            st.markdown("#### Cleaned email preview")
            preview = cleaned_export[["subject", "body", "labels", "is_job"]].copy().head(20)
            preview["is_job"] = preview["is_job"].map({0: "Not Job", 1: "Job"})
            st.dataframe(preview, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download cleaned dataset CSV",
                data=cleaned_export_csv,
                file_name="cleaned_email_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("#### Preprocessing JSON")
        st.json(summary)
def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Email job classification pipeline")
    parser.add_argument("--top-k-features", type=int, default=8, help="Number of top weighted features used for training.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory where metrics and plots are saved.")
    # FIX 2: parse_known_args instead of parse_args so that Streamlit's own
    # injected CLI flags (e.g. --server.port, --global.developmentMode) do not
    # cause argparse to raise an "unrecognized arguments" error and crash.
    args, _ = parser.parse_known_args()

    artifacts = run_pipeline(top_k_features=args.top_k_features, persist_dir=args.output_dir)
    best_row = artifacts.model_metrics.iloc[0]

    print("Pipeline complete")
    print(f"Rows after cleaning: {artifacts.preprocessing_summary['rows_after_basic_cleaning']}")
    print(f"Rows after outlier filter: {artifacts.preprocessing_summary['rows_after_outlier_filter']}")
    print(f"Highest weight feature: {artifacts.preprocessing_summary['highest_weight_feature']}")
    print(
        "Best model: "
        f"{best_row['model']} | "
        f"accuracy={best_row['accuracy']:.4f}, "
        f"precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, "
        f"f1={best_row['f1_score']:.4f}, "
        f"auc={best_row['auc_score']:.4f}"
    )
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


# FIX 3: Call streamlit_main() at module level when running under Streamlit
# (i.e. when the script is executed by `streamlit run`). The old code only
# called streamlit_main() if --streamlit was explicitly passed via argparse,
# which never happens when launched through the Streamlit server.
if st is not None and "__streamlit__" not in dir(st):
    # Detect Streamlit execution context: st.runtime.exists() is the canonical
    # way to check whether the script is being served by a Streamlit server.
    try:
        _running_in_streamlit = st.runtime.exists()
    except AttributeError:
        _running_in_streamlit = hasattr(st, "_is_running_with_streamlit")

    if _running_in_streamlit:
        streamlit_main()
    elif __name__ == "__main__":
        cli_main()
else:
    if __name__ == "__main__":
        cli_main()





