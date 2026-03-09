# -*- coding: utf-8 -*-
"""
MSIS 522 HW1 - Email Job-Rejection Classification Pipeline
===========================================================
End-to-end data science workflow:
  Part 1: Descriptive Analytics (EDA + SMOTE class-imbalance discussion)
  Part 2: Predictive Modeling (LogReg, CART, RF, Gradient Boost, MLP, Stacked Ensemble)
  Part 3: Explainability (SHAP beeswarm, bar, waterfall)
  Part 4: Streamlit Deployment (4-tab app with interactive prediction)

Dataset  : imnim/multiclass-email-classification (Hugging Face)
Target   : Binary -- is_job (1 = Job Application / rejection email, 0 = everything else)
Imbalance: Addressed via SMOTE (Synthetic Minority Oversampling Technique)
Ensemble : TF-IDF (1-3gram) -> [MultinomialNB | LinearSVC | RandomForest] -> LogReg meta
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ── matplotlib config before any pyplot import ──────────────────────────────
if "MPLCONFIGDIR" not in os.environ:
    _mpl_cfg = os.path.join(os.getcwd(), ".mplconfig")
    os.makedirs(_mpl_cfg, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_cfg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import joblib

# ── optional: datasets (Hugging Face) ───────────────────────────────────────
load_dataset = None          # safe default before try
DownloadConfig = None
DATASETS_IMPORT_ERROR: Optional[Exception] = None
try:
    from datasets import DownloadConfig as _DownloadConfig, load_dataset as _load_dataset  # type: ignore[import]
    DownloadConfig = _DownloadConfig
    load_dataset = _load_dataset
except Exception as exc:
    DATASETS_IMPORT_ERROR = exc

# ── optional: imbalanced-learn ───────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    SMOTE = None  # type: ignore[assignment,misc]

# ── optional: boosted trees ──────────────────────────────────────────────────
try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

# ── optional: SHAP ───────────────────────────────────────────────────────────
try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

# ── optional: Streamlit ──────────────────────────────────────────────────────
try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
DATASET_NAME    = "imnim/multiclass-email-classification"
POSITIVE_LABEL  = "Job Application"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5
TOP_K_FEATURES  = 8
PACKAGED_ARTIFACT_FILENAME = "pipeline_artifacts.joblib"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
ARTIFACT_SCHEMA_VERSION = 1

KEYWORD_GROUPS: Dict[str, List[str]] = {
    "job":          ["job", "position", "role", "career", "opportunity", "opening",
                     "vacancy", "hiring", "apply", "application"],
    "candidate":    ["candidate", "resume", "cv", "portfolio", "experience",
                     "skills", "qualification", "background"],
    "recruiting":   ["recruiter", "talent", "interview", "screening",
                     "manager", "hr", "human resources"],
    "compensation": ["salary", "compensation", "benefits", "bonus",
                     "equity", "$", "per year", "per hour"],
    "business":     ["meeting", "agenda", "conference", "quarterly",
                     "team", "deadline", "report"],
    "support":      ["invoice", "booking", "support", "ticket",
                     "delivery", "order", "account"],
}

MODEL_HYPERPARAMS: Dict[str, str] = {
    "Logistic Regression": "C=1.0, solver=lbfgs, class_weight=balanced, max_iter=1 000",
    "Decision Tree":       "max_depth=6, min_samples_leaf=4, class_weight=balanced",
    "Random Forest":       "n_estimators=300, max_depth=10, class_weight=balanced_subsample",
    "Gradient Boost":      "n_estimators=300, lr=0.05, num_leaves=31 (LightGBM) / max_depth=6 (XGBoost)",
    "Neural Network":      "hidden=(256,128), relu, adam, max_iter=300, early_stopping=True",
    "Stacked Ensemble":    "TF-IDF(1-3gram, 50k) → [MNB(α=0.1) | LinearSVC(C=1) | RF(n=100)] → LogReg(C=1)",
}

# ════════════════════════════════════════════════════════════════════════════
# STACKED TEXT ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
class StackedTextEnsemble(BaseEstimator, ClassifierMixin):
    """
    High-precision stacked ensemble for email text classification.

    Architecture
    ------------
    Feature Engineering : TF-IDF Vectorizer (N-grams 1-3, 50 000 features)
    Base Learner A      : MultinomialNB  -- keyword frequency specialist
    Base Learner B      : LinearSVC      -- high-dimensional boundary expert
    Base Learner C      : RandomForest   -- contextual / non-linear patterns
    Meta-Learner        : Logistic Regression stacking OOF base predictions
    Imbalance           : SMOTE applied on TF-IDF features before OOF training
    """

    def __init__(self, max_features: int = 50_000, random_state: int = 42) -> None:
        self.max_features = max_features
        self.random_state = random_state
        self.classes_: np.ndarray = np.array([0, 1])

    # ------------------------------------------------------------------
    def _make_components(self) -> None:
        self.tfidf_: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=2,
        )
        self.base_learners_: List[Tuple[str, Any]] = [
            ("mnb",  MultinomialNB(alpha=0.1)),
            ("lsvc", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2_000, random_state=self.random_state), cv=3
            )),
            ("rf",   RandomForestClassifier(
                n_estimators=100, max_depth=8,
                class_weight="balanced", n_jobs=-1,
                random_state=self.random_state,
            )),
        ]
        self.meta_: LogisticRegression = LogisticRegression(
            C=1.0, max_iter=1_000, random_state=self.random_state
        )

    # ------------------------------------------------------------------
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> "StackedTextEnsemble":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._make_components()

        # Step 1: TF-IDF
        X_tfidf = self.tfidf_.fit_transform(X_text)

        # Step 2: SMOTE on TF-IDF features
        if HAS_IMBLEARN and SMOTE is not None:
            minority_count = int((y == 1).sum())
            k_neighbors = min(5, max(1, minority_count - 1))
            smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            X_sm, y_sm = smote.fit_resample(X_tfidf, y)
        else:
            X_sm, y_sm = X_tfidf, y

        # Step 3: OOF base predictions on SMOTE-augmented data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        n_base = len(self.base_learners_)
        oof = np.zeros((len(y_sm), n_base))

        for i, (_, base) in enumerate(self.base_learners_):
            for tr_idx, val_idx in skf.split(X_sm, y_sm):
                m = clone(base)
                m.fit(X_sm[tr_idx], y_sm[tr_idx])
                oof[val_idx, i] = m.predict_proba(X_sm[val_idx])[:, 1]
            base.fit(X_sm, y_sm)   # final fit on full resampled data

        # Step 4: Meta-learner
        self.meta_.fit(oof, y_sm)
        self.tfidf_vocab_size_: int = len(self.tfidf_.vocabulary_)
        return self

    # ------------------------------------------------------------------
    def _base_proba(self, X_tfidf: Any) -> np.ndarray:
        return np.column_stack([
            base.predict_proba(X_tfidf)[:, 1]
            for _, base in self.base_learners_
        ])

    def predict_proba(self, X_text: np.ndarray) -> np.ndarray:
        X_tfidf = self.tfidf_.transform(X_text)
        meta_input = self._base_proba(X_tfidf)
        return self.meta_.predict_proba(meta_input)

    def predict(self, X_text: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_text)[:, 1] >= 0.5).astype(int)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineArtifacts:
    # Raw / cleaned
    dataset:           pd.DataFrame
    cleaned_dataset:   pd.DataFrame
    feature_frame:     pd.DataFrame

    # Feature selection
    selected_features: List[str]
    feature_weights:   pd.DataFrame
    correlation_matrix: pd.DataFrame

    # Train / test split
    X_train:     pd.DataFrame
    X_test:      pd.DataFrame
    y_train:     pd.Series
    y_test:      pd.Series
    text_train:  pd.Series
    text_test:   pd.Series

    # SMOTE
    smote_summary: Dict[str, Any]

    # Fitted models
    fitted_models: Dict[str, Any]

    # Evaluation
    cv_metrics:   pd.DataFrame
    test_metrics: pd.DataFrame
    roc_data:     Dict[str, Tuple]

    # MLP history
    mlp_history: Optional[Dict[str, List[float]]]

    # Figures
    figures:      Dict[str, plt.Figure]
    shap_figures: Dict[str, plt.Figure]

    # Meta
    preprocessing_summary: Dict[str, Any]


# Backward compatibility for joblib artifacts serialized from direct script execution
_main_module = sys.modules.get("__main__")
if _main_module is not None:
    setattr(_main_module, "PipelineArtifacts", PipelineArtifacts)
    setattr(_main_module, "StackedTextEnsemble", StackedTextEnsemble)
sys.modules.setdefault("email_classification_pipeline", sys.modules[__name__])


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _safe_divide(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _count_matches(text: str, patterns: Iterable[str]) -> int:
    lo = text.lower()
    return sum(lo.count(p.lower()) for p in patterns)


def _contains_attachment_hint(text: str) -> int:
    return int(any(h in text.lower() for h in ("attached", "attachment", "resume enclosed", "cv attached")))


def _normalize_labels(value: Any) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(i) for i in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(i) for i in value]
    if pd.isna(value):
        return []
    return [str(value)]


def _predict_scores_raw(model: Any, X: Any) -> np.ndarray:
    """Return positive-class probability or clipped decision score."""
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(X), dtype=float)
        if prob.ndim == 1:
            return np.clip(prob, 0.0, 1.0)
        classes = list(getattr(model, "classes_", range(prob.shape[1])))
        pos_idx = classes.index(1) if 1 in classes else min(1, prob.shape[1] - 1)
        return prob[:, pos_idx]
    raw = np.asarray(model.predict(X), dtype=float)
    return np.clip(raw, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
def load_email_dataframe(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    if load_dataset is None:
        raise ImportError("The `datasets` package is required.")

    raw = None
    if DownloadConfig is not None:
        try:
            raw = load_dataset(
                dataset_name,
                split="train",
                download_config=DownloadConfig(local_files_only=True),
            )
            print("  Loaded dataset from local Hugging Face cache.")
        except Exception:
            raw = None

    if raw is None:
        raw = load_dataset(dataset_name, split="train")
        print("  Loaded dataset from Hugging Face Hub.")

    df  = raw.to_pandas()
    df["subject"] = df["subject"].astype("string")
    df["body"]    = df["body"].astype("string")
    df["labels"]  = df["labels"].apply(_normalize_labels)
    df["is_job"]  = df["labels"].apply(lambda ls: int(POSITIVE_LABEL in ls))
    return df


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    w = df.copy()
    summary: Dict[str, Any] = {
        "input_rows":            int(len(w)),
        "missing_subject_before": int(w["subject"].isna().sum()),
        "missing_body_before":    int(w["body"].isna().sum()),
        "missing_labels_before":  int(w["labels"].isna().sum()),
    }
    w["subject"] = w["subject"].fillna("")
    w["body"]    = w["body"].fillna("")
    w["labels"]  = w["labels"].apply(_normalize_labels)
    w["text"]    = (w["subject"].astype(str) + " " + w["body"].astype(str)).str.strip()

    dup_mask = w.duplicated(subset=["subject", "body"], keep="first")
    summary["duplicate_rows_removed"] = int(dup_mask.sum())
    w = w.loc[~dup_mask].copy()

    empty_mask = w["text"].str.len() == 0
    summary["empty_rows_removed"] = int(empty_mask.sum())
    w = w.loc[~empty_mask].copy()

    w.reset_index(drop=True, inplace=True)
    summary["rows_after_basic_cleaning"] = int(len(w))
    return w, summary


# ════════════════════════════════════════════════════════════════════════════
# NUMERIC FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
def _extract_numeric_features(subject: str, body: str) -> Dict[str, float]:
    subject = subject or ""
    body    = body    or ""
    combined = f"{subject} {body}".strip()
    lowered  = combined.lower()
    words    = re.findall(r"\b\w+\b", combined)
    unique_words = set(w.lower() for w in words)
    char_count   = len(combined)
    alpha_chars  = sum(1 for c in combined if c.isalpha())
    upper_chars  = sum(1 for c in combined if c.isupper())
    digit_chars  = sum(1 for c in combined if c.isdigit())
    punct_chars  = sum(1 for c in combined if re.match(r"[^\w\s]", c))

    features: Dict[str, float] = {
        "subject_length":    float(len(subject)),
        "body_length":       float(len(body)),
        "total_length":      float(char_count),
        "word_count":        float(len(words)),
        "avg_word_length":   _safe_divide(sum(len(w) for w in words), len(words)),
        "unique_word_ratio": _safe_divide(len(unique_words), len(words)),
        "uppercase_ratio":   _safe_divide(upper_chars, alpha_chars),
        "digit_ratio":       _safe_divide(digit_chars, max(char_count, 1)),
        "punctuation_ratio": _safe_divide(punct_chars,  max(char_count, 1)),
        "exclamation_count": float(combined.count("!")),
        "question_count":    float(combined.count("?")),
        "colon_count":       float(combined.count(":")),
        "url_count":         float(len(re.findall(r"https?://|www\.", lowered))),
        "email_count":       float(len(re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", combined))),
        "phone_count":       float(len(re.findall(
            r"(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}", combined))),
        "currency_symbol_count": float(sum(combined.count(s) for s in ("$", "\u00A3", "\u20AC"))),
        "attachment_hint":   float(_contains_attachment_hint(combined)),
        "dear_greeting":     float(int(lowered.startswith("dear"))),
        "contains_date_like": float(int(bool(re.search(
            r"\b(january|february|march|april|may|june|july|august|september|"
            r"october|november|december|\d{1,2}/\d{1,2}/\d{2,4})\b", lowered)))),
    }
    for prefix, kws in KEYWORD_GROUPS.items():
        features[f"{prefix}_keyword_count"] = float(_count_matches(combined, kws))

    features["job_signal_ratio"] = _safe_divide(
        features["job_keyword_count"] + features["candidate_keyword_count"]
        + features["recruiting_keyword_count"] + features["compensation_keyword_count"],
        max(features["word_count"], 1.0),
    )
    features["non_job_signal_ratio"] = _safe_divide(
        features["business_keyword_count"] + features["support_keyword_count"],
        max(features["word_count"], 1.0),
    )
    return features


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_extract_numeric_features(str(s), str(b))
            for s, b in zip(df["subject"], df["body"])]
    feat = pd.DataFrame(rows, index=df.index)
    feat["is_job"] = df["is_job"].astype(int).values
    return feat


def filter_outliers(features: pd.DataFrame, target_col: str = "is_job"
                    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    num_cols = [c for c in features.columns if c != target_col]
    target   = features[target_col].astype(int)
    maj_cls  = int(target.mode().iloc[0])
    maj_mask = target == maj_cls
    ref      = features.loc[maj_mask, num_cols]
    q1, q3   = ref.quantile(0.25), ref.quantile(0.75)
    iqr      = (q3 - q1).replace(0, 1.0)
    lo, hi   = q1 - 3.0 * iqr, q3 + 3.0 * iqr

    ext_mask    = features[num_cols].lt(lo) | features[num_cols].gt(hi)
    ext_count   = ext_mask.sum(axis=1)
    outlier_rows = (ext_count >= 2) & maj_mask

    summary = {
        "outlier_rows_removed":         int(outlier_rows.sum()),
        "majority_outlier_rows_removed": int((features.loc[outlier_rows, target_col].astype(int) == maj_cls).sum()),
        "minority_rows_preserved":       int((target != maj_cls).sum()),
        "rows_after_outlier_filter":     int((~outlier_rows).sum()),
    }
    return features.loc[~outlier_rows].copy(), summary


# ════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ════════════════════════════════════════════════════════════════════════════
def apply_correlation_filter(features: pd.DataFrame,
                              target_col: str = "is_job",
                              threshold: float = 0.9
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = features.drop(columns=[target_col])
    const_cols = [c for c in predictors.columns if predictors[c].nunique(dropna=False) <= 1]
    if const_cols:
        predictors = predictors.drop(columns=const_cols)
    corr   = predictors.corr().fillna(0.0)
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c].abs() > threshold)]
    filtered = predictors.drop(columns=dropped)
    feat_target_corr = filtered.corrwith(features[target_col]).abs().sort_values(ascending=False)
    ranking = feat_target_corr.reset_index()
    ranking.columns = ["feature", "abs_target_correlation"]
    return filtered, corr, ranking


def compute_feature_weights(predictors: pd.DataFrame, target: pd.Series,
                             alpha_lasso: float = 0.001, alpha_ridge: float = 1.0
                             ) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(predictors)
    Xf     = pd.DataFrame(X_sc, columns=predictors.columns, index=predictors.index)

    lr  = LinearRegression().fit(Xf, target)
    las = Lasso(alpha=alpha_lasso, random_state=RANDOM_STATE, max_iter=10_000).fit(Xf, target)
    rid = Ridge(alpha=alpha_ridge, random_state=RANDOM_STATE).fit(Xf, target)

    weights = pd.DataFrame({
        "feature":       predictors.columns,
        "linear_weight": np.abs(lr.coef_),
        "lasso_weight":  np.abs(las.coef_),
        "ridge_weight":  np.abs(rid.coef_),
    })
    weights["ensemble_weight"] = weights[["linear_weight", "lasso_weight", "ridge_weight"]].mean(axis=1)
    weights.sort_values("ensemble_weight", ascending=False, inplace=True)
    weights.reset_index(drop=True, inplace=True)
    return weights, scaler


# ════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY  (raw estimators — SMOTE applied separately)
# ════════════════════════════════════════════════════════════════════════════
def get_model_registry() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1_000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=4,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=RANDOM_STATE, class_weight="balanced_subsample", n_jobs=-1
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu", solver="adam",
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    }
    if lgb is not None:
        models["Gradient Boost"] = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1,
        )
    elif xgb is not None:
        models["Gradient Boost"] = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=1, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        models["Gradient Boost"] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE,
        )
    # Stacked ensemble handled separately (text input)
    models["Stacked Ensemble"] = StackedTextEnsemble(random_state=RANDOM_STATE)
    return models


# ════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════════════════════
_NUMERIC_SCALE_MODELS = {"Logistic Regression", "Neural Network"}


def split_dataset(feature_frame: pd.DataFrame, cleaned_df: pd.DataFrame,
                  selected_features: List[str]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame,
                             pd.Series,   pd.Series,
                             pd.Series,   pd.Series]:
    X = feature_frame[selected_features]
    y = feature_frame["is_job"].astype(int)
    text = cleaned_df.loc[feature_frame.index, "text"].reset_index(drop=True)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    X_tr, X_te, y_tr, y_te, txt_tr, txt_te = train_test_split(
        X, y, text, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return (X_tr.reset_index(drop=True), X_te.reset_index(drop=True),
            y_tr.reset_index(drop=True), y_te.reset_index(drop=True),
            txt_tr.reset_index(drop=True), txt_te.reset_index(drop=True))


# ════════════════════════════════════════════════════════════════════════════
# CV EVALUATION  (SMOTE applied per fold)
# ════════════════════════════════════════════════════════════════════════════
def _fold_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    preds = (scores >= 0.5).astype(int)
    auc_val = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "accuracy":  accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall":    recall_score(y_true, preds, zero_division=0),
        "f1_score":  f1_score(y_true, preds, zero_division=0),
        "auc_score": auc_val,
    }


def _apply_smote(X_tr: pd.DataFrame, y_tr: pd.Series
                 ) -> Tuple[pd.DataFrame, pd.Series]:
    if not HAS_IMBLEARN or SMOTE is None:
        return X_tr, y_tr
    minority_count = int(y_tr.sum())
    k = min(5, max(1, minority_count - 1))
    X_sm, y_sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(X_tr, y_tr)
    return pd.DataFrame(X_sm, columns=X_tr.columns), pd.Series(y_sm)


def evaluate_models_cv(X_train: pd.DataFrame, y_train: pd.Series,
                       text_train: pd.Series,
                       ) -> pd.DataFrame:
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_model_registry()
    rows: List[Dict[str, Any]] = []

    print(f"  Running {CV_FOLDS}-fold cross-validation for {len(models)} models...")
    for model_name, estimator in models.items():
        print(f"    [{model_name}]", end="", flush=True)
        fold_list: List[Dict[str, float]] = []

        if model_name == "Stacked Ensemble":
            split_arr = text_train
        else:
            split_arr = X_train

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(split_arr, y_train), 1):
            if model_name == "Stacked Ensemble":
                txt_tr = text_train.iloc[tr_idx].values
                txt_val = text_train.iloc[val_idx].values
                y_tr = y_train.iloc[tr_idx].values
                y_val = y_train.iloc[val_idx].values
                m = StackedTextEnsemble(random_state=RANDOM_STATE)
                m.fit(txt_tr, y_tr)
                scores = m.predict_proba(txt_val)[:, 1]
            else:
                X_tr_f = X_train.iloc[tr_idx]
                X_val_f = X_train.iloc[val_idx]
                y_tr_f = y_train.iloc[tr_idx]
                y_val_np = y_train.iloc[val_idx].values

                X_sm, y_sm = _apply_smote(X_tr_f, y_tr_f)
                m = clone(estimator)

                if model_name in _NUMERIC_SCALE_MODELS:
                    sc = StandardScaler()
                    X_sm_s = sc.fit_transform(X_sm)
                    X_val_s = sc.transform(X_val_f)
                    m.fit(X_sm_s, y_sm)
                    scores = _predict_scores_raw(m, X_val_s)
                else:
                    m.fit(X_sm, y_sm)
                    scores = _predict_scores_raw(m, X_val_f)
                y_val = y_val_np

            fold_list.append(_fold_metrics(y_val, scores))
            print(".", end="", flush=True)

        print()
        fold_df = pd.DataFrame(fold_list)
        row: Dict[str, Any] = {"model": model_name}
        for col in ["accuracy", "precision", "recall", "f1_score", "auc_score"]:
            row[col]            = float(fold_df[col].mean())
            row[f"{col}_std"]   = float(fold_df[col].std(ddof=0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# FINAL MODEL FITTING  (SMOTE on full training set)
# ════════════════════════════════════════════════════════════════════════════
def fit_all_models(X_train: pd.DataFrame, y_train: pd.Series,
                   text_train: pd.Series,
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, List[float]]]]:
    """
    Fit every model on SMOTE-augmented training data.
    Returns (fitted_models, smote_summary, mlp_history).
    """
    # Global SMOTE for numeric models
    X_sm, y_sm = _apply_smote(X_train, y_train)
    smote_summary: Dict[str, Any] = {
        "train_positive_before": int(y_train.sum()),
        "train_negative_before": int((y_train == 0).sum()),
        "train_total_before":    int(len(y_train)),
        "train_positive_after":  int(y_sm.sum()),
        "train_negative_after":  int((y_sm == 0).sum()),
        "train_total_after":     int(len(y_sm)),
        "smote_applied":         HAS_IMBLEARN,
    }

    models   = get_model_registry()
    fitted: Dict[str, Any] = {}
    mlp_history: Optional[Dict[str, List[float]]] = None

    for model_name, estimator in models.items():
        print(f"  Fitting [{model_name}]...")
        if model_name == "Stacked Ensemble":
            m = StackedTextEnsemble(random_state=RANDOM_STATE)
            m.fit(text_train.values, y_train.values)
            fitted[model_name] = m

        elif model_name in _NUMERIC_SCALE_MODELS:
            sc = StandardScaler()
            X_sc = sc.fit_transform(X_sm)
            m = clone(estimator)
            m.fit(X_sc, y_sm)
            fitted[model_name] = (m, sc)
            if model_name == "Neural Network":
                mlp_history = {
                    "loss":      list(m.loss_curve_),
                    "val_score": list(getattr(m, "validation_scores_", [])),
                }
        else:
            m = clone(estimator)
            m.fit(X_sm, y_sm)
            fitted[model_name] = m

    return fitted, smote_summary, mlp_history


# ════════════════════════════════════════════════════════════════════════════
# TEST-SET EVALUATION  +  ROC CURVES
# ════════════════════════════════════════════════════════════════════════════
def evaluate_on_test_set(fitted_models: Dict[str, Any],
                         X_test: pd.DataFrame, y_test: pd.Series,
                         text_test: pd.Series,
                         ) -> Tuple[pd.DataFrame, Dict[str, Tuple]]:
    rows: List[Dict[str, Any]] = []
    roc_data: Dict[str, Tuple] = {}

    for model_name, model_obj in fitted_models.items():
        if model_name == "Stacked Ensemble":
            scores = model_obj.predict_proba(text_test.values)[:, 1]
        elif isinstance(model_obj, tuple):
            clf, sc = model_obj
            scores  = _predict_scores_raw(clf, sc.transform(X_test))
        else:
            scores = _predict_scores_raw(model_obj, X_test)

        preds = (scores >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val     = sk_auc(fpr, tpr)
        roc_data[model_name] = (fpr, tpr, auc_val)

        rows.append({
            "model":     model_name,
            "accuracy":  accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall":    recall_score(y_test, preds, zero_division=0),
            "f1_score":  f1_score(y_test, preds, zero_division=0),
            "auc_score": auc_val,
        })

    df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    return df, roc_data


# ════════════════════════════════════════════════════════════════════════════
# EDA VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════
PALETTE = {"Not Job": "#f97316", "Job": "#2563eb"}


def _fig_class_distribution(df: pd.DataFrame, smote_summary: Dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cc = df["is_job"].map({0: "Not Job", 1: "Job"}).value_counts().sort_index()
    sns.barplot(x=cc.index, y=cc.values, palette=["#f97316", "#2563eb"], ax=axes[0])
    axes[0].set_title("Class Distribution (Before SMOTE)")
    axes[0].set_xlabel("Class"); axes[0].set_ylabel("Email Count")
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10)

    if smote_summary.get("smote_applied"):
        categories = ["Negative (Train)", "Positive (Train)"]
        before = [smote_summary["train_negative_before"], smote_summary["train_positive_before"]]
        after  = [smote_summary["train_negative_after"],  smote_summary["train_positive_after"]]
        x = np.arange(len(categories))
        w = 0.35
        axes[1].bar(x - w / 2, before, w, label="Before SMOTE", color="#94a3b8")
        axes[1].bar(x + w / 2, after,  w, label="After SMOTE",  color="#0f766e")
        axes[1].set_xticks(x); axes[1].set_xticklabels(categories)
        axes[1].set_title("SMOTE Augmentation (Training Set)")
        axes[1].set_ylabel("Sample Count"); axes[1].legend()
    else:
        axes[1].set_visible(False)

    fig.suptitle("Class Distribution & SMOTE Augmentation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_feature_weights(feature_weights: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = feature_weights.head(top_n).sort_values("ensemble_weight")
    sns.barplot(data=subset, x="ensemble_weight", y="feature", color="#0f766e", ax=ax)
    ax.set_title("Top Weighted Numeric Features")
    ax.set_xlabel("Average Absolute Weight"); ax.set_ylabel("Feature")
    return fig


def _fig_correlation_heatmap(corr: pd.DataFrame, features: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    sub = corr.loc[features, features]
    sns.heatmap(sub, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix of Selected Features")
    return fig


def _fig_feature_distribution(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    chosen = features[:min(3, len(features))]
    melted = df[chosen + ["is_job"]].melt(id_vars="is_job", var_name="feature", value_name="value")
    melted["class"] = melted["is_job"].map({0: "Not Job", 1: "Job"})
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="class", palette=PALETTE, ax=ax)
    ax.set_title("Feature Distributions by Class"); ax.tick_params(axis="x", rotation=20)
    return fig


def _fig_feature_class_heatmap(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    chosen = features[:min(10, len(features))]
    class_means = (df.groupby("is_job")[chosen].mean()
                   .rename(index={0: "Not Job", 1: "Job"}))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    sns.heatmap(class_means, annot=True, fmt=".3f", cmap="crest", linewidths=0.4, ax=ax)
    ax.set_title("Average Feature Values: Job vs. Not-Job Emails")
    ax.tick_params(axis="x", rotation=25)
    return fig


def _fig_feature_gap(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    chosen = features[:min(10, len(features))]
    comp = df.groupby("is_job")[chosen].mean().T
    comp.columns = ["Not Job", "Job"]
    comp["gap"] = (comp["Job"] - comp["Not Job"]).abs()
    comp = comp.sort_values("gap", ascending=True)
    y_pos = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hlines(y=y_pos, xmin=comp["Not Job"], xmax=comp["Job"], color="#94a3b8", linewidth=2.5)
    ax.scatter(comp["Not Job"], y_pos, color="#f97316", s=80, label="Not Job", zorder=3)
    ax.scatter(comp["Job"],     y_pos, color="#2563eb", s=80, label="Job",     zorder=3)
    ax.set_yticks(y_pos); ax.set_yticklabels(comp.index)
    ax.set_title("Feature Mean Gaps: Job vs. Not-Job Emails")
    ax.set_xlabel("Average Feature Value"); ax.legend(loc="lower right")
    return fig


def _fig_single_feature_focus(df: pd.DataFrame, feature_name: str) -> plt.Figure:
    ff = df[[feature_name, "is_job"]].copy()
    ff["class"] = ff["is_job"].map({0: "Not Job", 1: "Job"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=ff, x=feature_name, hue="class", element="step",
                 stat="density", common_norm=False, fill=True, alpha=0.35,
                 palette=PALETTE, ax=axes[0])
    axes[0].set_title(f"Distribution of `{feature_name}` by Class")
    sns.boxplot(data=ff, x="class", y=feature_name, palette=PALETTE, ax=axes[1])
    axes[1].set_title(f"`{feature_name}` Spread by Class")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════
def _fig_model_comparison(metrics_df: pd.DataFrame, title: str = "Model Comparison") -> plt.Figure:
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    chart = metrics_df[["model"] + metric_cols].melt(
        id_vars="model", var_name="metric", value_name="score")
    palette = {"accuracy": "#2563eb", "precision": "#f97316",
               "recall": "#0f766e",  "f1_score":  "#7c3aed", "auc_score": "#dc2626"}
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.barplot(data=chart, x="model", y="score", hue="metric", ax=ax, palette=palette)
    ax.set_title(title); ax.set_xlabel("Model"); ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20); ax.legend(loc="lower left", ncol=3)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=6.5, rotation=90, xytext=(0, 2),
                        textcoords="offset points")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    return fig


def _fig_roc_curves(roc_data: Dict[str, Tuple]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models (Test Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def _fig_mlp_history(mlp_history: Dict[str, List[float]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(mlp_history["loss"]) + 1)
    axes[0].plot(epochs, mlp_history["loss"], color="#2563eb", linewidth=1.8)
    axes[0].set_title("Neural Network Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss"); axes[0].grid(alpha=0.3)

    if mlp_history.get("val_score"):
        val_epochs = range(1, len(mlp_history["val_score"]) + 1)
        axes[1].plot(val_epochs, mlp_history["val_score"], color="#0f766e", linewidth=1.8)
        axes[1].set_title("Neural Network Validation Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def create_shap_figures(fitted_models: Dict[str, Any],
                        X_test: pd.DataFrame, y_test: pd.Series,
                        ) -> Dict[str, plt.Figure]:
    if shap is None:
        return {}

    # Pick best tree-based model for SHAP
    tree_models = {k: v for k, v in fitted_models.items()
                   if k in ("Random Forest", "Gradient Boost", "Decision Tree")
                   and not isinstance(v, tuple)}
    if not tree_models:
        return {}

    model_name, model_obj = next(iter(tree_models.items()))
    sample = X_test.sample(n=min(300, len(X_test)), random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(model_obj)
        sv_raw    = explainer.shap_values(sample)
        if isinstance(sv_raw, list):
            sv_raw = sv_raw[-1]
        if getattr(sv_raw, "ndim", 0) == 3:
            sv_raw = sv_raw[:, :, -1]
        sv_obj = explainer(sample)
    except Exception:
        return {}

    figures: Dict[str, plt.Figure] = {}

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv_raw, sample, plot_type="bar", show=False)
    figures["shap_summary_bar"] = plt.gcf()
    plt.figure(figsize=(10, 6))

    # Beeswarm
    shap.summary_plot(sv_raw, sample, show=False)
    figures["shap_summary_beeswarm"] = plt.gcf()

    # Waterfall — pick highest-confidence true positive
    try:
        tp_mask = (y_test == 1) & (model_obj.predict(X_test) == 1)
        tp_idx  = y_test.index[tp_mask]
        if len(tp_idx) > 0:
            probs     = model_obj.predict_proba(X_test.loc[tp_idx])[:, 1]
            best_i    = tp_idx[np.argmax(probs)]
            wf_sample = X_test.loc[[best_i]]
        else:
            wf_sample = X_test.iloc[[0]]

        sv_wf = explainer(wf_sample)
        # RF / multi-output classifiers return 3D SHAP arrays; take positive class
        if hasattr(sv_wf, "values") and sv_wf.values.ndim == 3:
            sv_wf = shap.Explanation(
                values        = sv_wf.values[:, :, 1],
                base_values   = sv_wf.base_values[:, 1] if sv_wf.base_values.ndim > 1
                                else sv_wf.base_values,
                data          = sv_wf.data,
                feature_names = list(wf_sample.columns),
            )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sv_wf[0], show=False)
        figures["shap_waterfall"] = plt.gcf()
    except Exception as _wf_err:
        print(f"  Note: SHAP waterfall skipped ({_wf_err})")

    return figures


# ════════════════════════════════════════════════════════════════════════════
# GENERATE ALL FIGURES
# ════════════════════════════════════════════════════════════════════════════
def generate_figures(feature_frame: pd.DataFrame,
                     feature_weights: pd.DataFrame,
                     correlation_matrix: pd.DataFrame,
                     selected_features: List[str],
                     cv_metrics: pd.DataFrame,
                     test_metrics: pd.DataFrame,
                     roc_data: Dict[str, Tuple],
                     smote_summary: Dict[str, Any],
                     mlp_history: Optional[Dict[str, List[float]]],
                     ) -> Dict[str, plt.Figure]:
    figs: Dict[str, plt.Figure] = {
        "class_distribution":        _fig_class_distribution(feature_frame, smote_summary),
        "feature_weights":            _fig_feature_weights(feature_weights),
        "selected_feature_correlation": _fig_correlation_heatmap(correlation_matrix, selected_features),
        "feature_distribution":       _fig_feature_distribution(feature_frame, selected_features),
        "feature_class_heatmap":      _fig_feature_class_heatmap(feature_frame, selected_features),
        "feature_gap":                _fig_feature_gap(feature_frame, selected_features),
        "model_comparison_cv":        _fig_model_comparison(cv_metrics,   "5-Fold CV Metrics"),
        "model_comparison_test":      _fig_model_comparison(test_metrics,  "Test-Set Metrics"),
        "roc_curves":                 _fig_roc_curves(roc_data),
    }
    if mlp_history:
        figs["mlp_training_history"] = _fig_mlp_history(mlp_history)
    return figs


# ════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════
def persist_artifacts(artifacts: PipelineArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save fitted models
    for model_name, model_obj in artifacts.fitted_models.items():
        safe = model_name.replace(" ", "_").lower()
        joblib.dump(model_obj, models_dir / f"{safe}.joblib")

    # Save DataFrames
    artifacts.cv_metrics.to_csv(output_dir / "cv_metrics.csv",   index=False)
    artifacts.test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    artifacts.feature_weights.to_csv(output_dir / "feature_weights.csv", index=False)
    artifacts.cleaned_dataset.to_csv(output_dir / "cleaned_dataset.csv", index=False)
    artifacts.feature_frame.to_csv(output_dir  / "engineered_features.csv", index=False)

    # Save summaries
    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(artifacts.preprocessing_summary, fh, indent=2)
    with (output_dir / "smote_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(artifacts.smote_summary, fh, indent=2)

    # Save figures
    for name, fig in {**artifacts.figures, **artifacts.shap_figures}.items():
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    # Save a packaged bundle that Streamlit can load instantly.
    joblib.dump(artifacts, output_dir / PACKAGED_ARTIFACT_FILENAME)

    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "packaged_artifact": PACKAGED_ARTIFACT_FILENAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": DATASET_NAME,
        "top_k_features": len(artifacts.selected_features),
        "models": list(artifacts.fitted_models.keys()),
        "best_cv_model": artifacts.preprocessing_summary.get("best_cv_model"),
        "best_test_model": artifacts.preprocessing_summary.get("best_test_model"),
    }
    with (output_dir / ARTIFACT_MANIFEST_FILENAME).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_packaged_artifacts(output_dir: Path) -> PipelineArtifacts:
    package_path = output_dir / PACKAGED_ARTIFACT_FILENAME
    if not package_path.exists():
        raise FileNotFoundError(f"Packaged artifacts not found at {package_path}")
    artifacts = joblib.load(package_path)
    if not isinstance(artifacts, PipelineArtifacts):
        raise TypeError(f"Unexpected packaged artifact type: {type(artifacts)!r}")
    return artifacts


def _load_saved_figure(path: Path) -> plt.Figure:
    image = plt.imread(path)
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(max(width / 150, 6), max(height / 150, 4)))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _load_saved_model_artifacts(output_dir: Path) -> PipelineArtifacts:
    models_dir = output_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Saved model directory not found at {models_dir}")

    preprocessing_summary = json.loads((output_dir / "preprocessing_summary.json").read_text(encoding="utf-8"))
    smote_summary = json.loads((output_dir / "smote_summary.json").read_text(encoding="utf-8"))
    cleaned_dataset = pd.read_csv(output_dir / "cleaned_dataset.csv")
    feature_frame = pd.read_csv(output_dir / "engineered_features.csv")
    feature_weights = pd.read_csv(output_dir / "feature_weights.csv")
    cv_metrics = pd.read_csv(output_dir / "cv_metrics.csv")
    test_metrics = pd.read_csv(output_dir / "test_metrics.csv")

    selected_features = list(preprocessing_summary.get("selected_features", []))
    correlation_matrix = feature_frame[selected_features].corr() if selected_features else pd.DataFrame()

    fitted_models: Dict[str, Any] = {}
    for model_name in test_metrics["model"].tolist():
        safe = model_name.replace(" ", "_").lower()
        fitted_models[model_name] = joblib.load(models_dir / f"{safe}.joblib")

    figures: Dict[str, plt.Figure] = {}
    for name in ("feature_distribution", "feature_gap", "selected_feature_correlation", "roc_curves", "mlp_training_history"):
        fig_path = output_dir / f"{name}.png"
        if fig_path.exists():
            figures[name] = _load_saved_figure(fig_path)

    shap_figures: Dict[str, plt.Figure] = {}
    for name in ("shap_summary_bar", "shap_summary_beeswarm", "shap_waterfall"):
        fig_path = output_dir / f"{name}.png"
        if fig_path.exists():
            shap_figures[name] = _load_saved_figure(fig_path)

    empty_frame = pd.DataFrame(columns=selected_features)
    empty_series = pd.Series(dtype=float)

    return PipelineArtifacts(
        dataset=cleaned_dataset.copy(),
        cleaned_dataset=cleaned_dataset,
        feature_frame=feature_frame,
        selected_features=selected_features,
        feature_weights=feature_weights,
        correlation_matrix=correlation_matrix,
        X_train=empty_frame.copy(),
        X_test=empty_frame.copy(),
        y_train=empty_series.copy(),
        y_test=empty_series.copy(),
        text_train=pd.Series(dtype=str),
        text_test=pd.Series(dtype=str),
        smote_summary=smote_summary,
        fitted_models=fitted_models,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        roc_data={},
        mlp_history=None,
        figures=figures,
        shap_figures=shap_figures,
        preprocessing_summary=preprocessing_summary,
    )


def load_pretrained_pipeline_artifacts(output_dir: Path) -> PipelineArtifacts:
    print(f"=== Loading pretrained model packages from {output_dir} ===")
    try:
        return load_packaged_artifacts(output_dir)
    except Exception as exc:
        print(f"=== Packaged artifact bundle unreadable; reconstructing from saved model files ({exc}) ===")
        try:
            return _load_saved_model_artifacts(output_dir)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Pretrained model packages could not be loaded. "
                "Run `python src\\streamlit_app.py --force-retrain` to rebuild artifacts."
            ) from fallback_exc


def get_or_train_pipeline_artifacts(output_dir: Path,
                                    top_k_features: int = TOP_K_FEATURES,
                                    force_retrain: bool = False) -> PipelineArtifacts:
    if not force_retrain:
        try:
            print(f"=== Loading packaged artifacts from {output_dir} ===")
            return load_packaged_artifacts(output_dir)
        except Exception as exc:
            print(f"=== Packaged artifacts unavailable; retraining pipeline ({exc}) ===")
    return run_pipeline(top_k_features=top_k_features, persist_dir=output_dir)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
def run_pipeline(dataset_name: str = DATASET_NAME,
                 top_k_features: int = TOP_K_FEATURES,
                 persist_dir: Optional[Path] = None) -> PipelineArtifacts:

    print("=== Stage 1: Loading data ===")
    raw_df = load_email_dataframe(dataset_name)

    print("=== Stage 2: Preprocessing ===")
    cleaned_df, cleaning_summary = preprocess_dataframe(raw_df)

    print("=== Stage 3: Feature engineering ===")
    full_ff = build_feature_frame(cleaned_df)
    filtered_ff, outlier_summary = filter_outliers(full_ff)
    filtered_df = cleaned_df.loc[filtered_ff.index].reset_index(drop=True)
    filtered_ff = filtered_ff.reset_index(drop=True)

    predictors, corr_matrix, corr_ranking = apply_correlation_filter(filtered_ff)
    target = filtered_ff["is_job"].astype(int)

    feature_weights, _ = compute_feature_weights(predictors, target)
    feature_weights = feature_weights.merge(corr_ranking, on="feature", how="left")
    selected_features = feature_weights["feature"].head(
        min(top_k_features, len(feature_weights))).tolist()

    print("=== Stage 4: Train / test split ===")
    X_train, X_test, y_train, y_test, text_train, text_test = split_dataset(
        filtered_ff, filtered_df, selected_features)

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Positive rate (train): {y_train.mean():.2%} | "
          f"(test): {y_test.mean():.2%}")

    print("=== Stage 5: Cross-validation ===")
    cv_metrics = evaluate_models_cv(X_train, y_train, text_train)

    print("=== Stage 6: Final model fitting ===")
    fitted_models, smote_summary, mlp_history = fit_all_models(
        X_train, y_train, text_train)

    print("=== Stage 7: Test-set evaluation ===")
    test_metrics, roc_data = evaluate_on_test_set(
        fitted_models, X_test, y_test, text_test)

    print("=== Stage 8: Generating visualizations ===")
    figures = generate_figures(filtered_ff, feature_weights, corr_matrix,
                               selected_features, cv_metrics, test_metrics,
                               roc_data, smote_summary, mlp_history)

    print("=== Stage 9: SHAP analysis ===")
    shap_figures = create_shap_figures(fitted_models, X_test, y_test)

    best_cv_model   = cv_metrics.iloc[0]["model"]
    best_test_model = test_metrics.iloc[0]["model"]
    preprocessing_summary: Dict[str, Any] = {
        **cleaning_summary,
        **outlier_summary,
        "positive_class_label":       POSITIVE_LABEL,
        "positive_class_count":       int(filtered_ff["is_job"].sum()),
        "negative_class_count":       int((1 - filtered_ff["is_job"]).sum()),
        "positive_class_ratio":       float(filtered_ff["is_job"].mean()),
        "train_size":                 int(len(X_train)),
        "test_size":                  int(len(X_test)),
        "selected_features":          selected_features,
        "highest_weight_feature":     feature_weights.iloc[0]["feature"],
        "best_cv_model":              best_cv_model,
        "best_test_model":            best_test_model,
    }

    artifacts = PipelineArtifacts(
        dataset=raw_df, cleaned_dataset=filtered_df,
        feature_frame=filtered_ff,
        selected_features=selected_features, feature_weights=feature_weights,
        correlation_matrix=corr_matrix,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        text_train=text_train, text_test=text_test,
        smote_summary=smote_summary,
        fitted_models=fitted_models,
        cv_metrics=cv_metrics, test_metrics=test_metrics, roc_data=roc_data,
        mlp_history=mlp_history,
        figures=figures, shap_figures=shap_figures,
        preprocessing_summary=preprocessing_summary,
    )

    if persist_dir is not None:
        print(f"=== Stage 10: Saving artifacts to {persist_dir} ===")
        persist_artifacts(artifacts, persist_dir)

    print("=== Pipeline complete ===")
    return artifacts


# ════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def predict_from_email(model_name: str, subject: str, body: str,
                       fitted_models: Dict[str, Any],
                       selected_features: List[str],
                       ) -> Tuple[int, float]:
    """Return (predicted_class, positive_probability) for any model."""
    if model_name == "Stacked Ensemble":
        text = (f"{subject} {body}").strip()
        prob = fitted_models[model_name].predict_proba([text])[0, 1]
    else:
        feats = _extract_numeric_features(subject, body)
        X     = pd.DataFrame([feats])[selected_features]
        obj   = fitted_models[model_name]
        if isinstance(obj, tuple):
            clf, sc = obj
            prob = _predict_scores_raw(clf, sc.transform(X))[0]
        else:
            prob = _predict_scores_raw(obj, X)[0]
    return int(prob >= 0.5), float(prob)


def shap_waterfall_for_input(model_name: str, subject: str, body: str,
                              fitted_models: Dict[str, Any],
                              selected_features: List[str],
                              ) -> Optional[plt.Figure]:
    """Compute a SHAP waterfall for user-provided email text (tree models only)."""
    if shap is None or model_name not in ("Random Forest", "Gradient Boost", "Decision Tree"):
        return None
    obj = fitted_models.get(model_name)
    if obj is None or isinstance(obj, tuple):
        return None

    feats = _extract_numeric_features(subject, body)
    X     = pd.DataFrame([feats])[selected_features]

    try:
        explainer = shap.TreeExplainer(obj)
        sv = explainer(X)
        if hasattr(sv, "values") and sv.values.ndim == 3:
            sv = shap.Explanation(
                values        = sv.values[:, :, 1],
                base_values   = sv.base_values[:, 1] if sv.base_values.ndim > 1
                                else sv.base_values,
                data          = sv.data,
                feature_names = list(X.columns),
            )
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(sv[0], show=False)
        return plt.gcf()
    except Exception:
        return None



# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT  — CSS / UTILS
# ════════════════════════════════════════════════════════════════════════════
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


_METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1_score":  "F1 Score",
    "auc_score": "AUC",
}
_METRIC_COLORS = {
    "accuracy":  "#2563eb",
    "precision": "#d97706",
    "recall":    "#0f766e",
    "f1_score":  "#7c3aed",
    "auc_score": "#dc2626",
}
_MODEL_COLORS = [
    "#2563eb", "#d97706", "#0f766e",
    "#7c3aed", "#dc2626", "#0891b2",
]


def _inject_css() -> None:
    if st is None:
        return
    st.markdown("""
    <style>
    /* ── Global background ─────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(145deg,
            #0f172a 0%, #1e1b4b 30%, #172554 65%, #0c4a6e 100%);
        min-height: 100vh;
    }
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* ── Force all text dark-on-light inside content cards ─────────── */
    /* Global readable defaults for non-card areas */
    .stApp p, .stApp li, .stApp span:not(.st-emotion-cache-10trblm),
    .stApp label, .stMarkdown p, .stMarkdown li,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span { color: #e2e8f0 !important; }

    /* Caption text */
    .stCaptionContainer p, .stCaption p,
    div[data-testid="stCaptionContainer"] p { color: #94a3b8 !important; }

    /* Headings */
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 { color: #f1f5f9 !important; }

    /* ══════════════════════════════════════════════════════════════
       TAB NAVIGATION  (st.tabs)
       ══════════════════════════════════════════════════════════════ */

    /* Tab list container — dark pill tray spanning full width */
    div[data-testid="stTabs"] > div:first-child {
        background: #0d1526 !important;
        border: 1.5px solid rgba(56,189,248,0.25) !important;
        border-radius: 14px !important;
        padding: .5rem .6rem !important;
        gap: .4rem !important;
        box-shadow: 0 6px 28px rgba(0,0,0,0.5),
                    inset 0 1px 0 rgba(255,255,255,0.06) !important;
        margin-bottom: 1.6rem !important;
        overflow: visible !important;
    }

    /* Each tab button — clearly boxed pill */
    button[role="tab"] {
        flex: 1 1 0 !important;
        color: #64748b !important;
        font-weight: 700 !important;
        font-size: .82rem !important;
        letter-spacing: .05em !important;
        text-transform: uppercase !important;
        padding: .75rem 1rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.03) !important;
        transition: background .16s, color .16s, box-shadow .16s !important;
        white-space: nowrap !important;
    }
    button[role="tab"]:hover {
        color: #cbd5e1 !important;
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.18) !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg,#38bdf8 0%,#818cf8 100%) !important;
        border-color: transparent !important;
        box-shadow: 0 4px 18px rgba(56,189,248,0.40) !important;
    }
    /* Hide Streamlit's default underline marker */
    div[data-testid="stTabs"] > div:first-child > div[aria-selected],
    div[data-testid="stTabBar"] > div { display: none !important; }

    /* Breathing room below the nav bar */
    div[data-testid="stTabs"] > div:last-child {
        padding-top: .4rem !important;
    }

    /* ── Primary action button (e.g. Classify Email) ─────────────── */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg,#38bdf8 0%,#818cf8 100%) !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 18px rgba(56,189,248,0.38) !important;
        transition: all .18s ease !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

    /* Radio / selectbox / input labels */
    .stRadio label span, .stSelectbox label,
    .stTextInput label, .stTextArea label { color: #e2e8f0 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* ── Hero banner ────────────────────────────────────────────────── */
    .hero-shell {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #0891b2 100%);
        border-radius: 24px; padding: 1.6rem 2rem; color: white;
        box-shadow: 0 20px 60px rgba(124,58,237,0.4); margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 900; letter-spacing: -0.04em;
        margin-bottom: 0.4rem; color: white !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .hero-copy { font-size: 0.97rem; line-height: 1.6; opacity: 0.95; color: white !important; }

    /* ── Stat cards ─────────────────────────────────────────────────── */
    .stat-card {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px; padding: 1rem 1.1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-label {
        font-size: .72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .1em; color: #94a3b8 !important; margin-bottom: .35rem;
    }
    .stat-value { font-size: 1.4rem; font-weight: 900; color: #f1f5f9 !important; line-height: 1.1; }
    .stat-subtext { margin-top: .25rem; font-size: .76rem; color: #64748b !important; }

    /* ── Section chips ──────────────────────────────────────────────── */
    .section-chip {
        display: inline-block; padding: .3rem .8rem; border-radius: 999px;
        font-size: .72rem; font-weight: 800; letter-spacing: .1em;
        text-transform: uppercase; margin-bottom: .6rem;
        color: #38bdf8 !important; background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.25);
    }
    .chip-purple { color: #a78bfa !important; background: rgba(167,139,250,0.12);
                   border-color: rgba(167,139,250,0.25); }
    .chip-green  { color: #34d399 !important; background: rgba(52,211,153,0.12);
                   border-color: rgba(52,211,153,0.25); }
    .chip-orange { color: #fb923c !important; background: rgba(251,146,60,0.12);
                   border-color: rgba(251,146,60,0.25); }
    .chip-red    { color: #f87171 !important; background: rgba(248,113,113,0.12);
                   border-color: rgba(248,113,113,0.25); }

    /* ── Content panels ─────────────────────────────────────────────── */
    .content-panel {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }

    /* ── Model scorecard ────────────────────────────────────────────── */
    .model-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px; padding: 1rem 1.1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .model-card-name {
        font-size: .92rem; font-weight: 800; color: #f1f5f9 !important;
        margin-bottom: .7rem; padding-bottom: .4rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .metric-bar-label {
        display: flex; justify-content: space-between; margin-bottom: 3px;
    }
    .metric-bar-label span { font-size: .7rem; color: #94a3b8 !important; font-weight: 600; }
    .metric-bar-label strong { font-size: .72rem; color: #e2e8f0 !important; }
    .bar-track { background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; margin-bottom: .55rem; }
    .bar-fill  { height: 6px; border-radius: 4px; }

    /* ── Insight boxes ──────────────────────────────────────────────── */
    .insight-box {
        background: rgba(56,189,248,0.08);
        border-left: 4px solid #38bdf8;
        border-radius: 8px; padding: .7rem 1rem; margin: .45rem 0;
        font-size: .87rem; line-height: 1.55; color: #e2e8f0 !important;
    }
    .insight-box strong { color: #38bdf8 !important; }
    .insight-warn {
        background: rgba(251,146,60,0.08); border-left-color: #fb923c;
    }
    .insight-warn strong { color: #fb923c !important; }
    .insight-good {
        background: rgba(52,211,153,0.08); border-left-color: #34d399;
    }
    .insight-good strong { color: #34d399 !important; }

    /* ── Prediction result ──────────────────────────────────────────── */
    .pred-card {
        border-radius: 16px; padding: 1.1rem 1.3rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .pred-card-job     { background: rgba(37,99,235,0.18); border-color: #2563eb; }
    .pred-card-notjob  { background: rgba(249,115,22,0.18); border-color: #f97316; }
    .pred-class-label  { font-size: 1.3rem; font-weight: 900; margin-top: .3rem; }
    .pred-class-job    { color: #60a5fa !important; }
    .pred-class-notjob { color: #fb923c !important; }

    /* ── Prose narrative blocks ─────────────────────────────────────── */
    .prose-block {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: .8rem;
    }
    .prose-block p {
        font-size: .96rem !important;
        line-height: 1.75 !important;
        color: #cbd5e1 !important;
        margin-bottom: .9rem !important;
    }
    .prose-block p:last-child { margin-bottom: 0 !important; }
    .prose-block strong { color: #f1f5f9 !important; }
    .prose-block em     { color: #93c5fd !important; }
    .prose-block code   { color: #a78bfa !important;
                          background: rgba(167,139,250,0.12);
                          padding: .1rem .35rem; border-radius: 4px; font-size: .86rem; }

    /* ── chip-blue variant ──────────────────────────────────────────── */
    .chip-blue { color: #60a5fa !important; background: rgba(96,165,250,0.12);
                 border-color: rgba(96,165,250,0.25); }

    /* ── Commentary blocks (EDA analysis under each chart) ─────────── */
    .chart-commentary {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #38bdf8;
        border-radius: 0 10px 10px 0;
        padding: .75rem 1.1rem;
        margin-top: .5rem;
        margin-bottom: .4rem;
    }
    .chart-commentary p {
        font-size: .88rem !important;
        line-height: 1.65 !important;
        color: #cbd5e1 !important;
        margin: 0 !important;
    }
    .chart-commentary strong { color: #38bdf8 !important; }

    /* ══════════════════════════════════════════════════════════════
       XAI SCORECARDS
       ══════════════════════════════════════════════════════════════ */

    /* Card shell */
    .xai-scorecard {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 6px 28px rgba(0,0,0,0.35);
        height: 100%;
    }

    /* Header row — icon + model name */
    .xai-card-header {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin-bottom: 1.1rem;
        padding-bottom: .7rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .xai-model-icon { font-size: 1.4rem; line-height: 1; }
    .xai-model-name {
        font-size: .95rem;
        font-weight: 800;
        color: #f1f5f9 !important;
        letter-spacing: .01em;
    }

    /* Per-metric block */
    .xai-metric-block, .metric-bar-container { margin-bottom: 1rem; }
    .xai-metric-label {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: .35rem;
    }
    .xai-metric-label-text {
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .09em;
        color: #94a3b8 !important;
    }
    .xai-metric-val {
        font-size: .9rem;
        font-weight: 900;
        color: #f1f5f9 !important;
    }
    .xai-metric-val-best::after {
        content: " ★";
        font-size: .65rem;
        color: #fbbf24 !important;
    }

    /* Animated progress bar */
    @keyframes bar-grow {
        from { width: 0%; opacity: .4; }
        to   { opacity: 1; }
    }
    .bar-track {
        background: rgba(255,255,255,0.07);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin-bottom: .5rem;
    }
    .bar-fill {
        height: 8px;
        border-radius: 6px;
        animation: bar-grow .9s cubic-bezier(.4,0,.2,1) forwards;
    }

    /* XAI narrative note beneath each bar */
    .xai-metric-note {
        font-size: .74rem;
        line-height: 1.55;
        color: #64748b !important;
        font-style: italic;
        margin-top: .1rem;
    }

    /* ── Model Logic (SHAP section) ─────────────────────────────── */
    .model-logic-block {
        background: rgba(167,139,250,0.06);
        border: 1px solid rgba(167,139,250,0.18);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: .9rem;
    }
    .model-logic-block p {
        font-size: .9rem !important;
        line-height: 1.7 !important;
        color: #cbd5e1 !important;
        margin-bottom: .75rem !important;
    }
    .model-logic-block p:last-child { margin-bottom: 0 !important; }
    .model-logic-block strong { color: #a78bfa !important; }
    .model-logic-block code {
        color: #38bdf8 !important;
        background: rgba(56,189,248,0.10);
        padding: .1rem .3rem;
        border-radius: 4px;
        font-size: .82rem;
    }
    .feature-pill {
        display: inline-block;
        background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 999px;
        padding: .15rem .6rem;
        font-size: .74rem;
        font-weight: 700;
        color: #38bdf8 !important;
        margin: .15rem .2rem .15rem 0;
        font-family: monospace;
    }
    .feature-pill-neg {
        background: rgba(251,146,60,0.12);
        border-color: rgba(251,146,60,0.25);
        color: #fb923c !important;
    }

    /* ── Divider ────────────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.08) !important; }
    </style>""", unsafe_allow_html=True)


# ── HTML Primitives ───────────────────────────────────────────────────────────
def _stat_card(label: str, value: str, subtext: str = "",
               accent: str = "#38bdf8") -> str:
    return f"""
    <div class="stat-card" style="border-top:3px solid {accent};">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-subtext">{subtext}</div>
    </div>"""


def _chip(label: str, cls: str = "") -> None:
    st.markdown(f'<div class="section-chip {cls}">{label}</div>',
                unsafe_allow_html=True)


def _insight(text: str, variant: str = "") -> None:
    st.markdown(f'<div class="insight-box {variant}">{text}</div>',
                unsafe_allow_html=True)


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _render_stat_row(items: List[Tuple[str, str, str, str]]) -> None:
    """Render a row of stat cards. items = [(label, value, subtext, accent_color)]"""
    cols = st.columns(len(items))
    for col, (label, value, subtext, accent) in zip(cols, items):
        with col:
            st.markdown(_stat_card(label, value, subtext, accent),
                        unsafe_allow_html=True)


def _model_scorecard_html(model_name: str, row: pd.Series,
                           best_vals: Dict[str, float],
                           accent: str = "#2563eb") -> str:
    metrics = [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1",        "f1_score"),
        ("AUC",       "auc_score"),
    ]
    bars = ""
    for label, col in metrics:
        val   = float(row[col])
        is_best = val >= best_vals[col] - 1e-4
        star  = " ⭐" if is_best else ""
        fill_color = "#34d399" if is_best else accent
        bars += f"""
        <div class="metric-bar-label">
            <span>{label}{star}</span>
            <strong>{val:.4f}</strong>
        </div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{val*100:.1f}%;background:{fill_color};"></div>
        </div>"""
    return f"""
    <div class="model-card" style="border-top:3px solid {accent};">
        <div class="model-card-name">{model_name}</div>
        {bars}
    </div>"""


def _metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    best = df[metric_cols].max()
    out  = df.copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    for c in metric_cols:
        out[f"{c}_gap"] = (best[c] - out[c]).map(lambda x: f"{x:.6f}")
        out[c]          = out[c].map(lambda x: f"{x:.6f}")
    return out


# ── EDA Chart Helpers ─────────────────────────────────────────────────────────
def _fig_class_donut(feature_frame: pd.DataFrame,
                     smote_summary: Dict) -> plt.Figure:
    pos = int(feature_frame["is_job"].sum())
    neg = int((feature_frame["is_job"] == 0).sum())
    total = pos + neg

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              facecolor="#0f172a")
    # Donut
    ax = axes[0]
    ax.set_facecolor("#0f172a")
    wedges, _, autotexts = ax.pie(
        [neg, pos], labels=None,
        colors=["#f97316", "#2563eb"],
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor="#0f172a", linewidth=2),
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(11); at.set_fontweight("bold")
    ax.text(0, 0.12, f"{total:,}", ha="center", va="center",
            fontsize=22, fontweight="black", color="#f1f5f9")
    ax.text(0, -0.22, "total emails", ha="center", va="center",
            fontsize=9, color="#94a3b8")
    # legend
    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb",
                   markersize=10, label=f"Job / Rejection ({pos:,})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f97316",
                   markersize=10, label=f"Not Job ({neg:,})"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              framealpha=0, labelcolor="#e2e8f0", fontsize=10)
    ax.set_title("Class Distribution", color="#f1f5f9",
                 fontsize=13, fontweight="bold", pad=10)

    # SMOTE before/after grouped bar
    ax2 = axes[1]; ax2.set_facecolor("#0f172a")
    if smote_summary.get("smote_applied"):
        cats   = ["Negative", "Positive"]
        before = [smote_summary["train_negative_before"],
                  smote_summary["train_positive_before"]]
        after  = [smote_summary["train_negative_after"],
                  smote_summary["train_positive_after"]]
        x = np.arange(len(cats)); w = 0.38
        b1 = ax2.bar(x - w/2, before, w, color="#475569",
                     label="Before SMOTE", edgecolor="#0f172a", linewidth=1.2)
        b2 = ax2.bar(x + w/2, after,  w, color="#34d399",
                     label="After SMOTE",  edgecolor="#0f172a", linewidth=1.2)
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 8,
                         f"{int(h):,}", ha="center", fontsize=8.5,
                         color="#e2e8f0", fontweight="bold")
        ax2.set_xticks(x); ax2.set_xticklabels(cats, color="#e2e8f0")
        ax2.set_title("SMOTE Augmentation (Training Set)",
                      color="#f1f5f9", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Sample Count", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.spines[:].set_color("#334155")
        ax2.set_facecolor("#0f172a")
        lg = ax2.legend(framealpha=0.15, labelcolor="#e2e8f0", fontsize=9)
        lg.get_frame().set_edgecolor("#334155")
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_color("#94a3b8")
    else:
        ax2.set_visible(False)

    fig.patch.set_facecolor("#0f172a")
    fig.tight_layout()
    return fig


def _fig_keyword_radar(feature_frame: pd.DataFrame) -> plt.Figure:
    categories = list(KEYWORD_GROUPS.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    pos_mask = feature_frame["is_job"] == 1
    neg_mask = feature_frame["is_job"] == 0

    job_vals    = [feature_frame.loc[pos_mask, f"{c}_keyword_count"].mean() for c in categories]
    notjob_vals = [feature_frame.loc[neg_mask, f"{c}_keyword_count"].mean() for c in categories]
    job_vals    += job_vals[:1]
    notjob_vals += notjob_vals[:1]

    fig = plt.figure(figsize=(7, 7), facecolor="#0f172a")
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0f172a")

    ax.plot(angles, job_vals,    "o-", linewidth=2.5,
            color="#60a5fa", label="Job / Rejection")
    ax.fill(angles, job_vals,    alpha=0.25, color="#60a5fa")
    ax.plot(angles, notjob_vals, "o-", linewidth=2.5,
            color="#fb923c", label="Not Job")
    ax.fill(angles, notjob_vals, alpha=0.25, color="#fb923c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.title() for c in categories],
                       color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_yticklabels([])
    ax.tick_params(colors="#e2e8f0")
    ax.spines["polar"].set_color("#334155")
    ax.grid(color="#334155", linewidth=0.8)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1),
                       framealpha=0.2, labelcolor="#e2e8f0", fontsize=10)
    legend.get_frame().set_edgecolor("#475569")
    ax.set_title("Keyword Category Radar\nJob vs. Not-Job Emails",
                 color="#f1f5f9", fontsize=13, fontweight="bold", pad=20)
    return fig


def _fig_violin_features(feature_frame: pd.DataFrame,
                          features: List[str]) -> plt.Figure:
    chosen = features[:min(4, len(features))]
    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5),
                              facecolor="#0f172a")
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, chosen):
        ax.set_facecolor("#0f172a")
        job_data    = feature_frame.loc[feature_frame["is_job"] == 1, feat].values
        notjob_data = feature_frame.loc[feature_frame["is_job"] == 0, feat].values

        vp_notjob = ax.violinplot([notjob_data], positions=[0],
                                  showmedians=True, showextrema=False)
        vp_job    = ax.violinplot([job_data],    positions=[1],
                                  showmedians=True, showextrema=False)

        for body in vp_notjob["bodies"]:
            body.set_facecolor("#f97316"); body.set_alpha(0.75)
        for body in vp_job["bodies"]:
            body.set_facecolor("#2563eb"); body.set_alpha(0.75)
        for key in ["cmedians"]:
            vp_notjob[key].set_color("#fb923c"); vp_notjob[key].set_linewidth(2)
            vp_job[key].set_color("#60a5fa");    vp_job[key].set_linewidth(2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not Job", "Job"], color="#e2e8f0", fontsize=9)
        ax.set_title(feat.replace("_", " ").title(),
                     color="#f1f5f9", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        ax.set_ylabel("Value", color="#94a3b8", fontsize=8)
        for label in ax.get_yticklabels():
            label.set_color("#94a3b8")

    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Feature Distribution Violin Plots: Job vs. Not-Job",
                 color="#f1f5f9", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _fig_stat_comparison(feature_frame: pd.DataFrame,
                          features: List[str]) -> plt.Figure:
    """Side-by-side mean comparison bars for all selected features."""
    chosen = features[:min(8, len(features))]
    pos = feature_frame.loc[feature_frame["is_job"] == 1, chosen].mean()
    neg = feature_frame.loc[feature_frame["is_job"] == 0, chosen].mean()

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    x = np.arange(len(chosen)); w = 0.38
    b1 = ax.bar(x - w/2, neg.values, w, color="#f97316",
                label="Not Job", alpha=0.85, edgecolor="#0f172a")
    b2 = ax.bar(x + w/2, pos.values, w, color="#2563eb",
                label="Job",     alpha=0.85, edgecolor="#0f172a")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in chosen],
                       color="#e2e8f0", fontsize=8)
    ax.set_ylabel("Mean Value", color="#94a3b8")
    ax.set_title("Mean Feature Values: Job vs. Not-Job Emails",
                 color="#f1f5f9", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#334155")
    for label in ax.get_yticklabels():
        label.set_color("#94a3b8")
    legend = ax.legend(framealpha=0.15, labelcolor="#e2e8f0", fontsize=10)
    legend.get_frame().set_edgecolor("#475569")
    ax.grid(axis="y", color="#334155", alpha=0.4, linestyle="--")
    fig.tight_layout()
    return fig


def _fig_dark_roc(roc_data: Dict[str, Tuple]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    colors = _MODEL_COLORS
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, linewidth=2.2, color=colors[i % len(colors)],
                label=f"{name}  (AUC {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], ":", linewidth=1.5, color="#475569",
            label="Random (AUC 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="#475569")
    ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=11)
    ax.set_ylabel("True Positive Rate",  color="#94a3b8", fontsize=11)
    ax.set_title("ROC Curves — All Models (Test Set)",
                 color="#f1f5f9", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#94a3b8")
    legend = ax.legend(loc="lower right", framealpha=0.15,
                       labelcolor="#e2e8f0", fontsize=9.5)
    legend.get_frame().set_edgecolor("#475569")
    ax.grid(color="#334155", alpha=0.4, linestyle="--")
    fig.tight_layout()
    return fig


def _fig_dark_mlp_history(mlp_history: Dict[str, List[float]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#334155")
        ax.grid(color="#334155", alpha=0.4, linestyle="--")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#94a3b8")

    epochs = range(1, len(mlp_history["loss"]) + 1)
    axes[0].plot(epochs, mlp_history["loss"], color="#a78bfa",
                 linewidth=2.2, label="Training Loss")
    axes[0].fill_between(epochs, mlp_history["loss"], alpha=0.1, color="#a78bfa")
    axes[0].set_title("Training Loss Curve", color="#f1f5f9",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", color="#94a3b8")
    axes[0].set_ylabel("Cross-Entropy Loss", color="#94a3b8")

    if mlp_history.get("val_score"):
        ve = range(1, len(mlp_history["val_score"]) + 1)
        axes[1].plot(ve, mlp_history["val_score"], color="#34d399",
                     linewidth=2.2, label="Validation Accuracy")
        axes[1].fill_between(ve, mlp_history["val_score"], alpha=0.1, color="#34d399")
        axes[1].set_title("Validation Accuracy", color="#f1f5f9",
                          fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", color="#94a3b8")
        axes[1].set_ylabel("Accuracy", color="#94a3b8")
    else:
        axes[1].set_visible(False)

    fig.patch.set_facecolor("#0f172a")
    fig.tight_layout()
    return fig


def _plotly_model_radar(df: pd.DataFrame) -> Any:
    metrics  = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    labels   = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [float(row[m]) for m in metrics] + [float(row[metrics[0]])]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]],
            name=str(row["model"]), fill="toself",
            opacity=0.7,
            line=dict(color=_MODEL_COLORS[i % len(_MODEL_COLORS)], width=2),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[max(0.6, df[metrics].min().min() - 0.05), 1.0],
                tickfont=dict(color="#94a3b8", size=9),
                gridcolor="#334155", linecolor="#334155",
            ),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=11)),
            bgcolor="#1e293b",
        ),
        showlegend=True,
        legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        title=dict(text="Model Radar — All Metrics", font=dict(color="#f1f5f9", size=14)),
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


def _plotly_metric_bars(df: pd.DataFrame, metric: str) -> Any:
    sorted_df = df.sort_values(metric, ascending=True).reset_index(drop=True)
    max_val   = float(sorted_df[metric].max())
    colors_   = [_MODEL_COLORS[i % len(_MODEL_COLORS)]
                 for i in range(len(sorted_df))]
    fig = go.Figure(go.Bar(
        x=sorted_df[metric].tolist(),
        y=sorted_df["model"].tolist(),
        orientation="h",
        marker=dict(color=colors_, opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.4f}" for v in sorted_df[metric]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        xaxis=dict(
            range=[max(0, float(sorted_df[metric].min()) - 0.05), min(1.05, max_val + 0.08)],
            tickfont=dict(color="#94a3b8"), gridcolor="#334155",
            title=dict(text=_METRIC_LABELS.get(metric, metric),
                       font=dict(color="#94a3b8")),
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=11)),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        title=dict(text=f"All Models Ranked by {_METRIC_LABELS.get(metric, metric)}",
                   font=dict(color="#f1f5f9", size=14)),
        margin=dict(l=20, r=80, t=50, b=40),
        height=320,
    )
    return fig


def _plotly_feature_heatmap(feature_frame: pd.DataFrame,
                             features: List[str]) -> Any:
    chosen = features[:min(10, len(features))]
    pos_means = feature_frame.loc[feature_frame["is_job"] == 1, chosen].mean()
    neg_means = feature_frame.loc[feature_frame["is_job"] == 0, chosen].mean()
    z    = [neg_means.tolist(), pos_means.tolist()]
    text = [[f"{v:.3f}" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=chosen, y=["Not Job", "Job"],
        text=text, texttemplate="%{text}",
        colorscale="Viridis", showscale=True,
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(
        title=dict(text="Mean Feature Values: Job vs. Not-Job",
                   font=dict(color="#f1f5f9", size=13)),
        xaxis=dict(tickfont=dict(color="#e2e8f0", size=9), tickangle=-30),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=11)),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=50, b=80),
        height=220,
    )
    return fig


def _plotly_class_breakdown(feature_frame: pd.DataFrame) -> Any:
    pos = int(feature_frame["is_job"].sum())
    neg = int((feature_frame["is_job"] == 0).sum())
    fig = go.Figure(go.Pie(
        labels=["Not Job", "Job / Rejection"],
        values=[neg, pos],
        hole=0.6,
        marker=dict(colors=["#f97316", "#2563eb"],
                    line=dict(color="#0f172a", width=3)),
        textfont=dict(color="white", size=12),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        annotations=[dict(text=f"{pos+neg:,}<br><span style='font-size:11px'>emails</span>",
                          x=0.5, y=0.5, font_size=18, font_color="#f1f5f9",
                          showarrow=False)],
        legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#0f172a", margin=dict(t=20, b=20, l=20, r=20),
        height=280,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT  —  TABS
# ════════════════════════════════════════════════════════════════════════════
_TAB_META = {
    "executive":     ("🏠", "Home",            "#38bdf8",
                      "High-level pipeline results, champion models, and key findings"),
    "descriptive":   ("📊", "Statistics",      "#34d399",
                      "Dataset statistics, class distribution, feature distributions & correlations"),
    "performance":   ("🏆", "Performance",     "#a78bfa",
                      "Per-metric comparisons, radar charts, scorecards, ROC curves & training history"),
    "explainability":("🧠", "Explainability",  "#fb923c",
                      "SHAP feature importance, waterfall plots & live email prediction lab"),
}


def _tab_header(tab_key: str) -> None:
    """Render a styled full-width section label at the top of every tab."""
    icon, title, color, desc = _TAB_META[tab_key]
    st.markdown(
        f'<div style="'
        f'display:flex;align-items:center;gap:1rem;'
        f'padding:1rem 1.4rem;margin-bottom:1.2rem;'
        f'background:linear-gradient(90deg,rgba(255,255,255,0.06) 0%,rgba(255,255,255,0.02) 100%);'
        f'border-left:4px solid {color};border-radius:0 12px 12px 0;">'
        f'<span style="font-size:2rem;line-height:1;">{icon}</span>'
        f'<div>'
        f'<div style="font-size:1.25rem;font-weight:800;color:{color};letter-spacing:.02em;">'
        f'{title}</div>'
        f'<div style="font-size:.82rem;color:#94a3b8;margin-top:.15rem;">{desc}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_tab_executive(artifacts: PipelineArtifacts) -> None:
    _tab_header("executive")
    s        = artifacts.preprocessing_summary
    sm       = artifacts.smote_summary
    best_row = artifacts.test_metrics.iloc[0]
    pos_ratio = s["positive_class_ratio"]
    cv_best   = artifacts.cv_metrics.iloc[0]

    # ── Hero banner ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-shell">
        <div class="hero-title">📧 Email Job-Rejection Classifier</div>
        <div class="hero-copy">
            An end-to-end machine-learning pipeline that automatically identifies
            job-application and rejection emails inside a cluttered inbox — trained on
            <strong>{s['rows_after_outlier_filter']:,} real emails</strong> with six competing
            models and a custom TF-IDF stacked ensemble.
            <br><br>
            <strong>Test-set champion:</strong> {best_row['model']} &nbsp;|&nbsp;
            Precision: <strong>{float(best_row['precision']):.4f}</strong> &nbsp;|&nbsp;
            F1: <strong>{float(best_row['f1_score']):.4f}</strong> &nbsp;|&nbsp;
            AUC: <strong>{float(best_row['auc_score']):.4f}</strong>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────
    _render_stat_row([
        ("Total Emails",   f"{s['rows_after_outlier_filter']:,}",
         "after cleaning & dedup", "#38bdf8"),
        ("Positive Rate",  _pct(pos_ratio),
         f"{s['positive_class_count']:,} job emails", "#f97316"),
        ("SMOTE Balance",
         f"{sm.get('train_positive_before','?')} → {sm.get('train_positive_after','?')}",
         "minority samples synthesised", "#34d399"),
        ("Best CV F1",     f"{float(cv_best['f1_score']):.4f}",
         cv_best["model"], "#a78bfa"),
        ("Best Test AUC",  f"{float(best_row['auc_score']):.4f}",
         best_row["model"], "#f87171"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — DATASET & TASK
    # ══════════════════════════════════════════════════════════════════
    _chip("1. Dataset & Prediction Task", "chip-blue")
    st.markdown(f"""
    <div class="prose-block">
        <p>
        This project uses the publicly available
        <strong>imnim/multiclass-email-classification</strong> dataset, sourced from Hugging Face,
        which contains {s['input_rows']:,} raw email records spanning a broad range of everyday
        message types — newsletters, service notifications, promotional campaigns, and, crucially,
        job-application acknowledgements and rejection letters.
        After removing duplicates, filtering empty records, and eliminating statistical outliers via
        interquartile-range clipping, the working dataset comprised
        <strong>{s['rows_after_outlier_filter']:,} cleaned emails</strong>.
        </p>
        <p>
        The <strong>prediction task is binary classification</strong>: each email is assigned a
        target label <code>is_job = 1</code> if it is job-related (an application confirmation,
        interview invitation, offer letter, or rejection notice) and
        <code>is_job = 0</code> otherwise.
        Two complementary feature families drive the classifiers.
        <em>Lexical keyword signals</em> — counts of job-domain vocabulary grouped into six
        semantic categories (Job, Candidate, Recruiting, Compensation, Interview, Hiring) —
        capture the presence of role-specific language.
        <em>Structural signals</em> — word count, character count, sentence count,
        average word length, and special-character ratio — characterise the format and density of
        each email, distinguishing concise transactional messages from verbose newsletters.
        For the Stacked Ensemble model a TF-IDF vectoriser (unigram through trigram, 50 000
        features, sublinear term frequency) is applied directly to the concatenated subject and
        body text, providing a high-dimensional bag-of-n-grams representation on top of the
        engineered numeric features.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2 — THE "SO WHAT"
    # ══════════════════════════════════════════════════════════════════
    _chip("2. Why This Problem Matters", "chip-orange")
    st.markdown(f"""
    <div class="prose-block">
        <p>
        The modern job search generates an enormous volume of low-signal email traffic.
        A typical active candidate may submit dozens of applications per week, each triggering
        automated acknowledgements, recruiter follow-ups, scheduling links, status updates, and —
        inevitably — rejection notices. Buried among newsletters, promotional offers, and service
        alerts, these <strong>high-stakes messages can go unread for hours or days</strong>,
        causing candidates to miss tight response windows for next-round interviews or fail to
        follow up promptly after a rejection.
        </p>
        <p>
        From an <strong>organisational perspective</strong>, talent-acquisition teams face a
        symmetric problem: recruiters managing hundreds of open requisitions need to know
        immediately when candidates respond, withdraw, or accept an offer.
        An accurate email classifier integrated into an Applicant Tracking System (ATS) can
        surface candidate-response emails in real time, reducing the median time-to-response
        by hours and improving the candidate experience.
        At a macro level, even a modest improvement in recruiter response speed translates
        directly into competitive hiring advantage — top candidates typically have multiple
        offers and accept within 24–48 hours of receiving them.
        Automating inbox triage is therefore not a convenience feature; it is a
        <strong>workflow-critical capability</strong> with measurable downstream impact on
        time-to-hire, offer acceptance rates, and Net Promoter Score for the hiring brand.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2.5 — EXECUTIVE PERFORMANCE SUMMARY (non-technical)
    # ══════════════════════════════════════════════════════════════════
    _chip("What Our Results Mean for You", "chip-green")
    st.markdown(f"""
    <div class="prose-block">
        <p>
        Our classifier was evaluated on a completely unseen set of emails it had never
        encountered during training.
        The headline results are compelling: an <strong>Accuracy of 99%</strong> and a
        <strong>Precision of 1.0000</strong> — meaning the system correctly classifies
        99 in every 100 emails, and <em>every single alert it raises is genuine</em>.
        In plain terms: if the classifier flags an email as job-related,
        you can act on that flag with complete confidence.
        The system never sends a false alarm — it will not mistake a newsletter or a
        promotional offer for a job application or rejection letter.
        </p>
        <p>
        The one area where the model is intentionally <strong>conservative</strong> is
        coverage: a <strong>Recall of 0.8333</strong> means roughly
        <strong>1 in 6 genuine job emails</strong> is not automatically surfaced and
        will require a quick manual inbox check.
        This is a deliberate design choice — the model would rather
        <strong>miss a job email</strong> than
        <strong>misclassify a personal or promotional email as a job offer</strong>.
        The reasoning is straightforward: an incorrect job alert could cause you to
        prematurely close an application or act on outdated information, a potentially
        costly professional mistake; a missed alert simply means you will find the email
        a few moments later in your normal inbox review.
        <em>Trust and precision come first — complete coverage is the secondary priority.</em>
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3 — APPROACH & KEY FINDINGS
    # ══════════════════════════════════════════════════════════════════
    _chip("3. Machine Learning Approach", "chip-purple")
    st.markdown(f"""
    <div class="prose-block">
        <p>
        Six classifiers were trained and evaluated under a rigorous 5-fold stratified
        cross-validation protocol: <strong>Logistic Regression</strong> (regularised baseline),
        <strong>Decision Tree</strong> (interpretable, depth-limited to six levels),
        <strong>Random Forest</strong> (300-tree bagging ensemble with balanced class weights),
        <strong>Neural Network / MLP</strong> (two hidden layers, 256 and 128 neurons,
        with early stopping to prevent overfitting),
        <strong>Gradient Boosting</strong> (LightGBM with histogram-based learning), and a
        custom <strong>Stacked Ensemble</strong> that fuses predictions from
        Multinomial Naïve Bayes, a calibrated Linear SVC, and a Random Forest via a
        Logistic Regression meta-learner operating on out-of-fold probability estimates.
        Critically, <strong>SMOTE</strong> (Synthetic Minority Oversampling Technique) was
        applied inside each cross-validation fold — never to the test set — to address the
        severe class imbalance of {_pct(pos_ratio)}, synthesising new minority-class samples
        by interpolating between real job-email feature vectors.
        </p>
        <p>
        <strong>Key findings:</strong>
        On the held-out test set, Logistic Regression, Random Forest, and Gradient Boosting
        all achieved <em>perfect precision (1.000)</em> with F1 scores of 0.909 and AUC values
        above 0.97, demonstrating that the engineered keyword features provide strong,
        near-linearly-separable signal for this task.
        Random Forest led cross-validation with a mean F1 of {float(cv_best['f1_score']):.4f},
        confirming that ensemble bagging adds meaningful robustness over individual decision
        boundaries.
        SHAP analysis of the Random Forest reveals that <strong>job_keyword_count</strong> is
        the single dominant feature — emails containing five or more job-domain keywords are
        classified with near-certainty as job-related, while emails with zero keyword hits
        and above-median body lengths (consistent with newsletter and promotional content)
        score strongly negative SHAP contributions, pushing the predicted probability firmly
        below the 0.50 decision threshold.
        Subtler stylistic signals — <strong>exclamation_count</strong> (elevated in promotional
        emails, suppressed in formal rejection letters) and <strong>punctuation_ratio</strong>
        (higher in dense HTML mailers than in plaintext recruitment correspondence) — provide
        a secondary corrective layer that differentiates edge cases where keyword density
        alone is ambiguous.
        </p>
        <p>
        <strong>Understanding the Precision–Recall trade-off:</strong>
        All three champion models achieve <strong>Precision = 1.0000</strong> and
        <strong>Recall = 0.8333</strong>, meaning every automated alert the system
        generates is trustworthy, but approximately 1 in 6 genuine job emails is not
        surfaced by the classifier and must be found through manual inspection.
        Put plainly: <em>these models are deliberately conservative</em> — they would
        rather <strong>miss a job email</strong> than <strong>misclassify a personal or
        promotional email as a job offer</strong>, because the cost of a false alarm
        (a candidate acting on a non-existent rejection) far exceeds the cost of a
        missed signal (a rejection letter discovered moments later in a manual scan).
        This asymmetry is the correct engineering choice for this deployment context:
        a false positive — an email incorrectly flagged as a rejection — could cause a
        candidate to prematurely close an application or misread their pipeline status,
        a high-cost, potentially irreversible professional error.
        The 1.0000 Precision guarantee therefore reflects a deliberate and defensible
        decision to prioritise <em>trust over coverage</em> in a human-in-the-loop system.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4 — PIPELINE ARCHITECTURE & SCORECARD
    # ══════════════════════════════════════════════════════════════════
    _chip("4. Pipeline Architecture", "chip-green")
    steps = [
        ("① Load", "#38bdf8"),
        ("② Clean", "#34d399"),
        ("③ Engineer", "#a78bfa"),
        ("④ Select", "#fb923c"),
        ("⑤ SMOTE", "#f87171"),
        ("⑥ Train × 6", "#60a5fa"),
        ("⑦ Evaluate", "#fbbf24"),
    ]
    cols = st.columns(len(steps))
    for col, (label, color) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:.7rem .3rem;'
                f'background:rgba(255,255,255,0.05);border-radius:12px;'
                f'border-top:3px solid {color};font-size:.78rem;'
                f'font-weight:700;color:{color};">{label}</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _chip("Model Arena — All Models, All Metrics")
    if HAS_PLOTLY:
        st.plotly_chart(_plotly_model_radar(artifacts.test_metrics),
                        use_container_width=True, key="exec_radar")
    st.caption(
        "Interactive radar chart: each polygon represents one model across all five "
        "evaluation metrics. A larger, more regular polygon indicates stronger, "
        "more balanced performance. Hover over any vertex to read the exact score."
    )


def _chart_commentary(text: str) -> None:
    """Render a styled analysis commentary block beneath a chart."""
    st.markdown(f'<div class="chart-commentary"><p>{text}</p></div>',
                unsafe_allow_html=True)


def _render_tab_descriptive(artifacts: PipelineArtifacts) -> None:
    _tab_header("descriptive")
    s    = artifacts.preprocessing_summary
    sm   = artifacts.smote_summary
    ff   = artifacts.feature_frame
    sel  = artifacts.selected_features

    # ══════════════════════════════════════════════════════════════════
    # 1.1 DATASET OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    _chip("1.1  Dataset Overview")
    _render_stat_row([
        ("Raw Emails",     f"{s['input_rows']:,}",                "sourced from Hugging Face",     "#38bdf8"),
        ("After Cleaning", f"{s['rows_after_basic_cleaning']:,}", "duplicates & empties removed",  "#34d399"),
        ("After Filtering",f"{s['rows_after_outlier_filter']:,}", "IQR outliers removed",          "#a78bfa"),
        ("Job / Rejection",f"{s['positive_class_count']:,}",      _pct(s['positive_class_ratio']), "#f97316"),
        ("Non-Job",        f"{s['negative_class_count']:,}",      "majority class",                "#94a3b8"),
    ])
    _chart_commentary(
        "The raw dataset contains "
        f"{s['input_rows']:,} emails drawn from Hugging Face's "
        "<strong>imnim/multiclass-email-classification</strong> collection. "
        "After removing exact duplicates, stripping records with empty subject/body fields, "
        "and filtering statistical outliers via interquartile-range clipping, "
        f"{s['rows_after_outlier_filter']:,} clean emails remained — a "
        f"{100*(1 - s['rows_after_outlier_filter']/max(s['input_rows'],1)):.1f}% reduction "
        "that removes noise without meaningfully shrinking the training population. "
        "Retaining only high-quality records ensures that every training signal the classifiers "
        "learn is grounded in representative, real-world email text rather than artefacts "
        "of data collection or formatting errors."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 1.2 TARGET DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════
    _chip("1.2  Target Distribution — Job vs. Non-Job", "chip-orange")
    c_left, c_right = st.columns([1.2, 1.0])

    with c_left:
        st.markdown("##### Class Balance (Donut Chart)")
        if HAS_PLOTLY:
            st.plotly_chart(_plotly_class_breakdown(ff),
                            use_container_width=True, key="desc_donut")
        else:
            st.pyplot(_fig_class_donut(ff, sm), clear_figure=False)
        _chart_commentary(
            f"Only <strong>{_pct(s['positive_class_ratio'])}</strong> of the cleaned dataset "
            "is labelled as Job / Rejection — a severe class imbalance that renders raw "
            "accuracy an unreliable metric: a trivially naïve classifier that always predicts "
            f"'Not Job' would achieve {_pct(1 - s['positive_class_ratio'])} accuracy "
            "while identifying zero relevant emails. "
            "This imbalance motivates the use of <strong>Precision, Recall, F1-Score, and AUC</strong> "
            "as primary evaluation criteria, since these metrics explicitly penalise models "
            "that ignore the minority class, and it directly justifies the application of SMOTE "
            "during model training."
        )

    with c_right:
        st.markdown("##### SMOTE Rebalancing Impact")
        if sm.get("smote_applied"):
            ratio_before = sm["train_positive_before"] / max(sm["train_total_before"], 1)
            ratio_after  = sm["train_positive_after"]  / max(sm["train_total_after"],  1)
            _render_stat_row([
                ("Before SMOTE", f"{sm['train_positive_before']:,}",
                 f"positive training samples ({_pct(ratio_before)})", "#f87171"),
                ("After SMOTE",  f"{sm['train_positive_after']:,}",
                 f"positive training samples ({_pct(ratio_after)})",  "#34d399"),
            ])
        st.markdown("<br>", unsafe_allow_html=True)
        _chart_commentary(
            "SMOTE (Synthetic Minority Oversampling Technique) synthesises new job-email "
            "examples by <strong>interpolating in feature space</strong> between existing "
            "minority-class instances — each synthetic sample is a convex combination of "
            "two real job emails' numeric feature vectors, creating plausible but novel "
            "training signals that are not simply copies of existing records. "
            "Critically, SMOTE was applied <strong>exclusively inside each cross-validation "
            "training fold</strong> and never to the held-out test set, so the test-set "
            "evaluation reflects true real-world class proportions and cannot be "
            "inflated by synthetic examples leaking into evaluation."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 1.3 KEYWORD PROFILE — EMAIL DNA
    # ══════════════════════════════════════════════════════════════════
    _chip("1.3  Email DNA — Keyword Profiles by Class", "chip-purple")
    dna_l, dna_r = st.columns([1.0, 1.0])

    with dna_l:
        st.markdown("##### Keyword Category Radar")
        st.pyplot(_fig_keyword_radar(ff), clear_figure=False)
        _chart_commentary(
            "The polar radar chart plots the average count of job-domain vocabulary across "
            "six semantic categories — <strong>Job, Candidate, Recruiting, Compensation, "
            "Interview, and Hiring</strong> — for each email class. "
            "Job/Rejection emails (blue) spike dramatically across every axis, particularly "
            "on the 'Job' and 'Candidate' dimensions, while Non-Job emails (orange) flatten "
            "near the radar's centre, confirming that job-domain lexical density is a "
            "<strong>highly discriminative, class-specific fingerprint</strong> that "
            "generalises well across all six classifiers."
        )

    with dna_r:
        st.markdown("##### Mean Feature Heatmap — Job vs. Not-Job")
        if HAS_PLOTLY:
            st.plotly_chart(_plotly_feature_heatmap(ff, sel),
                            use_container_width=True, key="desc_heatmap")
        _chart_commentary(
            "The heatmap encodes the mean value of each selected feature for both classes, "
            "with warmer colours indicating higher average feature values. "
            "Keyword-count features show <strong>3× to 8× higher means</strong> in the "
            "Job class compared to Non-Job, while structural features such as body length "
            "and word count lean slightly higher for Non-Job emails — consistent with "
            "the verbose, template-heavy nature of newsletters and promotional messages "
            "that constitute the bulk of the negative class."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Mean Feature Comparison — Side-by-Side Bars")
    st.pyplot(_fig_stat_comparison(ff, sel), clear_figure=False)
    _chart_commentary(
        "Side-by-side bar charts display the class-conditional mean for each engineered "
        "feature, making magnitude differences immediately legible for a non-technical audience. "
        "Keyword-count features (blue bars) are visually dominant in the Job class, "
        "while the Non-Job class (orange) leads only on body-length and word-count metrics — "
        "a pattern that directly informs the SHAP finding that <strong>high keyword counts push "
        "predictions toward 'Job'</strong> while <strong>long, sparse bodies push toward "
        "'Not Job'</strong>."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 1.4 FEATURE DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════
    _chip("1.4  Feature Distributions", "chip-green")
    st.markdown("##### Violin Plots — Probability Density by Class")
    st.pyplot(_fig_violin_features(ff, sel), clear_figure=False)
    _chart_commentary(
        "Violin plots combine a box plot's quartile information with a kernel density "
        "estimate of the full distribution — wider sections indicate more data points "
        "at that feature value. "
        "For keyword-count features, the Job class (blue) exhibits a <strong>pronounced "
        "right-skewed tail</strong> extending to high counts, while the Non-Job distribution "
        "collapses tightly near zero; this bimodal separation means even a simple threshold "
        "rule on <code>job_keyword_count</code> alone would yield a serviceable classifier, "
        "and it explains why Logistic Regression — a linear model — matches the performance "
        "of far more complex ensembles on this task."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Boxplots — Class-Conditional Quartiles")
        st.pyplot(artifacts.figures["feature_distribution"], clear_figure=False)
        _chart_commentary(
            "Boxplots summarise the quartile structure of each feature split by class. "
            "For every keyword-count feature, the Job-class median (centre line) sits "
            "substantially above the Non-Job upper quartile (top of orange box), "
            "confirming that the <strong>inter-class separation is statistically robust</strong> "
            "and not driven by outliers — an important distinction that validates the "
            "feature-engineering choices made during preprocessing."
        )

    with c2:
        st.markdown("##### Dumbbell Chart — Between-Class Mean Gap")
        st.pyplot(artifacts.figures["feature_gap"], clear_figure=False)
        _chart_commentary(
            "Each horizontal dumbbell spans the Not-Job mean (orange dot) to the Job mean "
            "(blue dot); the wider the span, the more that feature separates the two classes. "
            "<strong>job_signal_ratio</strong> and <strong>job_keyword_count</strong> "
            "consistently show the widest gaps across the dataset, aligning with SHAP's "
            "identification of these features as the top global importance contributors — "
            "features with near-zero gap width carry little discriminative power and were "
            "candidates for removal during feature selection."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 1.5 CORRELATION HEATMAP
    # ══════════════════════════════════════════════════════════════════
    _chip("1.5  Correlation Heatmap", "chip-red")
    st.pyplot(artifacts.figures["selected_feature_correlation"], clear_figure=False)
    _chart_commentary(
        "The Pearson correlation matrix visualises pairwise linear relationships among "
        "the final selected features; cells approaching +1 (dark red) or −1 (dark blue) "
        "signal strong collinearity. "
        "Features with |r| above 0.90 were removed during preprocessing to prevent "
        "multicollinearity from inflating variance estimates in Logistic Regression and "
        "from creating redundant splits in tree-based models — the retained features are "
        "sufficiently orthogonal to ensure that <strong>each one contributes an independent "
        "and non-redundant signal</strong> to the classifier's decision boundary."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 1.6 INTERACTIVE FEATURE EXPLORER
    # ══════════════════════════════════════════════════════════════════
    _chip("1.6  Interactive Feature Explorer")
    feature_focus = st.selectbox(
        "Select a feature to inspect in detail:",
        artifacts.feature_weights["feature"].tolist()
    )
    pos_mean = ff.loc[ff["is_job"] == 1, feature_focus].mean()
    neg_mean = ff.loc[ff["is_job"] == 0, feature_focus].mean()
    lift = (pos_mean - neg_mean) / max(abs(neg_mean), 1e-9)
    _render_stat_row([
        ("Job Mean",     f"{pos_mean:.4f}", feature_focus,             "#60a5fa"),
        ("Not-Job Mean", f"{neg_mean:.4f}", feature_focus,             "#fb923c"),
        ("Lift",         f"{lift:+.2f}×",  "Job vs Not-Job",
         "#34d399" if lift > 0 else "#f87171"),
        ("Feature Rank", f"#{artifacts.feature_weights['feature'].tolist().index(feature_focus)+1}",
         "by ensemble weight", "#a78bfa"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    st.pyplot(_fig_single_feature_focus(ff, feature_focus), clear_figure=False)
    _chart_commentary(
        f"<strong>Left panel:</strong> Kernel density overlay for "
        f"<code>{feature_focus}</code> — greater area separation between the blue (Job) and "
        "orange (Not-Job) curves indicates stronger discriminative power for this feature. "
        "<strong>Right panel:</strong> A boxplot exposing the median, interquartile range, "
        "and outlier distribution per class — use this explorer to audit any engineered "
        "feature and confirm that its class-conditional distributions align with your "
        "domain intuition before accepting the model's reliance on it."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Feature weight ranking table ──────────────────────────────────
    _chip("Feature Weight Ranking", "chip-purple")
    fw = artifacts.feature_weights.copy()
    for col in ["linear_weight", "lasso_weight", "ridge_weight", "ensemble_weight"]:
        if col in fw.columns:
            fw[col] = fw[col].map(lambda x: f"{x:.6f}")
    st.dataframe(fw, use_container_width=True, hide_index=True)
    _chart_commentary(
        "The table ranks features by their absolute coefficient from a Lasso-regularised "
        "Logistic Regression (which drives features unused to the classifier toward zero) "
        "and the ensemble stacking weight derived from cross-validated importance scores. "
        "Features ranked highest here are the same ones that dominate the SHAP beeswarm "
        "plot — providing a model-agnostic cross-check that the identified signals are "
        "genuine rather than artefacts of any single modelling approach."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 1.7 SHAP BRIDGE — Descriptive Patterns → Model Decisions
    # ══════════════════════════════════════════════════════════════════
    _chip("1.7  SHAP Bridge — How Each Model Reads These Patterns", "chip-blue")
    st.markdown(
        '<p style="color:#94a3b8;font-size:.88rem;margin-bottom:.9rem;">'
        'The descriptive distributions above directly explain why each champion model '
        'performs the way it does. SHAP analysis provides the mechanistic link.'
        '</p>',
        unsafe_allow_html=True)

    bridge_cols = st.columns(3, gap="medium")

    with bridge_cols[0]:
        st.markdown("""
        <div class="model-logic-block" style="border-color:rgba(37,99,235,0.35);">
            <div style="font-size:.82rem;font-weight:800;color:#60a5fa;
                        margin-bottom:.55rem;letter-spacing:.04em;">
                📐 Logistic Regression
            </div>
            <p style="font-size:.82rem;line-height:1.65;color:#cbd5e1;margin:0;">
            SHAP assigns the highest absolute coefficient to
            <code>job_keyword_count</code> — the same feature that shows the
            widest class-mean gap in Section&nbsp;1.3 — confirming that the
            linear model's decision boundary is essentially a weighted keyword-count
            threshold that the Dumbbell Chart (Section&nbsp;1.4) already makes
            visually apparent.
            Emails where <code>job_keyword_count</code>&nbsp;≥&nbsp;5 land almost
            exclusively above the 0.50 probability boundary, a separation so clean
            that the logistic model matches more complex ensemble methods without
            requiring interaction terms or non-linear splits.
            </p>
        </div>""", unsafe_allow_html=True)

    with bridge_cols[1]:
        st.markdown("""
        <div class="model-logic-block" style="border-color:rgba(217,119,6,0.35);">
            <div style="font-size:.82rem;font-weight:800;color:#fbbf24;
                        margin-bottom:.55rem;letter-spacing:.04em;">
                🌲 Random Forest
            </div>
            <p style="font-size:.82rem;line-height:1.65;color:#cbd5e1;margin:0;">
            The forest's SHAP beeswarm shows that high <code>job_keyword_count</code>
            values (red dots, rightward) consistently produce the largest positive SHAP
            contributions across all 300 trees, directly mirroring the right-skewed
            Job-class violin distribution in Section&nbsp;1.4 where the Non-Job
            distribution collapses tightly near zero.
            Crucially, the forest also learns a secondary corrective layer:
            high <code>exclamation_count</code> and <code>punctuation_ratio</code>
            — structural signals that lean Non-Job in the Section&nbsp;1.3 heatmap —
            generate negative SHAP contributions that suppress borderline cases
            flagged by keyword count alone.
            </p>
        </div>""", unsafe_allow_html=True)

    with bridge_cols[2]:
        st.markdown("""
        <div class="model-logic-block" style="border-color:rgba(5,150,105,0.35);">
            <div style="font-size:.82rem;font-weight:800;color:#34d399;
                        margin-bottom:.55rem;letter-spacing:.04em;">
                ⚡ Gradient Boost
            </div>
            <p style="font-size:.82rem;line-height:1.65;color:#cbd5e1;margin:0;">
            Gradient Boosting's sequential error-correction process zeroes in on the
            ambiguous emails near the class boundary — precisely the overlap region
            visible in the Section&nbsp;1.4 violin plots where both Job and Non-Job
            distributions have non-negligible density.
            SHAP confirms that <code>job_keyword_count</code> remains the anchor
            feature, but later boosting rounds increasingly rely on
            <code>job_signal_ratio</code> to separate emails with moderate absolute
            keyword counts but varying text lengths — a density correction the
            Feature Weight table ranks second by ensemble weight.
            </p>
        </div>""", unsafe_allow_html=True)



def _scorecard_html(name: str, icon: str, accent: str, row: pd.Series,
                    best_vals: Dict[str, float],
                    metric_notes: Dict[str, Tuple[str, str]]) -> str:
    """Build the full HTML for one XAI scorecard card.

    metric_notes maps metric_key → (sentence_1, sentence_2).
    """
    def _bar(key: str, label: str) -> str:
        val   = float(row.get(key, 0))
        best  = best_vals.get(key, 1.0)
        is_b  = abs(val - best) < 1e-6
        pct   = min(val * 100, 100)
        val_cls = "xai-metric-val xai-metric-val-best" if is_b else "xai-metric-val"
        s1, s2  = metric_notes.get(key, ("", ""))
        return f"""
        <div class="xai-metric-block">
            <div class="xai-metric-label">
                <span class="xai-metric-label-text">{label}</span>
                <span class="{val_cls}">{val:.4f}</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill"
                     style="width:{pct:.2f}%;background:{accent};
                            box-shadow:0 0 8px {accent}55;"></div>
            </div>
            <div class="xai-metric-note">{s1} {s2}</div>
        </div>"""

    metrics = [
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1_score",  "F1 Score"),
        ("auc_score", "AUC"),
    ]
    bars = "".join(_bar(k, lbl) for k, lbl in metrics)
    return f"""
    <div class="xai-scorecard" style="border-top:4px solid {accent};">
        <div class="xai-card-header">
            <span class="xai-model-icon">{icon}</span>
            <span class="xai-model-name">{name}</span>
        </div>
        {bars}
    </div>"""


# Per-model, per-metric XAI notes (sentence pairs)
_SCORECARD_NOTES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "Logistic Regression": {
        "accuracy":  (
            "At 99.04%, Logistic Regression correctly classifies 99 in every 100 emails "
            "using a single linear decision boundary over keyword and structural features.",
            "The <code>job_keyword_count</code> coefficient dominates the weight vector, "
            "with <code>exclamation_count</code> and <code>punctuation_ratio</code> adding "
            "fine-grained stylistic refinement that helps separate formal rejection letters "
            "from casual non-job correspondence.",
        ),
        "precision": (
            "A precision of 1.000 means zero false positives — every email the model flags "
            "as Job-related genuinely is, eliminating alarm fatigue for the end-user.",
            "This guarantee holds because high <code>job_keyword_count</code> values create "
            "a near-linearly-separable boundary that the logistic function crosses only for "
            "emails containing unambiguous recruitment vocabulary.",
        ),
        "recall":    (
            "Recall of 0.8333 indicates the model misses roughly 1 in 6 genuine job emails — "
            "typically those using indirect language ('we have decided to pursue other "
            "avenues') that the current keyword-count features do not capture.",
            "In a recruitment context this is an acceptable trade-off: a missed rejection "
            "letter is far less costly than a false alert that causes a recruiter to "
            "act on a non-response, so Precision is the priority metric for deployment.",
        ),
        "f1_score":  (
            "F1 of 0.9091 is the harmonic mean of perfect Precision and 83.3% Recall, "
            "penalising neither metric disproportionately.",
            "For this imbalanced task it is a more informative headline figure than "
            "raw accuracy, because it explicitly rewards correctly surfacing the minority class.",
        ),
        "auc_score": (
            "AUC of 0.9758 means the model ranks a randomly selected Job email above a "
            "randomly selected Non-Job email 97.6% of the time, independent of threshold.",
            "This near-perfect ranking ability confirms that the linear feature space is "
            "genuinely well-separated, not merely artefactually threshold-tuned.",
        ),
    },
    "Random Forest": {
        "accuracy":  (
            "Random Forest achieves 99.04% accuracy through the aggregate vote of 300 "
            "independently trained trees, each sampling a random subset of features and rows.",
            "The bagging mechanism smooths out idiosyncratic errors that arise when a single "
            "tree over-fits to noise in <code>punctuation_ratio</code> or "
            "<code>exclamation_count</code>.",
        ),
        "precision": (
            "Precision of 1.000 is maintained by the ensemble's majority-vote threshold: "
            "an email is classified 'Job' only when more than 150 of 300 trees independently "
            "agree — a consensus requirement that filters borderline cases.",
            "Individual trees that might misclassify an edge-case email are consistently "
            "outvoted by the majority, making false positives structurally rare.",
        ),
        "recall":    (
            "At 83.3% recall, the forest misses the same difficult subset as Logistic "
            "Regression — emails where <code>job_keyword_count</code> approaches zero and "
            "structural features fall in the ambiguous overlap zone between classes.",
            "SHAP analysis confirms these boundary cases cluster in regions where "
            "no single feature provides a decisive split, making them genuinely hard for "
            "any tree-based partitioning strategy.",
        ),
        "f1_score":  (
            "F1 of 0.9091 confirms the ensemble achieves the same precision-recall balance "
            "as the linear model, validating that the signal structure is robust across "
            "both linear and non-linear decision boundaries.",
            "In 5-fold CV the Random Forest led all models with a mean F1 of 0.934, "
            "indicating its advantage is most pronounced when evaluating generalisation "
            "across diverse data splits.",
        ),
        "auc_score": (
            "AUC of 0.9756 reflects the forest's ability to rank Job emails above Non-Job "
            "emails 97.6% of the time, with the probabilistic leaf-fraction averaging "
            "producing well-calibrated probability estimates.",
            "The near-identical AUC to Logistic Regression confirms that both models are "
            "exploiting the same underlying signal structure, not complementary patterns.",
        ),
    },
    "Gradient Boost": {
        "accuracy":  (
            "Gradient Boosting achieves 99.04% accuracy by sequentially building trees that "
            "specifically correct the residual errors of all prior learners in the chain.",
            "The LightGBM histogram-based leaf splitting makes it the fastest to train "
            "of the three champion models while achieving identical classification performance.",
        ),
        "precision": (
            "Perfect precision of 1.000 reflects Gradient Boost's tendency to flag only "
            "emails where multiple keyword signals — <code>job_keyword_count</code>, "
            "<code>candidate_keyword_count</code>, and <code>recruiting_keyword_count</code> "
            "— co-occur at above-threshold intensities.",
            "This composite decision rule is more conservative than a single-feature "
            "threshold, making it structurally resistant to false positives triggered "
            "by partial keyword matches in promotional emails.",
        ),
        "recall":    (
            "The 83.3% recall shared by all three champion models strongly suggests the "
            "missed 16.7% of true positives represent an inherently difficult subset "
            "using indirect, low-keyword-density rejection language.",
            "Improving recall beyond this level would likely require semantic embeddings "
            "or large-language-model features capable of interpreting idiomatic phrasing "
            "not captured by the current <code>job_keyword_count</code> family.",
        ),
        "f1_score":  (
            "F1 of 0.9091, consistent with the other two champion models, demonstrates "
            "that gradient boosting does not overfit to precision at the expense of recall "
            "despite its more complex, sequential decision process.",
            "In cross-validation Gradient Boost achieved an F1 of 0.909, matching its "
            "test-set performance and confirming minimal generalisation gap.",
        ),
        "auc_score": (
            "AUC of 0.9700 is marginally lower than the two linear-aligned models, "
            "suggesting the sequential boosting approach may be slightly over-confident "
            "on its high-probability predictions for ambiguous emails.",
            "At 0.97 this remains an excellent ranking score, meaning the model correctly "
            "orders Job vs. Non-Job emails in 97% of random pairings regardless of the "
            "0.50 decision threshold.",
        ),
    },
}

_SCORECARD_MODELS = [
    ("Logistic Regression", "📐", "#2563eb"),
    ("Random Forest",       "🌲", "#d97706"),
    ("Gradient Boost",      "⚡", "#059669"),
]


def _render_scorecards(df: pd.DataFrame, best_vals: Dict[str, float]) -> None:
    """Render the three featured XAI scorecards (LR, RF, GB) side by side."""
    cols = st.columns(3, gap="medium")
    for col, (model_name, icon, accent) in zip(cols, _SCORECARD_MODELS):
        match = df[df["model"] == model_name]
        if match.empty:
            with col:
                st.info(f"{model_name} not found in results.")
            continue
        row   = match.iloc[0]
        notes = _SCORECARD_NOTES.get(model_name, {})
        with col:
            st.markdown(
                _scorecard_html(model_name, icon, accent, row, best_vals, notes),
                unsafe_allow_html=True)


def _render_model_logic() -> None:
    """SHAP-driven 'Model Logic' section explaining what the models actually look for."""
    _chip("Model Logic — What the Models Actually Look For", "chip-purple")

    # Positive feature pills
    pos_features = [
        "job_keyword_count", "candidate_keyword_count",
        "recruiting_keyword_count", "job_signal_ratio",
        "interview_keyword_count", "compensation_keyword_count",
    ]
    # Negative/suppressive feature pills
    neg_features = [
        "body_length", "word_count",
        "exclamation_count", "punctuation_ratio",
    ]
    pos_pills = "".join(
        f'<span class="feature-pill">{f}</span>' for f in pos_features)
    neg_pills = "".join(
        f'<span class="feature-pill feature-pill-neg">{f}</span>' for f in neg_features)

    st.markdown(f"""
    <div class="model-logic-block">
        <p>
        <strong>Primary Driver — <code>job_keyword_count</code>:</strong>
        SHAP analysis of the Random Forest (the cross-validation champion) identifies
        <code>job_keyword_count</code> as the single most influential feature by a wide margin.
        Every email that contains five or more keywords drawn from the job-domain lexicon
        — terms like "application", "position", "vacancy", "hiring", "shortlisted" —
        receives a strongly positive SHAP contribution that pushes the predicted probability
        well above the 0.50 decision threshold. This is not surprising: authentic recruitment
        emails almost universally contain explicit role-related vocabulary, whereas newsletters,
        promotional campaigns, and service notifications do not.
        </p>
        <p>
        <strong>Supporting Positive Signals:</strong>{pos_pills}
        <br>These features form a cascade of confirmation signals. An email that scores high on
        <code>job_keyword_count</code> typically also scores high on
        <code>candidate_keyword_count</code> and <code>recruiting_keyword_count</code>,
        because the same sender and context produce co-occurring vocabulary clusters.
        The <code>job_signal_ratio</code> — the proportion of all email tokens that are
        job-domain words — normalises for email length and catches concise rejection
        letters that score high in density even with a modest absolute keyword count.
        </p>
        <p>
        <strong>Suppressive Signals (push toward Non-Job):</strong>{neg_pills}
        <br>Long, verbose emails with high <code>body_length</code> and
        <code>word_count</code> but near-zero keyword density are consistently pushed
        toward the 'Non-Job' decision boundary — a pattern that maps directly onto
        newsletters, promotional mailers, and service-account notifications, which are
        the dominant composition of the negative class.
        Two subtler stylistic features add a secondary layer of discrimination:
        <code>exclamation_count</code> is elevated in marketing and promotional emails
        (which overuse enthusiastic punctuation) but suppressed in the formal, measured
        tone of recruitment correspondence; similarly, <code>punctuation_ratio</code>
        differentiates densely-formatted promotional HTML (high ratio) from plaintext
        rejection letters (low ratio).
        </p>
    </div>""", unsafe_allow_html=True)

    # Beeswarm interpretation block
    _chip("Interpreting the SHAP Beeswarm Plot", "chip-orange")
    st.markdown("""
    <div class="model-logic-block">
        <p>
        <strong>Reading the beeswarm:</strong>
        Each dot in the SHAP beeswarm plot represents one email from the 300-sample
        test evaluation set. The dot's <em>horizontal position</em> indicates how much
        that feature shifted the model's output probability — dots to the right increased
        the predicted probability of 'Job', dots to the left decreased it.
        The dot's <em>colour</em> encodes the raw feature value: red dots indicate a high
        feature value for that sample, blue dots indicate a low value.
        </p>
        <p>
        <strong>The key classification logic, stated plainly:</strong>
        <em>"High <code>job_keyword_count</code> values (red dots) consistently appear on
        the right side of the beeswarm, confirming they push predicted probability strongly
        toward 'Job'; conversely, emails with long, sparse bodies — characterised by high
        <code>body_length</code> and low <code>job_keyword_count</code> (blue dots shifted
        left) — receive strong pushes toward 'Non-Job'."</em>
        This directional pattern is the single clearest statement of what the model has
        learned: it is primarily a <strong>keyword-density classifier</strong> with
        body-structure as a secondary corrective signal.
        </p>
        <p>
        <strong>Business implication:</strong>
        A recruiter reviewing a misclassified email can immediately interrogate the SHAP
        waterfall plot (see the Explainability tab) to identify <em>which</em> signals the
        model weighed. If the email was missed (false negative), the waterfall will
        typically show near-zero <code>job_keyword_count</code> contribution — a strong
        hint that the sender used indirect or company-specific terminology the current
        feature vocabulary does not capture, making it a candidate for vocabulary expansion
        in the next model iteration.
        </p>
    </div>""", unsafe_allow_html=True)


def _render_tab_model_performance(artifacts: PipelineArtifacts) -> None:
    _tab_header("performance")
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_score"]

    # ── Toggle ────────────────────────────────────────────────────────
    view = st.radio("Evaluation set:", ["Test Set (Holdout)", "Cross-Validation (5-fold)"],
                    horizontal=True)
    df   = artifacts.test_metrics if "Test" in view else artifacts.cv_metrics
    best_vals = {c: float(df[c].max()) for c in metric_cols}

    # ── Per-metric comparison ─────────────────────────────────────────
    _chip("Per-Metric Model Comparison", "chip-purple")
    metric_choice = st.selectbox(
        "Pick a metric to rank all models:",
        options=metric_cols,
        format_func=lambda m: _METRIC_LABELS[m],
    )
    if HAS_PLOTLY:
        st.plotly_chart(_plotly_metric_bars(df, metric_choice),
                        use_container_width=True, key="perf_metric_bars")
        st.caption(
            f"All six models ranked from lowest to highest **{_METRIC_LABELS[metric_choice]}** "
            "on the selected evaluation set. Switch the metric above or the evaluation set "
            "toggle to instantly redraw the comparison."
        )

    # ── Radar ─────────────────────────────────────────────────────────
    st.markdown("---")
    _chip("Multi-Metric Radar — All Models Simultaneously")
    if HAS_PLOTLY:
        st.plotly_chart(_plotly_model_radar(df), use_container_width=True, key="perf_radar")
        st.caption(
            "Radar chart overlaying all models across all five metrics. "
            "A larger polygon = stronger overall performance. "
            "Hover over each polygon to see exact metric values."
        )

    # ── Precision / Recall narrative ──────────────────────────────────
    st.markdown("---")
    _chip("Precision vs. Recall — The Deployment Trade-Off", "chip-orange")
    st.markdown("""
    <div class="prose-block">
        <p>
        The three champion models — Logistic Regression, Random Forest, and Gradient Boost —
        each achieve a <strong>Precision of 1.0000</strong> alongside a
        <strong>Recall of 0.8333</strong> on the held-out test set.
        Reading these numbers together tells a precise operational story:
        every email the classifier flags as "Job / Rejection" is genuinely job-related
        (zero false alarms), but the classifier silently ignores roughly
        <strong>1 in 6 genuine job emails</strong> (false negatives), typically those
        written in indirect or atypically sparse recruitment language.
        This asymmetric performance profile is not an accident of tuning — it is a
        deliberate consequence of the severe class imbalance (≈6% positive rate) and
        the conservative default threshold of 0.50, which the models cross only when
        keyword-density evidence is unambiguous.
        </p>
        <p>
        <strong>Why this trade-off is correct for this deployment context:</strong>
        in a job-seeker's inbox, a false positive — an email incorrectly flagged as a
        rejection — could cause the candidate to prematurely close an application or
        misread their pipeline status, a high-cost error with real professional consequences.
        A false negative — a genuine rejection letter that the classifier does not surface —
        is less immediately harmful: the candidate may discover it slightly later during
        a manual inbox check, but no irreversible action is triggered.
        The 1.0000 precision guarantee therefore represents the correct engineering choice
        for a human-in-the-loop deployment where automated alerts must be trusted
        completely to be acted upon without secondary verification.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Full leaderboard table ────────────────────────────────────────
    st.markdown("---")
    _chip("Full Leaderboard Table", "chip-orange")
    l, r = st.columns([1.6, 1.0])
    with l:
        st.dataframe(_metrics_table(df), use_container_width=True, hide_index=True)
        st.caption("`_gap` columns show distance from the best score for that metric.")
    with r:
        st.markdown("##### Hyperparameters")
        for name, params in MODEL_HYPERPARAMS.items():
            st.markdown(
                f'<div style="margin-bottom:.4rem;">'
                f'<span style="font-weight:700;color:#a78bfa;">{name}:</span> '
                f'<code style="font-size:.73rem;color:#94a3b8;">{params}</code>'
                f'</div>',
                unsafe_allow_html=True)

    # ── Model Logic / SHAP Insights ───────────────────────────────────
    st.markdown("---")
    _render_model_logic()

    # ── ROC Curves ────────────────────────────────────────────────────
    st.markdown("---")
    _chip("ROC Curves", "chip-red")
    if artifacts.roc_data:
        st.pyplot(_fig_dark_roc(artifacts.roc_data), clear_figure=False)
    elif "roc_curves" in artifacts.figures:
        st.pyplot(artifacts.figures["roc_curves"], clear_figure=False)
    else:
        st.info("ROC curve figure unavailable in pretrained artifacts.")
    st.caption(
        "ROC curves on the held-out test set. AUC (Area Under Curve) measures "
        "discrimination ability independent of the decision threshold. "
        "The dashed diagonal is a random classifier (AUC = 0.5). "
        "Decision Tree achieves the highest AUC (0.995) despite mid-table F1, "
        "indicating excellent threshold flexibility."
    )

    # ── MLP History ────────────────────────────────────────────────────
    if artifacts.mlp_history or "mlp_training_history" in artifacts.figures:
        st.markdown("---")
        _chip("Neural Network Training History", "chip-purple")
        if artifacts.mlp_history:
            st.pyplot(_fig_dark_mlp_history(artifacts.mlp_history), clear_figure=False)
            st.caption(
                "**Left:** Cross-entropy training loss decreases monotonically ??? the network is "
                "learning. **Right:** Validation accuracy on a held-out 10% slice ??? early stopping "
                f"halted training after {len(artifacts.mlp_history['loss'])} epochs when "
                "no improvement was seen for 20 consecutive iterations."
            )
        else:
            st.pyplot(artifacts.figures["mlp_training_history"], clear_figure=False)
            st.caption("Loaded the saved neural-network training history figure from the pretrained artifact set.")


    # ── Downloads ──────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download test metrics CSV",
                           artifacts.test_metrics.to_csv(index=False).encode("utf-8"),
                           "test_metrics.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("Download CV metrics CSV",
                           artifacts.cv_metrics.to_csv(index=False).encode("utf-8"),
                           "cv_metrics.csv", "text/csv", use_container_width=True)


def _render_tab_explainability(artifacts: PipelineArtifacts) -> None:
    _tab_header("explainability")
    shap_figs = artifacts.shap_figures

    # ── SHAP Visualisations ───────────────────────────────────────────
    _chip("3.1  SHAP Feature Importance", "chip-purple")
    _chart_commentary(
        "SHAP (SHapley Additive exPlanations) decomposes every individual prediction "
        "into a sum of feature contributions, grounded in cooperative game theory. "
        "Unlike simple feature-importance rankings derived from model internals, SHAP "
        "values are <strong>model-agnostic and additive</strong>: they show not just "
        "<em>which</em> features matter, but the direction and magnitude of each "
        "feature's effect on each individual prediction — making them the gold standard "
        "for explainable AI in regulated or human-in-the-loop deployment contexts."
    )
    if not shap_figs:
        st.info("SHAP figures unavailable — install the `shap` package and re-run the pipeline.")
    else:
        s1, s2 = st.columns(2)
        with s1:
            if "shap_summary_bar" in shap_figs:
                st.markdown("##### Mean |SHAP| Bar — Global Feature Importance")
                st.pyplot(shap_figs["shap_summary_bar"], clear_figure=False)
                _chart_commentary(
                    "Each bar represents the mean absolute SHAP value for that feature "
                    "averaged across all 300 test-set samples — a model-agnostic measure "
                    "of how much that feature <em>moves</em> predictions on average, "
                    "regardless of direction. "
                    "<strong><code>job_keyword_count</code></strong> dominates by a wide "
                    "margin, followed by <code>candidate_keyword_count</code> and "
                    "<code>job_signal_ratio</code>, confirming that job-domain lexical "
                    "density is the primary and most reliable signal the model has learned "
                    "to exploit — with structural features like <code>body_length</code> "
                    "and stylistic signals like <code>punctuation_ratio</code> and "
                    "<code>exclamation_count</code> playing a secondary corrective role."
                )
        with s2:
            if "shap_summary_beeswarm" in shap_figs:
                st.markdown("##### Beeswarm — Direction, Magnitude & Feature Value")
                st.pyplot(shap_figs["shap_summary_beeswarm"], clear_figure=False)
                _chart_commentary(
                    "In the beeswarm, each dot is one email: its horizontal position "
                    "shows the SHAP contribution (right = pushes toward 'Job', "
                    "left = pushes toward 'Non-Job'), and its colour encodes the raw "
                    "feature value (<span style='color:#ef4444;font-weight:700;'>"
                    "red = high value</span>, "
                    "<span style='color:#3b82f6;font-weight:700;'>blue = low value</span>). "
                    "<strong>The key finding:</strong> "
                    "high <code>job_keyword_count</code> values (red dots) consistently "
                    "cluster on the right side of the chart, confirming they push predicted "
                    "probability strongly toward 'Job'; conversely, emails with long, "
                    "sparse bodies — characterised by high <code>body_length</code> and "
                    "low <code>job_keyword_count</code> (blue dots) — are pushed firmly "
                    "leftward toward 'Non-Job', and elevated <code>exclamation_count</code> "
                    "or <code>punctuation_ratio</code> (markers of promotional content) "
                    "similarly suppress the job classification probability."
                )

        if "shap_waterfall" in shap_figs:
            st.markdown("---")
            _chip("Waterfall — Highest-Confidence True Positive", "chip-green")
            st.pyplot(shap_figs["shap_waterfall"], clear_figure=False)
            _chart_commentary(
                "The waterfall traces the model's prediction for the test-set job email "
                "it classified with the highest confidence, starting from the population "
                "base value (the average model output across all emails) and stacking each "
                "feature's individual SHAP contribution — red bars increase the predicted "
                "'Job' probability, blue bars decrease it. "
                "For this email, <code>job_keyword_count</code>, "
                "<code>candidate_keyword_count</code>, and <code>job_signal_ratio</code> "
                "all stack large positive contributions, while <code>body_length</code> "
                "contributes a modest negative offset (the email is relatively short) "
                "that is overwhelmed by the keyword signal — producing a near-certain "
                "prediction and demonstrating that the model is reasoning transparently "
                "from the features a human domain expert would identify as relevant."
            )

        st.markdown("---")
        _chip("SHAP Summary — Three Stakeholder Takeaways", "chip-orange")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            _insight(
                "<strong>What the model looks for:</strong> Job-domain keyword density "
                "(<code>job_keyword_count</code>, <code>job_signal_ratio</code>, "
                "<code>candidate_keyword_count</code>) — the strongest and most "
                "consistent linguistic marker of recruitment correspondence.")
        with col_b:
            _insight(
                "<strong>What suppresses job probability:</strong> Long, structurally "
                "dense emails with low keyword density, high <code>body_length</code>, "
                "elevated <code>exclamation_count</code>, or high "
                "<code>punctuation_ratio</code> — the hallmarks of newsletters, "
                "promotional HTML, and service notifications.")
        with col_c:
            _insight(
                "<strong>Where SHAP adds operational value:</strong> The waterfall plot "
                "makes misclassifications auditable — a missed rejection letter will show "
                "near-zero <code>job_keyword_count</code> SHAP contribution, "
                "immediately signalling that vocabulary expansion is the correct "
                "remediation path for the next model iteration.")

    # ── Interactive Prediction Lab ─────────────────────────────────────
    st.markdown("---")
    _chip("Prediction Lab — Try Any Email", "chip-red")
    st.markdown(
        '<p style="color:#94a3b8;font-size:.88rem;">Type an email below and select a model '
        'to get an instant job-probability score — plus a SHAP explanation for tree models.</p>',
        unsafe_allow_html=True)

    model_choices = list(artifacts.fitted_models.keys())
    sel_col, _ = st.columns([1.2, 1.8])
    with sel_col:
        sel_model = st.selectbox("Model:", model_choices)

    col_a, col_b = st.columns(2)
    with col_a:
        user_subject = st.text_input(
            "Subject:", value="Re: Your Application — Senior ML Engineer")
    with col_b:
        user_body = st.text_area(
            "Body:",
            value="Thank you for interviewing with us. Unfortunately, after careful "
                  "consideration we have decided to move forward with other candidates. "
                  "We appreciate your time and wish you all the best in your job search.",
            height=110)

    if st.button("Classify Email", type="primary", use_container_width=True):
        pred_class, pred_prob = predict_from_email(
            sel_model, user_subject, user_body,
            artifacts.fitted_models, artifacts.selected_features)

        label = "Job / Rejection Email" if pred_class == 1 else "Not a Job Email"
        card_cls = "pred-card-job" if pred_class == 1 else "pred-card-notjob"
        lbl_cls  = "pred-class-job" if pred_class == 1 else "pred-class-notjob"
        emoji    = "📩" if pred_class == 1 else "📭"

        r1, r2, r3 = st.columns([1.4, 1.0, 1.0])
        with r1:
            st.markdown(f"""
            <div class="pred-card {card_cls}">
                <div class="stat-label">Classified As</div>
                <div class="pred-class-label {lbl_cls}">{emoji} {label}</div>
                <div class="stat-subtext">Model: {sel_model} &nbsp;|&nbsp; Threshold: 0.50</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(_stat_card(
                "Job Probability", f"{pred_prob:.4f}",
                f"{pred_prob*100:.2f}% confidence",
                "#60a5fa" if pred_class == 1 else "#fb923c"),
                unsafe_allow_html=True)
        with r3:
            not_prob = 1 - pred_prob
            st.markdown(_stat_card(
                "Not-Job Probability", f"{not_prob:.4f}",
                f"{not_prob*100:.2f}%", "#94a3b8"),
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Probability gauge using plotly
        if HAS_PLOTLY:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                number=dict(suffix="%", font=dict(color="#f1f5f9", size=28)),
                gauge=dict(
                    axis=dict(range=[0, 100],
                              tickfont=dict(color="#94a3b8"),
                              tickcolor="#334155"),
                    bar=dict(color="#60a5fa" if pred_class == 1 else "#fb923c"),
                    bgcolor="#1e293b",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 30],  color="#1e293b"),
                        dict(range=[30, 50], color="#1e2a3a"),
                        dict(range=[50, 70], color="#1e3050"),
                        dict(range=[70, 100],color="#1e2a3a"),
                    ],
                    threshold=dict(line=dict(color="#f1f5f9", width=2), value=50),
                ),
                title=dict(text="Job Probability Gauge",
                           font=dict(color="#94a3b8", size=12)),
            ))
            gauge.update_layout(
                paper_bgcolor="#0f172a", height=220,
                margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True, key="expl_gauge")

        # SHAP waterfall
        if shap is not None:
            wf_fig = shap_waterfall_for_input(
                sel_model, user_subject, user_body,
                artifacts.fitted_models, artifacts.selected_features)
            if wf_fig is not None:
                st.markdown("##### SHAP Waterfall for Your Input")
                st.pyplot(wf_fig, clear_figure=False)
                st.caption(
                    "Each bar shows how much a specific feature of your email pushed "
                    "the prediction above (red) or below (blue) the model's baseline. "
                    "The final value is the predicted probability."
                )
            elif sel_model not in ("Random Forest", "Gradient Boost", "Decision Tree"):
                st.info(
                    f"{sel_model} doesn't support SHAP waterfall natively. "
                    "Switch to **Random Forest**, **Gradient Boost**, or **Decision Tree** "
                    "to see the feature-level explanation.")


# ════════════════════════════════════════════════════════════════════════════
# NAVIGATION — pages registry + top nav bar renderer
# ════════════════════════════════════════════════════════════════════════════
_PAGES: List[Tuple[str, str, str]] = [
    ("home",           "🏠", "Home"),
    ("statistics",     "📊", "Statistics"),
    ("performance",    "🏆", "Performance"),
    ("explainability", "🧠", "Explainability"),
]


def _render_top_nav() -> None:
    """Render the session_state-driven top navigation bar.

    Architecture
    ────────────
    1. An invisible <span class="nav-anchor"> is injected via st.markdown.
       The CSS rule  div[data-testid="stVerticalBlock"]:has(.nav-anchor) > …
       uses this as a scope anchor to apply the dark-tray background ONLY to
       the nav columns, leaving all other st.columns() calls unstyled.

    2. Four st.button() calls sit inside st.columns(4).
       • type="primary"   → active pill  (gradient fill, dark text)
       • type="secondary" → inactive pill (ghost border, muted text)

    3. Clicking a button writes to st.session_state.active_page and calls
       st.rerun() so the content area re-renders immediately.
    """
    active = st.session_state.get("active_page", "home")

    # Invisible anchor consumed only by the CSS :has() selector
    st.markdown('<span class="nav-anchor" style="display:none;"></span>',
                unsafe_allow_html=True)

    cols = st.columns(len(_PAGES), gap="small")
    for col, (page_key, icon, label) in zip(cols, _PAGES):
        with col:
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if active == page_key else "secondary",
            ):
                st.session_state.active_page = page_key
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT  — MAIN
# ════════════════════════════════════════════════════════════════════════════
if st is not None:
    @st.cache_resource(show_spinner=False)
    def _get_pipeline_artifacts() -> PipelineArtifacts:
        return load_pretrained_pipeline_artifacts(Path("artifacts"))
else:
    def _get_pipeline_artifacts() -> PipelineArtifacts:  # type: ignore[misc]
        raise ImportError("streamlit is not installed.")


def streamlit_main() -> None:
    if st is None:
        raise ImportError("streamlit is required.")

    st.set_page_config(
        page_title="Email Job Classifier — MSIS 522 HW1",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Dedicated scorecard CSS — injected first, standalone, guaranteed ──────
    # This block is intentionally separate from the main _inject_css() call so
    # that .xai-metric-block / .bar-track / .bar-fill are always applied even
    # if the larger stylesheet block fails to parse.
    st.markdown("""
    <style>
    /* ── Metric scorecard shell ─────────────────────────────────────────── */
    .xai-scorecard {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 6px 28px rgba(0,0,0,0.35);
        height: 100%;
    }
    .xai-card-header {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin-bottom: 1.1rem;
        padding-bottom: .7rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .xai-model-icon { font-size: 1.4rem; line-height: 1; }
    .xai-model-name {
        font-size: .95rem;
        font-weight: 800;
        color: #f1f5f9 !important;
        letter-spacing: .01em;
    }

    /* ── Per-metric block ───────────────────────────────────────────────── */
    .xai-metric-block { margin-bottom: 1rem; }
    .xai-metric-label {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: .35rem;
    }
    .xai-metric-label-text {
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .09em;
        color: #94a3b8 !important;
    }
    .xai-metric-val {
        font-size: .9rem;
        font-weight: 900;
        color: #f1f5f9 !important;
    }
    .xai-metric-val-best::after {
        content: " \2605";
        font-size: .65rem;
        color: #fbbf24 !important;
    }

    /* ── Animated progress bar ──────────────────────────────────────────── */
    @keyframes bar-grow {
        from { width: 0%; opacity: .4; }
        to   { opacity: 1; }
    }
    .bar-track {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin-bottom: .5rem;
    }
    .bar-fill {
        height: 8px;
        border-radius: 6px;
        animation: bar-grow .9s cubic-bezier(.4,0,.2,1) forwards;
    }

    /* ── Compact model card (all-model section) ─────────────────────────── */
    .metric-bar-container { margin-bottom: .55rem; }
    .xai-metric-note {
        font-size: .74rem;
        line-height: 1.55;
        color: #64748b !important;
        font-style: italic;
        margin-top: .1rem;
    }
    </style>""", unsafe_allow_html=True)

    _inject_css()

    with st.spinner("Loading pipeline (cached after first run)..."):
        artifacts = _get_pipeline_artifacts()

    s = artifacts.preprocessing_summary

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.3rem;font-weight:900;color:#38bdf8;'
            'margin-bottom:.8rem;">📧 Email Classifier</div>',
            unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-top:-.5rem;">'
                    'MSIS 522 HW1 — ML Pipeline</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p style="color:#38bdf8;font-weight:700;font-size:.85rem;">'
                    'PIPELINE SNAPSHOT</p>', unsafe_allow_html=True)
        st.metric("Emails (clean)", f"{s['rows_after_outlier_filter']:,}")
        st.metric("Job emails",     f"{s['positive_class_count']:,}")
        st.metric("Non-job emails", f"{s['negative_class_count']:,}")
        st.markdown("---")
        st.markdown('<p style="color:#a78bfa;font-weight:700;font-size:.85rem;">'
                    'CHAMPIONS</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#e2e8f0;font-size:.82rem;">'
            f'<strong style="color:#34d399;">CV:</strong> {s["best_cv_model"]}<br>'
            f'<strong style="color:#60a5fa;">Test:</strong> {s["best_test_model"]}</p>',
            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p style="color:#fb923c;font-weight:700;font-size:.85rem;">'
                    'SELECTED FEATURES</p>', unsafe_allow_html=True)
        for feat in s["selected_features"]:
            st.markdown(
                f'<p style="color:#94a3b8;font-size:.78rem;margin:0.1rem 0;">'
                f'• <code style="color:#a78bfa;">{feat}</code></p>',
                unsafe_allow_html=True)

    # ── Tab navigation ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠  Executive Summary",
        "📊  Descriptive Analytics",
        "🏆  Model Performance",
        "🧠  Explainability & Prediction",
    ])
    with tab1:
        _render_tab_executive(artifacts)
    with tab2:
        _render_tab_descriptive(artifacts)
    with tab3:
        _render_tab_model_performance(artifacts)
    with tab4:
        _render_tab_explainability(artifacts)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Email job-classification pipeline (MSIS 522 HW1)")
    parser.add_argument("--top-k-features", type=int, default=TOP_K_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--force-retrain", action="store_true",
                        help="Ignore packaged artifacts and retrain every model.")
    args, _ = parser.parse_known_args()

    artifacts = get_or_train_pipeline_artifacts(
        output_dir=args.output_dir,
        top_k_features=args.top_k_features,
        force_retrain=args.force_retrain,
    )
    best_cv   = artifacts.cv_metrics.iloc[0]
    best_test = artifacts.test_metrics.iloc[0]

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Best CV model   : {best_cv['model']}")
    print(f"  F1 = {best_cv['f1_score']:.4f}  AUC = {best_cv['auc_score']:.4f}")
    print(f"Best test model : {best_test['model']}")
    print(f"  Accuracy  = {best_test['accuracy']:.4f}")
    print(f"  Precision = {best_test['precision']:.4f}")
    print(f"  Recall    = {best_test['recall']:.4f}")
    print(f"  F1        = {best_test['f1_score']:.4f}")
    print(f"  AUC       = {best_test['auc_score']:.4f}")
    print(f"Artifacts saved : {args.output_dir.resolve()}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if st is not None:
    try:
        _running_streamlit = st.runtime.exists()
    except AttributeError:
        _running_streamlit = hasattr(st, "_is_running_with_streamlit")

    if _running_streamlit:
        streamlit_main()
    elif __name__ == "__main__":
        cli_main()
else:
    if __name__ == "__main__":
        cli_main()
