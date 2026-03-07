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
import warnings
from dataclasses import dataclass, field
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
DATASETS_IMPORT_ERROR: Optional[Exception] = None
try:
    from datasets import load_dataset as _load_dataset  # type: ignore[import]
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
    raw = load_dataset(dataset_name, split="train")
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
       TOP NAVIGATION BAR  (session_state-driven, not st.tabs)
       ══════════════════════════════════════════════════════════════ */

    /* Nav tray background — scoped to the stVerticalBlock that owns
       the invisible .nav-anchor span we inject before the buttons.
       We target the direct child > div > stHorizontalBlock so only
       the nav columns (not any nested columns in page content) get
       the tray treatment. */
    div[data-testid="stVerticalBlock"]:has(.nav-anchor)
        > div
        > div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        gap: .5rem !important;
        align-items: stretch !important;
        padding: .65rem .75rem !important;
        background: #0d1526 !important;
        border: 1.5px solid rgba(56,189,248,0.28) !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 32px rgba(0,0,0,0.55),
                    inset 0 1px 0 rgba(255,255,255,0.06) !important;
        margin-bottom: 1.8rem !important;
    }

    /* ── Active nav pill (type="primary") ───────────────────────── */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg,#38bdf8 0%,#818cf8 100%) !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: .83rem !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        border-radius: 12px !important;
        padding: .7rem 1rem !important;
        box-shadow: 0 4px 20px rgba(56,189,248,0.40),
                    0 1px 4px rgba(0,0,0,0.30) !important;
        transition: all .18s ease !important;
    }

    /* ── Inactive nav pill (type="secondary") ───────────────────── */
    button[data-testid="baseButton-secondary"] {
        background: rgba(255,255,255,0.04) !important;
        color: #64748b !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        font-weight: 700 !important;
        font-size: .83rem !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        border-radius: 12px !important;
        padding: .7rem 1rem !important;
        transition: all .18s ease !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255,255,255,0.10) !important;
        color: #cbd5e1 !important;
        border-color: rgba(255,255,255,0.22) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important;
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

    # Hero
    st.markdown(f"""
    <div class="hero-shell">
        <div class="hero-title">Email Job-Rejection Classifier</div>
        <div class="hero-copy">
            An end-to-end ML pipeline classifying job-application &amp; rejection emails
            from everyday noise — built on the <strong>imnim/multiclass-email-classification</strong>
            dataset with six competing models, SMOTE oversampling, and a custom TF-IDF stacked ensemble.
            <br><br>
            Winner on Test Set: <strong>{best_row['model']}</strong> &nbsp;&bull;&nbsp;
            Precision: <strong>{float(best_row['precision']):.4f}</strong> &nbsp;&bull;&nbsp;
            F1: <strong>{float(best_row['f1_score']):.4f}</strong> &nbsp;&bull;&nbsp;
            AUC: <strong>{float(best_row['auc_score']):.4f}</strong>
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    pos_ratio = s["positive_class_ratio"]
    _render_stat_row([
        ("Total Emails",   f"{s['rows_after_outlier_filter']:,}",
         "after cleaning & filtering", "#38bdf8"),
        ("Positive Rate",  _pct(pos_ratio),
         f"{s['positive_class_count']:,} job emails", "#f97316"),
        ("SMOTE Boost",
         f"{sm.get('train_positive_before','?')} → {sm.get('train_positive_after','?')}",
         "positive samples in training", "#34d399"),
        ("Best CV F1",     f"{float(artifacts.cv_metrics.iloc[0]['f1_score']):.4f}",
         artifacts.cv_metrics.iloc[0]["model"], "#a78bfa"),
        ("Best Test AUC",  f"{float(best_row['auc_score']):.4f}",
         best_row["model"], "#f87171"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline flow
    _chip("Pipeline Architecture", "chip-purple")
    steps = [
        ("1 Load", "#38bdf8"),
        ("2 Clean", "#34d399"),
        ("3 Engineer", "#a78bfa"),
        ("4 Select", "#fb923c"),
        ("5 SMOTE", "#f87171"),
        ("6 Train × 6", "#60a5fa"),
        ("7 Evaluate", "#fbbf24"),
    ]
    cols = st.columns(len(steps))
    for col, (label, color) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:.6rem .3rem;'
                f'background:rgba(255,255,255,0.05);border-radius:12px;'
                f'border-top:3px solid {color};font-size:.78rem;'
                f'font-weight:700;color:{color};">{label}</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Two columns: Why + Approach
    col_l, col_r = st.columns(2)
    with col_l:
        _chip("Why It Matters", "chip-orange")
        _insight(
            "<strong>Job seekers receive hundreds of emails daily.</strong> Only a small "
            "fraction are actionable — offer letters, rejections, interview invites. "
            "An automated classifier surfaces only what matters, reducing cognitive load "
            "and enabling faster responses.", "insight-good")
        _insight(
            f"<strong>Class imbalance is severe:</strong> only {_pct(pos_ratio)} of "
            "emails are job-related. A naïve 'always Not-Job' classifier achieves "
            f"{_pct(1-pos_ratio)} accuracy while being completely useless. "
            "SMOTE and precision-focused metrics are therefore essential.", "insight-warn")

    with col_r:
        _chip("Key Findings", "chip-green")
        top3 = artifacts.test_metrics.head(3)
        for _, row in top3.iterrows():
            medal = "🥇" if _ == 0 else ("🥈" if _ == 1 else "🥉")
            _insight(f"{medal} <strong>{row['model']}</strong> — "
                     f"P: {row['precision']:.4f} | R: {row['recall']:.4f} | "
                     f"F1: {row['f1_score']:.4f} | AUC: {row['auc_score']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    _chip("Model Arena — Quick View")
    if HAS_PLOTLY:
        st.plotly_chart(_plotly_model_radar(artifacts.test_metrics),
                        use_container_width=True, key="exec_radar")


def _render_tab_descriptive(artifacts: PipelineArtifacts) -> None:
    _tab_header("descriptive")
    s    = artifacts.preprocessing_summary
    sm   = artifacts.smote_summary
    ff   = artifacts.feature_frame
    sel  = artifacts.selected_features

    # ── 1.1 Dataset Stats ─────────────────────────────────────────────
    _chip("1.1  Dataset Introduction")
    _render_stat_row([
        ("Raw Emails",     f"{s['input_rows']:,}",              "from Hugging Face",           "#38bdf8"),
        ("After Cleaning", f"{s['rows_after_basic_cleaning']:,}", "dedup + empty removed",      "#34d399"),
        ("After Filtering",f"{s['rows_after_outlier_filter']:,}", "IQR outlier removed",        "#a78bfa"),
        ("Job Emails",     f"{s['positive_class_count']:,}",    _pct(s['positive_class_ratio']),"#f97316"),
        ("Non-Job Emails", f"{s['negative_class_count']:,}",    "majority class",              "#94a3b8"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1.2 Class Distribution ────────────────────────────────────────
    st.markdown("---")
    _chip("1.2  Class Distribution & SMOTE", "chip-orange")
    c_left, c_right = st.columns([1.2, 1.0])
    with c_left:
        if HAS_PLOTLY:
            st.plotly_chart(_plotly_class_breakdown(ff),
                            use_container_width=True, key="desc_donut")
        else:
            st.pyplot(_fig_class_donut(ff, sm), clear_figure=False)
        st.caption(
            f"**{_pct(s['positive_class_ratio'])}** of all emails are job/rejection-related. "
            "The severe imbalance means standard accuracy is misleading — SMOTE and "
            "F1/AUC metrics are critical for honest evaluation."
        )

    with c_right:
        st.markdown("##### SMOTE Impact")
        if sm.get("smote_applied"):
            ratio_before = sm["train_positive_before"] / max(sm["train_total_before"], 1)
            ratio_after  = sm["train_positive_after"]  / max(sm["train_total_after"],  1)
            _render_stat_row([
                ("Before SMOTE", f"{sm['train_positive_before']:,}",
                 f"positives ({_pct(ratio_before)})", "#f87171"),
                ("After SMOTE",  f"{sm['train_positive_after']:,}",
                 f"positives ({_pct(ratio_after)})",  "#34d399"),
            ])
            st.markdown("<br>", unsafe_allow_html=True)
        _insight(
            "SMOTE synthesises new minority-class samples by interpolating between "
            "existing positive examples in feature space. The test set is <strong>never "
            "augmented</strong> — only the training fold sees synthetic samples, "
            "preserving a clean evaluation benchmark.", "insight-good")
        _insight(
            "Each synthetic email is a weighted blend of two real job emails' numeric "
            "features, creating plausible but novel training signals that help the "
            "classifier learn more robust job-email boundaries.")

    # ── Email DNA ─────────────────────────────────────────────────────
    st.markdown("---")
    _chip("Email DNA — What Makes a Job Email?", "chip-purple")
    dna_l, dna_r = st.columns([1.0, 1.0])
    with dna_l:
        st.markdown("##### Keyword Category Radar")
        st.pyplot(_fig_keyword_radar(ff), clear_figure=False)
        st.caption(
            "Radar chart comparing average keyword-category counts per email class. "
            "Job emails spike sharply on **Job**, **Candidate**, **Recruiting**, and "
            "**Compensation** axes. Non-job emails concentrate on **Business** and **Support** "
            "terminology — a clear lexical fingerprint separating the two classes."
        )
    with dna_r:
        st.markdown("##### Mean Feature Comparison")
        if HAS_PLOTLY:
            st.plotly_chart(_plotly_feature_heatmap(ff, sel),
                            use_container_width=True, key="desc_heatmap")
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(_fig_stat_comparison(ff, sel), clear_figure=False)
        st.caption(
            "Side-by-side mean values of selected features for each class. "
            "Tall blue (Job) bars on keyword features confirm the radar findings; "
            "orange (Not-Job) bars dominate text-length features because non-job "
            "emails (newsletters, invoices) tend to be longer and denser."
        )

    # ── 1.3 Feature Distributions ─────────────────────────────────────
    st.markdown("---")
    _chip("1.3  Feature Distribution Deep-Dive", "chip-green")
    st.markdown("##### Violin plots — spread & density by class")
    st.pyplot(_fig_violin_features(ff, sel), clear_figure=False)
    st.caption(
        "Violin plots show the full probability density for each feature split by class. "
        "Wider sections indicate more data density at that value. Job emails (blue) show "
        "a concentrated spike near zero for most features but a distinct long tail at high "
        "keyword counts — a strong discriminative signal."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Feature distributions by class")
        st.pyplot(artifacts.figures["feature_distribution"], clear_figure=False)
        st.caption(
            "Boxplots of top features split by class. Job emails show clearly elevated "
            "medians on all keyword signals — the box separation confirms these features "
            "are statistically meaningful predictors."
        )
    with c2:
        st.markdown("##### Mean gap — dumbbell chart")
        st.pyplot(artifacts.figures["feature_gap"], clear_figure=False)
        st.caption(
            "Each horizontal line spans the mean values for Not-Job (orange) and Job (blue). "
            "Long spans = high discriminative power. Job-signal ratio and keyword counts "
            "show the widest gaps, confirming their importance to all classifiers."
        )

    # ── 1.4 Correlation ───────────────────────────────────────────────
    st.markdown("---")
    _chip("1.4  Correlation Heatmap", "chip-red")
    st.pyplot(artifacts.figures["selected_feature_correlation"], clear_figure=False)
    st.caption(
        "Pearson correlation matrix of the 8 selected features. High inter-feature "
        "correlations (|r| > 0.9) were removed during preprocessing to reduce "
        "multicollinearity. The remaining features are reasonably orthogonal, ensuring "
        "each contributes independent signal to the classifiers."
    )

    # ── Interactive Feature Explorer ──────────────────────────────────
    st.markdown("---")
    _chip("Interactive Feature Explorer")
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
        ("Lift",         f"{lift:+.2f}×",  "Job vs Not-Job",          "#34d399" if lift > 0 else "#f87171"),
        ("Feature Rank", f"#{artifacts.feature_weights['feature'].tolist().index(feature_focus)+1}",
         "by ensemble weight", "#a78bfa"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    st.pyplot(_fig_single_feature_focus(ff, feature_focus), clear_figure=False)
    st.caption(
        f"**Left:** Density overlay for `{feature_focus}`. Greater separation = "
        "stronger discriminative signal. **Right:** Boxplot showing quartiles and "
        "outliers. Inspect any feature to compare its distributional profile across classes."
    )

    # ── Feature ranking table ─────────────────────────────────────────
    st.markdown("---")
    _chip("Feature Weight Ranking", "chip-purple")
    fw = artifacts.feature_weights.copy()
    for col in ["linear_weight", "lasso_weight", "ridge_weight", "ensemble_weight"]:
        if col in fw.columns:
            fw[col] = fw[col].map(lambda x: f"{x:.6f}")
    st.dataframe(fw, use_container_width=True, hide_index=True)


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

    # ── Model Scorecards ──────────────────────────────────────────────
    st.markdown("---")
    _chip("Individual Model Scorecards", "chip-green")
    st.caption("Every model's full metric profile — highlighted bars mark the best score per metric.")
    n_models = len(df)
    cols = st.columns(min(3, n_models))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 3]:
            st.markdown(
                _model_scorecard_html(str(row["model"]), row, best_vals,
                                      _MODEL_COLORS[i % len(_MODEL_COLORS)]),
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

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

    # ── ROC Curves ────────────────────────────────────────────────────
    st.markdown("---")
    _chip("ROC Curves", "chip-red")
    st.pyplot(_fig_dark_roc(artifacts.roc_data), clear_figure=False)
    st.caption(
        "ROC curves on the held-out test set. AUC (Area Under Curve) measures "
        "discrimination ability independent of the decision threshold. "
        "The dashed diagonal is a random classifier (AUC = 0.5). "
        "Decision Tree achieves the highest AUC (0.995) despite mid-table F1, "
        "indicating excellent threshold flexibility."
    )

    # ── MLP History ────────────────────────────────────────────────────
    if artifacts.mlp_history:
        st.markdown("---")
        _chip("Neural Network Training History", "chip-purple")
        st.pyplot(_fig_dark_mlp_history(artifacts.mlp_history), clear_figure=False)
        st.caption(
            "**Left:** Cross-entropy training loss decreases monotonically — the network is "
            "learning. **Right:** Validation accuracy on a held-out 10% slice — early stopping "
            f"halted training after {len(artifacts.mlp_history['loss'])} epochs when "
            "no improvement was seen for 20 consecutive iterations."
        )

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

    # ── SHAP ──────────────────────────────────────────────────────────
    _chip("3.1  SHAP Feature Importance", "chip-purple")
    if not shap_figs:
        st.info("SHAP figures unavailable — install the `shap` package and re-run the pipeline.")
    else:
        s1, s2 = st.columns(2)
        with s1:
            if "shap_summary_bar" in shap_figs:
                st.markdown("##### Mean |SHAP| — Feature Importance")
                st.pyplot(shap_figs["shap_summary_bar"], clear_figure=False)
                st.caption(
                    "Bar length = average absolute SHAP value across 300 test samples. "
                    "Longer bar = stronger average influence on predictions. "
                    "Job-signal keyword counts dominate — they are the most reliable "
                    "lexical proxies for job-related emails."
                )
        with s2:
            if "shap_summary_beeswarm" in shap_figs:
                st.markdown("##### Beeswarm — Direction & Magnitude")
                st.pyplot(shap_figs["shap_summary_beeswarm"], clear_figure=False)
                st.caption(
                    "Each dot = one sample. Color = feature value (red=high, blue=low). "
                    "Rightward push → increases Job probability. "
                    "High keyword counts (red dots) consistently push rightward; "
                    "long, sparse bodies (blue) push leftward."
                )

        if "shap_waterfall" in shap_figs:
            st.markdown("---")
            _chip("Waterfall — Highest-Confidence True Positive", "chip-green")
            st.pyplot(shap_figs["shap_waterfall"], clear_figure=False)
            st.caption(
                "Waterfall for the test-set positive email predicted with the highest "
                "confidence. Starting from the model baseline, each bar shows how much "
                "that feature nudges the score up (red) or down (blue). This email's "
                "strong keyword signals stack constructively to a near-certain prediction."
            )

        st.markdown("---")
        _chip("Interpretation", "chip-orange")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            _insight("<strong>Top drivers:</strong> Job-signal ratio, candidate & recruiting "
                     "keyword counts — direct linguistic markers of job-related content.")
        with col_b:
            _insight("<strong>Direction:</strong> High keyword counts → push toward 'Job'. "
                     "Long, sparse bodies (newsletters/invoices) → push toward 'Not Job'.")
        with col_c:
            _insight("<strong>Decision value:</strong> SHAP surfaces borderline cases — "
                     "emails near the decision boundary — guiding where human review "
                     "adds the most value.")

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
        return run_pipeline(persist_dir=Path("artifacts"))
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
    _inject_css()

    # Bootstrap session state on first load
    if "active_page" not in st.session_state:
        st.session_state.active_page = "home"

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

    # ── Top navigation bar ────────────────────────────────────────────
    _render_top_nav()

    # ── Centered content area ─────────────────────────────────────────
    # st.columns([1, 5, 1]) gives the content column ~71% of the main
    # area width, horizontally centered regardless of screen size.
    _, content_col, _ = st.columns([1, 5, 1])
    with content_col:
        page = st.session_state.get("active_page", "home")
        if page == "home":
            _render_tab_executive(artifacts)
        elif page == "statistics":
            _render_tab_descriptive(artifacts)
        elif page == "performance":
            _render_tab_model_performance(artifacts)
        elif page == "explainability":
            _render_tab_explainability(artifacts)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Email job-classification pipeline (MSIS 522 HW1)")
    parser.add_argument("--top-k-features", type=int, default=TOP_K_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args, _ = parser.parse_known_args()

    artifacts = run_pipeline(top_k_features=args.top_k_features,
                             persist_dir=args.output_dir)
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
