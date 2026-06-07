"""
MindCare AI ─ Module 2: Depression Prediction
==============================================
Automated preprocessing pipeline + multi-model comparison
(Logistic Regression, Random Forest, XGBoost, LightGBM).

Usage (Google Colab)
--------------------
    !python train_depression.py

Expected dataset layout
-----------------------
    project/datasets/student_depression.csv
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# ── optional boosting libraries ─────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ── local ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_logger, project_dirs,
    save_model, save_metrics, plot_confusion_matrix,
    print_metrics_table, load_csv_safe, summarise_dataframe,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR      = "project"
DIRS          = project_dirs(BASE_DIR)
LOGGER        = setup_logger("depression_prediction", log_dir=DIRS["logs"])

DATA_PATH     = os.path.join(DIRS["datasets"], "student_depression.csv")
MODEL_PATH    = os.path.join(DIRS["models"],   "depression_model.pkl")
METRICS_PATH  = os.path.join(DIRS["metrics"],  "depression_metrics.json")
CM_PATH       = os.path.join(DIRS["outputs"],  "depression_confusion_matrix.png")
COMPARE_PATH  = os.path.join(DIRS["outputs"],  "depression_model_comparison.png")

TARGET_COL    = "Depression"
DROP_COLS     = ["id"]          # non-informative identifier column
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
CV_FOLDS      = 5


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_depression_data(path: str) -> pd.DataFrame:
    """Load, validate, and do light cleaning of the depression dataset."""
    df = load_csv_safe(path, LOGGER)
    summarise_dataframe(df, "Student Depression Dataset")

    # Drop irrelevant columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Standardise column names: strip whitespace
    df.columns = df.columns.str.strip()

    LOGGER.info(f"Class distribution:\n{df[TARGET_COL].value_counts()}")
    LOGGER.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def identify_features(
    df: pd.DataFrame,
    target: str,
    num_unique_thresh: int = 15,
) -> Tuple[List[str], List[str]]:
    """
    Automatically split columns into numerical and categorical.

    Strategy
    --------
    - Numeric dtype  AND  >num_unique_thresh unique values  →  numerical
    - Everything else (object, bool, or low-cardinality int) →  categorical
    """
    feature_cols = [c for c in df.columns if c != target]
    numerical, categorical = [], []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > num_unique_thresh:
                numerical.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)

    LOGGER.info(f"Numerical features  ({len(numerical)}): {numerical}")
    LOGGER.info(f"Categorical features({len(categorical)}): {categorical}")
    return numerical, categorical


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    ColumnTransformer with:
      • Numerical  : SimpleImputer(median)  →  StandardScaler
      • Categorical: SimpleImputer(most_frequent)  →  OneHotEncoder
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def get_candidate_models() -> Dict[str, Any]:
    """Return a dict of {name: estimator} for all available classifiers."""
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        LOGGER.info("XGBoost detected and added to candidates.")
    else:
        LOGGER.warning("XGBoost not installed — skipping.")

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=63,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        LOGGER.info("LightGBM detected and added to candidates.")
    else:
        LOGGER.warning("LightGBM not installed — skipping.")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CROSS-VALIDATION & TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> Tuple[float, float]:
    """Return mean and std of stratified CV F1 (weighted)."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=skf, scoring="f1_weighted", n_jobs=-1,
    )
    return float(scores.mean()), float(scores.std())


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Compute full evaluation metrics on held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics: Dict[str, Any] = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred,
                            average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred,
                            average="weighted", zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred,
                            average="weighted", zero_division=0)), 4),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None

    return metrics, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    comparison: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Bar-chart comparing F1 / Accuracy / ROC-AUC across models."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    model_names  = list(comparison.keys())
    metric_keys  = ["accuracy", "f1_score", "roc_auc"]
    metric_labels= ["Accuracy", "F1 Score", "ROC-AUC"]
    palette      = sns.color_palette("Set2", len(metric_keys))

    x     = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        vals = [comparison[m].get(mkey, 0) or 0 for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=mlabel, color=palette[i])
        for bar, val in zip(bars, vals):
            if val:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Depression Prediction ─ Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Model comparison chart saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE IMPORTANCE (Random Forest / XGB / LGBM)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance_from_pipeline(
    pipeline: Pipeline,
    feature_names: List[str],
    save_path: str,
    model_name: str,
    top_n: int = 20,
) -> None:
    """Extract and plot feature importances from tree-based models."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    clf = pipeline.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        LOGGER.info(f"[{model_name}] No feature_importances_ — skipping importance plot.")
        return

    importances = clf.feature_importances_
    n = min(top_n, len(feature_names), len(importances))
    indices = np.argsort(importances)[-n:]
    top_names  = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, n // 2)))
    ax.barh(top_names, top_values, color="teal")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"[{model_name}] Top-{n} Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Feature importance chart saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN TRAINING ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_depression_model() -> Pipeline:
    """
    Full training workflow:
      load → feature identification → preprocessing → split →
      multi-model CV → best model selection → final evaluation →
      save model / metrics / visualisations.

    Returns
    -------
    Best sklearn Pipeline (preprocessor + classifier)
    """
    LOGGER.info("=" * 60)
    LOGGER.info("  MindCare AI ─ Depression Prediction Training")
    LOGGER.info("=" * 60)

    # ── 8.1  Load ────────────────────────────────────────────────────────────
    df = load_depression_data(DATA_PATH)

    # ── 8.2  Feature identification ──────────────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    numerical_features, categorical_features = identify_features(df, TARGET_COL)

    # ── 8.3  Train / test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    LOGGER.info(
        f"Split → Train: {len(X_train):,} | Test: {len(X_test):,}"
        f" | Positive rate (train): {y_train.mean():.2%}"
    )

    # ── 8.4  Preprocessor ────────────────────────────────────────────────────
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    # ── 8.5  Multi-model cross-validation ────────────────────────────────────
    candidate_models = get_candidate_models()
    cv_results: Dict[str, Dict[str, float]] = {}

    LOGGER.info(f"\nRunning {CV_FOLDS}-fold stratified CV for {len(candidate_models)} models …\n")

    for name, estimator in candidate_models.items():
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf",          estimator),
        ])
        t0 = time.time()
        mean_f1, std_f1 = cross_validate_model(full_pipeline, X_train, y_train)
        elapsed = time.time() - t0
        cv_results[name] = {"cv_f1_mean": round(mean_f1, 4), "cv_f1_std": round(std_f1, 4)}
        LOGGER.info(
            f"  {name:<25} CV F1 = {mean_f1:.4f} ± {std_f1:.4f}  ({elapsed:.1f}s)"
        )

    # ── 8.6  Select best model by CV F1 ─────────────────────────────────────
    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_f1_mean"])
    LOGGER.info(f"\nBest model selected: {best_name} "
                f"(CV F1 = {cv_results[best_name]['cv_f1_mean']:.4f})")

    # ── 8.7  Final fit on full training set ──────────────────────────────────
    best_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numerical_features, categorical_features)),
        ("clf",          candidate_models[best_name]),
    ])
    t0 = time.time()
    best_pipeline.fit(X_train, y_train)
    LOGGER.info(f"Final training completed in {time.time() - t0:.1f}s")

    # ── 8.8  Evaluate on test set ────────────────────────────────────────────
    metrics, y_pred = evaluate_on_test(best_pipeline, X_test, y_test)
    metrics["best_model"] = best_name
    metrics["cv_results"] = cv_results
    print_metrics_table(metrics, f"Depression Prediction ─ {best_name} ─ Test Metrics")

    # Collect test-set metrics for all models (for comparison chart)
    all_test_metrics: Dict[str, Dict[str, float]] = {}
    for name, estimator in candidate_models.items():
        p = Pipeline([
            ("preprocessor", build_preprocessor(numerical_features, categorical_features)),
            ("clf",          estimator),
        ])
        p.fit(X_train, y_train)
        m, _ = evaluate_on_test(p, X_test, y_test)
        all_test_metrics[name] = {k: v for k, v in m.items()
                                   if isinstance(v, (int, float))}

    # ── 8.9  Visualisations ──────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true=y_test.values,
        y_pred=y_pred,
        labels=[0, 1],
        title=f"Depression Prediction ─ {best_name} ─ Confusion Matrix",
        save_path=CM_PATH,
        figsize=(7, 6),
    )
    LOGGER.info(f"Confusion matrix saved → {CM_PATH}")

    plot_model_comparison(all_test_metrics, COMPARE_PATH)

    # Feature importance for tree-based best model
    try:
        preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
        feature_names_out   = list(preprocessor_fitted.get_feature_names_out())
        fi_path = os.path.join(DIRS["outputs"], "depression_feature_importance.png")
        plot_feature_importance_from_pipeline(
            best_pipeline, feature_names_out, fi_path, best_name
        )
    except Exception as e:
        LOGGER.warning(f"Feature importance plot skipped: {e}")

    # ── 8.10  Save ────────────────────────────────────────────────────────────
    save_model(best_pipeline, MODEL_PATH)
    LOGGER.info(f"Model saved → {MODEL_PATH}")

    save_metrics(metrics, METRICS_PATH)
    LOGGER.info(f"Metrics saved → {METRICS_PATH}")

    LOGGER.info("Depression Prediction training pipeline complete ✓")
    return best_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 9.  INFERENCE HELPER  (imported by ChatbotEngine)
# ══════════════════════════════════════════════════════════════════════════════

def predict_depression_risk(features: Dict[str, Any], pipeline: Pipeline) -> Dict[str, Any]:
    """
    Single-sample inference for depression risk.

    Parameters
    ----------
    features : dict mapping column names → raw values (same schema as training)
    pipeline : trained sklearn Pipeline

    Returns
    -------
    {
        "depression":    1,           # 0 or 1
        "risk_label":    "high",      # "low" | "moderate" | "high"
        "probability":   0.82
    }
    """
    input_df = pd.DataFrame([features])
    proba    = pipeline.predict_proba(input_df)[0][1]   # P(depression=1)
    pred     = int(proba >= 0.5)

    if proba < 0.35:
        label = "low"
    elif proba < 0.65:
        label = "moderate"
    else:
        label = "high"

    return {
        "depression":  pred,
        "risk_label":  label,
        "probability": round(float(proba), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_depression_model()
