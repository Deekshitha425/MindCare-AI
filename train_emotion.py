"""
MindCare AI ─ Module 1: Emotion Detection
==========================================
TF-IDF + LinearSVC sklearn Pipeline trained on the Kaggle Emotion Dataset.

Usage (Google Colab)
--------------------
    !python train_emotion.py

Expected dataset layout
-----------------------
    project/datasets/train.csv   (columns: Subtitle, Emotion)
    project/datasets/test.csv    (columns: Subtitle, Emotion)
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import logging
from typing import Dict, List, Tuple, Any

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Colab / servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# ── local ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_logger, project_dirs, preprocess_series,
    save_model, save_metrics, plot_confusion_matrix,
    print_metrics_table, load_csv_safe, summarise_dataframe,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = "project"
DIRS        = project_dirs(BASE_DIR)
LOGGER      = setup_logger("emotion_detection", log_dir=DIRS["logs"])

TRAIN_PATH  = os.path.join(DIRS["datasets"], "train.csv")
TEST_PATH   = os.path.join(DIRS["datasets"], "test.csv")

MODEL_PATH  = os.path.join(DIRS["models"],  "emotion_model.pkl")
METRICS_PATH= os.path.join(DIRS["metrics"], "emotion_metrics.json")
CM_PATH     = os.path.join(DIRS["outputs"], "emotion_confusion_matrix.png")
KW_PATH     = os.path.join(DIRS["outputs"], "emotion_top_keywords.png")

TEXT_COL    = "Subtitle"
LABEL_COL   = "Emotion"

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_validate(path: str, split_name: str) -> pd.DataFrame:
    """Load CSV and assert required columns exist."""
    df = load_csv_safe(path, LOGGER)
    for col in [TEXT_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Missing required column: '{col}'")
    LOGGER.info(f"[{split_name}] Class distribution:\n{df[LABEL_COL].value_counts()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Clean text and return (X, y)."""
    LOGGER.info("Preprocessing text …")
    X = preprocess_series(df[TEXT_COL])
    y = df[LABEL_COL].str.strip().str.lower()
    # Drop empty rows after cleaning
    mask  = X.str.len() > 0
    n_drop = (~mask).sum()
    if n_drop:
        LOGGER.warning(f"Dropping {n_drop} empty rows after text cleaning.")
    return X[mask], y[mask]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """
    TF-IDF Vectorizer  →  CalibratedLinearSVC

    CalibratedClassifierCV wraps LinearSVC so we get predict_proba
    (needed for downstream chatbot confidence scores).
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=50_000,
        sublinear_tf=True,       # apply log(1+tf)
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
    )
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
        cv=3,
    )
    return Pipeline([("tfidf", tfidf), ("clf", svc)])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TOP KEYWORDS PER EMOTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_keywords_per_emotion(
    pipeline: Pipeline,
    classes: List[str],
    save_path: str,
    top_n: int = 15,
) -> None:
    """
    Extract and visualise the top TF-IDF weighted keywords for each emotion.

    Works by inspecting the coef_ of the inner LinearSVC.
    """
    try:
        tfidf     = pipeline.named_steps["tfidf"]
        cal_clf   = pipeline.named_steps["clf"]
        base_svc  = cal_clf.estimator          # the underlying LinearSVC
        vocab     = tfidf.get_feature_names_out()
        coefs     = base_svc.coef_             # shape (n_classes, n_features)
        clf_classes = base_svc.classes_

        n_classes = len(clf_classes)
        fig, axes = plt.subplots(
            nrows=(n_classes + 1) // 2,
            ncols=2,
            figsize=(16, n_classes * 2.2),
        )
        axes = axes.flatten()
        palette = sns.color_palette("husl", n_classes)

        for idx, (cls, coef) in enumerate(zip(clf_classes, coefs)):
            top_idx   = np.argsort(coef)[-top_n:][::-1]
            top_words = vocab[top_idx]
            top_vals  = coef[top_idx]

            ax = axes[idx]
            ax.barh(top_words[::-1], top_vals[::-1], color=palette[idx])
            ax.set_title(f"Emotion: {cls.upper()}", fontweight="bold", fontsize=11)
            ax.set_xlabel("SVM Coefficient", fontsize=9)
            ax.tick_params(axis="y", labelsize=8)

        # hide any extra subplots
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Top Keywords per Emotion (LinearSVC Coefficients)",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info(f"Top-keywords chart saved → {save_path}")
    except Exception as exc:
        LOGGER.warning(f"Could not plot top keywords: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Generate predictions and return a metrics dict."""
    LOGGER.info("Evaluating on test set …")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # Build ordered label list (present in test set, ordered as EMOTION_CLASSES)
    present_labels = [c for c in EMOTION_CLASSES if c in y_test.unique()]

    metrics: Dict[str, Any] = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred,
                            labels=present_labels, average="weighted",
                            zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred,
                            labels=present_labels, average="weighted",
                            zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred,
                            labels=present_labels, average="weighted",
                            zero_division=0)), 4),
        "classification_report": classification_report(
            y_test, y_pred, labels=present_labels, zero_division=0
        ),
    }
    return metrics, y_pred, present_labels


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN TRAINING ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_emotion_model() -> Pipeline:
    """
    Full training workflow:
      load → preprocess → build pipeline → train → evaluate →
      save model / metrics / visualisations.

    Returns
    -------
    Trained sklearn Pipeline
    """
    LOGGER.info("=" * 60)
    LOGGER.info("  MindCare AI ─ Emotion Detection Training")
    LOGGER.info("=" * 60)

    # ── 6.1  Load data ───────────────────────────────────────────────────────
    train_df = load_and_validate(TRAIN_PATH, "TRAIN")
    test_df  = load_and_validate(TEST_PATH,  "TEST")
    summarise_dataframe(train_df, "Training Set")
    summarise_dataframe(test_df,  "Test Set")

    # ── 6.2  Preprocess ──────────────────────────────────────────────────────
    X_train, y_train = prepare_data(train_df)
    X_test,  y_test  = prepare_data(test_df)
    LOGGER.info(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    # ── 6.3  Build & train ───────────────────────────────────────────────────
    LOGGER.info("Building TF-IDF + LinearSVC pipeline …")
    pipeline = build_pipeline()

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    LOGGER.info(f"Training completed in {time.time() - t0:.1f}s")

    # ── 6.4  Evaluate ────────────────────────────────────────────────────────
    metrics, y_pred, present_labels = evaluate_model(pipeline, X_test, y_test)
    print_metrics_table(metrics, "Emotion Detection ─ Test Metrics")

    # ── 6.5  Confusion matrix ────────────────────────────────────────────────
    LOGGER.info("Generating confusion matrix …")
    plot_confusion_matrix(
        y_true=y_test.values,
        y_pred=y_pred,
        labels=present_labels,
        title="Emotion Detection ─ Confusion Matrix",
        save_path=CM_PATH,
        figsize=(10, 8),
        cmap="YlOrRd",
    )
    LOGGER.info(f"Confusion matrix saved → {CM_PATH}")

    # ── 6.6  Top keywords per emotion ────────────────────────────────────────
    plot_top_keywords_per_emotion(pipeline, present_labels, KW_PATH)

    # ── 6.7  Save model & metrics ────────────────────────────────────────────
    save_model(pipeline, MODEL_PATH)
    LOGGER.info(f"Model saved → {MODEL_PATH}")

    save_metrics(metrics, METRICS_PATH)
    LOGGER.info(f"Metrics saved → {METRICS_PATH}")

    LOGGER.info("Emotion Detection training pipeline complete ✓")
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 7.  INFERENCE HELPER  (imported by ChatbotEngine)
# ══════════════════════════════════════════════════════════════════════════════

def predict_emotion(text: str, pipeline: Pipeline) -> Dict[str, Any]:
    """
    Single-sample inference.

    Parameters
    ----------
    text     : raw user input string
    pipeline : trained sklearn Pipeline

    Returns
    -------
    {
        "emotion":      "joy",
        "confidence":   0.87,
        "all_probs":    {"joy": 0.87, "sadness": 0.05, …}
    }
    """
    from utils import clean_text   # local import to avoid circular dependency
    cleaned   = clean_text(text)
    proba     = pipeline.predict_proba([cleaned])[0]
    classes   = pipeline.classes_
    top_idx   = int(np.argmax(proba))

    return {
        "emotion":    classes[top_idx],
        "confidence": round(float(proba[top_idx]), 4),
        "all_probs":  {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_emotion_model()
