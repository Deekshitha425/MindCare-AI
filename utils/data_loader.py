"""
utils/data_loader.py
Cached loaders for models and assets.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import streamlit as st
from PIL import Image
from typing import Any, Optional


# ── Resolve paths relative to project root ────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


# ── Model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_emotion_model():
    path = _path("project", "models", "emotion_model.pkl")
    if not os.path.exists(path):
        st.error(f"Emotion model not found: {path}")
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_depression_model():
    path = _path("project", "models", "depression_model.pkl")
    if not os.path.exists(path):
        st.error(f"Depression model not found: {path}")
        return None
    return joblib.load(path)


# ── Image loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_image(filename: str) -> Optional[Image.Image]:
    path = _path("project", "outputs", filename)
    if not os.path.exists(path):
        return None
    return Image.open(path)


# ── Predict helpers ───────────────────────────────────────────────────────────

def predict_emotion(text: str, model) -> dict:
    """Run emotion model on text. Returns {emotion, confidence, all_probs}."""
    try:
        proba   = model.predict_proba([text])[0]
        classes = model.classes_
        idx     = proba.argmax()
        return {
            "emotion":    classes[idx],
            "confidence": float(proba[idx]),
            "all_probs":  {c: float(p) for c, p in zip(classes, proba)},
        }
    except Exception as e:
        return {"emotion": "neutral", "confidence": 0.0, "all_probs": {}, "error": str(e)}


def predict_depression(features: dict, model) -> dict:
    """Run depression model on feature dict. Returns {prediction, probability, risk_label}."""
    import pandas as pd
    try:
        df   = pd.DataFrame([features])
        prob = float(model.predict_proba(df)[0][1])
        pred = int(prob >= 0.5)
        if prob < 0.35:
            label = "Low"
        elif prob < 0.65:
            label = "Moderate"
        else:
            label = "High"
        return {"prediction": pred, "probability": round(prob, 4), "risk_label": label}
    except Exception as e:
        return {"prediction": 0, "probability": 0.0, "risk_label": "Unknown", "error": str(e)}
