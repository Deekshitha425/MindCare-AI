"""
utils/metrics_loader.py
Load model metrics dynamically from JSON files.
"""
import os, json
import streamlit as st
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _path(*parts):
    return os.path.join(ROOT, *parts)

@st.cache_data(show_spinner=False)
def load_emotion_metrics() -> Dict[str, Any]:
    path = _path("project", "metrics", "emotion_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_depression_metrics() -> Dict[str, Any]:
    path = _path("project", "metrics", "depression_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
