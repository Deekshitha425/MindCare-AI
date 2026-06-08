"""
components/cards.py
Reusable HTML card components.
"""
import streamlit as st


def kpi_card(icon: str, value: str, label: str, border_color: str = "#4F8EF7") -> str:
    return f"""
    <div class="kpi-card" style="border-top-color:{border_color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def feature_card(icon: str, title: str, desc: str, btn_label: str, page_key: str):
    col_btn = st.container()
    st.markdown(f"""
    <div class="feature-card">
        <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
        <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:6px;">{title}</div>
        <div style="font-size:0.85rem; color:#64748B; line-height:1.5;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def info_card(content: str):
    st.markdown(f'<div class="info-card">{content}</div>', unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def emotion_badge(emotion: str) -> str:
    cls_map = {
        "joy": "emotion-joy", "sadness": "emotion-sadness",
        "anger": "emotion-anger", "fear": "emotion-fear",
        "love": "emotion-love", "surprise": "emotion-surprise",
        "neutral": "emotion-neutral", "tired": "emotion-tired",
        "lonely": "emotion-lonely",
    }
    cls = cls_map.get(emotion.lower(), "emotion-neutral")
    label_map = {
        "joy": "😊 Joy", "sadness": "😢 Sadness", "anger": "😠 Anger",
        "fear": "😨 Fear", "love": "❤️ Love", "surprise": "😲 Surprise",
        "neutral": "😐 Neutral", "tired": "😴 Tired", "lonely": "🤍 Lonely",
    }
    label = label_map.get(emotion.lower(), f"🔵 {emotion.title()}")
    return f'<span class="emotion-badge {cls}">{label}</span>'


def risk_badge(risk: str) -> str:
    r = risk.lower()
    if r == "low":
        return '<span class="risk-low">✅ Low Risk</span>'
    elif r == "moderate":
        return '<span class="risk-mod">⚠️ Moderate Risk</span>'
    else:
        return '<span class="risk-high">🔴 High Risk</span>'
