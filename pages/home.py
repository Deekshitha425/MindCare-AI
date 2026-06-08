"""
pages/home.py — Home / Landing Page
"""
import streamlit as st
from components.cards import kpi_card, section_header
from utils.metrics_loader import load_emotion_metrics, load_depression_metrics


def render():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1.5rem;">
        <div class="hero-badge">AI-Powered Mental Health Support</div>
        <h1 style="font-size:2.6rem; font-weight:800; color:#1E293B; margin:0; line-height:1.2;">
            🧠 MindCare AI
        </h1>
        <p style="font-size:1.1rem; color:#64748B; margin-top:10px; max-width:560px; margin-left:auto; margin-right:auto; line-height:1.6;">
            Mental Health Assessment &amp; Emotion Support System
        </p>
        <p style="font-size:0.9rem; color:#94A3B8; max-width:520px; margin:6px auto 0;">
            AI-powered emotion detection and mental health risk assessment platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    em = load_emotion_metrics()
    dep = load_depression_metrics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "🎭",
            f"{em.get('accuracy', 0):.1%}",
            "Emotion Model Accuracy",
            "#4F8EF7",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "📊",
            f"{em.get('f1_score', 0):.1%}",
            "Emotion F1 Score",
            "#7EC8E3",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(
            "🧬",
            f"{dep.get('accuracy', 0):.1%}",
            "Depression Model Accuracy",
            "#A8E6CF",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "📈",
            f"{dep.get('roc_auc', 0):.1%}",
            "Depression ROC-AUC",
            "#F59E0B",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Cards ─────────────────────────────────────────────────────────
    section_header("What can MindCare AI do?")

    features = [
        ("💬", "Emotion Support Chat",
         "Talk freely and receive empathetic, emotion-aware responses powered by our local NLP model.",
         "Chat", "Start Chatting →"),
        ("📊", "Depression Risk Assessment",
         "Complete a structured assessment form and receive an instant AI-powered risk analysis.",
         "Assessment", "Take Assessment →"),
        ("📈", "Analytics Dashboard",
         "Explore trends in your emotion patterns and risk assessments through interactive charts.",
         "Analytics", "View Analytics →"),
        ("📋", "Assessment History",
         "Review past assessments, track changes over time, and export your data.",
         "History", "View History →"),
    ]

    c1, c2 = st.columns(2)
    cols = [c1, c2, c1, c2]
    for col, (icon, title, desc, page_key, btn_label) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom:1rem;">
                <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
                <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:6px;">{title}</div>
                <div style="font-size:0.85rem; color:#64748B; line-height:1.5; margin-bottom:12px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(btn_label, key=f"home_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    section_header("How It Works")
    s1, s2, s3 = st.columns(3)
    steps = [
        ("1️⃣", "Choose a Feature",
         "Select either the Emotion Support Chat or the Depression Risk Assessment from the sidebar."),
        ("2️⃣", "Interact with AI",
         "Chat freely or fill in the assessment form. The local AI models analyse your input instantly."),
        ("3️⃣", "Get Insights",
         "Receive emotion detection, risk scores, personalised recommendations, and track your history."),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3], steps):
        with col:
            st.markdown(f"""
            <div class="info-card" style="text-align:center;">
                <div style="font-size:1.8rem; margin-bottom:8px;">{num}</div>
                <div style="font-weight:700; margin-bottom:6px; color:#1E293B;">{title}</div>
                <div style="font-size:0.84rem; color:#64748B; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This application is intended for educational and self-awareness
        purposes only and is <b>not a substitute for professional medical advice</b>.
        If you are experiencing a mental health crisis, please contact a qualified mental health
        professional or a crisis helpline immediately.
    </div>
    """, unsafe_allow_html=True)
