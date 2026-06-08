"""
pages/about.py — About Page
Dynamic metrics loaded from JSON. Model images displayed in cards.
"""
import streamlit as st
from components.cards import section_header, kpi_card
from utils.metrics_loader import load_emotion_metrics, load_depression_metrics
from utils.data_loader import load_image


def _metric_row(label: str, value):
    if isinstance(value, float):
        disp = f"{value:.4f}  ({value:.1%})"
    else:
        disp = str(value)
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding:8px 0; border-bottom:1px solid #F1F5F9;">
        <span style="font-size:0.88rem; color:#64748B;">{label}</span>
        <span style="font-size:0.88rem; font-weight:700; color:#4F8EF7;">{disp}</span>
    </div>
    """, unsafe_allow_html=True)


def render():
    section_header("ℹ️ About MindCare AI")

    # ── Project Overview ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-card">
        <div style="font-size:1.05rem; font-weight:700; margin-bottom:10px; color:#1E293B;">
            🧠 Project Overview
        </div>
        <p style="color:#475569; line-height:1.7; font-size:0.9rem;">
            MindCare AI is a portfolio-quality, end-to-end machine learning system for
            <b>emotion detection</b> and <b>depression risk assessment</b>.
            It combines NLP, classical machine learning, and a context-aware chatbot
            — all running locally without any external AI APIs.
        </p>
        <p style="color:#475569; line-height:1.7; font-size:0.9rem; margin-top:6px;">
            Built to demonstrate real-world ML engineering skills including text classification,
            multi-model comparison, automated preprocessing pipelines, feature engineering,
            and modular software architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    section_header("📂 Datasets")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:8px; color:#1E293B;">🎭 Emotion Dataset</div>
            <div style="font-size:0.85rem; color:#475569; line-height:1.6;">
                <b>Source:</b> Kaggle Emotion Dataset<br>
                <b>Train:</b> 21,400 rows<br>
                <b>Test:</b> 5,400 rows<br>
                <b>Classes:</b> sadness · joy · love · anger · fear · surprise · neutral<br>
                <b>Features:</b> Raw subtitle / text input
            </div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:8px; color:#1E293B;">🧬 Student Depression Dataset</div>
            <div style="font-size:0.85rem; color:#475569; line-height:1.6;">
                <b>Source:</b> Kaggle Student Depression Dataset<br>
                <b>Rows:</b> 27,901<br>
                <b>Target:</b> Depression (0 / 1)<br>
                <b>Class Split:</b> 16,336 positive · 11,565 negative<br>
                <b>Features:</b> 16 demographic, academic &amp; lifestyle variables
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model 1: Emotion Detection ────────────────────────────────────────────
    section_header("🎭 Module 1 — Emotion Detection")
    em = load_emotion_metrics()

    m1, m2 = st.columns([1, 1.2])
    with m1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:10px; color:#1E293B;">Architecture</div>
            <div style="font-size:0.85rem; color:#475569; line-height:1.7;">
                <b>Vectoriser:</b> TF-IDF (unigrams + bigrams, 50k vocab)<br>
                <b>Classifier:</b> CalibratedLinearSVC<br>
                <b>Pipeline:</b> sklearn Pipeline<br>
                <b>Preprocessing:</b> Lowercasing, URL removal, stopword filtering, lemmatization
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-card"><div style="font-weight:700; margin-bottom:10px; color:#1E293B;">Test Set Metrics</div>', unsafe_allow_html=True)
        for label, key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 Score","f1_score")]:
            _metric_row(label, em.get(key, "N/A"))
        st.markdown('</div>', unsafe_allow_html=True)

    with m2:
        img = load_image("emotion_confusion_matrix.png")
        if img:
            st.markdown('<div class="info-card"><div style="font-weight:700; margin-bottom:8px;">Confusion Matrix</div>', unsafe_allow_html=True)
            st.image(img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model 2: Depression Prediction ────────────────────────────────────────
    section_header("🧬 Module 2 — Depression Prediction")
    dep = load_depression_metrics()
    best_model = dep.get("best_model", "XGBoost")

    pm1, pm2 = st.columns([1, 1.2])
    with pm1:
        st.markdown(f"""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:10px; color:#1E293B;">Architecture</div>
            <div style="font-size:0.85rem; color:#475569; line-height:1.7;">
                <b>Best Model:</b> <span style="color:#4F8EF7; font-weight:700;">{best_model}</span><br>
                <b>Preprocessing:</b> ColumnTransformer (median imputation + StandardScaler for numeric, mode imputation + OneHotEncoder for categorical)<br>
                <b>Selection:</b> 5-fold Stratified CV by F1 Score<br>
                <b>Candidates:</b> Logistic Regression · Random Forest · XGBoost · LightGBM
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-card"><div style="font-weight:700; margin-bottom:10px; color:#1E293B;">Test Set Metrics</div>', unsafe_allow_html=True)
        for label, key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 Score","f1_score"),("ROC-AUC","roc_auc")]:
            _metric_row(label, dep.get(key, "N/A"))
        st.markdown('</div>', unsafe_allow_html=True)

    with pm2:
        img = load_image("depression_confusion_matrix.png")
        if img:
            st.markdown('<div class="info-card"><div style="font-weight:700; margin-bottom:8px;">Confusion Matrix</div>', unsafe_allow_html=True)
            st.image(img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Additional model charts ────────────────────────────────────────────────
    img_col1, img_col2 = st.columns(2)
    for col, fname, title in [
        (img_col1, "depression_model_comparison.png", "Model Comparison"),
        (img_col2, "depression_feature_importance.png", "Feature Importance"),
    ]:
        img = load_image(fname)
        if img:
            with col:
                st.markdown(f'<div class="info-card"><div style="font-weight:700; margin-bottom:8px;">{title}</div>', unsafe_allow_html=True)
                st.image(img, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CV Results table ─────────────────────────────────────────────────────
    cv = dep.get("cv_results", {})
    if cv:
        section_header("📊 Cross-Validation Results")
        import pandas as pd
        cv_df = pd.DataFrame([
            {"Model": k, "CV F1 Mean": f"{v['cv_f1_mean']:.4f}", "CV F1 Std": f"±{v['cv_f1_std']:.4f}",
             "Best": "⭐" if k == best_model else ""}
            for k, v in cv.items()
        ])
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🛠️ Tech Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:8px;">ML & Data</div>
            <div style="font-size:0.84rem; color:#475569; line-height:1.8;">
                scikit-learn · XGBoost · LightGBM<br>
                pandas · numpy · joblib<br>
                NLTK · TF-IDF · LinearSVC
            </div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:8px;">Visualisation</div>
            <div style="font-size:0.84rem; color:#475569; line-height:1.8;">
                Plotly · Seaborn · Matplotlib<br>
                Streamlit · HTML/CSS<br>
                Power BI (integration ready)
            </div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; margin-bottom:8px;">Engineering</div>
            <div style="font-size:0.84rem; color:#475569; line-height:1.8;">
                Modular architecture · Type hints<br>
                Session state · Caching<br>
                JSON persistence · Error handling
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer" style="margin-top:1rem;">
        ⚠️ <b>Disclaimer:</b> MindCare AI is an academic portfolio project.
        It is <b>not a substitute</b> for professional mental health care.
        If you or someone you know is in crisis, please contact a qualified
        mental health professional or a crisis helpline immediately.
        <br><br>
        📞 iCall (India): <b>9152987821</b> &nbsp;|&nbsp;
        📞 Vandrevala Foundation: <b>1860-2662-345</b> (24/7)
    </div>
    """, unsafe_allow_html=True)
