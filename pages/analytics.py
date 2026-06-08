"""
pages/analytics.py — Analytics Dashboard
"""
import streamlit as st
from collections import Counter
import pandas as pd
from components.cards import section_header, kpi_card
from components.charts import (
    emotion_pie, emotion_trend_line, risk_donut,
    assessment_trend_bar, cv_comparison_bar,
)
from utils.history_manager import load_conversation_history, load_assessment_history
from utils.metrics_loader import load_depression_metrics




def render():
    section_header("📈 Analytics Dashboard")
    st.markdown(
        '<p style="color:#64748B; font-size:0.9rem; margin-top:-0.5rem; margin-bottom:1.4rem;">'
        'Interactive insights from your emotion chat history and depression assessments.'
        '</p>', unsafe_allow_html=True
    )

    chat_history       = load_conversation_history()
    assessment_history = load_assessment_history()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_assessments = len(assessment_history)
    avg_risk = (
        sum(r.get("probability", 0) for r in assessment_history) / total_assessments
        if total_assessments else 0
    )
    emotions = [
        r.get("emotion", {}).get("emotion", "neutral")
        if isinstance(r.get("emotion"), dict) else r.get("emotion", "neutral")
        for r in chat_history
    ]
    top_emotion = Counter(emotions).most_common(1)[0][0].title() if emotions else "N/A"
    high_risk_pct = (
        sum(1 for r in assessment_history if r.get("risk_level") == "High")
        / total_assessments * 100
        if total_assessments else 0
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("📋", str(total_assessments), "Total Assessments", "#4F8EF7"),
                    unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("📊", f"{avg_risk:.1%}", "Average Risk Score", "#F59E0B"),
                    unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("🎭", top_emotion, "Most Common Emotion", "#A8E6CF"),
                    unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("🔴", f"{high_risk_pct:.1f}%", "High Risk %", "#EF4444"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Emotion Charts ────────────────────────────────────────────────────────
    section_header("🎭 Emotion Analytics")

    if chat_history:
        ec1, ec2 = st.columns(2)
        with ec1:
            emotion_counts = Counter(emotions)
            st.plotly_chart(emotion_pie(dict(emotion_counts)),
                            use_container_width=True, config={"displayModeBar": False})
        with ec2:
            st.plotly_chart(emotion_trend_line(chat_history),
                            use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("💬 No chat history yet. Start a conversation in the Emotion Support Chat!")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Assessment Charts ─────────────────────────────────────────────────────
    section_header("🧬 Assessment Analytics")

    if assessment_history:
        ac1, ac2 = st.columns(2)
        with ac1:
            risk_counts = Counter(r.get("risk_level", "Unknown") for r in assessment_history)
            st.plotly_chart(risk_donut(dict(risk_counts)),
                            use_container_width=True, config={"displayModeBar": False})
        with ac2:
            st.plotly_chart(assessment_trend_bar(assessment_history),
                            use_container_width=True, config={"displayModeBar": False})

        # Risk probability timeline
        st.markdown("<br>", unsafe_allow_html=True)
        df_timeline = pd.DataFrame([
            {"Date": r["timestamp"][:10], "Probability": r.get("probability", 0),
             "Risk": r.get("risk_level", "Unknown")}
            for r in assessment_history
        ])
        import plotly.express as px
        RISK_COLORS = {"Low":"#22C55E","Moderate":"#F59E0B","High":"#EF4444"}
        fig = px.line(df_timeline, x="Date", y="Probability", color="Risk",
                      color_discrete_map=RISK_COLORS, markers=True,
                      title="Depression Risk Probability Over Time")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20,r=20,t=40,b=20),
            yaxis=dict(tickformat=".0%", range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("📊 No assessments yet. Take a Depression Risk Assessment to see analytics!")

    # ── CV Model comparison ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🤖 Model Training Comparison")
    dep_metrics = load_depression_metrics()
    cv_results  = dep_metrics.get("cv_results", {})
    if cv_results:
        st.plotly_chart(cv_comparison_bar(cv_results),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
        <div class="info-card">
            <b>Best Model Selected:</b>
            <span style="color:#4F8EF7; font-weight:700; font-size:1.05rem;">
                {dep_metrics.get('best_model', 'XGBoost')}
            </span>
            &nbsp;|&nbsp; CV F1: <b>{cv_results.get(dep_metrics.get('best_model','XGBoost'),{}).get('cv_f1_mean',0):.4f}</b>
        </div>
        """, unsafe_allow_html=True)

 