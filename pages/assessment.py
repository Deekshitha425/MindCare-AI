"""
pages/assessment.py — Depression Risk Assessment
Structured form → XGBoost model → risk output + recommendations.
ML models and metrics are NOT changed.
"""
import datetime
import streamlit as st
import plotly.graph_objects as go
from components.cards import section_header, risk_badge
from components.charts import probability_gauge
from utils.data_loader import load_depression_model, predict_depression
from utils.history_manager import save_assessment_record

RECOMMENDATIONS = {
    "Low": [
        "🌿 Keep up your self-care routine — it's clearly working.",
        "🧘 Try a 5-minute mindfulness or breathing exercise daily.",
        "📓 Keep a brief gratitude journal — 3 things you appreciated each day.",
        "🚶 Stay physically active with at least 20–30 minutes of movement.",
        "📱 Stay connected with friends or family, even with a short message.",
    ],
    "Moderate": [
        "💬 Consider speaking with a counsellor or therapist.",
        "😴 Establish a consistent sleep routine (same bedtime and wake time).",
        "🌬️ Practice deep breathing: 4 counts in, hold 4, exhale 6.",
        "🤝 Reach out to someone you trust and share how you're feeling.",
        "☕ Reduce caffeine and alcohol, which can amplify anxiety and low mood.",
        "🏃 Regular physical exercise has strong evidence for improving mood.",
    ],
    "High": [
        "🆘 Please consider reaching out to a mental health professional as soon as possible.",
        "📞 iCall (India): **9152987821**",
        "📞 Vandrevala Foundation: **1860-2662-345** (24/7)",
        "🤗 Talk to someone you trust — you don't have to go through this alone.",
        "🌱 Try grounding: name 5 things you see, 4 you touch, 3 you hear.",
        "💧 Prioritise basic self-care: hydration, regular meals, and rest.",
    ],
}

RISK_COLORS = {"Low": "#22C55E", "Moderate": "#F59E0B", "High": "#EF4444"}
RISK_BG     = {"Low": "#F0FDF4", "Moderate": "#FFFBEB", "High": "#FFF1F2"}


def render():
    section_header("📊 Depression Risk Assessment")
    st.markdown(
        '<p style="color:#64748B; font-size:0.9rem; margin-top:-0.5rem; margin-bottom:1.4rem;">'
        'Complete the form below. Our XGBoost model will analyse your responses '
        'and estimate your depression risk level. This is a self-screening tool only.'
        '</p>', unsafe_allow_html=True
    )

    model = load_depression_model()
    if model is None:
        st.error("Depression model not found. Please ensure depression_model.pkl exists in project/models/")
        return

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("assessment_form"):
        st.markdown("#### 👤 Personal Information")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with r1c2:
            age = st.slider("Age", 15, 60, 22)
        with r1c3:
            profession = st.selectbox("Profession", ["Student", "Working Professional"])

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            city = st.text_input("City", value="Hyderabad")
        with r2c2:
            degree = st.selectbox("Degree / Qualification",
                ["B.Tech", "B.Sc", "B.Com", "BA", "MBA", "MCA",
                 "M.Tech", "M.Sc", "PhD", "Class 12", "Other"])

        st.markdown("---")
        st.markdown("#### 📚 Academic & Work Factors")

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            academic_pressure = st.select_slider(
                "Academic Pressure", options=[1, 2, 3, 4, 5],
                value=3, format_func=lambda x: f"{x} — {'Low' if x<=2 else 'High' if x>=4 else 'Moderate'}"
            )
        with r3c2:
            work_pressure = st.select_slider(
                "Work Pressure", options=[1, 2, 3, 4, 5], value=2,
                format_func=lambda x: f"{x} — {'Low' if x<=2 else 'High' if x>=4 else 'Moderate'}"
            )
        with r3c3:
            work_study_hours = st.slider("Work / Study Hours per Day", 1, 16, 8)

        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            cgpa = st.slider("CGPA (out of 10)", 0.0, 10.0, 7.5, 0.1)
        with r4c2:
            study_satisfaction = st.select_slider(
                "Study Satisfaction", options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: f"{x} — {'Low' if x<=2 else 'High' if x>=4 else 'Moderate'}"
            )
        with r4c3:
            job_satisfaction = st.select_slider(
                "Job Satisfaction", options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: f"{x} — {'Low' if x<=2 else 'High' if x>=4 else 'Moderate'}"
            )

        st.markdown("---")
        st.markdown("#### 🏥 Health & Lifestyle")

        r5c1, r5c2, r5c3 = st.columns(3)
        with r5c1:
            sleep = st.selectbox("Sleep Duration",
                ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        with r5c2:
            diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        with r5c3:
            financial_stress = st.select_slider(
                "Financial Stress", options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: f"{x} — {'Low' if x<=2 else 'High' if x>=4 else 'Moderate'}"
            )

        r6c1, r6c2 = st.columns(2)
        with r6c1:
            suicidal = st.selectbox(
                "Have you ever had suicidal thoughts?",
                ["No", "Yes"],
                help="This information helps improve the accuracy of the risk assessment."
            )
        with r6c2:
            family_history = st.selectbox(
                "Family History of Mental Illness", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 Assess My Risk", use_container_width=True
        )

    # ── Results ───────────────────────────────────────────────────────────────
    if submitted:
        features = {
            "Gender":       gender,
            "Age":          age,
            "City":         city,
            "Profession":   profession,
            "Academic Pressure":  academic_pressure,
            "Work Pressure":      work_pressure,
            "CGPA":               cgpa,
            "Study Satisfaction": study_satisfaction,
            "Job Satisfaction":   job_satisfaction,
            "Sleep Duration":     sleep,
            "Dietary Habits":     diet,
            "Degree":             degree,
            "Have you ever had suicidal thoughts ?": suicidal,
            "Work/Study Hours":   float(work_study_hours),
            "Financial Stress":   financial_stress,
            "Family History of Mental Illness": family_history,
        }

        result = predict_depression(features, model)
        risk   = result["risk_label"]
        prob   = result["probability"]
        color  = RISK_COLORS.get(risk, "#64748B")
        bg     = RISK_BG.get(risk, "#F8FAFC")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        section_header("📋 Your Assessment Results")

        # Risk banner
        st.markdown(f"""
        <div style="background:{bg}; border:2px solid {color}; border-radius:14px;
                    padding:1.4rem 1.8rem; margin-bottom:1.2rem; text-align:center;">
            <div style="font-size:0.82rem; font-weight:600; color:{color};
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                Depression Risk Level
            </div>
            <div style="font-size:2.4rem; font-weight:800; color:{color}; line-height:1.1;">
                {risk} Risk
            </div>
            <div style="font-size:0.92rem; color:#64748B; margin-top:6px;">
                Probability Score: <b style="color:{color};">{prob:.1%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge + metrics
        g_col, m_col = st.columns([1.2, 1])
        with g_col:
            st.plotly_chart(probability_gauge(prob, risk),
                            use_container_width=True, config={"displayModeBar": False})
        with m_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Risk Level", risk)
            st.metric("Probability", f"{prob:.1%}")
            st.metric("Model", "XGBoost")

            if risk == "High":
                st.error("⚠️ Please consider seeking professional support.")
            elif risk == "Moderate":
                st.warning("💛 Consider talking to a counsellor.")
            else:
                st.success("✅ Low risk detected. Keep up the good work!")

        # Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("💡 Personalised Recommendations")
        recs = RECOMMENDATIONS.get(risk, RECOMMENDATIONS["Low"])
        rec_cols = st.columns(min(3, len(recs)))
        for i, rec in enumerate(recs):
            with rec_cols[i % len(rec_cols)]:
                st.markdown(f"""
                <div class="feature-card" style="min-height:90px; margin-bottom:8px;">
                    <div style="font-size:0.87rem; color:#1E293B; line-height:1.5;">{rec}</div>
                </div>
                """, unsafe_allow_html=True)

        # Save to user-specific history
        username = st.session_state.get("username", "guest")
        save_assessment_record({
            "username":     username,
            "timestamp":    datetime.datetime.now().isoformat(),
            "prediction":   result["prediction"],
            "probability":  prob,
            "risk_level":   risk,
            "input_features": features,
        })
        st.success(f"✅ Assessment saved to {username}'s history.")

        # Disclaimer
        st.markdown("""
        <div class="disclaimer" style="margin-top:1rem;">
            ⚠️ <b>Important:</b> This assessment is for self-awareness purposes only and is
            <b>not a clinical diagnosis</b>. Please consult a qualified mental health professional
            for a proper evaluation.
        </div>
        """, unsafe_allow_html=True)
