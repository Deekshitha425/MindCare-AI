"""
pages/history.py — Assessment History
"""
import io
import pandas as pd
import streamlit as st
from components.cards import section_header, risk_badge
from utils.history_manager import load_assessment_history


def render():
    section_header("📋 Assessment History")

    username = st.session_state.get("username", "guest")
    st.markdown(
        f'<p style="color:#64748B; font-size:0.9rem; margin-top:-0.5rem; margin-bottom:1.4rem;">'
        f'Showing past depression risk assessments for <b>{username}</b>.</p>',
        unsafe_allow_html=True,
    )

    history = load_assessment_history()

    if not history:
        st.info("📋 No assessments recorded yet. Take a Depression Risk Assessment to get started!")
        return

    # ── Build dataframe ───────────────────────────────────────────────────────
    rows = []
    for rec in history:
        ts  = rec.get("timestamp", "")[:16].replace("T", " ")
        row = {
            "Date":        ts,
            "Risk Level":  rec.get("risk_level", "Unknown"),
            "Probability": f"{rec.get('probability', 0):.1%}",
        }
        for k, v in rec.get("input_features", {}).items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        search = st.text_input("🔍 Search", placeholder="Filter by any value…")
    with ctrl2:
        risk_filter = st.selectbox("Filter by Risk", ["All", "High", "Moderate", "Low"])
    with ctrl3:
        sort_col = st.selectbox("Sort by", ["Date", "Risk Level", "Probability"])

    # Apply filters
    filtered = df.copy()
    if search:
        mask = filtered.apply(
            lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
        )
        filtered = filtered[mask]
    if risk_filter != "All":
        filtered = filtered[filtered["Risk Level"] == risk_filter]
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=(sort_col == "Date"))

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Records", len(history))
    with k2:
        st.metric("Shown", len(filtered))
    with k3:
        high_count = sum(1 for r in history if r.get("risk_level") == "High")
        st.metric("High Risk", high_count)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Table (no jinja2 / .style needed) ────────────────────────────────────
    display_cols = ["Date", "Risk Level", "Probability", "Age", "Gender",
                    "Sleep Duration", "Academic Pressure", "Financial Stress"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Colour the Risk Level column manually via a helper column
    def _risk_icon(val):
        return {"High": "🔴 High", "Moderate": "🟡 Moderate", "Low": "🟢 Low"}.get(val, val)

    display_df = filtered[display_cols].copy()
    if "Risk Level" in display_df.columns:
        display_df["Risk Level"] = display_df["Risk Level"].apply(_risk_icon)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Detail expander ───────────────────────────────────────────────────────
    if len(filtered) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("🔎 Detailed View")
        idx = st.selectbox(
            "Select record to view details",
            range(len(filtered)),
            format_func=lambda i: f"{filtered.iloc[i]['Date']} — {filtered.iloc[i]['Risk Level']}",
        )

        rec  = filtered.iloc[idx]
        risk = rec["Risk Level"]
        color = {"High": "#EF4444", "Moderate": "#F59E0B", "Low": "#22C55E"}.get(risk, "#64748B")

        st.markdown(f"""
        <div class="info-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-weight:700; font-size:1rem;">Assessment on {rec['Date']}</div>
                <span style="background:{color}20; color:{color}; padding:4px 12px;
                             border-radius:20px; font-weight:600; font-size:0.85rem;">
                    {risk} Risk — {rec['Probability']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        det_cols = st.columns(3)
        for i, (k, v) in enumerate(rec.items()):
            if k not in ("Date", "Risk Level", "Probability"):
                with det_cols[i % 3]:
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border-radius:8px; padding:8px 12px; margin-bottom:6px;">
                        <div style="font-size:0.72rem; color:#64748B; text-transform:uppercase; font-weight:600;">{k}</div>
                        <div style="font-size:0.92rem; font-weight:600; color:#1E293B;">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Export as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"mindcare_{st.session_state.get('username','user')}_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
