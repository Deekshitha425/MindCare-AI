"""
components/charts.py
Plotly chart builders for the Analytics Dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any

PALETTE = ["#4F8EF7","#7EC8E3","#A8E6CF","#F59E0B","#EF4444","#A78BFA","#FB7185"]

EMOTION_COLORS = {
    "joy":      "#F59E0B",
    "sadness":  "#3B82F6",
    "anger":    "#EF4444",
    "fear":     "#8B5CF6",
    "love":     "#EC4899",
    "surprise": "#F97316",
    "neutral":  "#94A3B8",
    "tired":    "#06B6D4",
    "lonely":   "#22C55E",
}

RISK_COLORS = {"Low": "#22C55E", "Moderate": "#F59E0B", "High": "#EF4444"}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#1E293B"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def emotion_pie(emotion_counts: Dict[str, int]) -> go.Figure:
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS.get(e, "#94A3B8") for e in labels]

    fig = go.Figure(go.Pie(
        labels=[l.title() for l in labels],
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(title="Emotion Distribution", **CHART_LAYOUT,
                      legend=dict(orientation="h", y=-0.15))
    return fig


def emotion_trend_line(history: List[Dict]) -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame([
        {"ts": r["timestamp"][:10],
         "emotion": r.get("emotion", {}).get("emotion", "neutral") if isinstance(r.get("emotion"), dict) else r.get("emotion", "neutral")}
        for r in history
    ])
    counts = df.groupby(["ts", "emotion"]).size().reset_index(name="count")
    fig = px.line(counts, x="ts", y="count", color="emotion",
                  color_discrete_map=EMOTION_COLORS,
                  markers=True, title="Emotion Trends Over Time")
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(title="Date", showgrid=False)
    fig.update_yaxes(title="Count", showgrid=True, gridcolor="#F1F5F9")
    return fig


def risk_donut(risk_counts: Dict[str, int]) -> go.Figure:
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = [RISK_COLORS.get(l, "#94A3B8") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(title="Risk Level Distribution", **CHART_LAYOUT,
                      legend=dict(orientation="h", y=-0.15))
    return fig


def assessment_trend_bar(history: List[Dict]) -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame([{"date": r["timestamp"][:10], "risk": r.get("risk_level","Unknown")} for r in history])
    counts = df.groupby(["date", "risk"]).size().reset_index(name="count")
    fig = px.bar(counts, x="date", y="count", color="risk",
                 color_discrete_map=RISK_COLORS,
                 barmode="stack", title="Assessment Frequency by Date")
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(title="Date", showgrid=False)
    fig.update_yaxes(title="Assessments", showgrid=True, gridcolor="#F1F5F9")
    return fig


def probability_gauge(probability: float, risk_label: str) -> go.Figure:
    color = RISK_COLORS.get(risk_label, "#94A3B8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B",
                     "tickfont": {"size": 11}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#DCFCE7"},
                {"range": [35, 65], "color": "#FEF3C7"},
                {"range": [65, 100],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": round(probability * 100, 1),
            },
        },
        title={"text": "Depression Risk Probability", "font": {"size": 14, "color": "#64748B"}},
    ))
    gauge_layout = {**CHART_LAYOUT, "margin": dict(l=30, r=30, t=50, b=10)}
    fig.update_layout(height=260, **gauge_layout)
    return fig


def cv_comparison_bar(cv_results: Dict[str, Dict]) -> go.Figure:
    models = list(cv_results.keys())
    means  = [cv_results[m]["cv_f1_mean"] for m in models]
    stds   = [cv_results[m]["cv_f1_std"]  for m in models]

    fig = go.Figure(go.Bar(
        x=models, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=PALETTE[:len(models)],
        text=[f"{v:.4f}" for v in means],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>F1: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Cross-Validation F1 Score — Model Comparison",
                      yaxis=dict(range=[0.8, 0.88], title="CV F1 Score"),
                      xaxis_title="Model",
                      **CHART_LAYOUT)
    return fig
