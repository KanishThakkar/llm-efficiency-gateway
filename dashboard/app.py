"""
Streamlit dashboard — LLM Efficiency Gateway
Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from efficiency_gateway.core.metrics_store import MetricsStore

st.set_page_config(
    page_title="LLM Efficiency Gateway",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------------
# load data
# ------------------------------------------------------------------
store = MetricsStore(db_path=str(ROOT / "data" / "metrics.db"))
summary = store.summary()
rows = store.fetch_all()
df = pd.DataFrame(rows) if rows else pd.DataFrame()

st.title("⚡ LLM Efficiency Gateway — Dashboard")

if df.empty:
    st.info("No runs recorded yet. Run `python scripts/run_workflow_step10.py` to generate data.")
    st.stop()

df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")
df["cache_hit_label"] = df["cache_hit"].map({1: "Cache Hit", 0: "LLM Call"})

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------
total = int(summary.get("total_runs", 0))
hits = int(summary.get("cache_hits", 0) or 0)
hit_rate = round(hits / total * 100, 1) if total else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Runs", total)
c2.metric("Cache Hit Rate", f"{hit_rate}%")
c3.metric("Avg Token Reduction", f"{summary.get('avg_token_reduction_pct', 0):.1f}%")
c4.metric("Avg Cost Reduction", f"{summary.get('avg_cost_reduction_pct', 0):.1f}%")
c5.metric("Avg Quality Score", f"{summary.get('avg_quality_score', 0):.3f}")
c6.metric("Avg Latency (ms)", f"{summary.get('avg_latency_ms', 0):.0f}")

st.markdown("---")

# ------------------------------------------------------------------
# row 1: pareto + token savings over time
# ------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Cost–Quality Pareto")
    pareto_df = df[df["selected_model"] != "cache"].copy()
    pareto_df["bubble_size"] = pareto_df["tokens_saved"].clip(lower=1)
    fig_pareto = px.scatter(
        pareto_df,
        x="estimated_cost",
        y="quality_score",
        color="selected_model",
        size="bubble_size",
        size_max=30,
        hover_data=["question", "token_reduction_pct", "cost_reduction_pct"],
        labels={
            "estimated_cost": "Estimated Cost (USD)",
            "quality_score": "Quality Score",
            "selected_model": "Model",
        },
        title="Each bubble = one run  |  size = tokens saved",
    )
    fig_pareto.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_pareto, use_container_width=True)

with col_right:
    st.subheader("Token Savings per Run")
    fig_tokens = px.bar(
        df.sort_values("timestamp_dt"),
        x="timestamp_dt",
        y=["baseline_input_tokens", "input_tokens"],
        barmode="group",
        labels={"value": "Tokens", "timestamp_dt": "Time", "variable": ""},
        color_discrete_map={
            "baseline_input_tokens": "#EF553B",
            "input_tokens": "#00CC96",
        },
        title="Baseline vs Optimised Input Tokens",
    )
    fig_tokens.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_tokens, use_container_width=True)

# ------------------------------------------------------------------
# row 2: model usage pie + latency histogram
# ------------------------------------------------------------------
col2_left, col2_right = st.columns(2)

with col2_left:
    st.subheader("Model Usage Breakdown")
    model_counts = df["selected_model"].value_counts().reset_index()
    model_counts.columns = ["model", "count"]
    fig_pie = px.pie(
        model_counts,
        names="model",
        values="count",
        hole=0.4,
        title="Requests routed to each model",
    )
    fig_pie.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2_right:
    st.subheader("Latency Distribution")
    fig_lat = px.histogram(
        df,
        x="latency_ms",
        color="cache_hit_label",
        nbins=20,
        labels={"latency_ms": "Latency (ms)", "cache_hit_label": ""},
        title="Cache hits are much faster",
    )
    fig_lat.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_lat, use_container_width=True)

# ------------------------------------------------------------------
# row 3: quality scores by model
# ------------------------------------------------------------------
st.subheader("Quality Score by Model")
fig_qual = px.box(
    df[df["selected_model"] != "cache"],
    x="selected_model",
    y="quality_score",
    color="selected_model",
    points="all",
    labels={"selected_model": "Model", "quality_score": "Quality Score"},
)
fig_qual.update_layout(showlegend=False, margin=dict(t=20, b=20))
st.plotly_chart(fig_qual, use_container_width=True)

# ------------------------------------------------------------------
# row 4: cost reduction timeline
# ------------------------------------------------------------------
st.subheader("Cost & Token Reduction Over Time")
fig_savings = go.Figure()
fig_savings.add_trace(go.Scatter(
    x=df["timestamp_dt"],
    y=df["token_reduction_pct"],
    mode="lines+markers",
    name="Token reduction %",
    line=dict(color="#636EFA"),
))
fig_savings.add_trace(go.Scatter(
    x=df["timestamp_dt"],
    y=df["cost_reduction_pct"],
    mode="lines+markers",
    name="Cost reduction %",
    line=dict(color="#EF553B"),
))
fig_savings.update_layout(
    yaxis_title="Reduction %",
    xaxis_title="Time",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_savings, use_container_width=True)

# ------------------------------------------------------------------
# recent runs table
# ------------------------------------------------------------------
st.subheader("Recent Runs")
display_cols = [
    "timestamp_dt", "question", "status", "cache_hit_label",
    "selected_model", "token_reduction_pct", "cost_reduction_pct",
    "quality_score", "latency_ms",
]
st.dataframe(
    df[display_cols].rename(columns={
        "timestamp_dt": "Time",
        "question": "Question",
        "status": "Status",
        "cache_hit_label": "Cache",
        "selected_model": "Model",
        "token_reduction_pct": "Token %",
        "cost_reduction_pct": "Cost %",
        "quality_score": "Quality",
        "latency_ms": "Latency ms",
    }).head(50),
    use_container_width=True,
)
