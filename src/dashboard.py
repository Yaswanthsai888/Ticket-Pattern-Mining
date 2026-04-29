"""
dashboard.py — Legacy vs DBB Ticket Pattern Mining Dashboard
=============================================================
Directly maps to Use Case 5 requirements:
  Tab 1  Migration Timeline          → "Distinguish Legacy vs DBB tickets"
  Tab 2  Pattern Discovery           → "Identify repeating patterns"
  Tab 3  Domain & Severity Analysis  → "Overlapping domains, modules"
  Tab 4  Cluster Deep-Dive           → "Same issue manifesting differently"
  Tab 5  Remediation Strategy        → "Shift-left, KB, automation"
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import argparse
import subprocess
import sys
from rag_pipeline import load_embedder as load_rag_embedder
from rag_pipeline import resolve_ticket

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploaded_datasets")
PIPELINE_STEPS = [
    "00 Classification",
    "01 Normalize",
    "02 Vectorize",
    "03 Clustering",
    "04 Metrics",
    "05 LLM Naming",
    "06 Executive Summary",
]

# ── Page Config ──
st.set_page_config(
    page_title="Legacy vs DBB - Ticket Pattern Mining",
    page_icon=":mag:",
    layout="wide",
)

# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
try:
    args, _ = parser.parse_known_args()
    base_dir = args.data_dir
except Exception:
    base_dir = DEFAULT_DATA_DIR


# ── Load Data ──
def load_data(base_dir):
    catalog = pd.read_csv(os.path.join(base_dir, "output", "cluster_catalog.csv"))
    pivot = pd.read_csv(os.path.join(base_dir, "output", "legacy_vs_dbb_pivot.csv"))
    tickets = pd.read_parquet(os.path.join(base_dir, "processed", "tickets_clustered.parquet"))
    summary_path = os.path.join(base_dir, "output", "executive_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return catalog, pivot, tickets, summary

@st.cache_resource
def load_embedder():
    return load_rag_embedder()

def get_available_datasets(base_dir):
    datasets = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, d)):
                if os.path.exists(os.path.join(base_dir, d, "output", "cluster_catalog.csv")):
                    datasets.append(d)
    return sorted(datasets)


def format_hours(hours):
    if pd.isna(hours):
        return "N/A"
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def build_pipeline_command(input_file, mode):
    return [
        sys.executable,
        os.path.join(PROJECT_ROOT, "run_pipeline.py"),
        input_file,
        "--output_dir",
        DEFAULT_DATA_DIR,
        "--mode",
        mode,
    ]


def start_pipeline_for_upload(input_file, mode):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    log_path = os.path.join(UPLOADS_DIR, f"{dataset_name}.pipeline.log")
    log_file = open(log_path, "w", encoding="utf-8")
    cmd = [
        *build_pipeline_command(input_file, mode),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()
    return process, log_path, dataset_name


def read_pipeline_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def pipeline_status():
    run = st.session_state.get("pipeline_run")
    if not run:
        return None

    process = run.get("process")
    returncode = process.poll()
    log = read_pipeline_log(run.get("log_path"))
    completed = "Pipeline completed successfully." in log
    failed = "Failed at step:" in log or (returncode is not None and returncode != 0)

    current_step_idx = -1
    for idx, step in enumerate(PIPELINE_STEPS):
        if step in log:
            current_step_idx = idx

    if completed:
        progress = 1.0
        label = "Pipeline complete"
    elif failed:
        progress = max((current_step_idx + 1) / len(PIPELINE_STEPS), 0.05)
        label = "Pipeline failed"
    elif current_step_idx >= 0:
        progress = min((current_step_idx + 0.35) / len(PIPELINE_STEPS), 0.98)
        label = f"Running {PIPELINE_STEPS[current_step_idx]}"
    else:
        progress = 0.03
        label = "Starting pipeline"

    return {
        **run,
        "returncode": returncode,
        "log": log,
        "progress": progress,
        "label": label,
        "completed": completed,
        "failed": failed,
        "running": returncode is None,
    }


@st.fragment(run_every="2s")
def render_pipeline_progress():
    status = pipeline_status()
    if not status:
        return False

    if status["completed"]:
        st.success(f"Pipeline completed for {status['dataset_name']}. You can select it from the sidebar.")
        st.progress(1.0, text="100% complete")
        st.session_state["last_pipeline_log"] = status["log"]
        st.session_state["selected_dataset"] = status["dataset_name"]
        st.session_state["pipeline_done_message"] = f"Pipeline completed for {status['dataset_name']}."
        st.session_state.pop("pipeline_run", None)
        st.rerun(scope="app")
        return False

    if status["failed"]:
        st.error(f"Pipeline failed for {status['dataset_name']}. Check the latest pipeline log in the sidebar.")
        st.progress(status["progress"], text=f"{int(status['progress'] * 100)}% - failed")
        st.session_state["last_pipeline_log"] = status["log"]
        st.session_state["pipeline_done_message"] = f"Pipeline failed for {status['dataset_name']}."
        st.session_state.pop("pipeline_run", None)
        st.rerun(scope="app")
        return False

    st.info(f"Processing {status['dataset_name']} in the background. Existing datasets remain available below.")
    st.progress(status["progress"], text=f"{int(status['progress'] * 100)}% - {status['label']}")
    return True


render_pipeline_progress()

if st.session_state.get("pipeline_done_message"):
    st.caption(st.session_state["pipeline_done_message"])

# ── Sidebar ──
st.sidebar.title("Ticket Pattern Mining")
st.sidebar.markdown("**Use Case 5**: Legacy vs DBB")

st.sidebar.subheader("Run New Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel or CSV",
    type=["xlsx", "xls", "csv"],
)
pipeline_mode = st.sidebar.selectbox("Pipeline Mode", ["POC", "PROD"])

if st.sidebar.button("Run Pipeline From Upload", use_container_width=True):
    if uploaded_file is None:
        st.sidebar.warning("Please upload a dataset first.")
    elif st.session_state.get("pipeline_run"):
        st.sidebar.warning("A dataset is already being processed. Please wait for it to finish.")
    else:
        saved_file = save_uploaded_file(uploaded_file)
        process, log_path, dataset_name = start_pipeline_for_upload(saved_file, pipeline_mode)
        st.session_state["pipeline_run"] = {
            "process": process,
            "log_path": log_path,
            "dataset_name": dataset_name,
        }
        st.session_state["last_pipeline_log"] = ""
        st.session_state.pop("pipeline_done_message", None)
        st.sidebar.success(f"Started pipeline for {dataset_name}.")
        st.rerun()

if st.session_state.get("last_pipeline_log"):
    with st.sidebar.expander("Latest Pipeline Log", expanded=False):
        st.text(st.session_state["last_pipeline_log"])

st.sidebar.divider()

datasets = get_available_datasets(base_dir)
if not datasets:
    # Fallback to root data folder if running legacy structure
    if os.path.exists(os.path.join(base_dir, "output", "cluster_catalog.csv")):
        datasets = ["Default"]
    else:
        st.error("No datasets found. Please run the pipeline first.")
        st.stop()

default_dataset = st.session_state.get("selected_dataset")
default_index = datasets.index(default_dataset) if default_dataset in datasets else 0
selected_dataset = st.sidebar.selectbox("Select Dataset", datasets, index=default_index)
st.session_state["selected_dataset"] = selected_dataset
if selected_dataset == "Default":
    dataset_dir = base_dir
else:
    dataset_dir = os.path.join(base_dir, selected_dataset)

try:
    catalog, pivot, tickets, exec_summary = load_data(dataset_dir)
except FileNotFoundError as e:
    st.error(f"Data not found: {e}. Run the full pipeline first.")
    st.stop()

# ── Prep ──
valid_catalog = catalog[catalog["Cluster_ID"] != -1].copy()
tickets["YearMonth"] = tickets["Created_Date"].dt.to_period("M").astype(str)

st.sidebar.divider()

# Quick KPIs in sidebar
total = len(tickets)
legacy_n = len(tickets[tickets["System_Type"] == "Legacy"])
dbb_n = len(tickets[tickets["System_Type"] == "DBB"])
clusters_n = len(valid_catalog)

st.sidebar.metric("Total Tickets Mined", f"{total:,}")
st.sidebar.metric("Legacy Tickets", f"{legacy_n:,}")
st.sidebar.metric("DBB Tickets", f"{dbb_n:,}")
st.sidebar.metric("Patterns Discovered", clusters_n)

verdict = exec_summary.get("legacy_to_dbb_verdict", "")
if verdict:
    color_map = {
        "DBB_REDUCED_ISSUES": "Green",
        "DBB_INCREASED_ISSUES": "Red",
        "MIXED_RESULTS": "Yellow",
        "INSUFFICIENT_DATA": "Grey",
    }
    status_label = color_map.get(verdict, "Grey")
    st.sidebar.markdown(f"### Verdict ({status_label}): {verdict.replace('_', ' ').title()}")

st.sidebar.divider()
st.sidebar.caption("Pipeline: classify -> normalize -> vectorize -> cluster -> metrics -> LLM naming -> summary")


# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Migration Timeline",
    "Pattern Discovery",
    "Domain & Severity",
    "Cluster Deep-Dive",
    "Remediation Strategy",
    "Smart Resolution (RAG)",
])


# ════════════════════════════════════════════════════════════════
#  TAB 1 — Migration Timeline
# ════════════════════════════════════════════════════════════════
with tab1:
    st.header("Migration Timeline: Legacy vs DBB Volume Over Time")
    st.caption("Does DBB reduce ticket volumes? This chart tells the story.")

    # Executive narrative
    narrative = exec_summary.get("executive_narrative", "")
    if narrative:
        st.info(narrative)

    # Monthly volume chart
    monthly = (
        tickets[tickets["System_Type"].isin(["Legacy", "DBB"])]
        .groupby(["YearMonth", "System_Type"])
        .size()
        .reset_index(name="Tickets")
    )

    fig = px.line(
        monthly,
        x="YearMonth",
        y="Tickets",
        color="System_Type",
        color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
        markers=True,
        title="Monthly Ticket Volume: Legacy vs DBB",
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Tickets",
        hovermode="x unified",
        legend_title_text="System",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative view
    st.subheader("Cumulative Ticket Growth")
    cum = monthly.copy()
    cum = cum.sort_values("YearMonth")
    cum["Cumulative"] = cum.groupby("System_Type")["Tickets"].cumsum()
    fig2 = px.area(
        cum, x="YearMonth", y="Cumulative", color="System_Type",
        color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
        title="Cumulative Ticket Growth Over Time",
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB 2 — Pattern Discovery
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("AI-Discovered Ticket Patterns")
    st.caption("Each card represents a cluster of semantically similar tickets identified by our NLP pipeline.")

    # Key Findings from LLM
    findings = exec_summary.get("key_findings", [])
    if findings:
        st.subheader("Key Findings")
        cols = st.columns(min(len(findings), 3))
        for i, f in enumerate(findings):
            with cols[i % 3]:
                impact_color = {"high": "High", "medium": "Medium", "low": "Low"}.get(f.get("impact", ""), "Unknown")
                st.markdown(f"**{impact_color} {f.get('title', '')}**")
                st.markdown(f"{f.get('detail', '')}")
        st.divider()

    # Cluster cards
    for _, row in valid_catalog.iterrows():
        cid = row["Cluster_ID"]
        name = row.get("Cluster_Name", row["Top_Keywords"])
        persona = row.get("Strategic_Persona", "")
        analysis = row.get("Analysis", "")
        rec = row.get("Recommendation", "")

        persona_prefix = f"{persona} - " if pd.notna(persona) and persona else ""
        with st.expander(f"{persona_prefix}Cluster {cid}: {name}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets", int(row["Size"]))
            c2.metric("Legacy", int(row["Frequency_Legacy"]))
            c3.metric("DBB", int(row["Frequency_DBB"]))

            # Smart verdict
            fl, fd = int(row["Frequency_Legacy"]), int(row["Frequency_DBB"])
            if fl == 0 and fd > 0:
                c4.metric("Trend", "New in DBB")
            elif fl > 0 and fd == 0:
                c4.metric("Trend", "Eliminated")
            elif fl > 0 and fd > 0:
                change = ((fd - fl) / fl) * 100
                c4.metric("Trend", f"{change:+.0f}%", delta=f"{change:+.0f}%", delta_color="inverse")
            else:
                c4.metric("Trend", "-")

            if pd.notna(analysis) and analysis:
                st.markdown(f"**Root Cause:** {analysis}")
            if pd.notna(rec) and rec:
                st.success(f"**Recommendation:** {rec}")

            st.markdown(f"*Keywords: {row['Top_Keywords']}*")

            # Domain
            st.caption(f"Domains: {row['Primary_Domains']}")

            # Per-cluster timeline
            cluster_tix = tickets[tickets["Cluster_ID"] == cid]
            if len(cluster_tix) > 5:
                ct = (
                    cluster_tix[cluster_tix["System_Type"].isin(["Legacy", "DBB"])]
                    .groupby(["YearMonth", "System_Type"])
                    .size()
                    .reset_index(name="Count")
                )
                if not ct.empty:
                    fig = px.bar(
                        ct, x="YearMonth", y="Count", color="System_Type",
                        color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
                        title=f"Volume Over Time - {name}",
                        barmode="group",
                    )
                    fig.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)

            # Sample tickets
            sample = cluster_tix[["Ticket_ID", "System_Type", "System_Subtype", "Short_Description", "Severity", "Reopen_Flag"]].head(5)
            st.dataframe(sample, hide_index=True)


# ════════════════════════════════════════════════════════════════
#  TAB 3 — Domain & Severity Analysis
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("Domain & Severity Health Check")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Tickets by Domain")
        domain_sys = (
            tickets[tickets["System_Type"].isin(["Legacy", "DBB"])]
            .groupby(["Domain", "System_Type"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            domain_sys, x="Count", y="Domain", color="System_Type",
            color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
            orientation="h", barmode="group",
            title="Ticket Volume by Domain: Legacy vs DBB",
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Severity Distribution")
        sev = (
            tickets[tickets["System_Type"].isin(["Legacy", "DBB"])]
            .groupby(["System_Type", "Severity"])
            .size()
            .reset_index(name="Count")
        )
        sev["Severity"] = sev["Severity"].map({1: "P1 - Critical", 2: "P2 - High", 3: "P3 - Medium", 4: "P4 - Low"})
        fig = px.bar(
            sev, x="Severity", y="Count", color="System_Type",
            color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
            barmode="group",
            title="Severity Breakdown: Legacy vs DBB",
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Domain health table from LLM
    domain_health = exec_summary.get("domain_health", [])
    if domain_health:
        st.subheader("Domain Health Scorecard (AI-Generated)")
        dh_df = pd.DataFrame(domain_health)
        verdict_map = {
            "improved": "Improved",
            "worsened": "Worsened",
            "new_in_dbb": "New in DBB",
            "legacy_only": "Legacy Only",
            "stable": "Stable",
        }
        if "verdict" in dh_df.columns:
            dh_df["verdict"] = dh_df["verdict"].map(lambda v: verdict_map.get(v, v))
        st.dataframe(dh_df, hide_index=True)

    # Reopen rates comparison
    st.subheader("Reopen Rates: Legacy vs DBB")
    reopen_data = (
        tickets[tickets["System_Type"].isin(["Legacy", "DBB"])]
        .groupby("System_Type")["Reopen_Flag"]
        .agg(["sum", "count"])
        .reset_index()
    )
    reopen_data.columns = ["System_Type", "Reopened_Tickets", "Total_Tickets"]
    reopen_data["Reopen_Rate"] = reopen_data["Reopened_Tickets"] / reopen_data["Total_Tickets"]
    c1, c2 = st.columns(2)
    for i, row in reopen_data.iterrows():
        with [c1, c2][i]:
            st.metric(
                f"{row['System_Type']} Reopen Rate",
                f"{row['Reopen_Rate']:.1%}",
                f"{int(row['Reopened_Tickets'])} of {int(row['Total_Tickets'])} tickets",
            )

    st.subheader("Repeat Defect / Reopened Ticket Drilldown")
    reopened = tickets[
        tickets["System_Type"].isin(["Legacy", "DBB"])
        & tickets["Reopen_Flag"].fillna(False)
    ].copy()
    if reopened.empty:
        st.info("No reopened tickets found in the selected dataset.")
    else:
        reopened_summary = (
            reopened.groupby("Cluster_ID")
            .agg(
                Reopened_Tickets=("Ticket_ID", "count"),
                Legacy_Reopened=("System_Type", lambda s: int((s == "Legacy").sum())),
                DBB_Reopened=("System_Type", lambda s: int((s == "DBB").sum())),
                Avg_Severity=("Severity", "mean"),
                Avg_MTTR_Hours=("Time_to_Resolve", "mean"),
            )
            .reset_index()
        )
        cluster_sizes = tickets.groupby("Cluster_ID")["Ticket_ID"].count().rename("Cluster_Tickets")
        reopened_summary = reopened_summary.merge(cluster_sizes, on="Cluster_ID", how="left")
        reopened_summary["Reopen_Rate"] = reopened_summary["Reopened_Tickets"] / reopened_summary["Cluster_Tickets"]
        cluster_names = valid_catalog.set_index("Cluster_ID")["Cluster_Name"].to_dict() if "Cluster_Name" in valid_catalog.columns else {}
        keyword_names = valid_catalog.set_index("Cluster_ID")["Top_Keywords"].to_dict() if "Top_Keywords" in valid_catalog.columns else {}
        reopened_summary["Pattern"] = reopened_summary["Cluster_ID"].map(cluster_names).fillna(reopened_summary["Cluster_ID"].map(keyword_names))
        reopened_summary["Pattern"] = reopened_summary["Pattern"].fillna("Noise / Unclustered")
        reopened_summary = reopened_summary.sort_values(["Reopened_Tickets", "Reopen_Rate"], ascending=False)

        fig = px.bar(
            reopened_summary.head(10),
            x="Reopened_Tickets",
            y="Pattern",
            color="DBB_Reopened",
            orientation="h",
            title="Top Reopened Patterns",
            color_continuous_scale="Reds",
        )
        fig.update_layout(template="plotly_dark", height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        display_reopen = reopened_summary[
            ["Pattern", "Reopened_Tickets", "Legacy_Reopened", "DBB_Reopened", "Reopen_Rate", "Avg_Severity", "Avg_MTTR_Hours"]
        ].copy()
        display_reopen["Reopen_Rate"] = display_reopen["Reopen_Rate"].map(lambda v: f"{v:.1%}")
        display_reopen["Avg_Severity"] = display_reopen["Avg_Severity"].round(2)
        display_reopen["Avg_MTTR_Hours"] = display_reopen["Avg_MTTR_Hours"].round(1)
        st.dataframe(display_reopen.head(15), hide_index=True)

        reopened_cols = [
            "Ticket_ID", "System_Type", "System_Subtype", "Domain", "OpCo",
            "Short_Description", "Severity", "Reopen_Count", "Time_to_Resolve",
        ]
        reopened_cols = [c for c in reopened_cols if c in reopened.columns]
        with st.expander("View reopened ticket details", expanded=False):
            st.dataframe(reopened[reopened_cols].sort_values("Reopen_Count", ascending=False).head(100), hide_index=True)

    st.subheader("Mean Time To Resolve (MTTR)")
    mttr_data = (
        tickets[tickets["System_Type"].isin(["Legacy", "DBB"])]
        .groupby("System_Type")["Time_to_Resolve"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    if not mttr_data.empty:
        c1, c2 = st.columns(2)
        mttr_cols = {"Legacy": c1, "DBB": c2}
        for _, row in mttr_data.iterrows():
            col = mttr_cols.get(row["System_Type"])
            if col is None:
                continue
            with col:
                st.metric(
                    f"{row['System_Type']} MTTR",
                    format_hours(row["mean"]),
                    f"Median {format_hours(row['median'])} across {int(row['count'])} resolved tickets",
                )

    heatmap_data = tickets.pivot_table(
        values="Ticket_ID", index="Domain", columns="System_Type",
        aggfunc="count", fill_value=0,
    )
    if "Legacy" in heatmap_data.columns and "DBB" in heatmap_data.columns:
        fig = px.imshow(
            heatmap_data[["Legacy", "DBB"]],
            text_auto=True,
            color_continuous_scale="RdYlGn_r",
            title="Ticket Volume Heatmap by Domain",
            aspect="auto",
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Same Failure Patterns Across OpCo / Country")
    opco_pattern_tickets = tickets[
        tickets["System_Type"].isin(["Legacy", "DBB"])
        & (tickets["Cluster_ID"] != -1)
        & tickets["OpCo"].notna()
    ].copy()
    if opco_pattern_tickets.empty:
        st.info("No clustered OpCo/country data available for this dataset.")
    else:
        cluster_names = valid_catalog.set_index("Cluster_ID")["Cluster_Name"].to_dict() if "Cluster_Name" in valid_catalog.columns else {}
        keyword_names = valid_catalog.set_index("Cluster_ID")["Top_Keywords"].to_dict() if "Top_Keywords" in valid_catalog.columns else {}
        opco_pattern_tickets["Pattern"] = opco_pattern_tickets["Cluster_ID"].map(cluster_names).fillna(
            opco_pattern_tickets["Cluster_ID"].map(keyword_names)
        )
        opco_pattern_tickets["Pattern"] = opco_pattern_tickets["Pattern"].fillna(
            opco_pattern_tickets["Cluster_ID"].map(lambda cid: f"Cluster {cid}")
        )
        opco_heatmap = opco_pattern_tickets.pivot_table(
            values="Ticket_ID",
            index="Pattern",
            columns="OpCo",
            aggfunc="count",
            fill_value=0,
        )
        top_patterns = opco_heatmap.sum(axis=1).sort_values(ascending=False).head(15).index
        top_opcos = opco_heatmap.sum(axis=0).sort_values(ascending=False).head(12).index
        opco_heatmap = opco_heatmap.loc[top_patterns, top_opcos]
        fig = px.imshow(
            opco_heatmap,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            title="Recurring Pattern Count by OpCo / Country",
            aspect="auto",
        )
        fig.update_layout(template="plotly_dark", height=max(450, 28 * len(opco_heatmap)))
        st.plotly_chart(fig, use_container_width=True)

        opco_summary = (
            opco_pattern_tickets.groupby(["Pattern", "OpCo", "System_Type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["Legacy", "DBB"]:
            if col not in opco_summary.columns:
                opco_summary[col] = 0
        opco_summary["Total"] = opco_summary["Legacy"] + opco_summary["DBB"]
        opco_summary = opco_summary.sort_values("Total", ascending=False)
        with st.expander("View OpCo / country recurrence table", expanded=False):
            st.dataframe(opco_summary[["Pattern", "OpCo", "Legacy", "DBB", "Total"]].head(100), hide_index=True)


# ════════════════════════════════════════════════════════════════
#  TAB 4 — Cluster Deep-Dive
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("Cluster Deep-Dive Explorer")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select Cluster")
        # Build readable options
        options = {}
        for _, row in valid_catalog.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            persona = row.get("Strategic_Persona", "")
            label = f"{name}"
            if pd.notna(persona) and persona:
                label = f"[{persona}] {name}"
            options[label] = row["Cluster_ID"]

        selected_label = st.selectbox("Choose a pattern", list(options.keys()))
        selected_cid = options[selected_label]
        cluster_info = valid_catalog[valid_catalog["Cluster_ID"] == selected_cid].iloc[0]

        st.metric("Size", int(cluster_info["Size"]))
        st.metric("Legacy Tickets", int(cluster_info["Frequency_Legacy"]))
        st.metric("DBB Tickets", int(cluster_info["Frequency_DBB"]))

    with col2:
        name = cluster_info.get("Cluster_Name", cluster_info["Top_Keywords"])
        persona = cluster_info.get("Strategic_Persona", "")

        if pd.notna(persona) and persona:
            st.markdown(f"### {persona}")
        st.markdown(f"## {name}")
        st.caption(f"TF-IDF Keywords: {cluster_info['Top_Keywords']}")
        st.caption(f"Domains: {cluster_info['Primary_Domains']}")

        analysis = cluster_info.get("Analysis", "")
        rec = cluster_info.get("Recommendation", "")
        if pd.notna(analysis) and analysis:
            st.markdown(f"**Root Cause Analysis:** {analysis}")
        if pd.notna(rec) and rec:
            st.info(f"**Prevention Strategy:** {rec}")

        mttr_legacy = cluster_info.get("AvgTTR_Legacy_Hours", float("nan"))
        mttr_dbb = cluster_info.get("AvgTTR_DBB_Hours", float("nan"))
        mttr_delta = cluster_info.get("AvgTTR_Delta_Hours", float("nan"))

        mttr_col1, mttr_col2, mttr_col3 = st.columns(3)
        mttr_col1.metric("Legacy MTTR", format_hours(mttr_legacy))
        mttr_col2.metric("DBB MTTR", format_hours(mttr_dbb))
        mttr_col3.metric(
            "MTTR Delta",
            format_hours(abs(mttr_delta)) if pd.notna(mttr_delta) else "N/A",
            "DBB slower" if pd.notna(mttr_delta) and mttr_delta > 0 else "DBB faster" if pd.notna(mttr_delta) and mttr_delta < 0 else "No comparison",
        )

        st.divider()

        # Timeline for this cluster
        cluster_tix = tickets[tickets["Cluster_ID"] == selected_cid]
        ct = (
            cluster_tix[cluster_tix["System_Type"].isin(["Legacy", "DBB"])]
            .groupby(["YearMonth", "System_Type"])
            .size()
            .reset_index(name="Count")
        )
        if not ct.empty:
            fig = px.bar(
                ct, x="YearMonth", y="Count", color="System_Type",
                color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
                title=f"Ticket Volume Over Time - {name}",
                barmode="group",
            )
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Sample tickets
        st.subheader("Sample Tickets")
        display_cols = ["Ticket_ID", "System_Type", "System_Subtype", "Domain", "Short_Description", "Severity", "Reopen_Flag"]
        display_cols = [c for c in display_cols if c in cluster_tix.columns]
        st.dataframe(cluster_tix[display_cols].head(15), hide_index=True)


# ════════════════════════════════════════════════════════════════
#  TAB 5 — Remediation Strategy
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("Remediation & Volume Reduction Strategy")
    st.caption("Actionable recommendations to reduce future ticket volumes.")

    # Shift-left opportunities from LLM
    shift_left = exec_summary.get("shift_left_opportunities", [])
    if shift_left:
        st.subheader("Shift-Left & Automation Opportunities")
        for opp in shift_left:
            with st.container():
                c1, c2, c3 = st.columns([2, 3, 1])
                c1.markdown(f"**{opp.get('pattern', '')}**")
                c2.markdown(opp.get("strategy", ""))
                c3.metric("Est. Reduction", opp.get("estimated_reduction", "?"))
            st.divider()

    # Pattern -> Root Cause -> Prevention Map (from cluster catalog)
    st.subheader("Pattern -> Root Cause -> Prevention Map")
    map_data = []
    for _, row in valid_catalog.iterrows():
        name = row.get("Cluster_Name", row["Top_Keywords"])
        analysis = row.get("Analysis", "")
        rec = row.get("Recommendation", "")
        persona = row.get("Strategic_Persona", "")
        if pd.notna(analysis) and analysis:
            map_data.append({
                "Pattern": name,
                "Persona": persona if pd.notna(persona) else "",
                "Root Cause": analysis,
                "Prevention": rec if pd.notna(rec) else "",
                "Legacy Tickets": int(row["Frequency_Legacy"]),
                "DBB Tickets": int(row["Frequency_DBB"]),
            })
    if map_data:
        st.dataframe(pd.DataFrame(map_data), hide_index=True)

    st.subheader("Legacy Patterns Reappeared in DBB")
    pollutants = valid_catalog[
        (valid_catalog["Frequency_Legacy"] > 0)
        & (valid_catalog["Frequency_DBB"] > 0)
    ].copy()
    if not pollutants.empty:
        pollutants["Change"] = ((pollutants["Frequency_DBB"] - pollutants["Frequency_Legacy"]) / pollutants["Frequency_Legacy"] * 100).round(0)
        pollutants["DBB Share"] = (
            pollutants["Frequency_DBB"]
            / (pollutants["Frequency_Legacy"] + pollutants["Frequency_DBB"])
        ).round(3)
        pollutants = pollutants.sort_values("Change", ascending=False)
        st.caption("These are patterns that existed in Legacy and still appeared in DBB. Positive change means DBB has more tickets than Legacy for the same recurring pattern.")

        pollution_rows = []
        for _, row in pollutants.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            change = row["Change"]
            pollution_rows.append({
                "Pattern": name,
                "Legacy Tickets": int(row["Frequency_Legacy"]),
                "DBB Tickets": int(row["Frequency_DBB"]),
                "DBB Share": f"{row['DBB Share']:.1%}",
                "Change vs Legacy": f"{change:+.0f}%",
                "Root Cause": row.get("Analysis", ""),
                "Prevention": row.get("Recommendation", ""),
            })
            if change > 0:
                st.error(f"**{name}**: Legacy pattern reappeared worse in DBB. DBB tickets are {change:.0f}% higher than Legacy ({int(row['Frequency_Legacy'])} -> {int(row['Frequency_DBB'])}).")
            else:
                st.warning(f"**{name}**: Legacy pattern still reappeared in DBB, but at {abs(change):.0f}% lower volume ({int(row['Frequency_Legacy'])} -> {int(row['Frequency_DBB'])}).")

        st.dataframe(pd.DataFrame(pollution_rows), hide_index=True)
    else:
        st.success("No clustered Legacy patterns reappeared in DBB for this dataset.")

    # New DBB-only issues
    new_dbb = valid_catalog[valid_catalog["Frequency_Legacy"] == 0]
    if not new_dbb.empty:
        st.subheader("Issues Introduced by DBB")
        for _, row in new_dbb.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            st.warning(f"**{name}**: {int(row['Frequency_DBB'])} tickets - this pattern did not exist in Legacy.")

    # Legacy issues eliminated
    eliminated = valid_catalog[valid_catalog["Frequency_DBB"] == 0]
    if not eliminated.empty:
        st.subheader("Legacy Issues Successfully Eliminated by DBB")
        for _, row in eliminated.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            st.success(f"**{name}**: {int(row['Frequency_Legacy'])} Legacy tickets -> 0 in DBB. This problem was solved.")


# ════════════════════════════════════════════════════════════════
#  TAB 6 — Smart Resolution (RAG)
# ════════════════════════════════════════════════════════════════
with tab6:
    st.header("Smart Resolution (RAG)")
    st.caption("Enter a new ticket description. The system will find similar historical tickets and suggest a resolution based on how they were solved.")

    new_ticket_text = st.text_area(
        "New Ticket Description",
        height=150,
        placeholder="E.g., User unable to sync data on Glassrun app...",
    )

    if st.button("Get AI Resolution"):
        if not new_ticket_text.strip():
            st.warning("Please enter a ticket description.")
        else:
            with st.spinner("Analyzing and retrieving similar tickets..."):
                try:
                    embedder = load_embedder()
                    recommendation, similar_tickets = resolve_ticket(
                        ticket_text=new_ticket_text,
                        tickets=tickets,
                        embedder=embedder,
                    )

                    st.subheader("AI Recommended Resolution")
                    st.write(recommendation)

                    st.divider()
                    st.subheader("Top Similar Historical Tickets")
                    display_cols = ["Similarity", "Ticket_ID", "Short_Description", "Resolution_Notes"]
                    avail_cols = [c for c in display_cols if c in similar_tickets.columns]
                    st.dataframe(similar_tickets[avail_cols], hide_index=True)
                except Exception as e:
                    st.error(f"Error calling RAG pipeline: {e}")
