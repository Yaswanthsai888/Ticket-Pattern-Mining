import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import argparse
import subprocess
import sys
import re
from html import escape
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


def format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def inject_dashboard_styles():
    st.markdown(
        """
        <style>
        :root {
            --panel-bg: rgba(15, 23, 42, 0.72);
            --panel-bg-soft: rgba(2, 6, 23, 0.42);
            --panel-border: rgba(148, 163, 184, 0.18);
            --text-strong: #f8fafc;
            --text-body: #dbe7f5;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.06), transparent 24%),
                linear-gradient(180deg, #020617 0%, #0f172a 45%, #111827 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 0.35rem;
        }
        [data-baseweb="tab"] {
            height: 42px;
            border-radius: 8px;
            padding: 0 16px;
            color: var(--text-muted);
            background: transparent;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: rgba(30, 41, 59, 0.95);
            color: var(--text-strong);
        }
        [data-testid="stMetric"] {
            background: var(--panel-bg-soft);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted);
        }
        [data-testid="stMetricValue"] {
            color: var(--text-strong);
        }
        [data-testid="stMetricDelta"] {
            color: #cbd5e1;
        }
        [data-testid="stDataFrame"], .stPlotlyChart {
            background: rgba(2, 6, 23, 0.18);
            border-radius: 10px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.38);
        }
        div[data-testid="stExpander"] details summary {
            background: rgba(15, 23, 42, 0.62);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(2,6,23,0.98) 100%);
            border-right: 1px solid var(--panel-border);
        }
        .section-panel {
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            background: var(--panel-bg);
            padding: 18px 20px;
            margin: 10px 0 16px;
        }
        .section-panel-title {
            color: var(--text-strong);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .section-panel-body {
            color: var(--text-body);
            line-height: 1.6;
        }
        .status-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            font-weight: 700;
            font-size: 0.82rem;
            margin-right: 8px;
            margin-bottom: 8px;
            border: 1px solid transparent;
        }
        .status-chip.info {
            color: #bfdbfe;
            background: rgba(59, 130, 246, 0.14);
            border-color: rgba(59, 130, 246, 0.28);
        }
        .status-chip.good {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.14);
            border-color: rgba(34, 197, 94, 0.3);
        }
        .status-chip.warn {
            color: #fde68a;
            background: rgba(245, 158, 11, 0.14);
            border-color: rgba(245, 158, 11, 0.3);
        }
        .status-chip.bad {
            color: #fecaca;
            background: rgba(239, 68, 68, 0.14);
            border-color: rgba(239, 68, 68, 0.3);
        }
        .insight-card {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
            padding: 18px 20px;
            margin: 10px 0 16px;
        }
        .insight-card strong,
        .mini-card strong {
            color: var(--text-strong);
        }
        .insight-label {
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .insight-title {
            color: var(--text-strong);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .insight-body {
            color: var(--text-body);
            line-height: 1.65;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 14px;
            margin: 12px 0 18px;
        }
        .mini-card {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg-soft);
            padding: 15px 16px;
            min-height: 100%;
        }
        .mini-card.root { border-left: 4px solid #38bdf8; }
        .mini-card.action { border-left: 4px solid #22c55e; }
        .mini-card.priority { border-left: 4px solid #f59e0b; }
        .mini-card-title {
            color: var(--text-strong);
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .clean-list {
            margin: 8px 0 0;
            padding-left: 19px;
            color: var(--text-body);
            line-height: 1.55;
        }
        .clean-list li { margin: 5px 0; }
        .priority-pill {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid rgba(245, 158, 11, 0.38);
            background: rgba(245, 158, 11, 0.12);
            color: #fde68a;
            padding: 5px 10px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_panel(title, body, tone="info"):
    tone_class = {
        "info": "info",
        "good": "good",
        "warn": "warn",
        "bad": "bad",
    }.get(tone, "info")
    st.markdown(
        (
            f'<div class="section-panel">'
            f'<span class="status-chip {tone_class}">{escape(title)}</span>'
            f'<div class="section-panel-body">{escape(body)}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )


def split_sentences(text, limit=None):
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[:limit] if limit else sentences


def numbered_items(text):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"\s*\(\d+\)\s*", text.strip())
    return [p.strip(" .;") for p in parts[1:] if p.strip(" .;")]


def section_between(text, start_label, end_labels):
    if not isinstance(text, str):
        return ""
    lower_text = text.lower()
    start = lower_text.find(start_label.lower())
    if start == -1:
        return ""
    start += len(start_label)
    end = len(text)
    for label in end_labels:
        idx = lower_text.find(label.lower(), start)
        if idx != -1:
            end = min(end, idx)
    return text[start:end].strip(" :.-")


def list_html(items):
    if not items:
        return ""
    return "<ul class='clean-list'>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def render_executive_narrative(narrative):
    paragraphs = [p.strip() for p in str(narrative).split("\n\n") if p.strip()]
    if not paragraphs:
        return

    labels = ["Executive Readout", "Patterns Needing Attention", "Recommended Action"]
    cards = []
    for idx, paragraph in enumerate(paragraphs[:3]):
        cards.append(
            (
                f'<div class="mini-card {"action" if idx == 2 else "root"}">'
                f'<div class="mini-card-title">{escape(labels[idx] if idx < len(labels) else "Insight")}</div>'
                f'<div class="insight-body">{escape(paragraph)}</div>'
                f'</div>'
            )
        )
    st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_cluster_spotlight_cards(rows):
    cards = []
    for _, row in rows.iterrows():
        legacy = int(row.get("Frequency_Legacy", 0))
        dbb = int(row.get("Frequency_DBB", 0))
        cards.append(
            (
                f'<div class="mini-card root">'
                f'<div class="mini-card-title">{escape(pattern_name(row))}</div>'
                f'<div style="margin-bottom:8px">'
                f'<span class="status-chip info">{int(row.get("Size", 0))} total tickets</span>'
                f'<span class="status-chip {"warn" if dbb > legacy else "good" if legacy > 0 and dbb == 0 else "info"}">{escape(pattern_status(row))}</span>'
                f'</div>'
                f'<div class="insight-body">Legacy: {legacy} | DBB: {dbb} | Change: {escape(pattern_change_text(legacy, dbb))}</div>'
                f'<div class="insight-body" style="margin-top:10px">Domains: {escape(str(row.get("Primary_Domains", "N/A")))}</div>'
                f'</div>'
            )
        )
    if cards:
        st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_pattern_insight(analysis, recommendation):
    summary = ""
    drivers = []
    closing = ""
    if isinstance(analysis, str) and analysis.strip():
        cause_label = "Common root causes include:"
        cause_idx = analysis.find(cause_label)
        if cause_idx != -1:
            summary = analysis[:cause_idx].strip()
            cause_block = analysis[cause_idx + len(cause_label):]
            repetitive_idx = cause_block.find("The repetitive nature")
            if repetitive_idx != -1:
                closing = cause_block[repetitive_idx:].strip()
                cause_block = cause_block[:repetitive_idx]
            drivers = numbered_items(cause_block)
        else:
            sentences = split_sentences(analysis)
            summary = " ".join(sentences[:2])
            drivers = sentences[2:6]

    immediate = section_between(recommendation, "Immediate actions", ["Long-term solutions", "Priority"])
    long_term = section_between(recommendation, "Long-term solutions", ["Priority"])
    priority = section_between(recommendation, "Priority", [])

    if not immediate and not long_term:
        rec_sentences = split_sentences(recommendation, 5)
        immediate_items = rec_sentences[:3]
        long_term_items = rec_sentences[3:]
    else:
        immediate_items = numbered_items(immediate)
        long_term_items = numbered_items(long_term)

    cards = []
    if summary:
        body = f"<div class='insight-body'>{escape(summary)}</div>"
        if closing:
            body += f"<div class='insight-body' style='margin-top:10px'>{escape(closing)}</div>"
        cards.append(
            f"<div class='mini-card root'><div class='mini-card-title'>Root Cause Summary</div>{body}</div>"
        )
    if drivers:
        cards.append(
            f"<div class='mini-card root'><div class='mini-card-title'>Likely Drivers</div>{list_html(drivers)}</div>"
        )
    if immediate_items:
        cards.append(
            f"<div class='mini-card action'><div class='mini-card-title'>Immediate Actions</div>{list_html(immediate_items)}</div>"
        )
    if long_term_items or priority:
        priority_html = f"<div class='priority-pill'>{escape(priority)}</div>" if priority else ""
        cards.append(
            f"<div class='mini-card priority'><div class='mini-card-title'>Prevention Plan</div>{priority_html}{list_html(long_term_items)}</div>"
        )

    if cards:
        st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def pattern_name(row):
    name = row.get("Cluster_Name", "")
    if pd.notna(name) and str(name).strip():
        return str(name)
    return str(row.get("Top_Keywords", "Unnamed pattern"))


def pattern_status(row):
    legacy = int(row.get("Frequency_Legacy", 0))
    dbb = int(row.get("Frequency_DBB", 0))
    if legacy == 0 and dbb > 0:
        return "New in DBB"
    if legacy > 0 and dbb == 0:
        return "Eliminated in DBB"
    if legacy > 0 and dbb > legacy:
        return "Worse in DBB"
    if legacy > 0 and dbb > 0:
        return "Still recurring"
    return "Insufficient comparison"


def pattern_change_text(legacy, dbb):
    if legacy == 0 and dbb > 0:
        return "New in DBB"
    if legacy > 0 and dbb == 0:
        return "Eliminated"
    if legacy > 0:
        return f"{((dbb - legacy) / legacy) * 100:+.0f}%"
    return "N/A"


def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def build_pipeline_command(input_file, mode):
    return [
        sys.executable,
        "-u",
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

inject_dashboard_styles()

# ── Prep ──
valid_catalog = catalog[catalog["Cluster_ID"] != -1].copy()
tickets["YearMonth"] = tickets["Created_Date"].dt.to_period("M").astype(str)

st.sidebar.divider()

# Quick KPIs in sidebar
total = len(tickets)
legacy_n = len(tickets[tickets["System_Type"] == "Legacy"])
dbb_n = len(tickets[tickets["System_Type"] == "DBB"])
clusters_n = len(valid_catalog)
unknown_n = len(tickets[tickets["System_Type"] == "Unknown"])
both_n = len(tickets[tickets["System_Type"] == "Both"])
clustered_n = len(tickets[tickets["Cluster_ID"] != -1]) if "Cluster_ID" in tickets.columns else 0

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

st.title("Legacy vs DBB Ticket Pattern Mining")
st.caption("Use this dashboard to answer three questions: what keeps repeating, whether DBB reduced it, and what should be prevented next.")

overview_cols = st.columns(5)
overview_cols[0].metric("Tickets Analyzed", f"{total:,}")
overview_cols[1].metric("Legacy", f"{legacy_n:,}")
overview_cols[2].metric("DBB", f"{dbb_n:,}")
overview_cols[3].metric("Recurring Patterns", f"{clusters_n:,}")
overview_cols[4].metric("Clustered Tickets", f"{clustered_n:,}", f"{format_pct(clustered_n / total) if total else 'N/A'} of all tickets")

if unknown_n or both_n:
    st.caption(
        f"Classification note: {unknown_n:,} tickets are Unknown and {both_n:,} tickets touch Both systems. "
        "Legacy vs DBB charts focus on tickets classified as Legacy or DBB."
    )


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
    st.header("Did DBB Reduce Ticket Volume?")
    st.caption("This view compares monthly ticket volume after each ticket is classified as Legacy or DBB. Lower DBB volume is better only when the same business scope is being compared.")

    # Executive narrative
    narrative = exec_summary.get("executive_narrative", "")
    if narrative:
        render_executive_narrative(narrative)

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
        title="Monthly Ticket Volume by System",
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
    st.subheader("Cumulative Ticket Load")
    st.caption("Cumulative view shows total support load over time. A steeper line means tickets are accumulating faster.")
    cum = monthly.copy()
    cum = cum.sort_values("YearMonth")
    cum["Cumulative"] = cum.groupby("System_Type")["Tickets"].cumsum()
    fig2 = px.area(
        cum, x="YearMonth", y="Cumulative", color="System_Type",
        color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
        title="Cumulative Tickets Over Time",
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB 2 — Pattern Discovery
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("What Problems Keep Repeating?")
    st.caption("This page surfaces the most repeated issues. Use Cluster Deep-Dive for timelines, root causes, and sample tickets.")

    # Key Findings from LLM
    findings = exec_summary.get("key_findings", [])
    if findings:
        st.subheader("Key Findings")
        cards = []
        tone_map = {"high": "bad", "medium": "warn", "low": "good"}
        impact_map = {"high": "High Impact", "medium": "Medium Impact", "low": "Positive Signal"}
        for f in findings:
            tone = tone_map.get(f.get("impact", ""), "info")
            impact_text = impact_map.get(f.get("impact", ""), "Insight")
            cards.append(
                (
                    f'<div class="mini-card {("priority" if tone == "warn" else "root" if tone == "info" else "action" if tone == "good" else "priority")}">'
                    f'<div class="mini-card-title">{escape(f.get("title", "Finding"))}</div>'
                    f'<div style="margin-bottom:8px"><span class="status-chip {tone}">{escape(impact_text)}</span></div>'
                    f'<div class="insight-body">{escape(f.get("detail", ""))}</div>'
                    f'</div>'
                )
            )
        st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
        st.divider()

    if not valid_catalog.empty:
        st.subheader("Pattern Summary")
        pattern_summary = valid_catalog.copy()
        pattern_summary["Pattern"] = pattern_summary.apply(pattern_name, axis=1)
        pattern_summary["Status"] = pattern_summary.apply(pattern_status, axis=1)
        pattern_summary["DBB vs Legacy"] = pattern_summary.apply(
            lambda r: pattern_change_text(int(r["Frequency_Legacy"]), int(r["Frequency_DBB"])),
            axis=1,
        )
        display_patterns = pattern_summary[
            [
                "Pattern", "Status", "Size", "Frequency_Legacy", "Frequency_DBB",
                "DBB vs Legacy", "AvgTTR_Delta_Hours", "ReopenRate_Delta",
            ]
        ].rename(columns={
            "Size": "Total Tickets",
            "Frequency_Legacy": "Legacy Tickets",
            "Frequency_DBB": "DBB Tickets",
            "AvgTTR_Delta_Hours": "DBB MTTR Difference (Hours)",
            "ReopenRate_Delta": "DBB Reopen Difference",
        })
        st.dataframe(display_patterns.head(12), hide_index=True, use_container_width=True)
        with st.expander("View full pattern summary", expanded=False):
            st.dataframe(display_patterns, hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Top Pattern Spotlights")
        spotlight_cols = [
            "Cluster_ID", "Cluster_Name", "Size", "Frequency_Legacy", "Frequency_DBB",
            "Primary_Domains", "Strategic_Persona",
        ]
        spotlight_df = pattern_summary.sort_values(["Size", "Frequency_DBB"], ascending=[False, False])[spotlight_cols].head(6)
        render_cluster_spotlight_cards(spotlight_df)
        render_section_panel(
            "Next Step",
            "Open Cluster Deep-Dive to inspect one pattern in detail, including likely drivers, prevention actions, monthly trend, and sample tickets.",
            tone="info",
        )


# ════════════════════════════════════════════════════════════════
#  TAB 3 — Domain & Severity Analysis
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("Where Is The Support Load Coming From?")
    st.caption("This page shows concentration, risk, fix quality, and recurrence across domains.")

    summary_row = st.columns(4)
    summary_row[0].metric("Domains Tracked", int(tickets["Domain"].nunique()))
    summary_row[1].metric("Legacy Tickets", f"{legacy_n:,}")
    summary_row[2].metric("DBB Tickets", f"{dbb_n:,}")
    summary_row[3].metric("Reopened Tickets", f"{int(tickets['Reopen_Flag'].fillna(False).sum()):,}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Ticket Volume by Business Domain")
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
            title="Legacy vs DBB Tickets by Domain",
        )
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Severity Mix")
        sev = (
            tickets[tickets["System_Type"].isin(["Legacy", "DBB"])]
            .groupby(["System_Type", "Severity"])
            .size()
            .reset_index(name="Count")
        )
        sev["Severity"] = sev["Severity"].map({1: "Low", 2: "Moderate", 3: "High", 4: "Critical"})
        fig = px.bar(
            sev, x="Severity", y="Count", color="System_Type",
            color_discrete_map={"Legacy": "#ef4444", "DBB": "#3b82f6"},
            barmode="group",
            title="Ticket Severity by System",
        )
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Domain health table from LLM
    domain_health = exec_summary.get("domain_health", [])
    if domain_health:
        st.subheader("Domain Health Scorecard")
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
        verdict_counts = dh_df["verdict"].value_counts().to_dict()
        summary_parts = [f"{count} {label}" for label, count in verdict_counts.items()]
        render_section_panel("Domain Readout", " | ".join(summary_parts), tone="info")
        st.dataframe(dh_df.head(10), hide_index=True, use_container_width=True)
        if len(dh_df) > 10:
            with st.expander("View full domain health table", expanded=False):
                st.dataframe(dh_df, hide_index=True, use_container_width=True)

    risk_col, quality_col = st.columns(2)

    # Reopen rates comparison
    with risk_col:
        st.subheader("Are Fixes Staying Fixed?")
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
                    f"{int(row['Reopened_Tickets'])} reopened",
                )

    with quality_col:
        st.subheader("How Long Do Tickets Take To Resolve?")
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
                        f"Median {format_hours(row['median'])}",
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
            title="Domain Heatmap: Where Legacy and DBB Tickets Concentrate",
            aspect="auto",
        )
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Recurrence Views")
    detail_tab1, detail_tab2 = st.tabs(["Reopened Patterns", "OpCo / Country Concentration"])

    with detail_tab1:
        reopened = tickets[
            tickets["System_Type"].isin(["Legacy", "DBB"])
            & tickets["Reopen_Flag"].fillna(False)
        ].copy()
        if reopened.empty:
            render_section_panel("No Reopens Found", "No reopened clustered tickets were found in the selected dataset.", tone="good")
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
                title="Reopened Tickets by Pattern",
                color_continuous_scale="Reds",
            )
            fig.update_layout(template="plotly_dark", height=360, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            display_reopen = reopened_summary[
                ["Pattern", "Reopened_Tickets", "Legacy_Reopened", "DBB_Reopened", "Reopen_Rate", "Avg_Severity", "Avg_MTTR_Hours"]
            ].copy()
            display_reopen["Reopen_Rate"] = display_reopen["Reopen_Rate"].map(lambda v: f"{v:.1%}")
            display_reopen["Avg_Severity"] = display_reopen["Avg_Severity"].round(2)
            display_reopen["Avg_MTTR_Hours"] = display_reopen["Avg_MTTR_Hours"].round(1)
            st.dataframe(display_reopen.head(8), hide_index=True, use_container_width=True)
            with st.expander("View full reopened pattern table", expanded=False):
                st.dataframe(display_reopen, hide_index=True, use_container_width=True)

    with detail_tab2:
        opco_pattern_tickets = tickets[
            tickets["System_Type"].isin(["Legacy", "DBB"])
            & (tickets["Cluster_ID"] != -1)
            & tickets["OpCo"].notna()
        ].copy()
        if opco_pattern_tickets.empty:
            render_section_panel("No OpCo Data", "No clustered OpCo or country data is available for this dataset.", tone="warn")
        else:
            opco_count = opco_pattern_tickets["OpCo"].nunique()
            if opco_count <= 1:
                render_section_panel(
                    "Single-OpCo Dataset",
                    "This upload mostly contains one OpCo/country, so this view shows concentration more than cross-country recurrence.",
                    tone="warn",
                )
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
            fig.update_layout(template="plotly_dark", height=max(380, 26 * len(opco_heatmap)))
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
            st.dataframe(opco_summary[["Pattern", "OpCo", "Legacy", "DBB", "Total"]].head(30), hide_index=True, use_container_width=True)
            if len(opco_summary) > 30:
                with st.expander("View full OpCo / country recurrence table", expanded=False):
                    st.dataframe(opco_summary[["Pattern", "OpCo", "Legacy", "DBB", "Total"]], hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB 4 — Cluster Deep-Dive
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("Pattern Deep-Dive")
    st.caption("Select one recurring pattern to inspect its volume, Legacy vs DBB behavior, likely cause, prevention action, and example tickets.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select Cluster")
        # Build readable options
        options = {}
        for _, row in valid_catalog.iterrows():
            name = pattern_name(row)
            persona = row.get("Strategic_Persona", "")
            label = f"{name}"
            if pd.notna(persona) and persona:
                label = f"[{persona}] {name}"
            options[label] = row["Cluster_ID"]

        selected_label = st.selectbox("Choose a pattern", list(options.keys()))
        selected_cid = options[selected_label]
        cluster_info = valid_catalog[valid_catalog["Cluster_ID"] == selected_cid].iloc[0]

        st.metric("Pattern Tickets", int(cluster_info["Size"]))
        st.metric("Legacy Tickets", int(cluster_info["Frequency_Legacy"]))
        st.metric("DBB Tickets", int(cluster_info["Frequency_DBB"]))
        st.metric("Status", pattern_status(cluster_info))

    with col2:
        name = pattern_name(cluster_info)
        persona = cluster_info.get("Strategic_Persona", "")

        if pd.notna(persona) and persona:
            st.markdown(f"### {persona}")
        st.markdown(f"## {name}")
        st.caption(f"Why grouped: shared keywords include {cluster_info['Top_Keywords']}")
        st.caption(f"Primary business domains: {cluster_info['Primary_Domains']}")

        analysis = cluster_info.get("Analysis", "")
        rec = cluster_info.get("Recommendation", "")
        if pd.notna(analysis) and analysis:
            render_pattern_insight(analysis, rec if pd.notna(rec) else "")

        mttr_legacy = cluster_info.get("AvgTTR_Legacy_Hours", float("nan"))
        mttr_dbb = cluster_info.get("AvgTTR_DBB_Hours", float("nan"))
        mttr_delta = cluster_info.get("AvgTTR_Delta_Hours", float("nan"))

        mttr_col1, mttr_col2, mttr_col3 = st.columns(3)
        mttr_col1.metric("Legacy Avg Resolve Time", format_hours(mttr_legacy))
        mttr_col2.metric("DBB Avg Resolve Time", format_hours(mttr_dbb))
        mttr_col3.metric(
            "DBB Resolve Difference",
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
                title=f"Monthly Ticket Count for Pattern: {name}",
                barmode="group",
            )
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Sample tickets
        st.subheader("Example Tickets Inside This Pattern")
        display_cols = ["Ticket_ID", "System_Type", "System_Subtype", "Domain", "Short_Description", "Severity", "Reopen_Flag"]
        display_cols = [c for c in display_cols if c in cluster_tix.columns]
        st.dataframe(cluster_tix[display_cols].head(8), hide_index=True, use_container_width=True)
        if len(cluster_tix) > 8:
            with st.expander("View more example tickets", expanded=False):
                st.dataframe(cluster_tix[display_cols].head(25), hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB 5 — Remediation Strategy
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("What Should We Fix First?")
    st.caption("This page turns recurring patterns into action: prevent repeat tickets, shift work left, create knowledge articles, or automate known fixes.")

    # Shift-left opportunities from LLM
    shift_left = exec_summary.get("shift_left_opportunities", [])
    if shift_left:
        st.subheader("Shift-Left And Automation Opportunities")
        st.caption("These are candidates where support effort can move earlier: self-service, monitoring, automation, or clearer L1 knowledge.")
        cards = []
        for opp in shift_left:
            cards.append(
                (
                    f'<div class="mini-card action">'
                    f'<div class="mini-card-title">{escape(opp.get("pattern", "Opportunity"))}</div>'
                    f'<div style="margin-bottom:8px"><span class="status-chip good">Estimated Reduction {escape(str(opp.get("estimated_reduction", "?")))}</span></div>'
                    f'<div class="insight-body">{escape(opp.get("strategy", ""))}</div>'
                    f'</div>'
                )
            )
        st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    # Pattern -> Root Cause -> Prevention Map (from cluster catalog)
    st.subheader("Pattern To Root Cause To Prevention")
    st.caption("Use this as the action backlog: each row connects a repeated issue to a likely cause and a prevention idea.")
    map_data = []
    for _, row in valid_catalog.iterrows():
        name = pattern_name(row)
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
        map_df = pd.DataFrame(map_data)
        map_df["Total Tickets"] = map_df["Legacy Tickets"] + map_df["DBB Tickets"]
        map_df = map_df.sort_values("Total Tickets", ascending=False)
        st.dataframe(map_df[["Pattern", "Persona", "Legacy Tickets", "DBB Tickets", "Total Tickets"]].head(8), hide_index=True, use_container_width=True)
        with st.expander("View full root cause and prevention map", expanded=False):
            st.dataframe(map_df.drop(columns=["Total Tickets"]), hide_index=True, use_container_width=True)

    st.subheader("Legacy Problems That Still Appear In DBB")
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
        st.caption("These patterns existed in Legacy and still appear in DBB. Positive change means DBB has more tickets than Legacy for the same recurring pattern.")

        pollution_rows = []
        for _, row in pollutants.iterrows():
            name = pattern_name(row)
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
        pollution_df = pd.DataFrame(pollution_rows)
        st.dataframe(pollution_df.head(8), hide_index=True, use_container_width=True)
        if len(pollution_df) > 8:
            with st.expander("View full carry-forward pattern table", expanded=False):
                st.dataframe(pollution_df, hide_index=True, use_container_width=True)
        top_pollutant = pollutants.iloc[0]
        top_name = pattern_name(top_pollutant)
        top_change = top_pollutant["Change"]
        if top_change > 0:
            render_section_panel(
                "Top Legacy Pollutant",
                f"{top_name} is reappearing worse in DBB, with DBB volume {top_change:.0f}% above Legacy.",
                tone="bad",
            )
        else:
            render_section_panel(
                "Top Carry-Forward Pattern",
                f"{top_name} still appears in DBB, but at {abs(top_change):.0f}% lower volume than Legacy.",
                tone="warn",
            )
    else:
        render_section_panel("Positive Signal", "No clustered Legacy patterns reappeared in DBB for this dataset.", tone="good")

    # New DBB-only issues
    new_dbb = valid_catalog[valid_catalog["Frequency_Legacy"] == 0]
    if not new_dbb.empty:
        st.subheader("New DBB-Only Patterns")
        cards = []
        for _, row in new_dbb.head(6).iterrows():
            name = pattern_name(row)
            cards.append(
                (
                    f'<div class="mini-card priority">'
                    f'<div class="mini-card-title">{escape(name)}</div>'
                    f'<div><span class="status-chip warn">{int(row["Frequency_DBB"])} DBB tickets</span></div>'
                    f'<div class="insight-body">This recurring pattern did not exist in Legacy within the current clustered data.</div>'
                    f'</div>'
                )
            )
        st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
        if len(new_dbb) > 6:
            new_dbb_table = new_dbb.copy()
            new_dbb_table["Pattern"] = new_dbb_table.apply(pattern_name, axis=1)
            with st.expander("View all DBB-only patterns", expanded=False):
                st.dataframe(new_dbb_table[["Pattern", "Frequency_DBB", "Primary_Domains"]], hide_index=True, use_container_width=True)

    # Legacy issues eliminated
    eliminated = valid_catalog[valid_catalog["Frequency_DBB"] == 0]
    if not eliminated.empty:
        st.subheader("Legacy Patterns Eliminated In DBB")
        cards = []
        for _, row in eliminated.head(6).iterrows():
            name = pattern_name(row)
            cards.append(
                (
                    f'<div class="mini-card action">'
                    f'<div class="mini-card-title">{escape(name)}</div>'
                    f'<div><span class="status-chip good">{int(row["Frequency_Legacy"])} Legacy tickets to 0 DBB tickets</span></div>'
                    f'<div class="insight-body">This is a positive migration signal: the recurring pattern does not appear in DBB.</div>'
                    f'</div>'
                )
            )
        st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
        if len(eliminated) > 6:
            eliminated_table = eliminated.copy()
            eliminated_table["Pattern"] = eliminated_table.apply(pattern_name, axis=1)
            with st.expander("View all eliminated legacy patterns", expanded=False):
                st.dataframe(eliminated_table[["Pattern", "Frequency_Legacy", "Primary_Domains"]], hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB 6 — Smart Resolution (RAG)
# ════════════════════════════════════════════════════════════════
with tab6:
    st.header("Resolve A New Ticket Using Past Tickets")
    st.caption("Paste a new ticket description. The system retrieves similar historical tickets and uses their resolution notes to suggest a next action.")

    new_ticket_text = st.text_area(
        "New Ticket Description",
        height=150,
        placeholder="E.g., User unable to sync data on Glassrun app...",
    )

    if st.button("Suggest Resolution From History"):
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

                    st.subheader("Suggested Resolution")
                    st.write(recommendation)

                    st.divider()
                    st.subheader("Historical Tickets Used As Evidence")
                    st.caption("Review these tickets to confirm the suggestion before applying it.")
                    display_cols = ["Similarity", "Ticket_ID", "Short_Description", "Resolution_Notes"]
                    avail_cols = [c for c in display_cols if c in similar_tickets.columns]
                    st.dataframe(similar_tickets[avail_cols], hide_index=True)
                except Exception as e:
                    st.error(f"Error calling RAG pipeline: {e}")
