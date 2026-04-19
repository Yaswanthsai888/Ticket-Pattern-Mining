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
import plotly.graph_objects as go
import numpy as np
import json
import os
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──
st.set_page_config(
    page_title="Legacy vs DBB — Ticket Pattern Mining",
    page_icon="🔍",
    layout="wide",
)

# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
try:
    args, _ = parser.parse_known_args()
    base_dir = args.data_dir
except Exception:
    base_dir = "data"


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
    return SentenceTransformer('all-mpnet-base-v2')

def get_available_datasets(base_dir):
    datasets = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, d)):
                if os.path.exists(os.path.join(base_dir, d, "output", "cluster_catalog.csv")):
                    datasets.append(d)
    return sorted(datasets)

# ── Sidebar ──
st.sidebar.title("🔍 Ticket Pattern Mining")
st.sidebar.markdown("**Use Case 5**: Legacy vs DBB")

datasets = get_available_datasets(base_dir)
if not datasets:
    # Fallback to root data folder if running legacy structure
    if os.path.exists(os.path.join(base_dir, "output", "cluster_catalog.csv")):
        datasets = ["Default"]
    else:
        st.error("No datasets found. Please run the pipeline first.")
        st.stop()

selected_dataset = st.sidebar.selectbox("📂 Select Dataset", datasets)
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
        "DBB_REDUCED_ISSUES": "🟢",
        "DBB_INCREASED_ISSUES": "🔴",
        "MIXED_RESULTS": "🟡",
        "INSUFFICIENT_DATA": "⚪",
    }
    icon = color_map.get(verdict, "⚪")
    st.sidebar.markdown(f"### {icon} Verdict: {verdict.replace('_', ' ').title()}")

st.sidebar.divider()
st.sidebar.caption("Pipeline: classify → normalize → vectorize → cluster → metrics → LLM naming → summary")


# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Migration Timeline",
    "🔬 Pattern Discovery",
    "🏢 Domain & Severity",
    "🔎 Cluster Deep-Dive",
    "🚀 Remediation Strategy",
    "🤖 Smart Resolution (RAG)"
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
        st.subheader("🔑 Key Findings")
        cols = st.columns(min(len(findings), 3))
        for i, f in enumerate(findings):
            with cols[i % 3]:
                impact_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(f.get("impact", ""), "⚪")
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

        with st.expander(f"{'🛡️ ' + persona + ' — ' if pd.notna(persona) and persona else ''}Cluster {cid}: {name}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets", int(row["Size"]))
            c2.metric("Legacy", int(row["Frequency_Legacy"]))
            c3.metric("DBB", int(row["Frequency_DBB"]))

            # Smart verdict
            fl, fd = int(row["Frequency_Legacy"]), int(row["Frequency_DBB"])
            if fl == 0 and fd > 0:
                c4.metric("Trend", "🆕 New in DBB")
            elif fl > 0 and fd == 0:
                c4.metric("Trend", "✅ Eliminated")
            elif fl > 0 and fd > 0:
                change = ((fd - fl) / fl) * 100
                c4.metric("Trend", f"{change:+.0f}%", delta=f"{change:+.0f}%", delta_color="inverse")
            else:
                c4.metric("Trend", "—")

            if pd.notna(analysis) and analysis:
                st.markdown(f"**🔍 Root Cause:** {analysis}")
            if pd.notna(rec) and rec:
                st.success(f"**💡 Recommendation:** {rec}")

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
                        title=f"Volume Over Time — {name}",
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
        # Add emojis
        verdict_map = {
            "improved": "✅ Improved",
            "worsened": "🔴 Worsened",
            "new_in_dbb": "🆕 New in DBB",
            "legacy_only": "📦 Legacy Only",
            "stable": "➡️ Stable",
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

    # Heatmap: Domain × Module
    st.subheader("Heatmap: Ticket Count by Domain × Module")
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
            st.markdown(f"### 🛡️ {persona}")
        st.markdown(f"## {name}")
        st.caption(f"TF-IDF Keywords: {cluster_info['Top_Keywords']}")
        st.caption(f"Domains: {cluster_info['Primary_Domains']}")

        analysis = cluster_info.get("Analysis", "")
        rec = cluster_info.get("Recommendation", "")
        if pd.notna(analysis) and analysis:
            st.markdown(f"**🔍 Root Cause Analysis:** {analysis}")
        if pd.notna(rec) and rec:
            st.info(f"**💡 Prevention Strategy:** {rec}")

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
                title=f"Ticket Volume Over Time — {name}",
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
    st.header("🚀 Remediation & Volume Reduction Strategy")
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

    # Pattern → Root Cause → Prevention Map (from cluster catalog)
    st.subheader("Pattern → Root Cause → Prevention Map")
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

    # Clusters where DBB didn't help (pollutants)
    st.subheader("⚠️ Where DBB Failed to Reduce Issues")
    pollutants = valid_catalog[
        (valid_catalog["Frequency_Legacy"] > 0)
        & (valid_catalog["Frequency_DBB"] > 0)
    ].copy()
    if not pollutants.empty:
        pollutants["Change"] = ((pollutants["Frequency_DBB"] - pollutants["Frequency_Legacy"]) / pollutants["Frequency_Legacy"] * 100).round(0)
        pollutants = pollutants.sort_values("Change", ascending=False)
        for _, row in pollutants.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            change = row["Change"]
            if change > 0:
                st.error(f"📈 **{name}**: DBB tickets are **{change:.0f}% higher** than Legacy ({int(row['Frequency_Legacy'])} → {int(row['Frequency_DBB'])})")
            else:
                st.success(f"📉 **{name}**: DBB tickets are **{abs(change):.0f}% lower** than Legacy ({int(row['Frequency_Legacy'])} → {int(row['Frequency_DBB'])})")

    # New DBB-only issues
    new_dbb = valid_catalog[valid_catalog["Frequency_Legacy"] == 0]
    if not new_dbb.empty:
        st.subheader("🆕 Issues Introduced by DBB")
        for _, row in new_dbb.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            st.warning(f"**{name}**: {int(row['Frequency_DBB'])} tickets — this pattern didn't exist in Legacy.")

    # Legacy issues eliminated
    eliminated = valid_catalog[valid_catalog["Frequency_DBB"] == 0]
    if not eliminated.empty:
        st.subheader("✅ Legacy Issues Successfully Eliminated by DBB")
        for _, row in eliminated.iterrows():
            name = row.get("Cluster_Name", row["Top_Keywords"])
            st.success(f"**{name}**: {int(row['Frequency_Legacy'])} Legacy tickets → 0 in DBB. This problem was solved.")


# ════════════════════════════════════════════════════════════════
#  TAB 6 — Smart Resolution (RAG)
# ════════════════════════════════════════════════════════════════
with tab6:
    st.header("🤖 Smart Resolution (RAG)")
    st.caption("Enter a new ticket description. The system will find similar historical tickets and suggest a resolution based on how they were solved.")
    
    new_ticket_text = st.text_area("New Ticket Description", height=150, placeholder="E.g., User unable to sync data on Glassrun app...")
    
    if st.button("Get AI Resolution"):
        if not new_ticket_text.strip():
            st.warning("Please enter a ticket description.")
        else:
            with st.spinner("Analyzing and retrieving similar tickets..."):
                embedder = load_embedder()
                new_emb = embedder.encode([new_ticket_text])
                
                # Calculate similarity
                embeddings = np.vstack(tickets['Embedding'].values)
                sims = cosine_similarity(new_emb, embeddings)[0]
                
                # Get top 5 indices
                top_indices = sims.argsort()[-5:][::-1]
                similar_tickets = tickets.iloc[top_indices].copy()
                similar_tickets['Similarity'] = sims[top_indices]
                
                # Prepare context for LLM
                context_str = ""
                for i, (_, row) in enumerate(similar_tickets.iterrows()):
                    context_str += f"--- Ticket {i+1} (Similarity: {row['Similarity']:.2f}) ---\n"
                    context_str += f"Description: {row.get('Short_Description', '')} {row.get('Description_Text', '')}\n"
                    context_str += f"Resolution Notes: {row.get('Resolution_Notes', '')}\n\n"
                    
                prompt = f"""You are an expert IT support agent.
                
A new ticket has been submitted:
"{new_ticket_text}"

Here are the most similar historical tickets and how they were resolved:
{context_str}

Based ONLY on the historical resolution notes, provide a concise, step-by-step recommended solution for the new ticket.
If the historical tickets do not contain a clear solution, state that human review is needed.
"""
                try:
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant",
                        temperature=0.2,
                        max_tokens=500
                    )
                    st.subheader("💡 AI Recommended Resolution")
                    st.write(response.choices[0].message.content)
                    
                    st.divider()
                    st.subheader("📚 Top Similar Historical Tickets")
                    display_cols = ["Similarity", "Ticket_ID", "Short_Description", "Resolution_Notes"]
                    avail_cols = [c for c in display_cols if c in similar_tickets.columns]
                    st.dataframe(similar_tickets[avail_cols], hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error calling LLM: {e}")
