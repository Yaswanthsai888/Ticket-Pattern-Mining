"""
06_executive_summary.py
-----------------------
Uses the Groq LLM to generate a data-driven executive narrative that ties
together all pipeline findings into a story aligned with Use Case 5.
Reads: cluster_catalog.csv, tickets_clustered.parquet
Writes: data/output/executive_summary.json
"""
import pandas as pd
import numpy as np
import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("groq not installed. pip install groq")
    sys.exit(1)


def generate_summary(catalog_file, tickets_file, output_file):
    print("Loading data for executive summary...")
    catalog = pd.read_csv(catalog_file)
    df = pd.read_parquet(tickets_file)

    # ── Build the data context the LLM will reason over ──
    total = len(df)
    legacy_count = len(df[df["System_Type"] == "Legacy"])
    dbb_count = len(df[df["System_Type"] == "DBB"])

    # Monthly trend
    df["YearMonth"] = df["Created_Date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby(["YearMonth", "System_Type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    monthly_summary = monthly.tail(12).to_string(index=False)

    # Domain breakdown
    domain_sys = (
        df.groupby(["Domain", "System_Type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    domain_summary = domain_sys.to_string(index=False)

    # Severity breakdown
    sev = (
        df.groupby(["System_Type", "Severity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    sev_summary = sev.to_string(index=False)

    # Reopen rates
    reopen_legacy = df[df["System_Type"] == "Legacy"]["Reopen_Flag"].mean()
    reopen_dbb = df[df["System_Type"] == "DBB"]["Reopen_Flag"].mean()

    # Cluster catalog (without noise)
    valid_cat = catalog[catalog["Cluster_ID"] != -1]
    cluster_lines = []
    for _, row in valid_cat.iterrows():
        persona = row.get("Strategic_Persona", "")
        persona_str = f" [{persona}]" if pd.notna(persona) and persona else ""
        cluster_lines.append(
            f"- Cluster {row['Cluster_ID']}{persona_str}: \"{row.get('Cluster_Name', row['Top_Keywords'])}\" | "
            f"Size={row['Size']}, Legacy={row['Frequency_Legacy']}, DBB={row['Frequency_DBB']}, "
            f"Analysis: {row.get('Analysis', 'N/A')}, "
            f"Recommendation: {row.get('Recommendation', 'N/A')}"
        )
    cluster_summary = "\n".join(cluster_lines)

    prompt = f"""You are a senior IT Service Management consultant presenting findings to C-level executives.

Below is data from a ticket-pattern-mining project comparing Legacy vs DBB (Digital Business Blueprint) systems.

## DATA CONTEXT

Total tickets: {total} | Legacy: {legacy_count} | DBB: {dbb_count}
Reopen Rate — Legacy: {reopen_legacy:.1%}, DBB: {reopen_dbb:.1%}

### Monthly Volume Trend (last 12 months)
{monthly_summary}

### Tickets by Domain × System
{domain_summary}

### Severity Breakdown
{sev_summary}

### Cluster Analysis (AI-identified patterns)
{cluster_summary}

## YOUR TASK

Produce a JSON object with these exact keys:

1. "executive_narrative": A 3-paragraph executive summary (plain text, no markdown). Paragraph 1: State the overall finding — did DBB reduce ticket volumes or not? Be specific with numbers. Paragraph 2: Highlight the top 3 most critical patterns discovered and what they mean for the business. Paragraph 3: Summarize the recommended actions.

2. "key_findings": A JSON array of 5 objects, each with "title" (short label), "detail" (1 sentence), and "impact" ("high"/"medium"/"low").

3. "shift_left_opportunities": A JSON array of 3 objects, each with "pattern" (cluster name), "strategy" (what to automate or shift left), and "estimated_reduction" (percentage of tickets this could eliminate, as a string like "40%").

4. "legacy_to_dbb_verdict": One of "DBB_REDUCED_ISSUES", "DBB_INCREASED_ISSUES", "MIXED_RESULTS", or "INSUFFICIENT_DATA".

5. "domain_health": A JSON array with one object per domain. Each object has "domain", "legacy_tickets" (int), "dbb_tickets" (int), "verdict" ("improved"/"worsened"/"new_in_dbb"/"legacy_only"/"stable").
"""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Cannot generate executive summary.")
        # Write a placeholder
        placeholder = {
            "executive_narrative": "Executive summary not generated — GROQ_API_KEY is required.",
            "key_findings": [],
            "shift_left_opportunities": [],
            "legacy_to_dbb_verdict": "INSUFFICIENT_DATA",
            "domain_health": [],
        }
        with open(output_file, "w") as f:
            json.dump(placeholder, f, indent=2)
        return

    client = Groq(api_key=api_key)

    print("Calling LLM for executive summary (this may take a few seconds)...")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior ITSM consultant. Output strictly valid JSON."},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    try:
        summary = json.loads(content)
    except json.JSONDecodeError:
        print(f"JSON decode error. Raw response:\n{content}")
        summary = {"executive_narrative": content, "key_findings": [], "shift_left_opportunities": [], "legacy_to_dbb_verdict": "INSUFFICIENT_DATA", "domain_health": []}

    print(f"Saving executive summary to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Verdict: {summary.get('legacy_to_dbb_verdict', 'N/A')}")
    print("Executive summary generation complete!")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 06_executive_summary.py <catalog_file> <tickets_file> <output_json>")
        sys.exit(1)
    generate_summary(sys.argv[1], sys.argv[2], sys.argv[3])
