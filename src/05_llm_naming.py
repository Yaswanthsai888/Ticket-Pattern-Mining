import json
import sys

import pandas as pd

from llm_gateway import generate_json

TOP_N = 15


def run_llm_naming(catalog_file, tickets_file):
    catalog = pd.read_csv(catalog_file)
    tickets = pd.read_parquet(tickets_file)

    for column in ["Cluster_Name", "Strategic_Persona", "Analysis", "Recommendation"]:
        if column not in catalog.columns:
            catalog[column] = ""
        catalog[column] = catalog[column].fillna("").astype(str)

    catalog["Cluster_Name"] = catalog["Top_Keywords"]

    valid = catalog[catalog["Cluster_ID"] != -1].copy()
    selected = valid.sort_values(
        by=["Business_Priority", "Size"],
        ascending=False,
    ).head(TOP_N)

    print(f"Analyzing only TOP {len(selected)} clusters")

    for _, row in selected.iterrows():
        cid = row["Cluster_ID"]

        samples = tickets[
            tickets["Cluster_ID"] == cid
        ]["Short_Description"].dropna().head(10).tolist()

        prompt = f"""
You are analyzing a recurring IT ticket cluster.
Return valid JSON only with these keys:
- Cluster_Name
- Analysis
- Recommendation

Keywords: {row['Top_Keywords']}
Domain: {row['Primary_Domains']}

Samples:
{chr(10).join(samples)}
"""

        try:
            data = generate_json(
                prompt,
                max_output_tokens=800,
            )

            catalog.loc[catalog["Cluster_ID"] == cid, "Cluster_Name"] = data.get(
                "Cluster_Name",
                row["Top_Keywords"],
            )
            catalog.loc[catalog["Cluster_ID"] == cid, "Analysis"] = data.get("Analysis", "")
            catalog.loc[catalog["Cluster_ID"] == cid, "Recommendation"] = data.get(
                "Recommendation",
                "",
            )

            print(f"Cluster {cid} done")

        except json.JSONDecodeError as exc:
            print(f"Cluster {cid} failed: invalid JSON response: {exc}")
        except Exception as exc:
            print(f"Cluster {cid} failed: {exc}")

    catalog.to_csv(catalog_file, index=False)
    print("LLM naming complete.")


if __name__ == "__main__":
    run_llm_naming(sys.argv[1], sys.argv[2])
