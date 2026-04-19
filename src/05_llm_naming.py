import pandas as pd
import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("Groq library not installed. Please run: pip install groq")
    sys.exit(1)

def run_llm_naming(catalog_file, tickets_file):
    print(f"Loading {catalog_file} and {tickets_file}...")
    catalog = pd.read_csv(catalog_file)
    tickets = pd.read_parquet(tickets_file)
    
    # Initialize new columns if they don't exist
    if 'Cluster_Name' not in catalog.columns:
        catalog['Cluster_Name'] = catalog['Top_Keywords'] # Default fallback
    if 'Strategic_Persona' not in catalog.columns:
        catalog['Strategic_Persona'] = ''
    if 'Analysis' not in catalog.columns:
        catalog['Analysis'] = ''
    if 'Recommendation' not in catalog.columns:
        catalog['Recommendation'] = ''
    
    # Filter out noise
    valid_clusters = catalog[catalog['Cluster_ID'] != -1].copy()
    
    # Assign personas
    selected_clusters = {} # persona -> cluster_id
    
    if 'Reduction_Rate' in valid_clusters.columns:
        pollutants = valid_clusters[(valid_clusters['Frequency_Legacy'] >= 5) & (valid_clusters['Frequency_DBB'] >= 5)]
        if not pollutants.empty:
            c_id = pollutants.sort_values('Reduction_Rate', ascending=True).iloc[0]['Cluster_ID']
            selected_clusters['The DBB Pollutant'] = c_id
        
    if 'ReopenRate_Delta' in valid_clusters.columns:
        chronic = valid_clusters.sort_values('ReopenRate_Delta', ascending=False)
        for _, row in chronic.iterrows():
            if row['Cluster_ID'] not in selected_clusters.values() and row['Size'] >= 3:
                selected_clusters['The Chronic Defect'] = row['Cluster_ID']
                break
                
    goldmine = valid_clusters.sort_values('Size', ascending=False)
    for _, row in goldmine.iterrows():
        if row['Cluster_ID'] not in selected_clusters.values():
            selected_clusters['The Automation Goldmine'] = row['Cluster_ID']
            break
                
    new_dbb = valid_clusters[valid_clusters['Frequency_Legacy'] == 0].sort_values('Frequency_DBB', ascending=False)
    for _, row in new_dbb.iterrows():
        if row['Cluster_ID'] not in selected_clusters.values() and row['Size'] >= 2:
            selected_clusters['The New DBB Headache'] = row['Cluster_ID']
            break
            
    legacy_only = valid_clusters[valid_clusters['Frequency_DBB'] == 0].sort_values('Frequency_Legacy', ascending=False)
    for _, row in legacy_only.iterrows():
        if row['Cluster_ID'] not in selected_clusters.values() and row['Size'] >= 2:
            selected_clusters['The Legacy Anchor'] = row['Cluster_ID']
            break

    # Fill missing
    remaining = valid_clusters[~valid_clusters['Cluster_ID'].isin(selected_clusters.values())].sort_values('Size', ascending=False)
    idx = 0
    personas = ['The DBB Pollutant', 'The Chronic Defect', 'The Automation Goldmine', 'The New DBB Headache', 'The Legacy Anchor']
    for p in personas:
        if p not in selected_clusters and idx < len(remaining):
            selected_clusters[p] = remaining.iloc[idx]['Cluster_ID']
            idx += 1

    # Map back cluster to persona
    cluster_to_persona = {cid: p for p, cid in selected_clusters.items()}
    
    # Set the persona in catalog
    for cid, persona in cluster_to_persona.items():
        catalog.loc[catalog['Cluster_ID'] == cid, 'Strategic_Persona'] = persona

    # LLM Call for ALL valid clusters
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nGROQ_API_KEY environment variable not set. Skipping LLM naming.")
        catalog.to_csv(catalog_file, index=False)
        return
        
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Groq Client error: {e}")
        catalog.to_csv(catalog_file, index=False)
        return
        
    print(f"\nAnalyzing ALL {len(valid_clusters)} valid clusters with LLM...")
    
    for _, row in valid_clusters.iterrows():
        cid = row['Cluster_ID']
        persona = cluster_to_persona.get(cid, "Standard Cluster")
        
        sample_tickets = tickets[tickets['Cluster_ID'] == cid]['Short_Description'].dropna().head(10).tolist()
        
        prompt = f"""
        I am mining tickets from an IT system to find recurring patterns. Analyze this cluster:
        
        System/Domain: {row['Primary_Domains']}
        Keywords: {row['Top_Keywords']}
        
        Sample Tickets:
        """ + "\n".join([f"- {t}" for t in sample_tickets]) + """
        
        Provide a JSON object with exactly these keys:
        - "Cluster_Name": A concise, highly specific name (3 to 6 words) describing the root problem.
        - "Analysis": A 1-sentence explanation of what is likely causing these tickets.
        - "Recommendation": A 1-sentence actionable recommendation (e.g., automation, shift-left, KB article) to prevent these tickets.
        """
        
        try:
            print(f"Analyzing Cluster {cid}...")
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert ITSM analyst. Output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant", 
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is None:
                content = "{}"
                
            try:
                data = json.loads(content)
                name = data.get("Cluster_Name", row['Top_Keywords'])
                analysis = data.get("Analysis", "No analysis provided.")
                recommendation = data.get("Recommendation", "No recommendation provided.")
            except json.JSONDecodeError:
                print(f"  JSON decode failed for response: {content}")
                name = row['Top_Keywords']
                analysis = ""
                recommendation = ""
                
            print(f"  -> Name: {name}")
            catalog.loc[catalog['Cluster_ID'] == cid, 'Cluster_Name'] = name
            catalog.loc[catalog['Cluster_ID'] == cid, 'Analysis'] = analysis
            catalog.loc[catalog['Cluster_ID'] == cid, 'Recommendation'] = recommendation
        except Exception as e:
            print(f"  Failed to analyze cluster {cid}: {e}")
            
    print(f"\nSaving named catalog to {catalog_file}...")
    catalog.to_csv(catalog_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 05_llm_naming.py <catalog_file> <tickets_file>")
        sys.exit(1)
    run_llm_naming(sys.argv[1], sys.argv[2])
