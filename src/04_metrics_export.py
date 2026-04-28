import pandas as pd
import numpy as np

import sys

def run_metrics(input_file, catalog_file, pivot_file):
    
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    catalog = pd.read_csv(catalog_file)
    
    # Exclude noise for cluster catalog metrics
    df_clusters = df[df['Cluster_ID'] != -1]
    
    print("Calculating intelligence metrics per cluster...")
    metrics = []
    
    for cluster_id in df_clusters['Cluster_ID'].unique():
        cluster_df = df_clusters[df_clusters['Cluster_ID'] == cluster_id]
        
        # Frequencies
        freq_legacy = len(cluster_df[cluster_df['System_Type'] == 'Legacy'])
        freq_dbb = len(cluster_df[cluster_df['System_Type'] == 'DBB'])
        
        # Reduction Rate
        if freq_legacy > 0:
            reduction_rate = 1 - (freq_dbb / freq_legacy)
        else:
            reduction_rate = np.nan # New in DBB
            
        # Impact
        avg_sev_legacy = cluster_df[cluster_df['System_Type'] == 'Legacy']['Severity'].mean() if freq_legacy > 0 else 0
        avg_sev_dbb = cluster_df[cluster_df['System_Type'] == 'DBB']['Severity'].mean() if freq_dbb > 0 else 0
        
        impact_legacy = freq_legacy * avg_sev_legacy
        impact_dbb = freq_dbb * avg_sev_dbb
        
        # TTR
        avg_ttr_legacy = cluster_df[cluster_df['System_Type'] == 'Legacy']['Time_to_Resolve'].mean()
        avg_ttr_dbb = cluster_df[cluster_df['System_Type'] == 'DBB']['Time_to_Resolve'].mean()
        
        ttr_delta = avg_ttr_dbb - avg_ttr_legacy if pd.notna(avg_ttr_dbb) and pd.notna(avg_ttr_legacy) else np.nan
        
        # Reopen Rate
        reopen_rate_legacy = cluster_df[cluster_df['System_Type'] == 'Legacy']['Reopen_Flag'].mean()
        reopen_rate_dbb = cluster_df[cluster_df['System_Type'] == 'DBB']['Reopen_Flag'].mean()
        
        reopen_delta = reopen_rate_dbb - reopen_rate_legacy if pd.notna(reopen_rate_dbb) and pd.notna(reopen_rate_legacy) else np.nan
        
        # Post Migration Noise Flag (if >20% of cluster tickets are marked as noise)
        noise_ratio = cluster_df['post_migration_noise'].mean()
        noise_flag = noise_ratio > 0.2 if pd.notna(noise_ratio) else False
        
        metrics.append({
            'Cluster_ID': cluster_id,
            'Frequency_Legacy': freq_legacy,
            'Frequency_DBB': freq_dbb,
            'Reduction_Rate': round(reduction_rate, 3) if pd.notna(reduction_rate) else np.nan,
            'Impact_Legacy': round(impact_legacy, 1),
            'Impact_DBB': round(impact_dbb, 1),
            'AvgTTR_Legacy_Hours': round(avg_ttr_legacy, 1) if pd.notna(avg_ttr_legacy) else np.nan,
            'AvgTTR_DBB_Hours': round(avg_ttr_dbb, 1) if pd.notna(avg_ttr_dbb) else np.nan,
            'AvgTTR_Delta_Hours': round(ttr_delta, 1) if pd.notna(ttr_delta) else np.nan,
            'ReopenRate_Delta': round(reopen_delta, 3) if pd.notna(reopen_delta) else np.nan,
            'PostMigrationNoiseFlag': noise_flag
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Refresh metric columns on reruns so the catalog does not accumulate _x/_y duplicates.
    metric_columns = [c for c in metrics_df.columns if c != 'Cluster_ID']
    metric_prefixes = tuple(metric_columns)
    stale_metric_columns = [
        c for c in catalog.columns
        if c in metric_columns
        or c.endswith('_x')
        or c.endswith('_y')
        or c.rsplit('_', 1)[0] in metric_columns
    ]
    catalog = catalog.drop(columns=stale_metric_columns, errors='ignore')

    # Merge with catalog
    final_catalog = catalog.merge(metrics_df, on='Cluster_ID', how='left')
    
    print(f"Saving final cluster catalog to {catalog_file}...")
    final_catalog.to_csv(catalog_file, index=False)
    
    # Pivot Table: Domain x Module x System_Type
    print("Generating Pivot Table...")
    pivot = pd.pivot_table(
        df,
        values='Ticket_ID',
        index=['Domain', 'Module'],
        columns=['System_Type'],
        aggfunc='count',
        fill_value=0
    ).reset_index()
    
    if 'Legacy' in pivot.columns and 'DBB' in pivot.columns:
        pivot['Reduction_Rate'] = 1 - (pivot['DBB'] / pivot['Legacy'].replace(0, np.nan))
    
    print(f"Saving Pivot Table to {pivot_file}...")
    pivot.to_csv(pivot_file, index=False)
    
    print("Metrics evaluation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 04_metrics_export.py <input_file> <catalog_file> <pivot_file>")
        sys.exit(1)
    run_metrics(sys.argv[1], sys.argv[2], sys.argv[3])
