import pandas as pd
import numpy as np
import sys
import re

def map_dataset_columns(df):
    """
    Heuristic-based column mapping to ensure the pipeline works across different dataset exports.
    """
    col_mapping = {}
    
    patterns = {
        'Number': r'^(number|ticket.*id|incident.*id|id)$',
        'Configuration item': r'^(configuration.*item|ci|asset|system|application)$',
        'Service offering': r'^(service.*offering|service|product)$',
        'Assignment group': r'^(assignment.*group|resolver.*group|assigned.*to.*group|team)$',
        'Business service': r'^(business.*service|domain|app.*service)$',
        'Short description': r'^(short.*description|title|summary|subject)$',
        'Description': r'^(description|details|issue)$',
        'Close notes': r'^(close.*notes|resolution|solution)$',
        'Additional comments': r'^(additional.*comments|comments|work.*notes)$',
        'Priority': r'^(priority|severity|urgency)$',
        'Created': r'^(created.*|opened.*|submitted.*date|date)$',
        'Closed': r'^(closed.*|resolved.*date|completion.*date)$',
        'Reopen count': r'^(reopen.*count|reopened|reopens)$',
        'Impacted OpCo': r'^(impacted.*opco|opco|company|country|location)$',
    }
    
    for target_col, pattern in patterns.items():
        for actual_col in df.columns:
            if actual_col in col_mapping.values():
                continue
            if re.match(pattern, str(actual_col).lower().strip()):
                col_mapping[target_col] = actual_col
                break
                
    print("Column Mapping Results:")
    for canonical, actual in col_mapping.items():
        print(f"  {canonical} <--- {actual}")
        df[f'norm_{canonical}'] = df[actual]
        
    # Fill in any missing canonical columns with empty strings/NaT to prevent KeyError
    for target_col in patterns.keys():
        if f'norm_{target_col}' not in df.columns:
            print(f"  Warning: No column found for '{target_col}'")
            df[f'norm_{target_col}'] = np.nan
            
    return df

def classify_tickets(input_file, output_file):
    print(f"Loading {input_file}...")
    if input_file.lower().endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        # Load first sheet by default
        xl = pd.ExcelFile(input_file)
        df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
        
    print("Mapping dataset columns...")
    df = map_dataset_columns(df)
    
    print("Classifying tickets...")
    df['System_Type'] = 'Unknown'
    df['System_Subtype'] = 'Unknown'
    df['Primary_System'] = 'None'
    df['label_confidence'] = 0.0
    df['label_source'] = 'none'
    df['mapping_version'] = 'v2.0' # Updated version with NLP mapping
    df['human_override_flag'] = False
    df['post_migration_noise'] = False
    
    def get_classification(row):
        # Use the mapped 'norm_' columns
        ci = str(row.get('norm_Configuration item', '')).lower()
        so = str(row.get('norm_Service offering', '')).lower()
        ag = str(row.get('norm_Assignment group', '')).lower()
        bs = str(row.get('norm_Business service', '')).lower()
        sd = str(row.get('norm_Short description', '')).lower()
        
        # Rule B: POSM Non-DBB
        if 'non-dbb' in so:
            return pd.Series(['Legacy', 'Legacy-Other', 'None', 1.0, 'rule-service'])
            
        # Rule C: Middleware / Both
        if bs == 'middleware' or 'digital integration' in ci or 'solace' in ci or 'commerce integration' in so or 'commerce integration' in ag or 'finance technology integration' in so:
            return pd.Series(['Both', 'Middleware-ESB', 'Integration', 1.0, 'rule-middleware'])
            
        # Priority 1: Configuration Item
        if 'otd production' in ci:
            if 'glassrun' in sd:
                return pd.Series(['DBB', 'DBB-GlassRun', 'None', 1.0, 'rule-config+text'])
            return pd.Series(['DBB', 'DBB-OTD', 'None', 1.0, 'rule-config'])
        elif 'omni production' in ci or 'tiger tribe' in ci:
            return pd.Series(['DBB', 'DBB-OMNI', 'None', 1.0, 'rule-config'])
        elif 'sem' in ci or 'dynamics 365' in ci:
            return pd.Series(['DBB', 'DBB-SEM', 'None', 1.0, 'rule-config'])
        elif 'virtocommerce' in ci or ('b2b' in ci and 'dot' in ci):
            return pd.Series(['DBB', 'DBB-DOT', 'None', 1.0, 'rule-config'])
        elif 'd&a hub' in ci:
            return pd.Series(['DBB', 'DBB-DA', 'None', 1.0, 'rule-config'])
        elif any(x in ci for x in ['erp sap', 'heicore', 'fiori', 'srm', 'qualass']):
            return pd.Series(['Legacy', 'Legacy-SAP', 'None', 1.0, 'rule-config'])
        elif any(x in ci for x in ['psub', 'pjkt', 'ijkt', 'isub']):
            return pd.Series(['Legacy', 'Legacy-Network', 'None', 1.0, 'rule-config'])
            
        # Priority 2: Service Offering
        if 'glassrun' in so or 'otd ' in so:
            return pd.Series(['DBB', 'DBB-OTD', 'None', 0.95, 'rule-service'])
        elif 'omni' in so:
            return pd.Series(['DBB', 'DBB-OMNI', 'None', 0.95, 'rule-service'])
        elif 'sem ' in so:
            return pd.Series(['DBB', 'DBB-SEM', 'None', 0.95, 'rule-service'])
        elif 'dot ' in so:
            return pd.Series(['DBB', 'DBB-DOT', 'None', 0.95, 'rule-service'])
        elif 'd&a hub' in so:
            return pd.Series(['DBB', 'DBB-DA', 'None', 0.95, 'rule-service'])
        elif 'commerce integration' in so:
            return pd.Series(['DBB', 'DBB-Commerce', 'None', 0.95, 'rule-service'])
        elif 'eazle' in so or 'b2gaas' in so:
            return pd.Series(['DBB', 'DBB-Eazle', 'None', 0.95, 'rule-service'])
        elif any(x in so for x in ['erp sap', 'ptp', 'rtr', 'dtw']):
            return pd.Series(['Legacy', 'Legacy-SAP', 'None', 0.95, 'rule-service'])
        elif any(x in so for x in ['ip networks', 'ntw ', 'heinet', 'sdwan']):
            return pd.Series(['Legacy', 'Legacy-Network', 'None', 0.95, 'rule-service'])
        elif any(x in so for x in ['windows ', 'hardware', 'remote access', 'ad ']):
            return pd.Series(['Legacy', 'Legacy-Workplace', 'None', 0.95, 'rule-service'])
        elif any(x in so for x in ['anti virus', 'wps ']):
            return pd.Series(['Legacy', 'Legacy-Security', 'None', 0.95, 'rule-service'])
        elif any(x in so for x in ['ms teams', 'sharepoint', 'onedrive']):
            return pd.Series(['Legacy', 'Legacy-Collaboration', 'None', 0.95, 'rule-service'])
        elif any(x in so for x in ['ifrs16', 'fin ', 'heifund']):
            return pd.Series(['Legacy', 'Legacy-Finance', 'None', 0.95, 'rule-service'])
            
        # Priority 3: Assignment Group
        if 'otd support' in ag or 'tiger tribe' in ag:
            subtype = 'DBB-OMNI' if 'tiger' in ag else 'DBB-OTD'
            return pd.Series(['DBB', subtype, 'None', 0.85, 'rule-assignment'])
        elif 'omni support' in ag:
            return pd.Series(['DBB', 'DBB-OMNI', 'None', 0.85, 'rule-assignment'])
        elif 'sem devops' in ag:
            return pd.Series(['DBB', 'DBB-SEM', 'None', 0.85, 'rule-assignment'])
        elif 'dot infosys' in ag:
            return pd.Series(['DBB', 'DBB-DOT', 'None', 0.85, 'rule-assignment'])
        elif 'd&a hub' in ag:
            return pd.Series(['DBB', 'DBB-DA', 'None', 0.85, 'rule-assignment'])
        elif 'gis orange' in ag:
            return pd.Series(['Legacy', 'Legacy-Network', 'None', 0.85, 'rule-assignment'])
        elif 'erp sap' in ag:
            return pd.Series(['Legacy', 'Legacy-SAP', 'None', 0.85, 'rule-assignment'])
        elif 't-systems' in ag or 'wpl ' in ag:
            return pd.Series(['Legacy', 'Legacy-Workplace', 'None', 0.85, 'rule-assignment'])
            
        # Priority 4: Business Service
        if bs in ['market to order (mto)', 'demand to warehouse (dtw)', 'market to cash (mtc)']:
            return pd.Series(['DBB', 'DBB-Other', 'None', 0.70, 'rule-business'])
        elif bs == 'network':
            return pd.Series(['Legacy', 'Legacy-Network', 'None', 0.70, 'rule-business'])
        elif bs in ['source to pay (stp)', 'record to report (rtr)', 'hosting sap']:
            return pd.Series(['Legacy', 'Legacy-SAP', 'None', 0.70, 'rule-business'])
        elif bs in ['workplace', 'security management', 'collaboration']:
            return pd.Series(['Legacy', 'Legacy-Workplace', 'None', 0.70, 'rule-business'])
        elif bs in ['service desk services', 'gsd (itsm) service', 'servicenow nextgen - siam']:
            return pd.Series(['Legacy', 'Legacy-ITSM', 'None', 0.70, 'rule-business'])
            
        # Priority 5: Text Signature
        if 'glassrun' in sd:
            return pd.Series(['DBB', 'DBB-GlassRun', 'None', 0.60, 'rule-text'])
        elif 'omni' in sd:
            return pd.Series(['DBB', 'DBB-OMNI', 'None', 0.60, 'rule-text'])
            
        return pd.Series(['Unknown', 'Unknown', 'None', 0.0, 'none'])

    # Apply classification
    new_cols = df.apply(get_classification, axis=1)
    df[['System_Type', 'System_Subtype', 'Primary_System', 'label_confidence', 'label_source']] = new_cols

    # Rule E: Reopen count heuristic
    if 'norm_Created' in df.columns and 'norm_Reopen count' in df.columns:
        created_dates = pd.to_datetime(df['norm_Created'], errors='coerce')
        dbb_min_date = created_dates[df['System_Type'] == 'DBB'].min()
        if pd.notna(dbb_min_date):
            reopens = pd.to_numeric(df['norm_Reopen count'], errors='coerce').fillna(0).astype(int)
            noise_mask = (reopens > 0) & \
                         (created_dates >= dbb_min_date) & \
                         (created_dates <= dbb_min_date + pd.Timedelta(days=90))
            df.loc[noise_mask, 'post_migration_noise'] = True

    print("Classification distribution:")
    print(df['System_Type'].value_counts())
    
    # Drop the temporary 'norm_' columns before saving
    cols_to_drop = [c for c in df.columns if c.startswith('norm_')]
    df = df.drop(columns=cols_to_drop)

    unknown_df = df[df['System_Type'] == 'Unknown'].copy()
    unknown_df['SME_Label'] = ''
    unknown_df['SME_Notes'] = ''
    unknown_df['override_date'] = ''
    
    print(f"\nSaving to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Classified Data', index=False)
        unknown_df.to_excel(writer, sheet_name='Unknown Review Queue', index=False)
        
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python 00_classify_tickets.py <input_file> <output_file>")
        sys.exit(1)
    classify_tickets(sys.argv[1], sys.argv[2])
