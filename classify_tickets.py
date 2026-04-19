import pandas as pd
import numpy as np

def classify_tickets():
    input_file = r'c:\Users\yaswa\OneDrive\Desktop\Projects\DBB VS Legacy\Indonesia_Incidents.xlsx'
    output_file = r'c:\Users\yaswa\OneDrive\Desktop\Projects\DBB VS Legacy\Indonesia_Incidents_Classified.xlsx'
    
    print(f"Loading {input_file}...")
    xl = pd.ExcelFile(input_file)
    df = pd.read_excel(xl, sheet_name='Sheet1')
    
    print("Classifying tickets...")
    
    # Initialize new columns
    df['System_Type'] = 'Unknown'
    df['System_Subtype'] = 'Unknown'
    df['Primary_System'] = 'None'
    df['label_confidence'] = 0.0
    df['label_source'] = 'none'
    df['mapping_version'] = 'v1.0'
    df['human_override_flag'] = False
    df['post_migration_noise'] = False
    
    # Pre-calculate first DBB ticket date for post_migration_noise
    # We will approximate this based on the earliest DBB ticket found
    
    def get_classification(row):
        ci = str(row.get('Configuration item', '')).lower()
        so = str(row.get('Service offering', '')).lower()
        ag = str(row.get('Assignment group', '')).lower()
        bs = str(row.get('Business service', '')).lower()
        sd = str(row.get('Short description', '')).lower()
        
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

    # Rule E: Reopen count heuristic (migration window noise)
    if 'Created' in df.columns and 'Reopen count' in df.columns:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        dbb_min_date = df[df['System_Type'] == 'DBB']['Created'].min()
        if not pd.isna(dbb_min_date):
            noise_mask = (df['Reopen count'].fillna(0).astype(int) > 0) & \
                         (df['Created'] >= dbb_min_date) & \
                         (df['Created'] <= dbb_min_date + pd.Timedelta(days=90))
            df.loc[noise_mask, 'post_migration_noise'] = True

    # Rearrange columns to place new ones after DBB Stream if it exists
    cols = list(df.columns)
    if 'DBB Stream' in cols:
        dbb_idx = cols.index('DBB Stream')
        # Remove newly added columns from current positions
        new_col_names = ['System_Type', 'System_Subtype', 'Primary_System', 'label_confidence', 'label_source', 'mapping_version', 'human_override_flag', 'post_migration_noise']
        for col in new_col_names:
            cols.remove(col)
        # Insert them right after DBB Stream
        for i, col in enumerate(new_col_names):
            cols.insert(dbb_idx + 1 + i, col)
        df = df[cols]

    print("Classification distribution:")
    print(df['System_Type'].value_counts())
    
    print("\nSubtype distribution:")
    print(df['System_Subtype'].value_counts().head(10))

    # Split into known and unknown
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
    classify_tickets()
