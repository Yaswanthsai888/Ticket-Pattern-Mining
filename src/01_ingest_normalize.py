import pandas as pd
import numpy as np
import sys

from schema_mapper import ColumnRule, map_columns

def map_dataset_columns(df):
    rules = {
        'Number': ColumnRule(
            aliases=('Number', 'Ticket ID', 'Incident ID', 'ID'),
            regexes=(r'^(number|ticket id|incident id|id)$',),
        ),
        'Business service': ColumnRule(
            aliases=('Business service', 'App service'),
            regexes=(r'^(business service|app service)$',),
            forbidden_aliases=('Domain', 'Domain Path', 'Business User', 'Business Impact', 'Business impact'),
        ),
        'System_Subtype': ColumnRule(
            aliases=('System_Subtype',),
            regexes=(r'^(system subtype)$',),
        ),
        'Impacted OpCo': ColumnRule(
            aliases=('Impacted OpCo', 'OpCo', 'Company', 'Country', 'Location'),
            regexes=(r'^(impacted opco|opco|company|country|location)$',),
        ),
        'Created': ColumnRule(
            aliases=('Created', 'Opened', 'Submitted Date'),
            regexes=(r'^(created|opened|submitted date|date)$',),
            forbidden_contains=('created by', 'opened by', 'updated', 'vendor opened', 'last reopened'),
            expected_type='datetime',
        ),
        'Closed': ColumnRule(
            aliases=('Closed', 'Resolved', 'Completion Date'),
            regexes=(r'^(closed|resolved|completion date)$',),
            forbidden_contains=('closed by', 'vendor closed', 'resolved by'),
            expected_type='datetime',
        ),
        'Priority': ColumnRule(
            aliases=('Priority', 'Severity', 'Urgency', 'Original Priority'),
            regexes=(r'^(priority|severity|urgency|original priority)$',),
        ),
        'Assignment group': ColumnRule(
            aliases=('Assignment group', 'Resolver group', 'Assigned to group', 'Team'),
            regexes=(r'^(assignment group|resolver group|assigned to group|team)$',),
            forbidden_contains=('assigned to', 'support group', 'project support group'),
        ),
        'Reopen count': ColumnRule(
            aliases=('Reopen count', 'Reopened', 'Reopens'),
            regexes=(r'^(reopen count|reopened|reopens)$',),
        ),
        'Description': ColumnRule(
            aliases=('Description', 'Details', 'Issue'),
            regexes=(r'^(description|details|issue)$',),
            forbidden_aliases=('Translated description',),
        ),
        'Short description': ColumnRule(
            aliases=('Short description', 'Title', 'Summary', 'Subject'),
            regexes=(r'^(short description|title|summary|subject)$',),
            forbidden_aliases=('Translated Short description',),
        ),
        'Close notes': ColumnRule(
            aliases=('Close notes', 'Resolution', 'Solution', 'Integrated Service Close Notes'),
            regexes=(r'^(close notes|resolution|solution|integrated service close notes)$',),
        ),
        'Additional comments': ColumnRule(
            aliases=('Additional comments', 'Comments and Work notes', 'Work notes', 'Comments'),
            regexes=(r'^(additional comments|comments and work notes|work notes|comments)$',),
            forbidden_aliases=('Comments for Knowledge Candidate',),
        ),
        'System_Type': ColumnRule(
            aliases=('System_Type',),
            regexes=(r'^(system type)$',),
        ),
        'Primary_System': ColumnRule(
            aliases=('Primary_System',),
            regexes=(r'^(primary system)$',),
        ),
        'label_confidence': ColumnRule(
            aliases=('label_confidence',),
            regexes=(r'^(label confidence)$',),
        ),
        'label_source': ColumnRule(
            aliases=('label_source',),
            regexes=(r'^(label source)$',),
        ),
        'post_migration_noise': ColumnRule(
            aliases=('post_migration_noise',),
            regexes=(r'^(post migration noise)$',),
        ),
    }
    print("Column Mapping Results:")
    df, _ = map_columns(df, rules, verbose=True)
    return df

def ingest_and_normalize(input_file, output_file):
    
    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file, sheet_name="Classified Data")
    
    print("Normalizing columns...")
    
    print("Mapping dataset columns...")
    df = map_dataset_columns(df)
    
    # 1. Base Identifiers
    df['Ticket_ID'] = df['norm_Number']
    
    # 2. Domain / Module / OpCo
    df['Domain'] = df['norm_Business service'].fillna('Unknown')
    df['Module'] = df['norm_System_Subtype']
    df['OpCo'] = df['norm_Impacted OpCo'].fillna('Pt Multi Bintang Indonesia Tbk')
    
    # 3. Timestamps
    df['Created_Date'] = pd.to_datetime(df['norm_Created'], errors='coerce')
    df['Closed_Date'] = pd.to_datetime(df['norm_Closed'], errors='coerce')

    # 4. Severity Mapping
    if df['norm_Priority'].notna().any():
        severity_map = {'1 - Critical': 4, '2 - High': 3, '3 - Moderate': 2, '4 - Low': 1}
        # Try map first, if not matching strings, try extracting digits
        mapped = df['norm_Priority'].map(severity_map)
        if mapped.isna().all():
            mapped = df['norm_Priority'].astype(str).str.extract(r'(\d+)').astype(float)
            mapped = 5 - mapped # 1=Critical->4
        df['Severity'] = mapped.fillna(1.0)
    else:
        df['Severity'] = 1.0

    # 5. Team & Routing
    df['Assignee_Team'] = df['norm_Assignment group'].fillna('Unknown')
    
    # 6. Reopen Metrics
    df['Reopen_Count'] = pd.to_numeric(df['norm_Reopen count'], errors='coerce').fillna(0)
    df['Reopen_Flag'] = df['Reopen_Count'] > 0
        
    # 7. Text Fields
    df['Description_Text'] = df['norm_Description'].fillna('') 
    df['Short_Description'] = df['norm_Short description'].fillna('')
    df['Resolution_Notes'] = df['norm_Close notes'].fillna('') + " " + df['norm_Additional comments'].fillna('')
    
    # Pull through pipeline fields
    df['System_Type'] = df['norm_System_Type']
    df['System_Subtype'] = df['norm_System_Subtype']
    df['Primary_System'] = df['norm_Primary_System']
    df['label_confidence'] = df['norm_label_confidence']
    df['label_source'] = df['norm_label_source']
    df['post_migration_noise'] = df['norm_post_migration_noise']
    

    
    # 8. Derived Time Metrics
    df['Time_to_Resolve'] = (df['Closed_Date'] - df['Created_Date']).dt.total_seconds() / 3600.0 # hours
    df['Time_to_Resolve'] = df['Time_to_Resolve'].clip(lower=0.1) # prevent 0 division
    
    df['Week'] = df['Created_Date'].dt.to_period('W').dt.start_time
    df['Month'] = df['Created_Date'].dt.to_period('M').dt.start_time

    # 9. Days from Migration Heuristic
    print("Calculating Inferred Migration Dates...")
    dbb_tickets = df[df['System_Type'] == 'DBB'].dropna(subset=['Created_Date'])
    
    if len(dbb_tickets) > 0:
        global_migration_date = dbb_tickets['Created_Date'].min()
    else:
        global_migration_date = pd.NaT

    migration_dates = {}
    for module in df['Module'].unique():
        if 'DBB' in str(module):
            first_date = df[(df['Module'] == module) & (df['System_Type'] == 'DBB')]['Created_Date'].min()
            if pd.notna(first_date):
                migration_dates[module] = first_date

    def calculate_days(row):
        if pd.isna(row['Created_Date']):
            return np.nan
        mod_date = migration_dates.get(row['Module'], global_migration_date)
        if pd.notna(mod_date):
            return (row['Created_Date'] - mod_date).days
        return np.nan

    df['Days_from_Migration'] = df.apply(calculate_days, axis=1)

    # Output selection
    canonical_cols = [
        'Ticket_ID', 'System_Type', 'System_Subtype', 'Primary_System', 
        'Domain', 'Module', 'OpCo', 'Short_Description', 'Description_Text', 'Resolution_Notes',
        'Created_Date', 'Closed_Date', 'Severity', 'Reopen_Count', 'Reopen_Flag', 
        'Assignee_Team', 'label_confidence', 'label_source', 'post_migration_noise',
        'Time_to_Resolve', 'Week', 'Month', 'Days_from_Migration'
    ]
    
    # Check if all exist, fill missing with None
    for c in canonical_cols:
        if c not in df.columns:
            print(f"Warning: Column {c} missing, creating empty.")
            df[c] = None
            
    df_out = df[canonical_cols]
    
    print(f"Saving canonical dataset ({len(df_out)} rows) to {output_file}...")
    df_out.to_parquet(output_file, engine='pyarrow', index=False)
    print("Ingestion & Normalization complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 01_ingest_normalize.py <input_file> <output_file>")
        sys.exit(1)
    ingest_and_normalize(sys.argv[1], sys.argv[2])
