import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

xl = pd.ExcelFile(r'c:\Users\yaswa\OneDrive\Desktop\Projects\DBB VS Legacy\Indonesia_Incidents.xlsx')
df = pd.read_excel(xl, sheet_name='Sheet1')

print(f"Total tickets: {len(df)}")

# ── CHECK 1: Ticket ID patterns ──────────────────────────────────────────────
print("\n=== CHECK 1: Ticket Number Patterns ===")
df['num_prefix'] = df['Number'].astype(str).str.extract(r'^([A-Z]+)')
print(df['num_prefix'].value_counts().to_string())
df['num_digits'] = pd.to_numeric(df['Number'].astype(str).str.extract(r'([0-9]+)$')[0], errors='coerce')
print(f"Ticket number range: {df['num_digits'].min():.0f} to {df['num_digits'].max():.0f}")

# ── CHECK 2: OMNI config items vs assignment groups ──────────────────────────
print("\n=== CHECK 2: OMNI Configuration Items - Who owns them? ===")
omni_mask = df['Configuration item'].str.contains('OMNI|Omni|omni', na=False, case=False)
omni_df = df[omni_mask][['Configuration item', 'Assignment group', 'Service offering', 'Business service']]
print(omni_df.value_counts().head(20).to_string())

# ── CHECK 3: Tiger Tribe config items ───────────────────────────────────────
print("\n=== CHECK 3: Tiger Tribe L2 - Configuration Items ===")
tt_mask = df['Assignment group'].str.contains('Tiger Tribe', na=False)
print(f"Tiger Tribe tickets: {tt_mask.sum()}")
tt_df = df[tt_mask][['Number', 'Configuration item', 'Service offering', 'Business service', 'Short description']]
print(tt_df[['Configuration item', 'Service offering', 'Business service']].value_counts().to_string())

# ── CHECK 4: OTD/GlassRun text signature ────────────────────────────────────
print("\n=== CHECK 4: OTD/GlassRun Text Signatures (Short Description) ===")
otd_mask = df['Short description'].str.contains('glassrun|GlassRun|Glassrun|OTD', na=False, case=False)
print(f"Tickets with OTD/GlassRun in short description: {otd_mask.sum()}")
print(df[otd_mask]['Short description'].value_counts().head(10).to_string())

# ── CHECK 5: Middleware/Integration signals ──────────────────────────────────
print("\n=== CHECK 5: Middleware / Integration Signals ===")
middleware_mask = (
    df['Business service'].str.contains('Middleware', na=False) |
    df['Service offering'].str.contains('Integration|Middleware|Boomi|SAP PO|PI/PO', na=False, case=False) |
    df['Configuration item'].str.contains('Integration|Middleware|Boomi|SAP PO', na=False, case=False) |
    df['Assignment group'].str.contains('Integration|Middleware|Digital Integration', na=False, case=False)
)
print(f"Middleware/Integration tickets: {middleware_mask.sum()}")
mid_df = df[middleware_mask][['Configuration item', 'Service offering', 'Business service', 'Assignment group']]
print(mid_df.value_counts().head(20).to_string())

# ── CHECK 6: Opened date range ───────────────────────────────────────────────
print("\n=== CHECK 6: Opened/Created Dates ===")
if 'Created' in df.columns:
    created = pd.to_datetime(df['Created'], errors='coerce')
    print(f"Created: min={created.min()}, max={created.max()}")

# Check Closed column
if 'Closed' in df.columns:
    closed = pd.to_datetime(df['Closed'], errors='coerce')
    print(f"Closed:  min={closed.min()}, max={closed.max()}")

# ── CHECK 7: Keyword frequency in Service offering - POSM Non-DBB ──────────
print("\n=== CHECK 7: Explicit Non-DBB References ===")
nondbb = df[df['Service offering'].str.contains('Non-DBB', na=False, case=False)]
print(f"Tickets with 'Non-DBB' in Service offering: {len(nondbb)}")
print(nondbb[['Number','Short description','Service offering','Assignment group']].to_string())

# ── CHECK 8: 'Both' candidates - integration tickets ─────────────────────────
print("\n=== CHECK 8: Potential 'Both' (Integration) Tickets ===")
both_mask = (
    df['Short description'].str.contains('integrat|sync|interface|API|middleware|boomi|SAP PO', na=False, case=False) |
    df['Service offering'].str.contains('Integration|Digital Integration Supply Chain|Commerce Integration', na=False, case=False)
)
print(f"Potential integration/both tickets: {both_mask.sum()}")
print(df[both_mask][['Service offering', 'Assignment group', 'Business service']].value_counts().head(15).to_string())

print("\n=== DONE ===")
