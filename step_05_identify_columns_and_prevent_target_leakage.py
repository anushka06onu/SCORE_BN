"""## 5. Identify columns and prevent target leakage — CPU

Only `Text` is used as input. `Action Needed` is explicitly excluded.
"""

column_lookup = {c.lower().strip(): c for c in df_raw.columns}

def find_column(possible_names):
    for name in possible_names:
        if name in column_lookup:
            return column_lookup[name]
    for lower, original in column_lookup.items():
        if any(name in lower for name in possible_names):
            return original
    return None

TEXT_COL = find_column(['text','query','question','healthcare query'])
LABEL_COL = find_column(['categories','category','label','severity','target'])
LEAK_COL = find_column(['action needed','action_needed','action'])

assert TEXT_COL and LABEL_COL, f'Could not identify columns. Available: {df_raw.columns.tolist()}'
print('Text column:', TEXT_COL)
print('Target column:', LABEL_COL)
print('Excluded leakage column:', LEAK_COL)

df = df_raw[[TEXT_COL, LABEL_COL]].copy()
df.columns = ['text', 'label_raw']
display(df.head())

