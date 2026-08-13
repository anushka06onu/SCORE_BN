"""## 7. Map ordered labels — CPU

The numeric order is `General=0`, `Routine=1`, `Urgent=2`, `Emergency=3`.
"""

print('Original labels and counts:')
display(df['label_raw'].astype(str).value_counts(dropna=False).to_frame('count'))

def canonical_label(value):
    s = re.sub(r'[_-]+', ' ', str(value).strip().lower())
    s = re.sub(r'\s+', ' ', s)
    if 'emergency' in s: return 'Emergency'
    if 'urgent' in s: return 'Urgent'
    if 'routine' in s: return 'Routine'
    if 'general' in s: return 'General Query'
    return None

df['label_name'] = df['label_raw'].map(canonical_label)
unmapped = df[df.label_name.isna()]['label_raw'].unique()
assert len(unmapped) == 0, f'Unmapped labels: {unmapped}. Update canonical_label().'

LABEL2ID = {'General Query':0, 'Routine':1, 'Urgent':2, 'Emergency':3}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}
df['label'] = df['label_name'].map(LABEL2ID).astype(int)
display(df['label_name'].value_counts().reindex(LABEL2ID).to_frame('count'))

