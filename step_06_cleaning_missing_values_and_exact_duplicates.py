"""## 6. Cleaning, missing values and exact duplicates — CPU"""

def normalize_for_dedup(text):
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

audit = {
    'original_rows': len(df),
    'missing_text': int(df['text'].isna().sum()),
    'missing_label': int(df['label_raw'].isna().sum())
}
df = df.dropna(subset=['text','label_raw']).copy()
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 0].copy()
df['text_normalized'] = df['text'].map(normalize_for_dedup)
audit['exact_normalized_duplicates'] = int(df.duplicated('text_normalized').sum())

# Conflicting duplicate labels must be inspected, not silently retained.
conflicts = (df.groupby('text_normalized')['label_raw'].nunique() > 1)
conflicting_texts = set(conflicts[conflicts].index)
print('Conflicting duplicate groups:', len(conflicting_texts))
if conflicting_texts:
    display(df[df.text_normalized.isin(conflicting_texts)].sort_values('text_normalized').head(30))

# Remove conflicting groups and keep one copy of consistent exact duplicates.
df = df[~df.text_normalized.isin(conflicting_texts)].drop_duplicates('text_normalized').reset_index(drop=True)
audit['clean_rows'] = len(df)
display(pd.Series(audit, name='value').to_frame())

