"""## 11. Leakage-safe stratified split — CPU

The untouched test set is created before any augmentation or vector fitting.
"""

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=SEED)

for name, split in [('train',train_df),('validation',val_df),('test',test_df)]:
    split = split.reset_index(drop=True)
    split[['text','label','label_name']].to_csv(PROJECT_DIR/f'data/splits/{name}.csv', index=False)
    print(name, len(split), split.label_name.value_counts(normalize=True).round(3).to_dict())

assert not set(train_df.text_normalized) & set(test_df.text_normalized)

