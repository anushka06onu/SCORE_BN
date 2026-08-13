"""## 12. Classical preprocessing and feature engineering — CPU"""

def clean_classical(text):
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' USER ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

X_train_text = train_df.text.map(clean_classical)
X_val_text = val_df.text.map(clean_classical)
X_test_text = test_df.text.map(clean_classical)
y_train, y_val, y_test = train_df.label.values, val_df.label.values, test_df.label.values

