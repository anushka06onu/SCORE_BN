"""## 4. Load the CSV/XLSX automatically — CPU"""

# Prefer XLSX because the official CSV currently contains a mixed-encoding byte.
candidate_files = list(raw_dir.rglob('*.xlsx')) + list(raw_dir.rglob('*.xls')) + list(raw_dir.rglob('*.csv'))
assert candidate_files, 'No CSV/XLSX found. Download and upload the dataset first.'

data_path = candidate_files[0]
if data_path.suffix.lower() == '.csv':
    try:
        df_raw = pd.read_csv(data_path)
    except UnicodeDecodeError:
        df_raw = pd.read_csv(data_path, encoding='cp1252')
else:
    df_raw = pd.read_excel(data_path)

df_raw.columns = [str(c).strip() for c in df_raw.columns]
print('Loaded:', data_path)
print('Shape:', df_raw.shape)
print('Columns:', df_raw.columns.tolist())
display(df_raw.head())

