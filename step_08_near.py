"""## 8. Near-duplicate audit — CPU

This samples suspiciously similar pairs. Review them before final reporting.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

char_audit = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=30000)
X_audit = char_audit.fit_transform(df['text_normalized'])
nn = NearestNeighbors(n_neighbors=2, metric='cosine', n_jobs=-1).fit(X_audit)
distances, indices = nn.kneighbors(X_audit)
similarities = 1 - distances[:,1]
near_rows = np.where(similarities >= 0.90)[0]
near_pairs = pd.DataFrame({
    'similarity': similarities[near_rows],
    'text_1': df.iloc[near_rows]['text'].values,
    'label_1': df.iloc[near_rows]['label_name'].values,
    'text_2': df.iloc[indices[near_rows,1]]['text'].values,
    'label_2': df.iloc[indices[near_rows,1]]['label_name'].values,
}).sort_values('similarity', ascending=False)
print('Potential near-duplicate rows:', len(near_pairs))
display(near_pairs.head(30))
near_pairs.to_csv(PROJECT_DIR/'results/near_duplicate_audit.csv', index=False)

