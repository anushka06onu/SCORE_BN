"""## 10. Word frequency, n-grams, word clouds and co-occurrence — CPU"""

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

tokens = [tok for text in df.text for tok in re.findall(r'[\u0980-\u09FFA-Za-z]+', text.lower()) if len(tok)>1]
display(pd.DataFrame(Counter(tokens).most_common(30), columns=['token','count']))

def top_ngrams(texts, n=2, top_k=20):
    vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(n,n), min_df=2)
    matrix = vec.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    names = np.asarray(vec.get_feature_names_out())
    idx = counts.argsort()[::-1][:top_k]
    return pd.DataFrame({'ngram':names[idx], 'count':counts[idx]})

for n in [1,2,3]:
    print(f'Top {n}-grams')
    display(top_ngrams(df.text, n))

# WordCloud may not shape Bangla perfectly, but it fulfills the visual EDA requirement.
for label_name in LABEL2ID:
    text_blob = ' '.join(df.loc[df.label_name==label_name, 'text'])
    wc = WordCloud(width=1000, height=500, background_color='white', collocations=False).generate(text_blob)
    plt.figure(figsize=(10,4)); plt.imshow(wc); plt.axis('off'); plt.title(label_name); plt.show()

# Co-occurrence heatmap for 20 frequent tokens.
vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=20, binary=True)
Xc = vec.fit_transform(df.text)
co = (Xc.T @ Xc).toarray(); np.fill_diagonal(co, 0)
plt.figure(figsize=(12,10))
sns.heatmap(co, xticklabels=vec.get_feature_names_out(), yticklabels=vec.get_feature_names_out(), cmap='Blues')
plt.title('Top-token co-occurrence'); plt.tight_layout(); plt.show()

