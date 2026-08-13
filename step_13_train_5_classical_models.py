"""## 13. Train 5 classical models — CPU

Families: probabilistic, linear, margin-based, tree and boosted ensemble.
"""

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib, time

word_tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98, sublinear_tf=True, max_features=40000)
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=40000)
features = FeatureUnion([('word',word_tfidf),('char',char_tfidf)])
Xtr = features.fit_transform(X_train_text)
Xva = features.transform(X_val_text)
Xte = features.transform(X_test_text)

classical_models = {
    'MultinomialNB': MultinomialNB(alpha=0.5),
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'LinearSVM': CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'), cv=3),
    'RandomForest': RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=SEED),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05, subsample=.8,
                             colsample_bytree=.8, objective='multi:softprob', eval_metric='mlogloss',
                             random_state=SEED, n_jobs=-1)
}

results = []
trained_models = {}
for name, model in classical_models.items():
    start = time.time(); model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)
    p,r,f,_ = precision_recall_fscore_support(y_test,pred,average='macro',zero_division=0)
    results.append({'model':name,'accuracy':accuracy_score(y_test,pred),'macro_precision':p,
                    'macro_recall':r,'macro_f1':f,'roc_auc_ovr':roc_auc_score(y_test,prob,multi_class='ovr'),
                    'seconds':time.time()-start})
    trained_models[name]=model
display(pd.DataFrame(results).sort_values('macro_f1',ascending=False))
joblib.dump(features, PROJECT_DIR/'models/tfidf_features.joblib')
joblib.dump(trained_models['LinearSVM'], PROJECT_DIR/'models/linear_svm.joblib')

