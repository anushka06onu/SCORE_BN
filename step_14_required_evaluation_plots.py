"""## 14. Required evaluation plots — CPU"""

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize

best_classical_name = max(trained_models, key=lambda n: precision_recall_fscore_support(
    y_test, trained_models[n].predict(Xte), average='macro', zero_division=0)[2])
best_classical = trained_models[best_classical_name]
pred = best_classical.predict(Xte); prob = best_classical.predict_proba(Xte)
print('Best classical:', best_classical_name)
print(classification_report(y_test,pred,target_names=[ID2LABEL[i] for i in range(4)],zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=[ID2LABEL[i] for i in range(4)],cmap='Blues')
plt.xticks(rotation=25); plt.tight_layout(); plt.show()

y_bin = label_binarize(y_test, classes=[0,1,2,3])
for i in range(4):
    RocCurveDisplay.from_predictions(y_bin[:,i], prob[:,i], name=ID2LABEL[i])
plt.plot([0,1],[0,1],'--',color='gray'); plt.title('One-vs-rest ROC curves'); plt.show()

import joblib
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

BACKUP_DIR = Path('/content/drive/MyDrive/SCORE_BN_Checkpoints')
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Save TF-IDF feature extractor
joblib.dump(
    features,
    BACKUP_DIR / 'tfidf_features.joblib'
)

# Save transformed datasets
joblib.dump(
    {
        'Xtr': Xtr,
        'Xva': Xva,
        'Xte': Xte,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    },
    BACKUP_DIR / 'transformed_data.joblib'
)

# Save already trained classical models
joblib.dump(
    trained_models,
    BACKUP_DIR / 'classical_models.joblib'
)

# Save dataset splits
train_df.to_csv(BACKUP_DIR / 'train.csv', index=False)
val_df.to_csv(BACKUP_DIR / 'validation.csv', index=False)
test_df.to_csv(BACKUP_DIR / 'test.csv', index=False)

print("Checkpoint saved successfully to Google Drive.")

