"""## 15. Tune three models — CPU for SVM/XGBoost; T4 for transformer later"""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier
import joblib

# ============================================================
# 1. FAST LINEAR SVM TUNING
# ============================================================

print("Starting Linear SVM tuning...")
svm_start_time = time.time()

svm_search = RandomizedSearchCV(
    estimator=LinearSVC(
        class_weight='balanced',
        random_state=SEED
    ),
    param_distributions={
        'C': [
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.0,
            5.0
        ]
    },
    n_iter=5,
    scoring='f1_macro',
    cv=2,
    random_state=SEED,
    n_jobs=-1,
    verbose=2,
    refit=True
)

svm_search.fit(Xtr, y_train)

svm_minutes = (time.time() - svm_start_time) / 60

print("\nBest SVM parameters:")
print(svm_search.best_params_)

print("Best SVM cross-validation macro-F1:")
print(round(svm_search.best_score_, 4))

print("SVM tuning time:")
print(round(svm_minutes, 2), "minutes")


# ============================================================
# 2. CHECK XGBOOST VERSION AND GPU
# ============================================================

import xgboost as xgb
import torch

print("\nXGBoost version:", xgb.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. FAST GPU XGBOOST TUNING
# ============================================================

print("\nStarting XGBoost GPU tuning...")
xgb_start_time = time.time()

xgb_base = XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',

    # Use histogram-based training
    tree_method='hist',

    # Explicitly use the connected T4 GPU
    device='cuda',

    random_state=SEED,

    # Keep this at 1 because several simultaneous CV jobs
    # competing for one GPU can cause memory problems.
    n_jobs=1
)

xgb_parameter_space = {
    'n_estimators': [
        100,
        150,
        200
    ],
    'max_depth': [
        3,
        5,
        7
    ],
    'learning_rate': [
        0.03,
        0.05,
        0.1
    ],
    'subsample': [
        0.8,
        1.0
    ],
    'colsample_bytree': [
        0.8,
        1.0
    ],
    'min_child_weight': [
        1,
        3
    ]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_parameter_space,

    # Five candidates × two folds = ten fits
    n_iter=5,
    cv=2,

    scoring='f1_macro',
    random_state=SEED,
    n_jobs=1,
    verbose=2,
    refit=True,
    error_score='raise'
)

xgb_search.fit(Xtr, y_train)

xgb_minutes = (time.time() - xgb_start_time) / 60

print("\nBest XGBoost parameters:")
print(xgb_search.best_params_)

print("Best XGBoost cross-validation macro-F1:")
print(round(xgb_search.best_score_, 4))

print("XGBoost tuning time:")
print(round(xgb_minutes, 2), "minutes")


# ============================================================
# 4. EVALUATE TUNED MODELS ON UNTOUCHED TEST SET
# ============================================================

tuned_models = {
    'Tuned Linear SVM': svm_search.best_estimator_,
    'Tuned XGBoost': xgb_search.best_estimator_
}

tuned_results = []

for model_name, model in tuned_models.items():

    predictions = model.predict(Xte)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average='macro',
            zero_division=0
        )
    )

    tuned_results.append({
        'model': model_name,
        'accuracy': accuracy_score(
            y_test,
            predictions
        ),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1
    })

tuned_results_df = pd.DataFrame(tuned_results)

display(
    tuned_results_df.sort_values(
        by='macro_f1',
        ascending=False
    )
)


# ============================================================
# 5. SAVE TUNED MODELS DIRECTLY TO GOOGLE DRIVE
# ============================================================

joblib.dump(
    svm_search.best_estimator_,
    BACKUP_DIR / 'tuned_linear_svm.joblib'
)

joblib.dump(
    xgb_search.best_estimator_,
    BACKUP_DIR / 'tuned_xgboost.joblib'
)

tuned_results_df.to_csv(
    BACKUP_DIR / 'tuned_model_results.csv',
    index=False
)

print("\nTuned models and results saved to:")
print(BACKUP_DIR)

