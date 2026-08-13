"""## 25. Combined comparison table and exports — CPU"""

# ============================================================
# SECTION 25: COMBINE AND SAVE ALL MODEL RESULTS
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Output directory
# ------------------------------------------------------------

FINAL_RESULTS_DIR = (
    BACKUP_DIR /
    'final_results'
)

FINAL_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 2. Reload transformed test data
# ------------------------------------------------------------

transformed_data = joblib.load(
    BACKUP_DIR /
    'transformed_data.joblib'
)

Xte = transformed_data['Xte']
y_test = np.asarray(
    transformed_data['y_test']
)

# ------------------------------------------------------------
# 3. Reload and evaluate original classical models
# ------------------------------------------------------------

trained_models = joblib.load(
    BACKUP_DIR /
    'classical_models.joblib'
)

classical_result_rows = []

for model_name, model in trained_models.items():

    print(
        "Evaluating:",
        model_name
    )

    predictions = model.predict(
        Xte
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average='macro',
            zero_division=0
        )
    )

    result = {
        'model': model_name,
        'stage': 'Baseline',
        'accuracy': accuracy_score(
            y_test,
            predictions
        ),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'roc_auc_ovr': np.nan
    }

    # Probability-based ROC-AUC
    if hasattr(
        model,
        'predict_proba'
    ):

        try:

            probabilities = (
                model.predict_proba(
                    Xte
                )
            )

            result[
                'roc_auc_ovr'
            ] = roc_auc_score(
                y_test,
                probabilities,
                multi_class='ovr',
                average='macro'
            )

        except Exception as error:

            print(
                f"ROC-AUC unavailable for "
                f"{model_name}: {error}"
            )

    classical_result_rows.append(
        result
    )

classical_results_df = pd.DataFrame(
    classical_result_rows
)

# ------------------------------------------------------------
# 4. Reload tuned classical results
# ------------------------------------------------------------

TUNED_RESULTS_PATH = (
    BACKUP_DIR /
    'tuned_model_results.csv'
)

if TUNED_RESULTS_PATH.exists():

    tuned_results_df = pd.read_csv(
        TUNED_RESULTS_PATH
    )

    tuned_results_df[
        'stage'
    ] = 'Tuned'

else:

    print(
        "Warning: tuned_model_results.csv "
        "was not found."
    )

    tuned_results_df = pd.DataFrame()

# ------------------------------------------------------------
# 5. Reload CNN, BiLSTM and BiGRU results
# ------------------------------------------------------------

DEEP_RESULTS_PATH = (
    BACKUP_DIR /
    'deep_model_results.csv'
)

if DEEP_RESULTS_PATH.exists():

    deep_results_df = pd.read_csv(
        DEEP_RESULTS_PATH
    )

    deep_results_df[
        'stage'
    ] = 'Deep Learning'

else:

    print(
        "Warning: deep_model_results.csv "
        "was not found."
    )

    deep_results_df = pd.DataFrame()

# ------------------------------------------------------------
# 6. Reload SCORE-BN test results
# ------------------------------------------------------------

SCORE_METRICS_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'test_metrics.json'
)

if SCORE_METRICS_PATH.exists():

    with open(
        SCORE_METRICS_PATH,
        'r',
        encoding='utf-8'
    ) as file:

        score_metrics = json.load(
            file
        )

    score_results_df = pd.DataFrame([
        {
            'model': 'SCORE-BN',
            'stage': 'Proposed Model',
            **score_metrics
        }
    ])

else:

    print(
        "Warning: SCORE-BN test metrics "
        "were not found."
    )

    score_results_df = pd.DataFrame()

# ------------------------------------------------------------
# 7. Try to reload BanglaBERT baseline metrics
# ------------------------------------------------------------

baseline_metric_files = list(
    BACKUP_DIR.rglob(
        'banglabert_baseline/test_metrics.json'
    )
)

banglabert_results_df = pd.DataFrame()

if baseline_metric_files:

    with open(
        baseline_metric_files[0],
        'r',
        encoding='utf-8'
    ) as file:

        raw_bert_metrics = json.load(
            file
        )

    banglabert_row = {
        'model': 'BanglaBERT',
        'stage': 'Transformer Baseline',
        'accuracy': raw_bert_metrics.get(
            'eval_accuracy',
            raw_bert_metrics.get(
                'accuracy',
                np.nan
            )
        ),
        'macro_precision': (
            raw_bert_metrics.get(
                'eval_macro_precision',
                raw_bert_metrics.get(
                    'macro_precision',
                    np.nan
                )
            )
        ),
        'macro_recall': (
            raw_bert_metrics.get(
                'eval_macro_recall',
                raw_bert_metrics.get(
                    'macro_recall',
                    np.nan
                )
            )
        ),
        'macro_f1': (
            raw_bert_metrics.get(
                'eval_macro_f1',
                raw_bert_metrics.get(
                    'macro_f1',
                    np.nan
                )
            )
        ),
        'roc_auc_ovr': (
            raw_bert_metrics.get(
                'eval_roc_auc_ovr',
                raw_bert_metrics.get(
                    'roc_auc_ovr',
                    np.nan
                )
            )
        )
    }

    banglabert_results_df = (
        pd.DataFrame(
            [banglabert_row]
        )
    )

else:

    print(
        "Note: BanglaBERT baseline metrics "
        "file was not found. Other results "
        "will still be combined."
    )

# ------------------------------------------------------------
# 8. Combine available result tables
# ------------------------------------------------------------

available_tables = [
    table
    for table in [
        classical_results_df,
        tuned_results_df,
        deep_results_df,
        banglabert_results_df,
        score_results_df
    ]
    if not table.empty
]

all_results = pd.concat(
    available_tables,
    ignore_index=True,
    sort=False
)

# Arrange the important columns first
important_columns = [
    'model',
    'stage',
    'accuracy',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'roc_auc_ovr',
    'ordinal_mae',
    'under_prioritisation_rate',
    'severe_error_rate',
    'cross_script_agreement',
    'romanized_macro_f1'
]

ordered_columns = [
    column
    for column in important_columns
    if column in all_results.columns
]

remaining_columns = [
    column
    for column in all_results.columns
    if column not in ordered_columns
]

all_results = all_results[
    ordered_columns
    +
    remaining_columns
]

all_results = all_results.sort_values(
    by='macro_f1',
    ascending=False,
    na_position='last'
).reset_index(drop=True)

print(
    "\nFinal model comparison:"
)

display(
    all_results.round(4)
)

# ------------------------------------------------------------
# 9. Save final model-comparison table
# ------------------------------------------------------------

MODEL_COMPARISON_PATH = (
    FINAL_RESULTS_DIR /
    'model_comparison.csv'
)

all_results.to_csv(
    MODEL_COMPARISON_PATH,
    index=False,
    encoding='utf-8-sig'
)

# ------------------------------------------------------------
# 10. Save label mapping
# ------------------------------------------------------------

with open(
    FINAL_RESULTS_DIR /
    'label_mapping.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        LABEL2ID,
        file,
        indent=2
    )

# ------------------------------------------------------------
# 11. Reconstruct and save the cleaned dataset
# ------------------------------------------------------------

clean_dataset = pd.concat(
    [
        train_df.assign(
            split='train'
        ),
        val_df.assign(
            split='validation'
        ),
        test_df.assign(
            split='test'
        )
    ],
    ignore_index=True
)

columns_to_save = [
    column
    for column in [
        'text',
        'label',
        'label_name',
        'split'
    ]
    if column in clean_dataset.columns
]

clean_dataset[
    columns_to_save
].to_csv(
    FINAL_RESULTS_DIR /
    'clean_dataset_with_splits.csv',
    index=False,
    encoding='utf-8-sig'
)

# ------------------------------------------------------------
# 12. Save final dataset audit
# ------------------------------------------------------------

dataset_audit = {
    'original_dataset_samples': 5263,
    'clean_samples_used': int(
        len(clean_dataset)
    ),
    'training_samples': int(
        len(train_df)
    ),
    'validation_samples': int(
        len(val_df)
    ),
    'test_samples': int(
        len(test_df)
    ),
    'input_column': 'Text',
    'target_column': 'Categories',
    'excluded_leakage_column': (
        'Action Needed'
    ),
    'number_of_classes': 4,
    'label_mapping': LABEL2ID,
    'split_ratio': '70/15/15',
    'duplicate_safe_split': True
}

with open(
    FINAL_RESULTS_DIR /
    'dataset_audit.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        dataset_audit,
        file,
        indent=2
    )

print(
    "\nSection 25 completed."
)

print(
    "Model comparison saved to:",
    MODEL_COMPARISON_PATH
)

print(
    "Clean dataset saved to:",
    FINAL_RESULTS_DIR /
    'clean_dataset_with_splits.csv'
)

print(
    "Dataset audit saved to:",
    FINAL_RESULTS_DIR /
    'dataset_audit.json'
)

