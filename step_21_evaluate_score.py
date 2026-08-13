"""## 21. Evaluate SCORE-BN and cross-script agreement — T4 GPU"""

# ============================================================
# SECTION 21: SCORE-BN TEST AND CROSS-SCRIPT EVALUATION
# ============================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    cohen_kappa_score
)

# ------------------------------------------------------------
# 1. Load the best SCORE-BN model
# ------------------------------------------------------------

BEST_MODEL_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'best_score_bn.pt'
)

if not BEST_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best model not found: {BEST_MODEL_PATH}"
    )

best_model_weights = torch.load(
    BEST_MODEL_PATH,
    map_location=device
)

score_model.load_state_dict(
    best_model_weights
)

score_model.to(device)
score_model.eval()

print("Best SCORE-BN model loaded:")
print(BEST_MODEL_PATH)

# ------------------------------------------------------------
# 2. Prepare original and Romanized test texts
# ------------------------------------------------------------

original_texts = (
    test_df['text']
    .fillna('')
    .astype(str)
    .tolist()
)

romanized_texts = [
    romanize_bangla(text)
    for text in original_texts
]

y_test = (
    test_df['label']
    .astype(int)
    .to_numpy()
)

print("Test samples:", len(original_texts))
print("Romanized samples:", len(romanized_texts))

# ------------------------------------------------------------
# 3. Memory-safe prediction function
# ------------------------------------------------------------

def score_predict(
    texts,
    batch_size=16
):

    score_model.eval()
    all_probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch_texts = texts[
                start:
                start + batch_size
            ]

            encoded_batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            encoded_batch = {
                key: value.to(device)
                for key, value
                in encoded_batch.items()
            }

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                logits, _ = score_model(
                    encoded_batch
                )

            probabilities = F.softmax(
                logits.float(),
                dim=-1
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

    return np.vstack(
        all_probabilities
    )

# ------------------------------------------------------------
# 4. Generate predictions
# ------------------------------------------------------------

print("\nPredicting original test queries...")

score_probabilities = score_predict(
    original_texts
)

score_predictions = (
    score_probabilities.argmax(
        axis=1
    )
)

print("Predicting Romanized test queries...")

romanized_probabilities = score_predict(
    romanized_texts
)

romanized_predictions = (
    romanized_probabilities.argmax(
        axis=1
    )
)

print("Predictions completed.")

# Keep notebook-compatible variable names
score_prob = score_probabilities
score_pred = score_predictions
roman_prob = romanized_probabilities
roman_pred = romanized_predictions

# ------------------------------------------------------------
# 5. Calculate evaluation metrics
# ------------------------------------------------------------

precision, recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        score_predictions,
        average='macro',
        zero_division=0
    )
)

ordinal_mae = np.abs(
    y_test -
    score_predictions
).mean()

under_prioritisation_rate = (
    score_predictions <
    y_test
).mean()

over_prioritisation_rate = (
    score_predictions >
    y_test
).mean()

severe_error_rate = (
    np.abs(
        y_test -
        score_predictions
    )
    >= 2
).mean()

cross_script_agreement = (
    score_predictions
    ==
    romanized_predictions
).mean()

romanized_accuracy = accuracy_score(
    y_test,
    romanized_predictions
)

romanized_precision, romanized_recall, romanized_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        romanized_predictions,
        average='macro',
        zero_division=0
    )
)

try:

    original_roc_auc = roc_auc_score(
        y_test,
        score_probabilities,
        multi_class='ovr',
        average='macro'
    )

except ValueError:

    original_roc_auc = float('nan')

try:

    romanized_roc_auc = roc_auc_score(
        y_test,
        romanized_probabilities,
        multi_class='ovr',
        average='macro'
    )

except ValueError:

    romanized_roc_auc = float('nan')

quadratic_weighted_kappa = (
    cohen_kappa_score(
        y_test,
        score_predictions,
        weights='quadratic'
    )
)

score_metrics = {
    'accuracy': accuracy_score(
        y_test,
        score_predictions
    ),
    'macro_precision': precision,
    'macro_recall': recall,
    'macro_f1': macro_f1,
    'roc_auc_ovr': original_roc_auc,
    'ordinal_mae': ordinal_mae,
    'under_prioritisation_rate': (
        under_prioritisation_rate
    ),
    'over_prioritisation_rate': (
        over_prioritisation_rate
    ),
    'severe_error_rate': (
        severe_error_rate
    ),
    'quadratic_weighted_kappa': (
        quadratic_weighted_kappa
    ),
    'cross_script_agreement': (
        cross_script_agreement
    ),
    'romanized_accuracy': (
        romanized_accuracy
    ),
    'romanized_macro_precision': (
        romanized_precision
    ),
    'romanized_macro_recall': (
        romanized_recall
    ),
    'romanized_macro_f1': (
        romanized_f1
    ),
    'romanized_roc_auc_ovr': (
        romanized_roc_auc
    )
}

# ------------------------------------------------------------
# 6. Display results
# ------------------------------------------------------------

print("\nSCORE-BN evaluation results:")

score_metrics_df = (
    pd.Series(
        score_metrics,
        name='SCORE-BN'
    )
    .to_frame(
        name='value'
    )
)

display(score_metrics_df)

class_names = [
    ID2LABEL[index]
    for index in range(4)
]

print(
    "\nClassification report "
    "for original test queries:\n"
)

print(
    classification_report(
        y_test,
        score_predictions,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0,
        digits=4
    )
)

print(
    "\nClassification report "
    "for Romanized test queries:\n"
)

print(
    classification_report(
        y_test,
        romanized_predictions,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0,
        digits=4
    )
)

# ------------------------------------------------------------
# 7. Confusion matrices
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    score_predictions,
    labels=[0, 1, 2, 3],
    display_labels=class_names,
    cmap='Purples',
    ax=axes[0],
    colorbar=False
)

axes[0].set_title(
    'SCORE-BN: Original Bangla'
)

axes[0].tick_params(
    axis='x',
    rotation=25
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    romanized_predictions,
    labels=[0, 1, 2, 3],
    display_labels=class_names,
    cmap='Oranges',
    ax=axes[1],
    colorbar=False
)

axes[1].set_title(
    'SCORE-BN: Romanized Bangla'
)

axes[1].tick_params(
    axis='x',
    rotation=25
)

plt.tight_layout()

CONFUSION_FIGURE_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'score_bn_confusion_matrices.png'
)

plt.savefig(
    CONFUSION_FIGURE_PATH,
    dpi=200,
    bbox_inches='tight'
)

plt.show()

# ------------------------------------------------------------
# 8. Save predictions for error analysis
# ------------------------------------------------------------

prediction_results = pd.DataFrame({
    'text': original_texts,
    'romanized_text': romanized_texts,
    'true_label_id': y_test,
    'true_label': [
        ID2LABEL[int(label)]
        for label in y_test
    ],
    'original_prediction_id': (
        score_predictions
    ),
    'original_prediction': [
        ID2LABEL[int(label)]
        for label in score_predictions
    ],
    'romanized_prediction_id': (
        romanized_predictions
    ),
    'romanized_prediction': [
        ID2LABEL[int(label)]
        for label
        in romanized_predictions
    ],
    'cross_script_agreement': (
        score_predictions
        ==
        romanized_predictions
    ),
    'ordinal_error': (
        np.abs(
            y_test -
            score_predictions
        )
    ),
    'under_prioritised': (
        score_predictions <
        y_test
    )
})

for class_index in range(4):

    prediction_results[
        f'probability_{ID2LABEL[class_index]}'
    ] = score_probabilities[
        :,
        class_index
    ]

prediction_results.to_csv(
    BACKUP_DIR /
    'score_bn' /
    'test_predictions_and_errors.csv',
    index=False,
    encoding='utf-8-sig'
)

np.save(
    BACKUP_DIR /
    'score_bn' /
    'original_test_probabilities.npy',
    score_probabilities
)

np.save(
    BACKUP_DIR /
    'score_bn' /
    'romanized_test_probabilities.npy',
    romanized_probabilities
)

# ------------------------------------------------------------
# 9. Save metrics
# ------------------------------------------------------------

serializable_metrics = {
    key: float(value)
    for key, value
    in score_metrics.items()
}

with open(
    BACKUP_DIR /
    'score_bn' /
    'test_metrics.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        serializable_metrics,
        file,
        indent=2
    )

print("\nSection 21 completed.")

print(
    "Metrics saved to:",
    BACKUP_DIR /
    'score_bn' /
    'test_metrics.json'
)

print(
    "Error-analysis file saved to:",
    BACKUP_DIR /
    'score_bn' /
    'test_predictions_and_errors.csv'
)

