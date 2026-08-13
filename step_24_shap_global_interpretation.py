"""## 24. SHAP global interpretation — CPU

This explains a linear SVM/logistic-style TF-IDF model. Limit samples to keep Colab memory manageable.
"""

# ============================================================
# SECTION 24: SHAP GLOBAL EXPLAINABILITY
# ============================================================

!pip -q install shap

import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ------------------------------------------------------------
# 1. Output directory
# ------------------------------------------------------------

XAI_DIR = (
    BACKUP_DIR /
    'explainability'
)

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 2. Reload saved classical models
# ------------------------------------------------------------

CLASSICAL_MODELS_PATH = (
    BACKUP_DIR /
    'classical_models.joblib'
)

TRANSFORMED_DATA_PATH = (
    BACKUP_DIR /
    'transformed_data.joblib'
)

TFIDF_PATH = (
    BACKUP_DIR /
    'tfidf_features.joblib'
)

if not CLASSICAL_MODELS_PATH.exists():
    raise FileNotFoundError(
        f"Classical models not found: "
        f"{CLASSICAL_MODELS_PATH}"
    )

if not TRANSFORMED_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Transformed data not found: "
        f"{TRANSFORMED_DATA_PATH}"
    )

if not TFIDF_PATH.exists():
    raise FileNotFoundError(
        f"TF-IDF features not found: "
        f"{TFIDF_PATH}"
    )

trained_models = joblib.load(
    CLASSICAL_MODELS_PATH
)

transformed_data = joblib.load(
    TRANSFORMED_DATA_PATH
)

features = joblib.load(
    TFIDF_PATH
)

print(
    "Available classical models:",
    list(trained_models.keys())
)

# ------------------------------------------------------------
# 3. Retrieve Logistic Regression and TF-IDF matrices
# ------------------------------------------------------------

if (
    'LogisticRegression'
    not in trained_models
):
    raise KeyError(
        "LogisticRegression is missing from "
        "classical_models.joblib."
    )

lr_model = trained_models[
    'LogisticRegression'
]

Xtr = transformed_data[
    'Xtr'
]

Xte = transformed_data[
    'Xte'
]

y_test_shap = np.asarray(
    transformed_data[
        'y_test'
    ]
)

feature_names = (
    features.get_feature_names_out()
)

print("Training matrix:", Xtr.shape)
print("Test matrix:", Xte.shape)
print(
    "Number of TF-IDF features:",
    len(feature_names)
)

# ------------------------------------------------------------
# 4. Select memory-safe SHAP samples
# ------------------------------------------------------------

# Use a small background sample to keep memory usage low.
rng = np.random.default_rng(
    SEED
)

background_size = min(
    200,
    Xtr.shape[0]
)

explanation_size = min(
    100,
    Xte.shape[0]
)

background_indices = rng.choice(
    Xtr.shape[0],
    size=background_size,
    replace=False
)

explanation_indices = np.arange(
    explanation_size
)

background = Xtr[
    background_indices
]

explain_X = Xte[
    explanation_indices
]

print(
    "Background samples:",
    background.shape[0]
)

print(
    "Explained test samples:",
    explain_X.shape[0]
)

# ------------------------------------------------------------
# 5. Calculate SHAP values
# ------------------------------------------------------------

print("\nCalculating SHAP values...")

shap_explainer = shap.LinearExplainer(
    lr_model,
    background
)

shap_explanation = shap_explainer(
    explain_X
)

print(
    "SHAP value shape:",
    shap_explanation.values.shape
)

# ------------------------------------------------------------
# 6. Prepare multiclass SHAP values
# ------------------------------------------------------------

shap_array = (
    shap_explanation.values
)

class_names = [
    ID2LABEL[index]
    for index in range(4)
]

# Different SHAP versions can return multiclass values
# in different formats.
if shap_array.ndim == 3:

    # Expected shape:
    # samples × features × classes
    shap_values_for_plot = [
        shap_array[
            :,
            :,
            class_index
        ]
        for class_index in range(
            shap_array.shape[2]
        )
    ]

elif shap_array.ndim == 2:

    shap_values_for_plot = (
        shap_array
    )

else:

    raise ValueError(
        f"Unexpected SHAP shape: "
        f"{shap_array.shape}"
    )

# ------------------------------------------------------------
# 7. Create global SHAP summary plot
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 8)
)

shap.summary_plot(
    shap_values_for_plot,
    explain_X,
    feature_names=feature_names,
    class_names=class_names,
    max_display=20,
    plot_type='bar',
    show=False
)

plt.title(
    'Global SHAP Feature Importance — '
    'Logistic Regression'
)

plt.tight_layout()

SHAP_SUMMARY_PATH = (
    XAI_DIR /
    'shap_global_summary.png'
)

plt.savefig(
    SHAP_SUMMARY_PATH,
    dpi=200,
    bbox_inches='tight'
)

plt.show()

# ------------------------------------------------------------
# 8. Save coefficient-based feature table
# ------------------------------------------------------------

# Logistic-regression coefficients provide a useful
# tabular companion to the SHAP plot.

coefficient_rows = []

for class_index in range(
    lr_model.coef_.shape[0]
):

    coefficients = (
        lr_model.coef_[
            class_index
        ]
    )

    top_positive_indices = np.argsort(
        coefficients
    )[-20:][::-1]

    for rank, feature_index in enumerate(
        top_positive_indices,
        start=1
    ):

        coefficient_rows.append({
            'class_id': class_index,
            'class_name': (
                ID2LABEL[class_index]
            ),
            'rank': rank,
            'feature': (
                feature_names[
                    feature_index
                ]
            ),
            'coefficient': float(
                coefficients[
                    feature_index
                ]
            )
        })

coefficient_table = pd.DataFrame(
    coefficient_rows
)

coefficient_table.to_csv(
    XAI_DIR /
    'logistic_regression_top_features.csv',
    index=False,
    encoding='utf-8-sig'
)

display(
    coefficient_table.groupby(
        'class_name'
    ).head(10)
)

print("\nSHAP analysis completed.")

print(
    "SHAP figure saved to:",
    SHAP_SUMMARY_PATH
)

print(
    "Feature table saved to:",
    XAI_DIR /
    'logistic_regression_top_features.csv'
)

