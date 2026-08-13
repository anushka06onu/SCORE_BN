## 23. LIME explainability — CPU or T4
"""

# ============================================================
# SECTION 23: LIME EXPLAINABILITY
# ============================================================

!pip -q install lime

import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer

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
# 2. Reload TF-IDF features and tuned XGBoost
# ------------------------------------------------------------

TFIDF_PATH = (
    BACKUP_DIR /
    'tfidf_features.joblib'
)

XGBOOST_PATH = (
    BACKUP_DIR /
    'tuned_xgboost.joblib'
)

if not TFIDF_PATH.exists():
    raise FileNotFoundError(
        f"TF-IDF file not found: {TFIDF_PATH}"
    )

if not XGBOOST_PATH.exists():
    raise FileNotFoundError(
        f"XGBoost model not found: {XGBOOST_PATH}"
    )

features = joblib.load(
    TFIDF_PATH
)

lime_model = joblib.load(
    XGBOOST_PATH
)

print(
    "TF-IDF extractor loaded successfully."
)

print(
    "Tuned XGBoost model loaded successfully."
)

# ------------------------------------------------------------
# 3. Define the same classical preprocessing
# ------------------------------------------------------------

def clean_classical(text):

    text = unicodedata.normalize(
        'NFKC',
        str(text)
    )

    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' URL ',
        text
    )

    text = re.sub(
        r'@[A-Za-z0-9_]+',
        ' USER ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip().lower()

# ------------------------------------------------------------
# 4. LIME prediction function
# ------------------------------------------------------------

def classical_predict_proba(
    text_list
):

    cleaned_texts = [
        clean_classical(text)
        for text in text_list
    ]

    transformed_texts = (
        features.transform(
            cleaned_texts
        )
    )

    probabilities = (
        lime_model.predict_proba(
            transformed_texts
        )
    )

    return probabilities

# ------------------------------------------------------------
# 5. Class names
# ------------------------------------------------------------

class_names = [
    ID2LABEL[index]
    for index in range(4)
]

explainer = LimeTextExplainer(
    class_names=class_names,
    split_expression=r'\s+',
    random_state=SEED
)

# ------------------------------------------------------------
# 6. Select one correctly classified example
# ------------------------------------------------------------

all_classical_probabilities = (
    classical_predict_proba(
        test_df['text']
        .fillna('')
        .astype(str)
        .tolist()
    )
)

all_classical_predictions = (
    all_classical_probabilities.argmax(
        axis=1
    )
)

true_labels = (
    test_df['label']
    .astype(int)
    .to_numpy()
)

correct_indices = np.where(
    all_classical_predictions
    ==
    true_labels
)[0]

if len(correct_indices) > 0:
    example_index = int(
        correct_indices[0]
    )
else:
    example_index = 0

example_text = str(
    test_df.iloc[
        example_index
    ]['text']
)

true_label_id = int(
    test_df.iloc[
        example_index
    ]['label']
)

predicted_label_id = int(
    all_classical_predictions[
        example_index
    ]
)

print(
    "Example index:",
    example_index
)

print(
    "True class:",
    ID2LABEL[
        true_label_id
    ]
)

print(
    "Predicted class:",
    ID2LABEL[
        predicted_label_id
    ]
)

# ------------------------------------------------------------
# 7. Generate LIME explanation
# ------------------------------------------------------------

explanation = (
    explainer.explain_instance(
        example_text,
        classical_predict_proba,
        labels=[
            predicted_label_id
        ],
        num_features=12,
        num_samples=500
    )
)

print(
    "\nInfluential words and weights:"
)

for word, weight in explanation.as_list(
    label=predicted_label_id
):
    print(
        f"{word}: {weight:.4f}"
    )

# ------------------------------------------------------------
# 8. Plot the explanation
# ------------------------------------------------------------

figure = explanation.as_pyplot_figure(
    label=predicted_label_id
)

plt.title(
    "LIME Explanation: "
    f"{ID2LABEL[predicted_label_id]}"
)

plt.tight_layout()

LIME_FIGURE_PATH = (
    XAI_DIR /
    'lime_example.png'
)

plt.savefig(
    LIME_FIGURE_PATH,
    dpi=200,
    bbox_inches='tight'
)

plt.show()

# ------------------------------------------------------------
# 9. Save interactive HTML explanation
# ------------------------------------------------------------

LIME_HTML_PATH = (
    XAI_DIR /
    'lime_example.html'
)

explanation.save_to_file(
    str(LIME_HTML_PATH)
)

# ------------------------------------------------------------
# 10. Save explanation values as CSV
# ------------------------------------------------------------

lime_values = pd.DataFrame(
    explanation.as_list(
        label=predicted_label_id
    ),
    columns=[
        'word_or_phrase',
        'importance_weight'
    ]
)

lime_values[
    'true_class'
] = ID2LABEL[
    true_label_id
]

lime_values[
    'predicted_class'
] = ID2LABEL[
    predicted_label_id
]

lime_values.to_csv(
    XAI_DIR /
    'lime_feature_weights.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\nLIME explanation completed.")

print(
    "HTML explanation saved to:",
    LIME_HTML_PATH
)

print(
    "Figure saved to:",
    LIME_FIGURE_PATH
)

