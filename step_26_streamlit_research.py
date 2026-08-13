"""## 26. Streamlit research-demo application — CPU after training"""

# ============================================================
# SECTION 26A: PREPARE AND TEST APPLICATION ASSETS
# ============================================================

import re
import shutil
import unicodedata
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# ------------------------------------------------------------
# 1. Define paths
# ------------------------------------------------------------

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

APP_DIR = (
    BACKUP_DIR /
    'streamlit_app'
)

APP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TFIDF_SOURCE = (
    BACKUP_DIR /
    'tfidf_features.joblib'
)

XGBOOST_SOURCE = (
    BACKUP_DIR /
    'tuned_xgboost.joblib'
)

TFIDF_APP_PATH = (
    APP_DIR /
    'tfidf_features.joblib'
)

XGBOOST_APP_PATH = (
    APP_DIR /
    'tuned_xgboost.joblib'
)

# ------------------------------------------------------------
# 2. Check source files
# ------------------------------------------------------------

print(
    "TF-IDF file exists:",
    TFIDF_SOURCE.exists()
)

print(
    "XGBoost file exists:",
    XGBOOST_SOURCE.exists()
)

if not TFIDF_SOURCE.exists():
    raise FileNotFoundError(
        f"Missing file: {TFIDF_SOURCE}"
    )

if not XGBOOST_SOURCE.exists():
    raise FileNotFoundError(
        f"Missing file: {XGBOOST_SOURCE}"
    )

# ------------------------------------------------------------
# 3. Load saved assets
# ------------------------------------------------------------

feature_extractor = joblib.load(
    TFIDF_SOURCE
)

application_model = joblib.load(
    XGBOOST_SOURCE
)

# Force XGBoost to use CPU in the application.
# This makes deployment easier.
try:
    application_model.set_params(
        device='cpu'
    )
except Exception as error:
    print(
        "CPU-setting notice:",
        error
    )

print(
    "Feature extractor loaded:",
    type(feature_extractor)
)

print(
    "Model loaded:",
    type(application_model)
)

# ------------------------------------------------------------
# 4. Define the exact preprocessing function
# ------------------------------------------------------------

def clean_application_text(text):

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
# 5. Test the model before creating Streamlit
# ------------------------------------------------------------

sample_query = str(
    test_df.iloc[0]['text']
)

cleaned_sample = clean_application_text(
    sample_query
)

sample_features = (
    feature_extractor.transform(
        [cleaned_sample]
    )
)

sample_probabilities = (
    application_model.predict_proba(
        sample_features
    )[0]
)

sample_prediction = int(
    sample_probabilities.argmax()
)

LABELS = {
    0: 'General Query',
    1: 'Routine',
    2: 'Urgent',
    3: 'Emergency'
}

print(
    "\nTest prediction:",
    LABELS[sample_prediction]
)

print(
    "Probabilities:",
    sample_probabilities
)

print(
    "Probability sum:",
    sample_probabilities.sum()
)

if len(sample_probabilities) != 4:
    raise ValueError(
        "Expected four class probabilities."
    )

if not np.isclose(
    sample_probabilities.sum(),
    1.0,
    atol=1e-4
):
    raise ValueError(
        "Probabilities do not sum to one."
    )

# ------------------------------------------------------------
# 6. Save CPU-compatible application assets
# ------------------------------------------------------------

joblib.dump(
    feature_extractor,
    TFIDF_APP_PATH
)

joblib.dump(
    application_model,
    XGBOOST_APP_PATH
)

print(
    "\nSECTION 26A COMPLETED"
)

print(
    "Application assets saved to:",
    APP_DIR
)

# ============================================================
# SECTION 26B: CREATE APP.PY
# ============================================================

from pathlib import Path

APP_DIR = (
    BACKUP_DIR /
    'streamlit_app'
)

app_code = r'''
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=(
        "Bangla Healthcare Query "
        "Severity Classifier"
    ),
    page_icon="📝",
    layout="centered"
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #5f6368;
        margin-bottom: 1.2rem;
    }

    .result-box {
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1px solid #d8dce3;
        background-color: #f8f9fb;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
    Bangla Healthcare Query Severity Classification
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    NLP research demonstration using real-world Bangla text
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "Research demonstration only. This application is not "
    "a medical diagnosis or clinical decision system. "
    "For actual health concerns, consult a qualified "
    "healthcare professional."
)

# ============================================================
# Paths and assets
# ============================================================

APP_DIRECTORY = Path(
    __file__
).resolve().parent

TFIDF_PATH = (
    APP_DIRECTORY /
    'tfidf_features.joblib'
)

MODEL_PATH = (
    APP_DIRECTORY /
    'tuned_xgboost.joblib'
)

@st.cache_resource
def load_assets():

    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            "tfidf_features.joblib is missing."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "tuned_xgboost.joblib is missing."
        )

    extractor = joblib.load(
        TFIDF_PATH
    )

    classifier = joblib.load(
        MODEL_PATH
    )

    try:
        classifier.set_params(
            device='cpu'
        )
    except Exception:
        pass

    return extractor, classifier

try:

    feature_extractor, model = (
        load_assets()
    )

except Exception as error:

    st.error(
        "The application could not load "
        "the saved model files."
    )

    st.exception(
        error
    )

    st.stop()

# ============================================================
# Preprocessing
# ============================================================

def clean_text(text):

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

# ============================================================
# Labels
# ============================================================

LABELS = {
    0: 'General Query',
    1: 'Routine',
    2: 'Urgent',
    3: 'Emergency'
}

CLASS_ORDER = [
    'General Query',
    'Routine',
    'Urgent',
    'Emergency'
]

# ============================================================
# Example inputs
# ============================================================

example_option = st.selectbox(
    "Optional example:",
    [
        "Write my own query",
        "General health-information example",
        "Routine consultation example"
    ]
)

example_texts = {
    "Write my own query": "",
    "General health-information example":
        "সুস্থ থাকার জন্য প্রতিদিন কত ঘণ্টা ঘুমানো উচিত?",
    "Routine consultation example":
        "কয়েক দিন ধরে হালকা মাথাব্যথা হচ্ছে।"
}

default_text = example_texts[
    example_option
]

# ============================================================
# Input interface
# ============================================================

query = st.text_area(
    "Enter a Bangla healthcare-related query:",
    value=default_text,
    height=160,
    placeholder=(
        "এখানে বাংলা স্বাস্থ্য-সম্পর্কিত "
        "প্রশ্ন লিখুন..."
    )
)

classify_button = st.button(
    "Classify Query",
    type="primary",
    use_container_width=True
)

# ============================================================
# Prediction
# ============================================================

if classify_button:

    if not query.strip():

        st.error(
            "Please enter a query before "
            "clicking Classify Query."
        )

    else:

        cleaned_query = clean_text(
            query
        )

        transformed_query = (
            feature_extractor.transform(
                [cleaned_query]
            )
        )

        probabilities = (
            model.predict_proba(
                transformed_query
            )[0]
        )

        if len(probabilities) != 4:

            st.error(
                "The model returned an unexpected "
                "number of classes."
            )

            st.stop()

        predicted_id = int(
            np.argmax(
                probabilities
            )
        )

        predicted_label = LABELS[
            predicted_id
        ]

        confidence = float(
            probabilities[
                predicted_id
            ]
        )

        st.markdown(
            f"""
            <div class="result-box">
            <b>Predicted research category:</b>
            {predicted_label}<br>
            <b>Model confidence:</b>
            {confidence:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

        probability_table = pd.DataFrame({
            'Severity category': CLASS_ORDER,
            'Probability': [
                float(
                    probabilities[index]
                )
                for index in range(4)
            ]
        })

        st.subheader(
            "Prediction probabilities"
        )

        st.bar_chart(
            probability_table.set_index(
                'Severity category'
            ),
            y='Probability'
        )

        display_table = (
            probability_table.copy()
        )

        display_table[
            'Probability'
        ] = display_table[
            'Probability'
        ].map(
            lambda value: f"{value:.2%}"
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

        if confidence < 0.60:

            st.info(
                "This prediction has relatively low "
                "confidence and should be referred "
                "for human review."
            )

        with st.expander(
            "View normalized model input"
        ):

            st.write(
                cleaned_query
            )

# ============================================================
# Project information
# ============================================================

st.divider()

with st.expander(
    "About the project"
):

    st.markdown(
        """
        The project investigates severity classification
        for real-world Bangla healthcare queries.

        The four ordered categories are:

        1. General Query
        2. Routine
        3. Urgent
        4. Emergency

        The complete experimental study compares traditional
        machine-learning models, CNN, BiLSTM, BiGRU,
        BanglaBERT, and the proposed SCORE-BN framework.
        """
    )

with st.expander(
    "Why does the application use XGBoost?"
):

    st.markdown(
        """
        The interactive application uses tuned XGBoost with
        word-level and character-level TF-IDF features because
        it is lightweight and suitable for deployment.

        SCORE-BN remains the proposed research model evaluated
        in the experimental part of the project.
        """
    )

with st.expander(
    "Limitations"
):

    st.markdown(
        """
        Predictions can be incorrect for incomplete queries,
        unusual spelling, Romanized Bangla, code-mixed input,
        or text outside the training distribution.

        This application must not be used as a substitute for
        professional healthcare assessment.
        """
    )
'''

APP_FILE = (
    APP_DIR /
    'app.py'
)

APP_FILE.write_text(
    app_code,
    encoding='utf-8'
)

print(
    "app.py created:"
)

print(APP_FILE)

# ============================================================
# SECTION 26C: CREATE DEPLOYMENT FILES
# ============================================================

import importlib.metadata

# Read installed versions to reduce model-compatibility errors.
streamlit_version = (
    importlib.metadata.version(
        'streamlit'
    )
    if importlib.util.find_spec(
        'streamlit'
    )
    else None
)

sklearn_version = (
    importlib.metadata.version(
        'scikit-learn'
    )
)

xgboost_version = (
    importlib.metadata.version(
        'xgboost'
    )
)

joblib_version = (
    importlib.metadata.version(
        'joblib'
    )
)

pandas_version = (
    importlib.metadata.version(
        'pandas'
    )
)

numpy_version = (
    importlib.metadata.version(
        'numpy'
    )
)

scipy_version = (
    importlib.metadata.version(
        'scipy'
    )
)

requirements_lines = [
    (
        f"streamlit=={streamlit_version}"
        if streamlit_version
        else "streamlit"
    ),
    f"scikit-learn=={sklearn_version}",
    f"xgboost=={xgboost_version}",
    f"joblib=={joblib_version}",
    f"pandas=={pandas_version}",
    f"numpy=={numpy_version}",
    f"scipy=={scipy_version}"
]

requirements_text = '\n'.join(
    requirements_lines
)

REQUIREMENTS_FILE = (
    APP_DIR /
    'requirements.txt'
)

REQUIREMENTS_FILE.write_text(
    requirements_text,
    encoding='utf-8'
)

README_FILE = (
    APP_DIR /
    'README.md'
)

# ============================================================
# INSTALL AND CONFIGURE BANGLA FONTS IN GOOGLE COLAB
# ============================================================

!apt-get update -qq
!apt-get install -y -qq fonts-noto-core fonts-noto-extra

import os
import glob
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import font_manager

# Clear Matplotlib's old font cache
cache_directory = matplotlib.get_cachedir()

for cache_file in glob.glob(
    os.path.join(
        cache_directory,
        'fontlist-*.json'
    )
):
    try:
        os.remove(cache_file)
    except OSError:
        pass

# Re-register all available system fonts
for font_path in font_manager.findSystemFonts(
    fontpaths=None,
    fontext='ttf'
):
    try:
        font_manager.fontManager.addfont(
            font_path
        )
    except Exception:
        pass

# Find available Noto Bengali fonts
bangla_font_paths = [
    path
    for path in font_manager.findSystemFonts(
        fontpaths=None,
        fontext='ttf'
    )
    if (
        'Bengali' in path
        or
        'Bangla' in path
    )
]

print(
    "Bangla fonts found:"
)

for path in bangla_font_paths[:20]:
    print(path)

if not bangla_font_paths:
    raise FileNotFoundError(
        "No Bangla font was found."
    )

BANGLA_FONT_PATH = (
    bangla_font_paths[0]
)

BANGLA_FONT_NAME = (
    font_manager.FontProperties(
        fname=BANGLA_FONT_PATH
    ).get_name()
)

BANGLA_FONT = (
    font_manager.FontProperties(
        fname=BANGLA_FONT_PATH
    )
)

# Configure Matplotlib globally
plt.rcParams[
    'font.family'
] = BANGLA_FONT_NAME

plt.rcParams[
    'font.sans-serif'
] = [
    BANGLA_FONT_NAME,
    'Noto Sans Bengali',
    'DejaVu Sans'
]

plt.rcParams[
    'axes.unicode_minus'
] = False

print(
    "\nSelected Bangla font:",
    BANGLA_FONT_NAME
)

print(
    "Font path:",
    BANGLA_FONT_PATH
)

# ============================================================
# TEST BANGLA FONT RENDERING
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    figsize=(11, 3)
)

ax.text(
    0.5,
    0.60,
    "বাংলা স্বাস্থ্যসেবা প্রশ্নের তীব্রতা শ্রেণিবিন্যাস",
    fontproperties=BANGLA_FONT,
    fontsize=22,
    horizontalalignment='center',
    verticalalignment='center'
)

ax.text(
    0.5,
    0.30,
    "সাধারণ প্রশ্ন • নিয়মিত • জরুরি • অত্যন্ত জরুরি",
    fontproperties=BANGLA_FONT,
    fontsize=17,
    horizontalalignment='center',
    verticalalignment='center'
)

ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# REGENERATE LIME FIGURE WITH BANGLA FONT
# ============================================================

import matplotlib.pyplot as plt

# If your variable is named exp instead of explanation
if (
    'explanation' not in globals()
    and
    'exp' in globals()
):
    explanation = exp

if 'explanation' not in globals():
    raise NameError(
        "The LIME explanation object is missing. "
        "Rerun the corrected LIME cell first."
    )

if 'predicted_label_id' not in globals():

    if 'all_classical_predictions' in globals():

        predicted_label_id = int(
            all_classical_predictions[
                example_index
            ]
        )

    else:

        raise NameError(
            "predicted_label_id is missing. "
            "Rerun the corrected LIME cell."
        )

lime_figure = (
    explanation.as_pyplot_figure(
        label=predicted_label_id
    )
)

lime_axis = (
    lime_figure.axes[0]
)

# Apply the Bangla font to all y-axis labels
for label in lime_axis.get_yticklabels():

    label.set_fontproperties(
        BANGLA_FONT
    )

    label.set_fontsize(
        12
    )

# Apply font to x-axis labels if necessary
for label in lime_axis.get_xticklabels():

    label.set_fontproperties(
        BANGLA_FONT
    )

lime_axis.set_title(
    "LIME Explanation: "
    f"{ID2LABEL[predicted_label_id]}",
    fontproperties=BANGLA_FONT,
    fontsize=16
)

lime_axis.set_xlabel(
    "Contribution to Prediction",
    fontproperties=BANGLA_FONT
)

plt.tight_layout()

LIME_FIXED_PATH = (
    BACKUP_DIR /
    'explainability' /
    'lime_example_bangla_fixed.png'
)

plt.savefig(
    LIME_FIXED_PATH,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

print(
    "Corrected LIME figure saved to:"
)

print(LIME_FIXED_PATH)

# ============================================================
# MANUAL LIME PLOT WITH BANGLA FONT
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

lime_items = explanation.as_list(
    label=predicted_label_id
)

lime_plot_df = pd.DataFrame(
    lime_items,
    columns=[
        'feature',
        'importance'
    ]
)

# Reverse order for a readable horizontal chart
lime_plot_df = (
    lime_plot_df
    .sort_values(
        'importance',
        ascending=True
    )
)

colors = [
    '#198754'
    if value > 0
    else '#DC3545'
    for value in lime_plot_df[
        'importance'
    ]
]

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.barh(
    lime_plot_df[
        'feature'
    ],
    lime_plot_df[
        'importance'
    ],
    color=colors
)

ax.axvline(
    0,
    color='black',
    linewidth=0.8
)

ax.set_title(
    "LIME Explanation: "
    f"{ID2LABEL[predicted_label_id]}",
    fontproperties=BANGLA_FONT,
    fontsize=17
)

ax.set_xlabel(
    "Contribution to Prediction",
    fontproperties=BANGLA_FONT,
    fontsize=12
)

for label in ax.get_yticklabels():

    label.set_fontproperties(
        BANGLA_FONT
    )

    label.set_fontsize(
        12
    )

for label in ax.get_xticklabels():

    label.set_fontproperties(
        BANGLA_FONT
    )

plt.tight_layout()

LIME_MANUAL_PATH = (
    BACKUP_DIR /
    'explainability' /
    'lime_explanation_final.png'
)

plt.savefig(
    LIME_MANUAL_PATH,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

print(
    "Final LIME figure saved to:"
)

print(LIME_MANUAL_PATH)

# ============================================================
# REGENERATE SHAP FIGURE WITH BANGLA FONT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import shap

# Remove FeatureUnion prefixes for cleaner display
clean_feature_names = np.array([
    str(feature)
    .replace(
        'word__',
        ''
    )
    .replace(
        'char__',
        ''
    )
    for feature in feature_names
])

plt.figure(
    figsize=(13, 9)
)

shap.summary_plot(
    shap_values_for_plot,
    explain_X,
    feature_names=clean_feature_names,
    class_names=class_names,
    max_display=20,
    plot_type='bar',
    show=False
)

current_figure = plt.gcf()

# Apply the font to every axis and every label
for axis in current_figure.axes:

    for label in axis.get_yticklabels():

        label.set_fontproperties(
            BANGLA_FONT
        )

        label.set_fontsize(
            11
        )

    for label in axis.get_xticklabels():

        label.set_fontproperties(
            BANGLA_FONT
        )

    axis.title.set_fontproperties(
        BANGLA_FONT
    )

    axis.xaxis.label.set_fontproperties(
        BANGLA_FONT
    )

    axis.yaxis.label.set_fontproperties(
        BANGLA_FONT
    )

plt.suptitle(
    "Global SHAP Feature Importance — Logistic Regression",
    fontproperties=BANGLA_FONT,
    fontsize=18,
    y=1.02
)

plt.tight_layout()

SHAP_FIXED_PATH = (
    BACKUP_DIR /
    'explainability' /
    'shap_global_summary_bangla_fixed.png'
)

plt.savefig(
    SHAP_FIXED_PATH,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

print(
    "Corrected SHAP figure saved to:"
)

print(SHAP_FIXED_PATH)

for axis in plt.gcf().axes:

    for label in axis.get_xticklabels():
        label.set_fontproperties(
            BANGLA_FONT
        )

    for label in axis.get_yticklabels():
        label.set_fontproperties(
            BANGLA_FONT
        )

plt.savefig(
    "figure_name.png",
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

# ============================================================
# FINAL PROJECT BACKUP BEFORE CLOSING GOOGLE COLAB
# ============================================================

import os
import json
import shutil
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Confirm Google Drive location
# ------------------------------------------------------------

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

if not BACKUP_DIR.exists():
    raise FileNotFoundError(
        f"Google Drive checkpoint folder was not found:\n"
        f"{BACKUP_DIR}"
    )

FINAL_EXPORT_DIR = (
    BACKUP_DIR /
    'final_export'
)

FINAL_FIGURE_DIR = (
    FINAL_EXPORT_DIR /
    'figures'
)

FINAL_TABLE_DIR = (
    FINAL_EXPORT_DIR /
    'tables'
)

FINAL_ARRAY_DIR = (
    FINAL_EXPORT_DIR /
    'arrays'
)

FINAL_REPORT_DIR = (
    FINAL_EXPORT_DIR /
    'reports'
)

FINAL_METADATA_DIR = (
    FINAL_EXPORT_DIR /
    'metadata'
)

for directory in [
    FINAL_EXPORT_DIR,
    FINAL_FIGURE_DIR,
    FINAL_TABLE_DIR,
    FINAL_ARRAY_DIR,
    FINAL_REPORT_DIR,
    FINAL_METADATA_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

print("Main checkpoint folder:")
print(BACKUP_DIR)

print("\nFinal export folder:")
print(FINAL_EXPORT_DIR)

# ------------------------------------------------------------
# 2. Save every currently open Matplotlib figure
# ------------------------------------------------------------

open_figure_numbers = plt.get_fignums()

print(
    "\nOpen Matplotlib figures:",
    len(open_figure_numbers)
)

for figure_number in open_figure_numbers:

    figure = plt.figure(
        figure_number
    )

    figure_path = (
        FINAL_FIGURE_DIR /
        f'open_figure_{figure_number}.png'
    )

    figure.savefig(
        figure_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    print(
        "Saved open figure:",
        figure_path.name
    )

# ------------------------------------------------------------
# 3. Copy figures/reports from temporary Colab storage
# ------------------------------------------------------------

allowed_extensions = {
    '.png',
    '.jpg',
    '.jpeg',
    '.svg',
    '.pdf',
    '.html'
}

temporary_search_roots = [
    Path('/content/score_bn'),
    Path('/content/results'),
    Path('/content/figures')
]

copied_temporary_files = []

for search_root in temporary_search_roots:

    if not search_root.exists():
        continue

    for source_file in search_root.rglob('*'):

        if (
            source_file.is_file()
            and
            source_file.suffix.lower()
            in allowed_extensions
        ):

            relative_name = (
                str(
                    source_file.relative_to(
                        search_root
                    )
                )
                .replace('/', '__')
            )

            destination_file = (
                FINAL_FIGURE_DIR /
                (
                    search_root.name
                    +
                    '__'
                    +
                    relative_name
                )
            )

            shutil.copy2(
                source_file,
                destination_file
            )

            copied_temporary_files.append(
                str(destination_file)
            )

print(
    "\nTemporary figures/reports copied:",
    len(copied_temporary_files)
)

# ------------------------------------------------------------
# 4. Save important in-memory DataFrames
# ------------------------------------------------------------

dataframe_variables = [
    'df',
    'train_df',
    'val_df',
    'test_df',
    'paired_train',
    'paired_validation',
    'paired_test',
    'near_pairs',
    'classical_results_df',
    'tuned_results_df',
    'deep_results_df',
    'banglabert_results_df',
    'score_results_df',
    'tuning_results_df',
    'all_results',
    'score_metrics_df',
    'prediction_results',
    'lime_values',
    'coefficient_table',
    'training_history'
]

saved_dataframes = []
missing_dataframes = []

for variable_name in dataframe_variables:

    if variable_name not in globals():

        missing_dataframes.append(
            variable_name
        )

        continue

    value = globals()[
        variable_name
    ]

    try:

        if isinstance(
            value,
            pd.DataFrame
        ):

            output_path = (
                FINAL_TABLE_DIR /
                f'{variable_name}.csv'
            )

            value.to_csv(
                output_path,
                index=False,
                encoding='utf-8-sig'
            )

            saved_dataframes.append(
                variable_name
            )

        elif isinstance(
            value,
            pd.Series
        ):

            output_path = (
                FINAL_TABLE_DIR /
                f'{variable_name}.csv'
            )

            value.to_csv(
                output_path,
                encoding='utf-8-sig'
            )

            saved_dataframes.append(
                variable_name
            )

        elif isinstance(
            value,
            list
        ):

            output_path = (
                FINAL_TABLE_DIR /
                f'{variable_name}.csv'
            )

            pd.DataFrame(
                value
            ).to_csv(
                output_path,
                index=False,
                encoding='utf-8-sig'
            )

            saved_dataframes.append(
                variable_name
            )

    except Exception as error:

        print(
            f"Could not save {variable_name}:",
            error
        )

print(
    "\nSaved table variables:",
    saved_dataframes
)

# Missing in-memory variables are not automatically a problem,
# because most results were already saved during training.
print(
    "Unavailable in-memory tables:",
    missing_dataframes
)

# ------------------------------------------------------------
# 5. Save important in-memory arrays
# ------------------------------------------------------------

array_variables = [
    'y_train',
    'y_val',
    'y_test',
    'score_probabilities',
    'score_predictions',
    'romanized_probabilities',
    'romanized_predictions',
    'score_prob',
    'score_pred',
    'roman_prob',
    'roman_pred',
    'test_probabilities',
    'test_predictions'
]

saved_arrays = []

for variable_name in array_variables:

    if variable_name not in globals():
        continue

    try:

        value = np.asarray(
            globals()[
                variable_name
            ]
        )

        output_path = (
            FINAL_ARRAY_DIR /
            f'{variable_name}.npy'
        )

        np.save(
            output_path,
            value
        )

        saved_arrays.append(
            variable_name
        )

    except Exception as error:

        print(
            f"Could not save {variable_name}:",
            error
        )

print(
    "\nSaved array variables:",
    saved_arrays
)

# ------------------------------------------------------------
# 6. Save important dictionaries/metadata
# ------------------------------------------------------------

metadata_to_save = {}

possible_metadata = [
    'LABEL2ID',
    'ID2LABEL',
    'audit',
    'dataset_audit',
    'score_metrics',
    'configuration',
    'best_hyperparameter_summary',
    'best_configuration'
]

for variable_name in possible_metadata:

    if variable_name in globals():

        value = globals()[
            variable_name
        ]

        if isinstance(
            value,
            dict
        ):

            metadata_to_save[
                variable_name
            ] = value

def make_json_serializable(value):

    if isinstance(
        value,
        dict
    ):

        return {
            str(key): make_json_serializable(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            make_json_serializable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.floating
        )
    ):

        return value.item()

    if isinstance(
        value,
        np.ndarray
    ):

        return value.tolist()

    if isinstance(
        value,
        Path
    ):

        return str(value)

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ) or value is None:

        return value

    return str(value)

metadata_output_path = (
    FINAL_METADATA_DIR /
    'in_memory_metadata.json'
)

with open(
    metadata_output_path,
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        make_json_serializable(
            metadata_to_save
        ),
        file,
        ensure_ascii=False,
        indent=2
    )

print(
    "\nMetadata saved to:",
    metadata_output_path
)

# ------------------------------------------------------------
# 7. Save a final copy of important existing result files
# ------------------------------------------------------------

important_existing_patterns = [
    'deep_model_results.csv',
    'tuned_model_results.csv',
    'all_trial_results.csv',
    'best_hyperparameters.json',
    'test_metrics.json',
    'training_history.csv',
    'configuration.json',
    'test_predictions_and_errors.csv',
    'model_comparison.csv',
    'dataset_audit.json',
    'label_mapping.json',
    'lime_example.html',
    'lime_feature_weights.csv',
    'logistic_regression_top_features.csv'
]

copied_existing_results = []

for filename_pattern in important_existing_patterns:

    matching_files = list(
        BACKUP_DIR.rglob(
            filename_pattern
        )
    )

    for source_file in matching_files:

        if not source_file.is_file():
            continue

        relative_name = (
            str(
                source_file.relative_to(
                    BACKUP_DIR
                )
            )
            .replace('/', '__')
        )

        destination_file = (
            FINAL_REPORT_DIR /
            relative_name
        )

        # Do not copy a file onto itself.
        if (
            source_file.resolve()
            !=
            destination_file.resolve()
        ):

            shutil.copy2(
                source_file,
                destination_file
            )

        copied_existing_results.append(
            str(source_file)
        )

print(
    "\nImportant existing result files copied:",
    len(copied_existing_results)
)

# ------------------------------------------------------------
# 8. Verify essential files
# ------------------------------------------------------------

essential_files = {
    'Train split':
        BACKUP_DIR / 'train.csv',

    'Validation split':
        BACKUP_DIR / 'validation.csv',

    'Test split':
        BACKUP_DIR / 'test.csv',

    'TF-IDF extractor':
        BACKUP_DIR / 'tfidf_features.joblib',

    'Classical models':
        BACKUP_DIR / 'classical_models.joblib',

    'Tuned SVM':
        BACKUP_DIR / 'tuned_linear_svm.joblib',

    'Tuned XGBoost':
        BACKUP_DIR / 'tuned_xgboost.joblib',

    'CNN model':
        BACKUP_DIR / 'CNN_final.keras',

    'BiLSTM model':
        BACKUP_DIR / 'BiLSTM_final.keras',

    'BiGRU model':
        BACKUP_DIR / 'BiGRU_final.keras',

    'BanglaBERT final model':
        (
            BACKUP_DIR /
            'banglabert_baseline' /
            'final_model' /
            'model.safetensors'
        ),

    'Transformer tuning result':
        (
            BACKUP_DIR /
            'transformer_tuning' /
            'best_hyperparameters.json'
        ),

    'Best tuned transformer':
        (
            BACKUP_DIR /
            'transformer_tuning' /
            'best_tuned_model' /
            'model.safetensors'
        ),

    'SCORE-BN model':
        (
            BACKUP_DIR /
            'score_bn' /
            'best_score_bn.pt'
        ),

    'SCORE-BN metrics':
        (
            BACKUP_DIR /
            'score_bn' /
            'test_metrics.json'
        ),

    'SCORE-BN error analysis':
        (
            BACKUP_DIR /
            'score_bn' /
            'test_predictions_and_errors.csv'
        ),

    'Final model comparison':
        (
            BACKUP_DIR /
            'final_results' /
            'model_comparison.csv'
        ),

    'LIME figure':
        (
            BACKUP_DIR /
            'explainability' /
            'lime_explanation_final.png'
        ),

    'SHAP figure':
        (
            BACKUP_DIR /
            'explainability' /
            'shap_global_summary_bangla_fixed.png'
        )
}

verification_rows = []

for description, file_path in essential_files.items():

    exists = file_path.exists()

    size_bytes = (
        file_path.stat().st_size
        if exists
        else 0
    )

    verification_rows.append({
        'item': description,
        'path': str(file_path),
        'exists': exists,
        'size_bytes': size_bytes
    })

verification_df = pd.DataFrame(
    verification_rows
)

display(
    verification_df
)

verification_df.to_csv(
    FINAL_METADATA_DIR /
    'essential_file_verification.csv',
    index=False,
    encoding='utf-8-sig'
)

missing_essential_files = (
    verification_df.loc[
        ~verification_df[
            'exists'
        ],
        'item'
    ]
    .tolist()
)

print(
    "\nMissing essential items:",
    missing_essential_files
)

# ------------------------------------------------------------
# 9. Create complete file inventory
# ------------------------------------------------------------

inventory_rows = []

for file_path in BACKUP_DIR.rglob('*'):

    if not file_path.is_file():
        continue

    # The ZIP is stored elsewhere, so it cannot include itself.
    relative_path = (
        file_path.relative_to(
            BACKUP_DIR
        )
    )

    inventory_rows.append({
        'relative_path': str(
            relative_path
        ),
        'extension': (
            file_path.suffix.lower()
        ),
        'size_bytes': (
            file_path.stat().st_size
        ),
        'size_mb': round(
            file_path.stat().st_size /
            1024**2,
            3
        )
    })

inventory_df = (
    pd.DataFrame(
        inventory_rows
    )
    .sort_values(
        'relative_path'
    )
    .reset_index(drop=True)
)

inventory_path = (
    FINAL_METADATA_DIR /
    'complete_file_inventory.csv'
)

inventory_df.to_csv(
    inventory_path,
    index=False,
    encoding='utf-8-sig'
)

print(
    "\nTotal files in checkpoint folder:",
    len(inventory_df)
)

print(
    "Total saved size:",
    round(
        inventory_df[
            'size_bytes'
        ].sum() /
        1024**3,
        2
    ),
    "GB"
)

print(
    "Inventory saved to:",
    inventory_path
)

# ------------------------------------------------------------
# 10. Create a lightweight submission ZIP
# ------------------------------------------------------------

EXPORTS_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Exports'
)

EXPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LIGHT_PACKAGE_DIR = Path(
    '/content/SCORE_BN_Submission_Package'
)

LIGHT_PACKAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Include final results, figures, reports and Streamlit files.
light_sources = [
    (
        BACKUP_DIR /
        'final_results'
    ),
    (
        BACKUP_DIR /
        'explainability'
    ),
    (
        BACKUP_DIR /
        'score_bn'
    ),
    (
        BACKUP_DIR /
        'transformer_tuning'
    ),
    (
        BACKUP_DIR /
        'streamlit_app'
    ),
    FINAL_EXPORT_DIR
]

for source_directory in light_sources:

    if not source_directory.exists():
        continue

    destination_directory = (
        LIGHT_PACKAGE_DIR /
        source_directory.name
    )

    shutil.copytree(
        source_directory,
        destination_directory,
        dirs_exist_ok=True
    )

# Add dataset splits and result tables.
individual_files = [
    BACKUP_DIR / 'train.csv',
    BACKUP_DIR / 'validation.csv',
    BACKUP_DIR / 'test.csv',
    BACKUP_DIR / 'deep_model_results.csv',
    BACKUP_DIR / 'tuned_model_results.csv'
]

for source_file in individual_files:

    if source_file.exists():

        shutil.copy2(
            source_file,
            LIGHT_PACKAGE_DIR /
            source_file.name
        )

light_zip_base = (
    '/content/SCORE_BN_SUBMISSION_PACKAGE'
)

light_zip_path = shutil.make_archive(
    light_zip_base,
    'zip',
    root_dir=LIGHT_PACKAGE_DIR
)

light_drive_zip = (
    EXPORTS_DIR /
    'SCORE_BN_SUBMISSION_PACKAGE.zip'
)

shutil.copy2(
    light_zip_path,
    light_drive_zip
)

print(
    "\nLightweight submission ZIP created:"
)

print(light_drive_zip)

print(
    "ZIP size:",
    round(
        light_drive_zip.stat().st_size /
        1024**2,
        2
    ),
    "MB"
)

# ------------------------------------------------------------
# 11. Optional complete ZIP containing all saved models
# ------------------------------------------------------------

print(
    "\nCreating the complete model backup ZIP."
)

print(
    "This may take several minutes because it includes "
    "BanglaBERT checkpoints and all trained models."
)

complete_zip_base = (
    '/content/SCORE_BN_COMPLETE_BACKUP'
)

complete_zip_path = shutil.make_archive(
    complete_zip_base,
    'zip',
    root_dir=BACKUP_DIR
)

complete_drive_zip = (
    EXPORTS_DIR /
    'SCORE_BN_COMPLETE_BACKUP.zip'
)

shutil.copy2(
    complete_zip_path,
    complete_drive_zip
)

print(
    "\nComplete backup ZIP created:"
)

print(complete_drive_zip)

print(
    "Complete ZIP size:",
    round(
        complete_drive_zip.stat().st_size /
        1024**3,
        2
    ),
    "GB"
)

# ------------------------------------------------------------
# 12. Final status
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "FINAL GOOGLE DRIVE BACKUP COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nDownloadable files:"
)

print(
    "1.",
    light_drive_zip
)

print(
    "2.",
    complete_drive_zip
)

print(
    "\nEssential items missing:",
    missing_essential_files
)

print(
    "\nDo not close Colab until both ZIP files "
    "are visible in Google Drive."
)

# ============================================================
# REPORT ASSET SETUP AND BANGLA FONT
# ============================================================

!pip -q install wordcloud openpyxl
!apt-get update -qq
!apt-get install -y -qq fonts-noto-core fonts-noto-extra

import os
import re
import glob
import json
import shutil
import warnings
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    roc_curve
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

REPORT_ASSET_DIR = (
    BACKUP_DIR /
    'report_assets'
)

FIGURE_DIR = (
    REPORT_ASSET_DIR /
    'figures'
)

TABLE_DIR = (
    REPORT_ASSET_DIR /
    'tables'
)

for directory in [
    REPORT_ASSET_DIR,
    FIGURE_DIR,
    TABLE_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

assert BACKUP_DIR.exists(), (
    f"Checkpoint folder not found: {BACKUP_DIR}"
)

# ------------------------------------------------------------
# Install/register Bangla font
# ------------------------------------------------------------

for font_path in font_manager.findSystemFonts(
    fontext='ttf'
):
    try:
        font_manager.fontManager.addfont(
            font_path
        )
    except Exception:
        pass

bangla_font_paths = [
    path
    for path in font_manager.findSystemFonts(
        fontext='ttf'
    )
    if (
        'Bengali' in path
        or
        'Bangla' in path
    )
]

if not bangla_font_paths:
    raise FileNotFoundError(
        "No Bangla font was found."
    )

# Prefer Noto Sans Bengali when available
preferred_fonts = [
    path
    for path in bangla_font_paths
    if 'NotoSansBengali' in path.replace(
        ' ',
        ''
    )
]

BANGLA_FONT_PATH = (
    preferred_fonts[0]
    if preferred_fonts
    else bangla_font_paths[0]
)

BANGLA_FONT = (
    font_manager.FontProperties(
        fname=BANGLA_FONT_PATH
    )
)

BANGLA_FONT_NAME = (
    BANGLA_FONT.get_name()
)

plt.rcParams.update({
    'font.family': BANGLA_FONT_NAME,
    'font.sans-serif': [
        BANGLA_FONT_NAME,
        'Noto Sans Bengali',
        'DejaVu Sans'
    ],
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 300
})

print("Bangla font:", BANGLA_FONT_NAME)
print("Font path:", BANGLA_FONT_PATH)
print("Figure directory:", FIGURE_DIR)
print("Table directory:", TABLE_DIR)

# ============================================================
# RELOAD DATA AND PREPARE REPORT FEATURES
# ============================================================

LABEL2ID = {
    'General Query': 0,
    'Routine': 1,
    'Urgent': 2,
    'Emergency': 3
}

ID2LABEL = {
    value: key
    for key, value in LABEL2ID.items()
}

CLASS_ORDER = [
    'General Query',
    'Routine',
    'Urgent',
    'Emergency'
]

# ------------------------------------------------------------
# Load dataset splits
# ------------------------------------------------------------

train_df = pd.read_csv(
    BACKUP_DIR /
    'train.csv'
)

val_df = pd.read_csv(
    BACKUP_DIR /
    'validation.csv'
)

test_df = pd.read_csv(
    BACKUP_DIR /
    'test.csv'
)

train_df['split'] = 'Train'
val_df['split'] = 'Validation'
test_df['split'] = 'Test'

df_report = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)

df_report['text'] = (
    df_report['text']
    .fillna('')
    .astype(str)
)

df_report['label'] = (
    df_report['label']
    .astype(int)
)

if 'label_name' not in df_report.columns:
    df_report['label_name'] = (
        df_report['label']
        .map(ID2LABEL)
    )

# ------------------------------------------------------------
# Text features
# ------------------------------------------------------------

def normalize_text(text):

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


def tokenize_text(text):

    return re.findall(
        r'[\u0980-\u09FFA-Za-z]+',
        str(text).lower()
    )


df_report['normalized_text'] = (
    df_report['text']
    .map(normalize_text)
)

df_report['char_length'] = (
    df_report['text']
    .str.len()
)

df_report['word_length'] = (
    df_report['text']
    .str.split()
    .str.len()
)

df_report['bangla_chars'] = (
    df_report['text']
    .apply(
        lambda text: len(
            re.findall(
                r'[\u0980-\u09FF]',
                text
            )
        )
    )
)

df_report['latin_chars'] = (
    df_report['text']
    .apply(
        lambda text: len(
            re.findall(
                r'[A-Za-z]',
                text
            )
        )
    )
)

df_report['latin_ratio'] = (
    df_report['latin_chars']
    /
    (
        df_report['bangla_chars']
        +
        df_report['latin_chars']
        +
        1
    )
)

df_report['digit_count'] = (
    df_report['text']
    .str.count(
        r'[0-9০-৯]'
    )
)

df_report['question_mark_count'] = (
    df_report['text']
    .str.count(
        r'[?？]'
    )
)

df_report['token_list'] = (
    df_report['text']
    .map(tokenize_text)
)

vocabulary = {
    token
    for tokens in df_report[
        'token_list'
    ]
    for token in tokens
}

print("Total clean samples:", len(df_report))
print("Vocabulary size:", len(vocabulary))
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# REPORT-SAVING HELPER FUNCTIONS
# ============================================================

generated_assets = []

def save_figure(
    figure,
    filename
):

    output_path = (
        FIGURE_DIR /
        filename
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    generated_assets.append(
        str(output_path)
    )

    print(
        "Saved figure:",
        output_path.name
    )

    return output_path


def save_table_csv(
    dataframe,
    filename
):

    output_path = (
        TABLE_DIR /
        filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig'
    )

    generated_assets.append(
        str(output_path)
    )

    print(
        "Saved table:",
        output_path.name
    )

    return output_path


def save_table_image(
    dataframe,
    filename,
    title,
    max_rows=30,
    font_size=9
):

    table_df = (
        dataframe
        .head(max_rows)
        .copy()
    )

    for column in table_df.columns:

        if pd.api.types.is_float_dtype(
            table_df[column]
        ):

            table_df[column] = (
                table_df[column]
                .round(4)
            )

    figure_height = max(
        2.5,
        0.42 * len(table_df) + 1.8
    )

    figure_width = max(
        9,
        1.7 * len(table_df.columns)
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height
        )
    )

    axis.axis('off')

    axis.set_title(
        title,
        fontsize=14,
        fontweight='bold',
        pad=14,
        fontproperties=BANGLA_FONT
    )

    table = axis.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        colLoc='center',
        loc='center'
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        font_size
    )

    table.scale(
        1,
        1.35
    )

    for (
        row_index,
        column_index
    ), cell in table.get_celld().items():

        cell.get_text().set_fontproperties(
            BANGLA_FONT
        )

        if row_index == 0:

            cell.set_facecolor(
                '#4C72B0'
            )

            cell.get_text().set_color(
                'white'
            )

            cell.get_text().set_weight(
                'bold'
            )

        elif row_index % 2 == 0:

            cell.set_facecolor(
                '#EEF3F8'
            )

    save_figure(
        figure,
        filename
    )

    plt.show()

    return figure

# ============================================================
# DATASET AND PREPROCESSING TABLES
# ============================================================

# ------------------------------------------------------------
# Table 1: Dataset summary
# ------------------------------------------------------------

dataset_summary = pd.DataFrame({
    'Property': [
        'Dataset name',
        'Domain',
        'Task',
        'Language',
        'Data source',
        'Original samples',
        'Clean samples used',
        'Input feature',
        'Target',
        'Excluded leakage column',
        'Number of classes',
        'Train samples',
        'Validation samples',
        'Test samples'
    ],
    'Value': [
        'Bangla Healthcare Severity Dataset',
        'Healthcare NLP',
        'Ordinal multiclass text classification',
        'Bangla',
        'Public Facebook and YouTube discussions',
        5263,
        len(df_report),
        'Text',
        'Categories',
        'Action Needed',
        4,
        len(train_df),
        len(val_df),
        len(test_df)
    ]
})

save_table_csv(
    dataset_summary,
    'table_01_dataset_summary.csv'
)

save_table_image(
    dataset_summary,
    'table_01_dataset_summary.png',
    'Table 1: Dataset Summary',
    max_rows=20,
    font_size=9
)

# ------------------------------------------------------------
# Table 2: Class distribution
# ------------------------------------------------------------

class_distribution = (
    df_report[
        'label_name'
    ]
    .value_counts()
    .reindex(
        CLASS_ORDER
    )
    .rename_axis(
        'Severity Class'
    )
    .reset_index(
        name='Samples'
    )
)

class_distribution[
    'Percentage'
] = (
    class_distribution[
        'Samples'
    ]
    /
    len(df_report)
    *
    100
).round(2)

save_table_csv(
    class_distribution,
    'table_02_class_distribution.csv'
)

save_table_image(
    class_distribution,
    'table_02_class_distribution.png',
    'Table 2: Class Distribution'
)

# ------------------------------------------------------------
# Table 3: Split distribution
# ------------------------------------------------------------

split_distribution = (
    pd.crosstab(
        df_report[
            'split'
        ],
        df_report[
            'label_name'
        ]
    )
    .reindex(
        columns=CLASS_ORDER
    )
    .reset_index()
)

split_distribution[
    'Total'
] = split_distribution[
    CLASS_ORDER
].sum(axis=1)

save_table_csv(
    split_distribution,
    'table_03_split_distribution.csv'
)

save_table_image(
    split_distribution,
    'table_03_split_distribution.png',
    'Table 3: Train, Validation and Test Distribution'
)

# ------------------------------------------------------------
# Table 4: Preprocessing summary
# ------------------------------------------------------------

preprocessing_summary = pd.DataFrame({
    'Operation': [
        'Missing-value handling',
        'Unicode normalization',
        'URL normalization',
        'User-mention normalization',
        'Whitespace normalization',
        'Exact duplicate removal',
        'Normalized duplicate removal',
        'Tokenization',
        'Stopword analysis',
        'Stemming/Lemmatization',
        'Class imbalance handling',
        'Data splitting',
        'Leakage prevention'
    ],
    'Implementation': [
        'Rows with missing text/labels removed',
        'NFKC normalization',
        'URLs replaced with URL token',
        'Mentions replaced with USER token',
        'Repeated whitespace collapsed',
        'Duplicate text removed before split',
        'Normalized duplicates removed before split',
        'Regex and model-specific tokenization',
        'Bangla stopword frequency analysed',
        'Not applied to transformer input; subword tokenization retained',
        'Stratification and class-weighted classical models',
        '70% train, 15% validation, 15% test',
        'Action Needed excluded; transforms fitted on training data'
    ]
})

save_table_csv(
    preprocessing_summary,
    'table_04_preprocessing_summary.csv'
)

save_table_image(
    preprocessing_summary,
    'table_04_preprocessing_summary.png',
    'Table 4: Preprocessing Pipeline',
    max_rows=20,
    font_size=8
)

# ------------------------------------------------------------
# Table 5: Feature engineering
# ------------------------------------------------------------

feature_engineering_table = pd.DataFrame({
    'Feature Type': [
        'Bag-of-Words',
        'Word TF-IDF',
        'Character TF-IDF',
        'Trainable embeddings',
        'Transformer embeddings',
        'Length features',
        'Script features',
        'Domain-related patterns'
    ],
    'Used With': [
        'Classical baseline analysis',
        'NB, LR, SVM, RF, XGBoost',
        'LR, SVM, RF, XGBoost',
        'CNN, BiLSTM, BiGRU',
        'BanglaBERT and SCORE-BN',
        'EDA/domain analysis',
        'Bangla/Latin character analysis',
        'Digits, question marks and text patterns'
    ],
    'Justification': [
        'Simple lexical baseline',
        'Captures informative words and n-grams',
        'Robust to spelling and word variation',
        'Learns task-specific representations',
        'Provides contextual language representations',
        'Measures query complexity',
        'Measures script and code-mixing variation',
        'Captures query-specific structural signals'
    ]
})

save_table_csv(
    feature_engineering_table,
    'table_05_feature_engineering.csv'
)

save_table_image(
    feature_engineering_table,
    'table_05_feature_engineering.png',
    'Table 5: Feature Engineering Methods',
    font_size=8
)

# ------------------------------------------------------------
# Table 6: Model families
# ------------------------------------------------------------

model_family_table = pd.DataFrame({
    'Model': [
        'Multinomial Naive Bayes',
        'Logistic Regression',
        'Linear SVM',
        'Random Forest',
        'XGBoost',
        'Text-CNN',
        'BiLSTM',
        'BiGRU',
        'BanglaBERT',
        'SCORE-BN'
    ],
    'Family': [
        'Probabilistic',
        'Linear',
        'Margin-based',
        'Tree ensemble',
        'Boosted ensemble',
        'Convolutional neural network',
        'Recurrent neural network',
        'Recurrent neural network',
        'Transformer',
        'Proposed transformer framework'
    ],
    'Feature Representation': [
        'TF-IDF',
        'Word + character TF-IDF',
        'Word + character TF-IDF',
        'Word + character TF-IDF',
        'Word + character TF-IDF',
        'Trainable token embeddings',
        'Trainable token embeddings',
        'Trainable token embeddings',
        'Contextual subword embeddings',
        'Original + Romanized paired contextual embeddings'
    ]
})

save_table_csv(
    model_family_table,
    'table_06_model_families.csv'
)

save_table_image(
    model_family_table,
    'table_06_model_families.png',
    'Table 6: Evaluated Models and Algorithm Families',
    font_size=8
)

# ============================================================
# REQUIRED EXPLORATORY-DATA-ANALYSIS FIGURES
# ============================================================

# ------------------------------------------------------------
# Figure 1: Class distribution
# ------------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(9, 6)
)

sns.barplot(
    data=class_distribution,
    x='Severity Class',
    y='Samples',
    order=CLASS_ORDER,
    palette='viridis',
    ax=axis
)

axis.set_title(
    'Distribution of Healthcare Severity Classes',
    fontsize=16,
    fontweight='bold'
)

axis.set_xlabel(
    'Severity Class'
)

axis.set_ylabel(
    'Number of Queries'
)

axis.tick_params(
    axis='x',
    rotation=15
)

for container in axis.containers:
    axis.bar_label(
        container,
        padding=3
    )

plt.tight_layout()

save_figure(
    figure,
    'figure_01_class_distribution.png'
)

plt.show()

# ------------------------------------------------------------
# Figure 2: Word-length distribution
# ------------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(11, 6)
)

sns.histplot(
    data=df_report,
    x='word_length',
    hue='label_name',
    hue_order=CLASS_ORDER,
    bins=40,
    element='step',
    common_norm=False,
    ax=axis
)

axis.set_title(
    'Text-Length Distribution by Severity Class',
    fontsize=16,
    fontweight='bold'
)

axis.set_xlabel(
    'Number of Words'
)

axis.set_ylabel(
    'Number of Queries'
)

plt.tight_layout()

save_figure(
    figure,
    'figure_02_word_length_distribution.png'
)

plt.show()

# ------------------------------------------------------------
# Figure 3: Character-length box plot
# ------------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(10, 6)
)

sns.boxplot(
    data=df_report,
    x='label_name',
    y='char_length',
    order=CLASS_ORDER,
    palette='Set2',
    ax=axis
)

axis.set_title(
    'Character-Length Distribution by Class',
    fontsize=16,
    fontweight='bold'
)

axis.set_xlabel(
    'Severity Class'
)

axis.set_ylabel(
    'Number of Characters'
)

axis.tick_params(
    axis='x',
    rotation=15
)

plt.tight_layout()

save_figure(
    figure,
    'figure_03_character_length_boxplot.png'
)

plt.show()

# ------------------------------------------------------------
# Figure 4: Script composition
# ------------------------------------------------------------

script_summary = (
    df_report
    .groupby(
        'label_name'
    )[
        [
            'bangla_chars',
            'latin_chars'
        ]
    ]
    .mean()
    .reindex(
        CLASS_ORDER
    )
    .reset_index()
)

script_long = script_summary.melt(
    id_vars='label_name',
    var_name='Character Type',
    value_name='Average Count'
)

figure, axis = plt.subplots(
    figsize=(10, 6)
)

sns.barplot(
    data=script_long,
    x='label_name',
    y='Average Count',
    hue='Character Type',
    ax=axis
)

axis.set_title(
    'Average Bangla and Latin Characters by Class',
    fontsize=16,
    fontweight='bold'
)

axis.set_xlabel(
    'Severity Class'
)

axis.set_ylabel(
    'Average Character Count'
)

axis.tick_params(
    axis='x',
    rotation=15
)

plt.tight_layout()

save_figure(
    figure,
    'figure_04_script_composition.png'
)

plt.show()

# ------------------------------------------------------------
# Figure 5: Data split sizes
# ------------------------------------------------------------

split_size_table = (
    df_report[
        'split'
    ]
    .value_counts()
    .reindex(
        [
            'Train',
            'Validation',
            'Test'
        ]
    )
    .rename_axis(
        'Split'
    )
    .reset_index(
        name='Samples'
    )
)

figure, axis = plt.subplots(
    figsize=(8, 5)
)

sns.barplot(
    data=split_size_table,
    x='Split',
    y='Samples',
    palette='Blues_d',
    ax=axis
)

axis.set_title(
    'Dataset Split Sizes',
    fontsize=16,
    fontweight='bold'
)

for container in axis.containers:
    axis.bar_label(
        container,
        padding=3
    )

plt.tight_layout()

save_figure(
    figure,
    'figure_05_dataset_split_sizes.png'
)

plt.show()

# ============================================================
# WORD CLOUDS AND CO-OCCURRENCE PLOT
# ============================================================

# ------------------------------------------------------------
# Figure 8: One word cloud per class
# ------------------------------------------------------------

figure, axes = plt.subplots(
    2,
    2,
    figsize=(18, 11)
)

for axis, class_name in zip(
    axes.flatten(),
    CLASS_ORDER
):

    class_tokens = [
        token
        for tokens in df_report.loc[
            df_report[
                'label_name'
            ]
            ==
            class_name,
            'token_list'
        ]
        for token in tokens
        if (
            len(token) > 1
            and
            token not in BANGLA_STOPWORDS
        )
    ]

    class_text = ' '.join(
        class_tokens
    )

    word_cloud = WordCloud(
        width=1000,
        height=550,
        background_color='white',
        font_path=BANGLA_FONT_PATH,
        collocations=False,
        max_words=150,
        colormap='viridis'
    ).generate(
        class_text
    )

    axis.imshow(
        word_cloud,
        interpolation='bilinear'
    )

    axis.axis('off')

    axis.set_title(
        class_name,
        fontsize=16,
        fontweight='bold'
    )

plt.suptitle(
    'Word Clouds by Healthcare Severity Class',
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout()

save_figure(
    figure,
    'figure_08_wordclouds_by_class.png'
)

plt.show()

# ------------------------------------------------------------
# Figure 9: Token co-occurrence heatmap
# ------------------------------------------------------------

top_cooccurrence_words = [
    word
    for word, count
    in content_counts.most_common(
        15
    )
]

binary_vectorizer = CountVectorizer(
    vocabulary=top_cooccurrence_words,
    token_pattern=r'(?u)\b\w+\b',
    binary=True
)

cooccurrence_matrix_source = (
    binary_vectorizer.fit_transform(
        df_report[
            'normalized_text'
        ]
    )
)

cooccurrence_matrix = (
    cooccurrence_matrix_source.T
    @
    cooccurrence_matrix_source
).toarray()

np.fill_diagonal(
    cooccurrence_matrix,
    0
)

cooccurrence_table = pd.DataFrame(
    cooccurrence_matrix,
    index=top_cooccurrence_words,
    columns=top_cooccurrence_words
)

cooccurrence_table.to_csv(
    TABLE_DIR /
    'table_10_cooccurrence_matrix.csv',
    encoding='utf-8-sig'
)

figure, axis = plt.subplots(
    figsize=(13, 11)
)

sns.heatmap(
    cooccurrence_table,
    cmap='YlGnBu',
    annot=True,
    fmt='d',
    linewidths=0.3,
    ax=axis
)

axis.set_title(
    'Frequent-Word Co-occurrence Matrix',
    fontsize=16,
    fontweight='bold'
)

for label in (
    axis.get_xticklabels()
    +
    axis.get_yticklabels()
):
    label.set_fontproperties(
        BANGLA_FONT
    )

axis.tick_params(
    axis='x',
    rotation=45
)

axis.tick_params(
    axis='y',
    rotation=0
)

plt.tight_layout()

save_figure(
    figure,
    'figure_09_word_cooccurrence_heatmap.png'
)

plt.show()

# ============================================================
# TEXT STATISTICS AND VOCABULARY TABLES
# ============================================================

text_statistics = (
    df_report
    .groupby(
        'label_name'
    )[
        [
            'word_length',
            'char_length',
            'bangla_chars',
            'latin_chars',
            'latin_ratio'
        ]
    ]
    .agg(
        [
            'mean',
            'median',
            'std',
            'min',
            'max'
        ]
    )
    .round(2)
    .reindex(
        CLASS_ORDER
    )
)

# Flatten multi-level column names
text_statistics.columns = [
    f'{feature}_{statistic}'
    for feature, statistic
    in text_statistics.columns
]

text_statistics = (
    text_statistics
    .reset_index()
    .rename(
        columns={
            'label_name':
            'Severity Class'
        }
    )
)

save_table_csv(
    text_statistics,
    'table_11_text_statistics_by_class.csv'
)

# A smaller version for a readable report image
compact_text_statistics = (
    df_report
    .groupby(
        'label_name'
    )
    .agg(
        Samples=(
            'text',
            'size'
        ),
        Mean_Words=(
            'word_length',
            'mean'
        ),
        Median_Words=(
            'word_length',
            'median'
        ),
        Mean_Characters=(
            'char_length',
            'mean'
        ),
        Vocabulary=(
            'text',
            lambda series: len({
                token
                for text in series
                for token in tokenize_text(
                    text
                )
            })
        ),
        Mean_Latin_Ratio=(
            'latin_ratio',
            'mean'
        )
    )
    .reindex(
        CLASS_ORDER
    )
    .round(3)
    .reset_index()
    .rename(
        columns={
            'label_name':
            'Severity Class'
        }
    )
)

save_table_csv(
    compact_text_statistics,
    'table_12_compact_text_statistics.csv'
)

save_table_image(
    compact_text_statistics,
    'table_12_compact_text_statistics.png',
    'Table 12: Text Statistics by Severity Class',
    font_size=8
)

vocabulary_summary = pd.DataFrame({
    'Measure': [
        'Total clean samples',
        'Vocabulary size',
        'Total tokens',
        'Unique content words',
        'Average words per query',
        'Median words per query',
        'Maximum words in a query'
    ],
    'Value': [
        len(df_report),
        len(vocabulary),
        len(all_tokens),
        len(content_counts),
        round(
            df_report[
                'word_length'
            ].mean(),
            2
        ),
        round(
            df_report[
                'word_length'
            ].median(),
            2
        ),
        int(
            df_report[
                'word_length'
            ].max()
        )
    ]
})

save_table_csv(
    vocabulary_summary,
    'table_13_vocabulary_summary.csv'
)

save_table_image(
    vocabulary_summary,
    'table_13_vocabulary_summary.png',
    'Table 13: Vocabulary and Token Summary'
)

# ============================================================
# MODEL COMPARISON AND TUNING FIGURES
# ============================================================

MODEL_COMPARISON_PATH = (
    BACKUP_DIR /
    'final_results' /
    'model_comparison.csv'
)

if not MODEL_COMPARISON_PATH.exists():
    raise FileNotFoundError(
        "Run corrected Section 25 first. "
        "model_comparison.csv is missing."
    )

model_comparison = pd.read_csv(
    MODEL_COMPARISON_PATH
)

save_table_csv(
    model_comparison,
    'table_14_complete_model_comparison.csv'
)

display_columns = [
    column
    for column in [
        'model',
        'stage',
        'accuracy',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'roc_auc_ovr'
    ]
    if column in model_comparison.columns
]

save_table_image(
    model_comparison[
        display_columns
    ],
    'table_14_complete_model_comparison.png',
    'Table 14: Complete Model Comparison',
    max_rows=20,
    font_size=7
)

# ------------------------------------------------------------
# Figure 10: Macro-F1 comparison
# ------------------------------------------------------------

plot_model_results = (
    model_comparison
    .dropna(
        subset=[
            'macro_f1'
        ]
    )
    .sort_values(
        'macro_f1',
        ascending=True
    )
)

figure, axis = plt.subplots(
    figsize=(12, 8)
)

sns.barplot(
    data=plot_model_results,
    x='macro_f1',
    y='model',
    hue='stage',
    dodge=False,
    palette='Set2',
    ax=axis
)

axis.set_title(
    'Macro-F1 Comparison Across NLP Models',
    fontsize=16,
    fontweight='bold'
)

axis.set_xlabel(
    'Macro-F1'
)

axis.set_ylabel(
    'Model'
)

axis.set_xlim(
    0,
    1
)

for container in axis.containers:
    axis.bar_label(
        container,
        fmt='%.3f',
        padding=3
    )

plt.tight_layout()

save_figure(
    figure,
    'figure_10_model_macro_f1_comparison.png'
)

plt.show()

# ------------------------------------------------------------
# Figure 11: Accuracy, precision, recall and F1
# ------------------------------------------------------------

metric_columns = [
    column
    for column in [
        'accuracy',
        'macro_precision',
        'macro_recall',
        'macro_f1'
    ]
    if column in model_comparison.columns
]

multi_metric_table = (
    model_comparison[
        [
            'model'
        ]
        +
        metric_columns
    ]
    .dropna(
        subset=[
            'macro_f1'
        ]
    )
    .set_index(
        'model'
    )
)

figure, axis = plt.subplots(
    figsize=(15, 8)
)

multi_metric_table.plot(
    kind='bar',
    ax=axis,
    width=0.82,
    colormap='viridis'
)

axis.set_title(
    'Accuracy, Precision, Recall and F1 Comparison',
    fontsize=16,
    fontweight='bold'
)

axis.set_xlabel(
    'Model'
)

axis.set_ylabel(
    'Score'
)

axis.set_ylim(
    0,
    1
)

axis.tick_params(
    axis='x',
    rotation=45
)

axis.legend(
    title='Metric',
    bbox_to_anchor=(
        1.02,
        1
    ),
    loc='upper left'
)

plt.tight_layout()

save_figure(
    figure,
    'figure_11_multi_metric_model_comparison.png'
)

plt.show()

# ------------------------------------------------------------
# Transformer tuning trials
# ------------------------------------------------------------

TRANSFORMER_TRIAL_PATH = (
    BACKUP_DIR /
    'transformer_tuning' /
    'all_trial_results.csv'
)

if TRANSFORMER_TRIAL_PATH.exists():

    transformer_trials = pd.read_csv(
        TRANSFORMER_TRIAL_PATH
    )

    save_table_csv(
        transformer_trials,
        'table_15_transformer_tuning_trials.csv'
    )

    save_table_image(
        transformer_trials,
        'table_15_transformer_tuning_trials.png',
        'Table 15: BanglaBERT Hyperparameter-Tuning Results',
        font_size=7
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    sns.barplot(
        data=transformer_trials,
        x='trial',
        y='validation_macro_f1',
        palette='flare',
        ax=axis
    )

    axis.set_title(
        'BanglaBERT Tuning Trial Comparison',
        fontsize=16,
        fontweight='bold'
    )

    axis.set_xlabel(
        'Trial'
    )

    axis.set_ylabel(
        'Validation Macro-F1'
    )

    for container in axis.containers:
        axis.bar_label(
            container,
            fmt='%.4f',
            padding=3
        )

    plt.tight_layout()

    save_figure(
        figure,
        'figure_12_transformer_tuning_comparison.png'
    )

    plt.show()

# ------------------------------------------------------------
# Classical tuning table
# ------------------------------------------------------------

CLASSICAL_TUNING_PATH = (
    BACKUP_DIR /
    'tuned_model_results.csv'
)

if CLASSICAL_TUNING_PATH.exists():

    classical_tuning = pd.read_csv(
        CLASSICAL_TUNING_PATH
    )

    save_table_csv(
        classical_tuning,
        'table_16_classical_tuning_results.csv'
    )

    save_table_image(
        classical_tuning,
        'table_16_classical_tuning_results.png',
        'Table 16: Tuned SVM and XGBoost Results',
        font_size=8
    )

# ============================================================
# SCORE-BN EVALUATION FIGURES AND TABLES
# ============================================================

SCORE_METRICS_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'test_metrics.json'
)

SCORE_ERROR_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'test_predictions_and_errors.csv'
)

ORIGINAL_PROBABILITY_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'original_test_probabilities.npy'
)

ROMANIZED_PROBABILITY_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'romanized_test_probabilities.npy'
)

if not SCORE_ERROR_PATH.exists():
    raise FileNotFoundError(
        "SCORE-BN prediction file is missing. "
        "Run corrected Section 21 first."
    )

score_errors = pd.read_csv(
    SCORE_ERROR_PATH
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

    score_metric_table = pd.DataFrame({
        'Metric': list(
            score_metrics.keys()
        ),
        'Value': list(
            score_metrics.values()
        )
    })

    save_table_csv(
        score_metric_table,
        'table_17_score_bn_metrics.csv'
    )

    save_table_image(
        score_metric_table,
        'table_17_score_bn_metrics.png',
        'Table 17: SCORE-BN Evaluation Metrics',
        max_rows=30,
        font_size=8
    )

# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------

true_labels = (
    score_errors[
        'true_label_id'
    ]
    .astype(int)
    .to_numpy()
)

original_predictions = (
    score_errors[
        'original_prediction_id'
    ]
    .astype(int)
    .to_numpy()
)

romanized_predictions = (
    score_errors[
        'romanized_prediction_id'
    ]
    .astype(int)
    .to_numpy()
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)

ConfusionMatrixDisplay.from_predictions(
    true_labels,
    original_predictions,
    labels=[
        0,
        1,
        2,
        3
    ],
    display_labels=CLASS_ORDER,
    cmap='Purples',
    colorbar=False,
    ax=axes[0]
)

axes[0].set_title(
    'SCORE-BN: Original Bangla'
)

axes[0].tick_params(
    axis='x',
    rotation=25
)

ConfusionMatrixDisplay.from_predictions(
    true_labels,
    romanized_predictions,
    labels=[
        0,
        1,
        2,
        3
    ],
    display_labels=CLASS_ORDER,
    cmap='Oranges',
    colorbar=False,
    ax=axes[1]
)

axes[1].set_title(
    'SCORE-BN: Romanized Bangla'
)

axes[1].tick_params(
    axis='x',
    rotation=25
)

plt.tight_layout()

save_figure(
    figure,
    'figure_13_score_bn_confusion_matrices.png'
)

plt.show()

# ------------------------------------------------------------
# ROC curves
# ------------------------------------------------------------

if ORIGINAL_PROBABILITY_PATH.exists():

    original_probabilities = np.load(
        ORIGINAL_PROBABILITY_PATH
    )

    binary_labels = label_binarize(
        true_labels,
        classes=[
            0,
            1,
            2,
            3
        ]
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    for class_index in range(4):

        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                binary_labels[
                    :,
                    class_index
                ],
                original_probabilities[
                    :,
                    class_index
                ]
            )
        )

        class_auc = auc(
            false_positive_rate,
            true_positive_rate
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=(
                f'{ID2LABEL[class_index]} '
                f'(AUC={class_auc:.3f})'
            )
        )

    axis.plot(
        [
            0,
            1
        ],
        [
            0,
            1
        ],
        linestyle='--',
        color='gray'
    )

    axis.set_title(
        'SCORE-BN One-vs-Rest ROC Curves',
        fontsize=16,
        fontweight='bold'
    )

    axis.set_xlabel(
        'False Positive Rate'
    )

    axis.set_ylabel(
        'True Positive Rate'
    )

    axis.legend(
        loc='lower right'
    )

    plt.tight_layout()

    save_figure(
        figure,
        'figure_14_score_bn_roc_curves.png'
    )

    plt.show()

# ------------------------------------------------------------
# Original vs Romanized comparison
# ------------------------------------------------------------

if SCORE_METRICS_PATH.exists():

    robustness_table = pd.DataFrame({
        'Condition': [
            'Original Bangla',
            'Romanized Bangla'
        ],
        'Accuracy': [
            score_metrics.get(
                'accuracy',
                np.nan
            ),
            score_metrics.get(
                'romanized_accuracy',
                np.nan
            )
        ],
        'Macro Precision': [
            score_metrics.get(
                'macro_precision',
                np.nan
            ),
            score_metrics.get(
                'romanized_macro_precision',
                np.nan
            )
        ],
        'Macro Recall': [
            score_metrics.get(
                'macro_recall',
                np.nan
            ),
            score_metrics.get(
                'romanized_macro_recall',
                np.nan
            )
        ],
        'Macro F1': [
            score_metrics.get(
                'macro_f1',
                np.nan
            ),
            score_metrics.get(
                'romanized_macro_f1',
                np.nan
            )
        ],
        'ROC-AUC': [
            score_metrics.get(
                'roc_auc_ovr',
                np.nan
            ),
            score_metrics.get(
                'romanized_roc_auc_ovr',
                np.nan
            )
        ]
    })

    save_table_csv(
        robustness_table,
        'table_18_original_romanized_comparison.csv'
    )

    save_table_image(
        robustness_table,
        'table_18_original_romanized_comparison.png',
        'Table 18: Original and Romanized Test Performance',
        font_size=8
    )

    robustness_plot = (
        robustness_table
        .set_index(
            'Condition'
        )[
            [
                'Accuracy',
                'Macro Precision',
                'Macro Recall',
                'Macro F1',
                'ROC-AUC'
            ]
        ]
    )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    robustness_plot.plot(
        kind='bar',
        ax=axis,
        width=0.75,
        colormap='Set2'
    )

    axis.set_title(
        'Original vs Romanized Bangla Performance',
        fontsize=16,
        fontweight='bold'
    )

    axis.set_xlabel(
        'Input Condition'
    )

    axis.set_ylabel(
        'Score'
    )

    axis.set_ylim(
        0,
        1
    )

    axis.tick_params(
        axis='x',
        rotation=0
    )

    axis.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc='upper left'
    )

    plt.tight_layout()

    save_figure(
        figure,
        'figure_15_cross_script_robustness.png'
    )

    plt.show()

# ============================================================
# TRAINING HISTORY AND ERROR ANALYSIS
# ============================================================

# ------------------------------------------------------------
# SCORE-BN training history
# ------------------------------------------------------------

SCORE_HISTORY_PATH = (
    BACKUP_DIR /
    'score_bn' /
    'training_history.csv'
)

if SCORE_HISTORY_PATH.exists():

    score_history = pd.read_csv(
        SCORE_HISTORY_PATH
    )

    save_table_csv(
        score_history,
        'table_19_score_bn_training_history.csv'
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    axes[0].plot(
        score_history[
            'epoch'
        ],
        score_history[
            'training_loss'
        ],
        marker='o',
        label='Training Loss'
    )

    axes[0].plot(
        score_history[
            'epoch'
        ],
        score_history[
            'validation_loss'
        ],
        marker='o',
        label='Validation Loss'
    )

    axes[0].set_title(
        'SCORE-BN Training and Validation Loss'
    )

    axes[0].set_xlabel(
        'Epoch'
    )

    axes[0].set_ylabel(
        'Loss'
    )

    axes[0].legend()

    axes[1].plot(
        score_history[
            'epoch'
        ],
        score_history[
            'validation_macro_f1'
        ],
        marker='o',
        color='#2A9D8F'
    )

    axes[1].set_title(
        'SCORE-BN Validation Macro-F1'
    )

    axes[1].set_xlabel(
        'Epoch'
    )

    axes[1].set_ylabel(
        'Macro-F1'
    )

    axes[1].set_ylim(
        0,
        1
    )

    plt.tight_layout()

    save_figure(
        figure,
        'figure_16_score_bn_training_history.png'
    )

    plt.show()

# ------------------------------------------------------------
# CNN, BiLSTM and BiGRU training curves
# ------------------------------------------------------------

deep_history_files = {
    'CNN':
        BACKUP_DIR /
        'CNN_training_history.csv',

    'BiLSTM':
        BACKUP_DIR /
        'BiLSTM_training_history.csv',

    'BiGRU':
        BACKUP_DIR /
        'BiGRU_training_history.csv'
}

figure, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)

any_deep_history = False

for model_name, history_path in deep_history_files.items():

    if not history_path.exists():
        continue

    any_deep_history = True

    history = pd.read_csv(
        history_path
    )

    epochs = np.arange(
        1,
        len(history) + 1
    )

    if 'loss' in history.columns:

        axes[0].plot(
            epochs,
            history[
                'loss'
            ],
            marker='o',
            label=(
                f'{model_name} train'
            )
        )

    if 'val_loss' in history.columns:

        axes[0].plot(
            epochs,
            history[
                'val_loss'
            ],
            linestyle='--',
            label=(
                f'{model_name} validation'
            )
        )

    if 'accuracy' in history.columns:

        axes[1].plot(
            epochs,
            history[
                'accuracy'
            ],
            marker='o',
            label=(
                f'{model_name} train'
            )
        )

    if 'val_accuracy' in history.columns:

        axes[1].plot(
            epochs,
            history[
                'val_accuracy'
            ],
            linestyle='--',
            label=(
                f'{model_name} validation'
            )
        )

if any_deep_history:

    axes[0].set_title(
        'Deep-Model Loss Curves'
    )

    axes[0].set_xlabel(
        'Epoch'
    )

    axes[0].set_ylabel(
        'Loss'
    )

    axes[0].legend(
        fontsize=8
    )

    axes[1].set_title(
        'Deep-Model Accuracy Curves'
    )

    axes[1].set_xlabel(
        'Epoch'
    )

    axes[1].set_ylabel(
        'Accuracy'
    )

    axes[1].legend(
        fontsize=8
    )

    plt.tight_layout()

    save_figure(
        figure,
        'figure_17_deep_model_training_curves.png'
    )

    plt.show()

else:

    plt.close(
        figure
    )

# ------------------------------------------------------------
# Error analysis
# ------------------------------------------------------------

error_summary = pd.DataFrame({
    'Error Type': [
        'Correct predictions',
        'Under-prioritised',
        'Over-prioritised',
        'Severe ordinal errors',
        'Cross-script disagreements'
    ],
    'Samples': [
        int(
            (
                original_predictions
                ==
                true_labels
            ).sum()
        ),
        int(
            (
                original_predictions
                <
                true_labels
            ).sum()
        ),
        int(
            (
                original_predictions
                >
                true_labels
            ).sum()
        ),
        int(
            (
                np.abs(
                    original_predictions
                    -
                    true_labels
                )
                >= 2
            ).sum()
        ),
        int(
            (
                original_predictions
                !=
                romanized_predictions
            ).sum()
        )
    ]
})

save_table_csv(
    error_summary,
    'table_20_score_bn_error_summary.csv'
)

save_table_image(
    error_summary,
    'table_20_score_bn_error_summary.png',
    'Table 20: SCORE-BN Error Analysis'
)

figure, axis = plt.subplots(
    figsize=(10, 6)
)

sns.barplot(
    data=error_summary,
    x='Samples',
    y='Error Type',
    palette='rocket',
    ax=axis
)

axis.set_title(
    'SCORE-BN Prediction and Error Summary',
    fontsize=16,
    fontweight='bold'
)

for container in axis.containers:
    axis.bar_label(
        container,
        padding=3
    )

plt.tight_layout()

save_figure(
    figure,
    'figure_18_score_bn_error_summary.png'
)

plt.show()

# ============================================================
# COLLECT XAI FIGURES AND CREATE FINAL ASSET CHECKLIST
# ============================================================

EXPLAINABILITY_DIR = (
    BACKUP_DIR /
    'explainability'
)

# Copy corrected LIME/SHAP figures into the report directory
xai_candidates = [
    'lime_explanation_final.png',
    'lime_example_bangla_fixed.png',
    'lime_example.png',
    'shap_global_summary_bangla_fixed.png',
    'shap_global_summary.png'
]

for filename in xai_candidates:

    source_path = (
        EXPLAINABILITY_DIR /
        filename
    )

    if source_path.exists():

        destination_path = (
            FIGURE_DIR /
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        generated_assets.append(
            str(destination_path)
        )

        print(
            "Copied XAI figure:",
            filename
        )

# Copy XAI tables/HTML
xai_report_candidates = [
    'lime_example.html',
    'lime_feature_weights.csv',
    'logistic_regression_top_features.csv'
]

for filename in xai_report_candidates:

    source_path = (
        EXPLAINABILITY_DIR /
        filename
    )

    if source_path.exists():

        destination_path = (
            TABLE_DIR /
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        generated_assets.append(
            str(destination_path)
        )

        print(
            "Copied XAI asset:",
            filename
        )

# ------------------------------------------------------------
# Required-report asset checklist
# ------------------------------------------------------------

required_report_assets = [
    {
        'PDF Requirement':
        'Dataset description',

        'Recommended Asset':
        'table_01_dataset_summary.png'
    },
    {
        'PDF Requirement':
        'Target/class analysis',

        'Recommended Asset':
        'figure_01_class_distribution.png'
    },
    {
        'PDF Requirement':
        'Text-length distribution',

        'Recommended Asset':
        'figure_02_word_length_distribution.png'
    },
    {
        'PDF Requirement':
        'Vocabulary size',

        'Recommended Asset':
        'table_13_vocabulary_summary.png'
    },
    {
        'PDF Requirement':
        'Word frequency',

        'Recommended Asset':
        'figure_06_top_content_words.png'
    },
    {
        'PDF Requirement':
        'Stopword analysis',

        'Recommended Asset':
        'figure_07_stopword_frequency.png'
    },
    {
        'PDF Requirement':
        'N-gram analysis',

        'Recommended Asset':
        'figure_bigram_frequency.png'
    },
    {
        'PDF Requirement':
        'Word cloud',

        'Recommended Asset':
        'figure_08_wordclouds_by_class.png'
    },
    {
        'PDF Requirement':
        'Co-occurrence plot',

        'Recommended Asset':
        'figure_09_word_cooccurrence_heatmap.png'
    },
    {
        'PDF Requirement':
        'Preprocessing',

        'Recommended Asset':
        'table_04_preprocessing_summary.png'
    },
    {
        'PDF Requirement':
        'Feature engineering',

        'Recommended Asset':
        'table_05_feature_engineering.png'
    },
    {
        'PDF Requirement':
        'Eight models/five families',

        'Recommended Asset':
        'table_06_model_families.png'
    },
    {
        'PDF Requirement':
        'Hyperparameter tuning',

        'Recommended Asset':
        'figure_12_transformer_tuning_comparison.png'
    },
    {
        'PDF Requirement':
        'Model comparison',

        'Recommended Asset':
        'figure_11_multi_metric_model_comparison.png'
    },
    {
        'PDF Requirement':
        'Confusion matrix',

        'Recommended Asset':
        'figure_13_score_bn_confusion_matrices.png'
    },
    {
        'PDF Requirement':
        'ROC-AUC',

        'Recommended Asset':
        'figure_14_score_bn_roc_curves.png'
    },
    {
        'PDF Requirement':
        'Explainable AI—LIME',

        'Recommended Asset':
        'lime_explanation_final.png'
    },
    {
        'PDF Requirement':
        'Explainable AI—SHAP',

        'Recommended Asset':
        'shap_global_summary_bangla_fixed.png'
    },
    {
        'PDF Requirement':
        'Proposed-model robustness',

        'Recommended Asset':
        'figure_15_cross_script_robustness.png'
    }
]

asset_checklist = pd.DataFrame(
    required_report_assets
)

asset_checklist[
    'Exists'
] = asset_checklist[
    'Recommended Asset'
].apply(
    lambda filename: (
        FIGURE_DIR /
        filename
    ).exists()
    or
    (
        TABLE_DIR /
        filename
    ).exists()
)

save_table_csv(
    asset_checklist,
    'report_requirement_asset_checklist.csv'
)

display(
    asset_checklist
)

# ------------------------------------------------------------
# Complete inventory
# ------------------------------------------------------------

inventory_rows = []

for folder_name, folder_path in [
    (
        'Figure',
        FIGURE_DIR
    ),
    (
        'Table',
        TABLE_DIR
    )
]:

    for file_path in folder_path.rglob(
        '*'
    ):

        if file_path.is_file():

            inventory_rows.append({
                'Type': folder_name,
                'Filename': file_path.name,
                'Path': str(
                    file_path
                ),
                'Size KB': round(
                    file_path.stat().st_size
                    /
                    1024,
                    2
                )
            })

report_asset_inventory = (
    pd.DataFrame(
        inventory_rows
    )
    .sort_values(
        [
            'Type',
            'Filename'
        ]
    )
    .reset_index(
        drop=True
    )
)

save_table_csv(
    report_asset_inventory,
    'complete_report_asset_inventory.csv'
)

print(
    "\nTotal report figures:",
    len(
        list(
            FIGURE_DIR.glob(
                '*'
            )
        )
    )
)

print(
    "Total report tables/assets:",
    len(
        list(
            TABLE_DIR.glob(
                '*'
            )
        )
    )
)

print(
    "\nReport assets saved in:"
)

print(REPORT_ASSET_DIR)

# ============================================================
# CREATE DOWNLOADABLE REPORT-ASSETS ZIP
# ============================================================

import shutil
from pathlib import Path

EXPORTS_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Exports'
)

EXPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

temporary_zip_base = (
    '/content/SCORE_BN_REPORT_ASSETS'
)

temporary_zip_path = shutil.make_archive(
    temporary_zip_base,
    'zip',
    root_dir=REPORT_ASSET_DIR
)

drive_zip_path = (
    EXPORTS_DIR /
    'SCORE_BN_REPORT_ASSETS.zip'
)

shutil.copy2(
    temporary_zip_path,
    drive_zip_path
)

print(
    "Report-assets ZIP created:"
)

print(drive_zip_path)

print(
    "ZIP size:",
    round(
        drive_zip_path.stat().st_size
        /
        1024**2,
        2
    ),
    "MB"
)

print(
    "\nYou may now run the final complete-backup "
    "cell and then turn off Colab."
)

# ============================================================
# DEFINITIVE BANGLA FONT REPAIR
# ============================================================

!apt-get update -qq
!apt-get install -y -qq fonts-noto-core fonts-noto-extra fontconfig
!fc-cache -f -v > /dev/null

import os
import glob
import re
import unicodedata
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager

# ------------------------------------------------------------
# 1. Use the exact regular Bengali font
# ------------------------------------------------------------

possible_font_paths = [
    Path(
        '/usr/share/fonts/truetype/noto/'
        'NotoSansBengali-Regular.ttf'
    ),
    Path(
        '/usr/share/fonts/truetype/noto/'
        'NotoSerifBengali-Regular.ttf'
    )
]

BANGLA_FONT_PATH = next(
    (
        path
        for path in possible_font_paths
        if path.exists()
    ),
    None
)

if BANGLA_FONT_PATH is None:

    discovered_fonts = [
        Path(path)
        for path
        in font_manager.findSystemFonts(
            fontext='ttf'
        )
        if (
            'NotoSansBengali-Regular'
            in Path(path).name
        )
    ]

    if not discovered_fonts:
        raise FileNotFoundError(
            "Noto Sans Bengali Regular was not found."
        )

    BANGLA_FONT_PATH = discovered_fonts[0]

# ------------------------------------------------------------
# 2. Clear the old Matplotlib font cache
# ------------------------------------------------------------

for cache_file in glob.glob(
    os.path.join(
        matplotlib.get_cachedir(),
        'fontlist-*.json'
    )
):
    try:
        os.remove(cache_file)
    except OSError:
        pass

# Explicitly register this exact file
font_manager.fontManager.addfont(
    str(BANGLA_FONT_PATH)
)

BANGLA_FONT = font_manager.FontProperties(
    fname=str(BANGLA_FONT_PATH)
)

BANGLA_FONT_NAME = BANGLA_FONT.get_name()

# Reapply Seaborn first, then override its font settings
sns.set_theme(
    style='whitegrid',
    font=BANGLA_FONT_NAME
)

plt.rcParams.update({
    'font.family': BANGLA_FONT_NAME,
    'font.sans-serif': [
        BANGLA_FONT_NAME,
        'Noto Sans Bengali',
        'DejaVu Sans'
    ],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

print("Selected font:", BANGLA_FONT_NAME)
print("Exact file:", BANGLA_FONT_PATH)

assert (
    BANGLA_FONT_PATH.name
    ==
    'NotoSansBengali-Regular.ttf'
), "The regular Bengali font was not selected."

# ============================================================
# VERIFY BANGLA RENDERING BEFORE REGENERATING EVERYTHING
# ============================================================

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

FONT_TEST_DIR = (
    BACKUP_DIR /
    'font_test'
)

FONT_TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

bangla_test_text = (
    "বাংলা স্বাস্থ্যসেবা প্রশ্নের "
    "তীব্রতা শ্রেণিবিন্যাস"
)

figure, axis = plt.subplots(
    figsize=(12, 3)
)

axis.text(
    0.5,
    0.65,
    bangla_test_text,
    fontproperties=BANGLA_FONT,
    fontsize=24,
    ha='center',
    va='center'
)

axis.text(
    0.5,
    0.30,
    "সাধারণ প্রশ্ন • নিয়মিত • জরুরি • অত্যন্ত জরুরি",
    fontproperties=BANGLA_FONT,
    fontsize=18,
    ha='center',
    va='center'
)

axis.axis('off')

font_test_path = (
    FONT_TEST_DIR /
    'bangla_font_test.png'
)

figure.savefig(
    font_test_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

print("Font-test image saved to:")
print(font_test_path)

!pip -q install -U plotly kaleido

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

XAI_DIR = (
    BACKUP_DIR /
    'explainability'
)

REPORT_FIGURE_DIR = (
    BACKUP_DIR /
    'report_assets' /
    'figures'
)

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BANGLA_PLOT_FONT = (
    'Noto Sans Bengali, Nirmala UI, '
    'Vrinda, sans-serif'
)

print("Output directory:", XAI_DIR)

# ============================================================
# REBUILD LIME USING BROWSER-BASED BANGLA TEXT RENDERING
# ============================================================

LIME_CSV_PATH = (
    XAI_DIR /
    'lime_feature_weights.csv'
)

if not LIME_CSV_PATH.exists():
    raise FileNotFoundError(
        f"LIME values were not found: {LIME_CSV_PATH}"
    )

lime_df = pd.read_csv(
    LIME_CSV_PATH
)

print("LIME columns:", lime_df.columns.tolist())

# Detect columns safely
feature_column = next(
    column
    for column in lime_df.columns
    if column.lower() in [
        'word_or_phrase',
        'feature',
        'word',
        'token'
    ]
)

weight_column = next(
    column
    for column in lime_df.columns
    if column.lower() in [
        'importance_weight',
        'importance',
        'weight',
        'value'
    ]
)

lime_plot_df = (
    lime_df[
        [
            feature_column,
            weight_column
        ]
    ]
    .dropna()
    .sort_values(
        weight_column,
        ascending=True
    )
    .tail(15)
)

predicted_class = (
    lime_df[
        'predicted_class'
    ].iloc[0]
    if 'predicted_class' in lime_df.columns
    else 'Prediction'
)

bar_colors = [
    '#198754'
    if value >= 0
    else '#DC3545'
    for value in lime_plot_df[
        weight_column
    ]
]

lime_figure = go.Figure()

lime_figure.add_trace(
    go.Bar(
        x=lime_plot_df[
            weight_column
        ],
        y=lime_plot_df[
            feature_column
        ],
        orientation='h',
        marker_color=bar_colors,
        text=[
            f'{value:.4f}'
            for value in lime_plot_df[
                weight_column
            ]
        ],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Contribution: %{x:.4f}'
            '<extra></extra>'
        )
    )
)

lime_figure.add_vline(
    x=0,
    line_width=1,
    line_color='black'
)

lime_figure.update_layout(
    title=(
        f'LIME Explanation: '
        f'{predicted_class}'
    ),
    xaxis_title='Contribution to Prediction',
    yaxis_title='Word or Phrase',
    template='plotly_white',
    width=1200,
    height=750,
    margin=dict(
        l=250,
        r=100,
        t=90,
        b=80
    ),
    font=dict(
        family=BANGLA_PLOT_FONT,
        size=17,
        color='black'
    ),
    title_font=dict(
        family=BANGLA_PLOT_FONT,
        size=24
    )
)

lime_figure.show()

# HTML always preserves browser-based Bengali shaping
LIME_HTML_PATH = (
    XAI_DIR /
    'lime_explanation_bangla_correct.html'
)

lime_figure.write_html(
    LIME_HTML_PATH,
    include_plotlyjs='cdn'
)

# Static PNG for the report
LIME_PNG_PATH = (
    XAI_DIR /
    'lime_explanation_bangla_correct.png'
)

try:
    lime_figure.write_image(
        LIME_PNG_PATH,
        width=1200,
        height=750,
        scale=2
    )

    shutil.copy2(
        LIME_PNG_PATH,
        REPORT_FIGURE_DIR /
        LIME_PNG_PATH.name
    )

    print("LIME PNG saved:", LIME_PNG_PATH)

except Exception as error:
    print("PNG export failed:", error)
    print("HTML was still saved correctly:", LIME_HTML_PATH)

# Commented out IPython magic to ensure Python compatibility.
# ============================================================
# REBUILD GLOBAL SHAP FIGURE WITH CORRECT BANGLA SHAPING
# ============================================================

required_shap_variables = [
    'shap_values_for_plot',
    'feature_names',
    'class_names'
]

missing_shap_variables = [
    variable
    for variable in required_shap_variables
    if variable not in globals()
]

if missing_shap_variables:
    raise NameError(
        f"Missing SHAP variables: {missing_shap_variables}. "
        "Rerun the corrected SHAP calculation cell first."
    )

# Remove FeatureUnion prefixes
clean_feature_names = np.array([
    str(feature)
    .replace('word__', '')
    .replace('char__', '')
    for feature in feature_names
])

# Convert SHAP output to samples × features × classes
if isinstance(
    shap_values_for_plot,
    list
):

    shap_by_class = [
        np.asarray(values)
        for values in shap_values_for_plot
    ]

elif (
    isinstance(
        shap_values_for_plot,
        np.ndarray
    )
    and
    shap_values_for_plot.ndim == 3
):

    shap_by_class = [
        shap_values_for_plot[
            :,
            :,
            class_index
        ]
        for class_index in range(
            shap_values_for_plot.shape[2]
        )
    ]

elif (
    hasattr(
        shap_values_for_plot,
        'values'
    )
):

    shap_array = np.asarray(
        shap_values_for_plot.values
    )

    if shap_array.ndim == 3:

        shap_by_class = [
            shap_array[
                :,
                :,
                class_index
            ]
            for class_index in range(
                shap_array.shape[2]
            )
        ]

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

else:
    raise ValueError(
        "Unsupported SHAP output format."
    )

# Mean absolute SHAP contribution by feature and class
mean_absolute_shap = np.column_stack([
    np.abs(
        class_values
    ).mean(axis=0)
    for class_values in shap_by_class
])

global_importance = (
    mean_absolute_shap.sum(axis=1)
)

top_feature_indices = np.argsort(
    global_importance
)[-20:]

top_features = clean_feature_names[
    top_feature_indices
]

class_colors = [
    '#1685E5',
    '#B94BD3',
    '#FF0051',
    '#00A02B'
]

shap_figure = go.Figure()

for class_index, class_name in enumerate(
    class_names
):

    shap_figure.add_trace(
        go.Bar(
            name=class_name,
            x=mean_absolute_shap[
                top_feature_indices,
                class_index
            ],
            y=top_features,
            orientation='h',
            marker_color=class_colors[
                class_index
#                 %
                len(class_colors)
            ],
            hovertemplate=(
                '<b>%{y}</b><br>'
                +
                f'Class: {class_name}<br>'
                +
                'Mean |SHAP|: %{x:.5f}'
                +
                '<extra></extra>'
            )
        )
    )

shap_figure.update_layout(
    title=(
        'Global SHAP Feature Importance — '
        'Logistic Regression'
    ),
    xaxis_title='Mean Absolute SHAP Value',
    yaxis_title='Feature',
    barmode='stack',
    template='plotly_white',
    width=1400,
    height=950,
    margin=dict(
        l=300,
        r=100,
        t=100,
        b=80
    ),
    legend_title='Severity Class',
    font=dict(
        family=BANGLA_PLOT_FONT,
        size=16,
        color='black'
    ),
    title_font=dict(
        family=BANGLA_PLOT_FONT,
        size=24
    )
)

shap_figure.show()

SHAP_HTML_PATH = (
    XAI_DIR /
    'shap_global_bangla_correct.html'
)

shap_figure.write_html(
    SHAP_HTML_PATH,
    include_plotlyjs='cdn'
)

SHAP_PNG_PATH = (
    XAI_DIR /
    'shap_global_bangla_correct.png'
)

try:
    shap_figure.write_image(
        SHAP_PNG_PATH,
        width=1400,
        height=950,
        scale=2
    )

    shutil.copy2(
        SHAP_PNG_PATH,
        REPORT_FIGURE_DIR /
        SHAP_PNG_PATH.name
    )

    print("SHAP PNG saved:", SHAP_PNG_PATH)

except Exception as error:
    print("PNG export failed:", error)
    print("HTML was still saved correctly:", SHAP_HTML_PATH)

# ============================================================
# BANGLA-SAFE REPORT ASSET SETUP
# ============================================================

!pip -q install -U plotly kaleido

import re
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

FIXED_REPORT_DIR = (
    BACKUP_DIR /
    'report_assets_bangla_fixed'
)

FIXED_FIGURE_DIR = (
    FIXED_REPORT_DIR /
    'figures'
)

FIXED_TABLE_PNG_DIR = (
    FIXED_REPORT_DIR /
    'tables_png'
)

FIXED_TABLE_CSV_DIR = (
    FIXED_REPORT_DIR /
    'tables_csv'
)

FIXED_HTML_DIR = (
    FIXED_REPORT_DIR /
    'html'
)

for directory in [
    FIXED_REPORT_DIR,
    FIXED_FIGURE_DIR,
    FIXED_TABLE_PNG_DIR,
    FIXED_TABLE_CSV_DIR,
    FIXED_HTML_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# Browser fallback font list
BANGLA_FONT_FAMILY = (
    'Noto Sans Bengali, Nirmala UI, '
    'Vrinda, Arial Unicode MS, sans-serif'
)

LABEL2ID = {
    'General Query': 0,
    'Routine': 1,
    'Urgent': 2,
    'Emergency': 3
}

ID2LABEL = {
    value: key
    for key, value in LABEL2ID.items()
}

CLASS_ORDER = [
    'General Query',
    'Routine',
    'Urgent',
    'Emergency'
]

# ------------------------------------------------------------
# Load saved dataset splits
# ------------------------------------------------------------

train_df = pd.read_csv(
    BACKUP_DIR /
    'train.csv'
)

val_df = pd.read_csv(
    BACKUP_DIR /
    'validation.csv'
)

test_df = pd.read_csv(
    BACKUP_DIR /
    'test.csv'
)

train_df['split'] = 'Train'
val_df['split'] = 'Validation'
test_df['split'] = 'Test'

df_report = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)

df_report['text'] = (
    df_report['text']
    .fillna('')
    .astype(str)
)

df_report['label'] = (
    df_report['label']
    .astype(int)
)

if 'label_name' not in df_report.columns:

    df_report['label_name'] = (
        df_report['label']
        .map(ID2LABEL)
    )

def normalize_text(text):

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


def tokenize_bangla(text):

    return re.findall(
        r'[\u0980-\u09FFA-Za-z]+',
        str(text).lower()
    )


df_report['normalized_text'] = (
    df_report['text']
    .map(normalize_text)
)

df_report['tokens'] = (
    df_report['text']
    .map(tokenize_bangla)
)

df_report['word_length'] = (
    df_report['text']
    .str.split()
    .str.len()
)

df_report['char_length'] = (
    df_report['text']
    .str.len()
)

BANGLA_STOPWORDS = {
    'আমি', 'আমার', 'আমাকে', 'আমাদের',
    'আপনি', 'আপনার', 'সে', 'তার',
    'এই', 'ওই', 'এটা', 'সেটা',
    'কি', 'কী', 'কেন', 'কিভাবে',
    'কোন', 'কখন', 'কোথায়',
    'এবং', 'ও', 'আর', 'কিন্তু',
    'যে', 'যদি', 'তবে', 'তাহলে',
    'হয়', 'হয়', 'হচ্ছে', 'হবে',
    'হয়েছে', 'হয়েছে', 'ছিল',
    'আছে', 'নেই', 'না', 'নয়',
    'জন্য', 'থেকে', 'সাথে',
    'দিকে', 'মধ্যে', 'উপর',
    'একটি', 'একটা', 'কিছু',
    'খুব', 'অনেক', 'বা',
    'তো', 'এর', 'কে', 'তে',
    'র', 'টি'
}

all_tokens = [
    token
    for tokens in df_report['tokens']
    for token in tokens
    if len(token) > 1
]

content_tokens = [
    token
    for token in all_tokens
    if token not in BANGLA_STOPWORDS
]

token_counts = Counter(all_tokens)
content_counts = Counter(content_tokens)

print("Clean samples:", len(df_report))
print("Directories created:", FIXED_REPORT_DIR)

# ============================================================
# BANGLA-SAFE PLOTLY SAVE FUNCTIONS
# ============================================================

saved_assets = []
failed_assets = []

def apply_bangla_layout(
    figure,
    title,
    width=1200,
    height=750
):

    figure.update_layout(
        title=title,
        template='plotly_white',
        width=width,
        height=height,
        font=dict(
            family=BANGLA_FONT_FAMILY,
            size=16,
            color='black'
        ),
        title_font=dict(
            family=BANGLA_FONT_FAMILY,
            size=24,
            color='black'
        ),
        margin=dict(
            l=180,
            r=80,
            t=100,
            b=90
        )
    )

    return figure


def save_plotly_figure(
    figure,
    filename,
    width=1200,
    height=750
):

    html_path = (
        FIXED_HTML_DIR /
        filename.replace(
            '.png',
            '.html'
        )
    )

    png_path = (
        FIXED_FIGURE_DIR /
        filename
    )

    # Browser-readable HTML
    figure.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True
    )

    try:

        figure.write_image(
            png_path,
            width=width,
            height=height,
            scale=2
        )

        saved_assets.append(
            str(png_path)
        )

        print(
            "PNG saved:",
            png_path.name
        )

    except Exception as error:

        failed_assets.append({
            'file': filename,
            'error': str(error)
        })

        print(
            "PNG export failed:",
            filename
        )

        print(
            "HTML was saved:",
            html_path
        )

    saved_assets.append(
        str(html_path)
    )

    return png_path


def create_plotly_table(
    dataframe,
    title,
    filename,
    max_rows=30
):

    table_df = (
        dataframe
        .head(max_rows)
        .copy()
    )

    for column in table_df.columns:

        if pd.api.types.is_float_dtype(
            table_df[column]
        ):

            table_df[column] = (
                table_df[column]
                .round(4)
            )

    row_colors = [
        '#FFFFFF'
        if index % 2 == 0
        else '#EEF3F8'
        for index in range(
            len(table_df)
        )
    ]

    figure = go.Figure(
        data=[
            go.Table(
                columnwidth=[
                    1.3
                    for _ in table_df.columns
                ],
                header=dict(
                    values=[
                        f'<b>{column}</b>'
                        for column in table_df.columns
                    ],
                    fill_color='#3E6E9E',
                    font=dict(
                        family=BANGLA_FONT_FAMILY,
                        size=15,
                        color='white'
                    ),
                    align='center',
                    height=38
                ),
                cells=dict(
                    values=[
                        table_df[
                            column
                        ].astype(str)
                        for column in table_df.columns
                    ],
                    fill_color=[
                        row_colors
                        for _ in table_df.columns
                    ],
                    font=dict(
                        family=BANGLA_FONT_FAMILY,
                        size=14,
                        color='black'
                    ),
                    align='center',
                    height=34
                )
            )
        ]
    )

    height = max(
        400,
        95 + 36 * len(table_df)
    )

    figure.update_layout(
        title=title,
        width=max(
            1000,
            190 * len(table_df.columns)
        ),
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20
        ),
        font=dict(
            family=BANGLA_FONT_FAMILY
        )
    )

    csv_path = (
        FIXED_TABLE_CSV_DIR /
        filename.replace(
            '.png',
            '.csv'
        )
    )

    table_df.to_csv(
        csv_path,
        index=False,
        encoding='utf-8-sig'
    )

    html_path = (
        FIXED_HTML_DIR /
        filename.replace(
            '.png',
            '.html'
        )
    )

    png_path = (
        FIXED_TABLE_PNG_DIR /
        filename
    )

    figure.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True
    )

    try:

        figure.write_image(
            png_path,
            width=figure.layout.width,
            height=height,
            scale=2
        )

        saved_assets.append(
            str(png_path)
        )

        print(
            "Table PNG saved:",
            png_path.name
        )

    except Exception as error:

        failed_assets.append({
            'file': filename,
            'error': str(error)
        })

        print(
            "Table PNG export failed:",
            filename
        )

    saved_assets.extend([
        str(csv_path),
        str(html_path)
    ])

    return figure

# ============================================================
# REBUILD ALL SAVED TABLES WITH CORRECT BANGLA RENDERING
# ============================================================

old_table_directories = [
    BACKUP_DIR /
    'report_assets' /
    'tables',

    BACKUP_DIR /
    'final_results',

    BACKUP_DIR /
    'explainability'
]

table_csv_files = []

for directory in old_table_directories:

    if directory.exists():

        table_csv_files.extend(
            directory.glob(
                '*.csv'
            )
        )

# Remove duplicate paths
table_csv_files = list(
    dict.fromkeys(
        table_csv_files
    )
)

print(
    "CSV tables found:",
    len(table_csv_files)
)

for csv_path in table_csv_files:

    try:

        table_df = pd.read_csv(
            csv_path
        )

        if table_df.empty:
            continue

        output_name = (
            'fixed_'
            +
            csv_path.stem
            +
            '.png'
        )

        create_plotly_table(
            table_df,
            title=csv_path.stem.replace(
                '_',
                ' '
            ).title(),
            filename=output_name,
            max_rows=30
        )

    except Exception as error:

        failed_assets.append({
            'file': str(csv_path),
            'error': str(error)
        })

        print(
            "Could not rebuild:",
            csv_path.name,
            error
        )

print(
    "\nFinished rebuilding saved tables."
)

# ============================================================
# GENERATE ESSENTIAL PROJECT TABLES
# ============================================================

dataset_summary = pd.DataFrame({
    'Property': [
        'Dataset',
        'Domain',
        'Problem Type',
        'Language',
        'Original Samples',
        'Clean Samples',
        'Training Samples',
        'Validation Samples',
        'Test Samples',
        'Input',
        'Target',
        'Excluded Leakage Column'
    ],
    'Value': [
        'Bangla Healthcare Severity Dataset',
        'Healthcare NLP',
        'Ordinal Multiclass Classification',
        'Bangla',
        5263,
        len(df_report),
        len(train_df),
        len(val_df),
        len(test_df),
        'Text',
        'Categories',
        'Action Needed'
    ]
})

create_plotly_table(
    dataset_summary,
    'Dataset Summary',
    'table_01_dataset_summary_fixed.png'
)

class_distribution = (
    df_report['label_name']
    .value_counts()
    .reindex(CLASS_ORDER)
    .rename_axis(
        'Severity Class'
    )
    .reset_index(
        name='Samples'
    )
)

class_distribution[
    'Percentage'
] = (
    class_distribution[
        'Samples'
    ]
    /
    len(df_report)
    *
    100
).round(2)

create_plotly_table(
    class_distribution,
    'Class Distribution',
    'table_02_class_distribution_fixed.png'
)

split_distribution = (
    pd.crosstab(
        df_report['split'],
        df_report['label_name']
    )
    .reindex(
        columns=CLASS_ORDER
    )
    .reset_index()
)

split_distribution['Total'] = (
    split_distribution[
        CLASS_ORDER
    ].sum(axis=1)
)

create_plotly_table(
    split_distribution,
    'Dataset Split Distribution',
    'table_03_split_distribution_fixed.png'
)

preprocessing_table = pd.DataFrame({
    'Operation': [
        'Missing Values',
        'Duplicate Removal',
        'Unicode Normalization',
        'URL Normalization',
        'Mention Normalization',
        'Tokenization',
        'Stopword Analysis',
        'TF-IDF Encoding',
        'Transformer Tokenization',
        'Data Splitting',
        'Leakage Prevention'
    ],
    'Implementation': [
        'Missing text and target rows removed',
        'Exact and normalized duplicates removed',
        'NFKC normalization',
        'URLs replaced with URL token',
        'Mentions replaced with USER token',
        'Regex and model-specific tokenization',
        'Bangla stopword frequency analysed',
        'Word and character TF-IDF',
        'BanglaBERT subword tokenizer',
        '70% train, 15% validation, 15% test',
        'Action Needed excluded from input'
    ]
})

create_plotly_table(
    preprocessing_table,
    'Preprocessing Pipeline',
    'table_04_preprocessing_pipeline_fixed.png'
)

model_family_table = pd.DataFrame({
    'Model': [
        'Multinomial Naive Bayes',
        'Logistic Regression',
        'Linear SVM',
        'Random Forest',
        'XGBoost',
        'Text-CNN',
        'BiLSTM',
        'BiGRU',
        'BanglaBERT',
        'SCORE-BN'
    ],
    'Family': [
        'Probabilistic',
        'Linear',
        'Margin-based',
        'Tree Ensemble',
        'Boosted Ensemble',
        'Convolutional Neural Network',
        'Recurrent Neural Network',
        'Recurrent Neural Network',
        'Transformer',
        'Proposed Transformer Framework'
    ]
})

create_plotly_table(
    model_family_table,
    'Models and Algorithm Families',
    'table_05_model_families_fixed.png'
)

# ============================================================
# BANGLA WORD, STOPWORD AND N-GRAM FIGURES
# ============================================================

top_words = pd.DataFrame(
    content_counts.most_common(20),
    columns=[
        'Word',
        'Frequency'
    ]
).sort_values(
    'Frequency'
)

word_figure = go.Figure(
    go.Bar(
        x=top_words['Frequency'],
        y=top_words['Word'],
        orientation='h',
        marker_color='#2A9D8F',
        text=top_words['Frequency'],
        textposition='outside'
    )
)

apply_bangla_layout(
    word_figure,
    'Most Frequent Content Words',
    width=1200,
    height=850
)

word_figure.update_xaxes(
    title='Frequency'
)

word_figure.update_yaxes(
    title='Word'
)

word_figure.show()

save_plotly_figure(
    word_figure,
    'figure_01_top_content_words_fixed.png',
    width=1200,
    height=850
)

# ------------------------------------------------------------
# Stopword figure
# ------------------------------------------------------------

stopword_counts = Counter(
    token
    for token in all_tokens
    if token in BANGLA_STOPWORDS
)

top_stopwords = pd.DataFrame(
    stopword_counts.most_common(20),
    columns=[
        'Stopword',
        'Frequency'
    ]
).sort_values(
    'Frequency'
)

if not top_stopwords.empty:

    stopword_figure = go.Figure(
        go.Bar(
            x=top_stopwords[
                'Frequency'
            ],
            y=top_stopwords[
                'Stopword'
            ],
            orientation='h',
            marker_color='#457B9D',
            text=top_stopwords[
                'Frequency'
            ],
            textposition='outside'
        )
    )

    apply_bangla_layout(
        stopword_figure,
        'Most Frequent Bangla Stopwords',
        width=1200,
        height=850
    )

    stopword_figure.update_xaxes(
        title='Frequency'
    )

    stopword_figure.update_yaxes(
        title='Stopword'
    )

    stopword_figure.show()

    save_plotly_figure(
        stopword_figure,
        'figure_02_stopword_frequency_fixed.png',
        width=1200,
        height=850
    )

# ============================================================
# N-GRAM ANALYSIS WITH BANGLA-SAFE RENDERING
# ============================================================

from sklearn.feature_extraction.text import CountVectorizer

def calculate_top_ngrams(
    texts,
    n,
    top_k=20
):

    vectorizer = CountVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        ngram_range=(n, n),
        min_df=2
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    frequencies = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    indices = frequencies.argsort()[
        ::-1
    ][
        :top_k
    ]

    return pd.DataFrame({
        'N-gram': names[indices],
        'Frequency': frequencies[indices]
    })


for n, name in [
    (1, 'Unigram'),
    (2, 'Bigram'),
    (3, 'Trigram')
]:

    ngram_df = calculate_top_ngrams(
        df_report[
            'normalized_text'
        ],
        n,
        top_k=20
    )

    ngram_df.to_csv(
        FIXED_TABLE_CSV_DIR /
        f'{name.lower()}_frequency.csv',
        index=False,
        encoding='utf-8-sig'
    )

    plot_df = ngram_df.sort_values(
        'Frequency'
    )

    figure = go.Figure(
        go.Bar(
            x=plot_df[
                'Frequency'
            ],
            y=plot_df[
                'N-gram'
            ],
            orientation='h',
            marker_color='#6A4C93',
            text=plot_df[
                'Frequency'
            ],
            textposition='outside'
        )
    )

    apply_bangla_layout(
        figure,
        f'Top {name} Frequencies',
        width=1300,
        height=900
    )

    figure.update_xaxes(
        title='Frequency'
    )

    figure.update_yaxes(
        title=name
    )

    figure.show()

    save_plotly_figure(
        figure,
        f'figure_{name.lower()}_frequency_fixed.png',
        width=1300,
        height=900
    )

# ============================================================
# BANGLA-SAFE CO-OCCURRENCE HEATMAP
# ============================================================

top_cooccurrence_words = [
    word
    for word, count
    in content_counts.most_common(
        15
    )
]

co_vectorizer = CountVectorizer(
    vocabulary=top_cooccurrence_words,
    token_pattern=r'(?u)\b\w+\b',
    binary=True
)

co_source = co_vectorizer.fit_transform(
    df_report[
        'normalized_text'
    ]
)

co_matrix = (
    co_source.T
    @
    co_source
).toarray()

np.fill_diagonal(
    co_matrix,
    0
)

cooccurrence_df = pd.DataFrame(
    co_matrix,
    index=top_cooccurrence_words,
    columns=top_cooccurrence_words
)

cooccurrence_df.to_csv(
    FIXED_TABLE_CSV_DIR /
    'cooccurrence_matrix.csv',
    encoding='utf-8-sig'
)

heatmap_figure = go.Figure(
    data=go.Heatmap(
        z=co_matrix,
        x=top_cooccurrence_words,
        y=top_cooccurrence_words,
        colorscale='Blues',
        text=co_matrix,
        texttemplate='%{text}',
        hovertemplate=(
            '%{y} | %{x}<br>'
            'Co-occurrence: %{z}'
            '<extra></extra>'
        )
    )
)

apply_bangla_layout(
    heatmap_figure,
    'Frequent-Word Co-occurrence Matrix',
    width=1200,
    height=1050
)

heatmap_figure.update_xaxes(
    tickangle=-45
)

heatmap_figure.show()

save_plotly_figure(
    heatmap_figure,
    'figure_cooccurrence_heatmap_fixed.png',
    width=1200,
    height=1050
)

# ============================================================
# BANGLA-SAFE FREQUENCY CLOUD
# ============================================================

cloud_words = pd.DataFrame(
    content_counts.most_common(80),
    columns=[
        'Word',
        'Frequency'
    ]
)

minimum_frequency = (
    cloud_words[
        'Frequency'
    ].min()
)

maximum_frequency = (
    cloud_words[
        'Frequency'
    ].max()
)

def cloud_font_size(frequency):

    if maximum_frequency == minimum_frequency:
        return 28

    return int(
        18
        +
        48
        *
        (
            frequency
            -
            minimum_frequency
        )
        /
        (
            maximum_frequency
            -
            minimum_frequency
        )
    )


colors = [
    '#264653',
    '#2A9D8F',
    '#E9C46A',
    '#F4A261',
    '#E76F51',
    '#6A4C93',
    '#1982C4'
]

word_spans = []

for index, row in cloud_words.iterrows():

    size = cloud_font_size(
        row['Frequency']
    )

    color = colors[
        index % len(colors)
    ]

    word_spans.append(
        f'''
        <span
            title="Frequency: {row["Frequency"]}"
            style="
                font-size:{size}px;
                color:{color};
                margin:10px 14px;
                display:inline-block;
                line-height:1.2;
                font-family:{BANGLA_FONT_FAMILY};
            "
        >
            {row["Word"]}
        </span>
        '''
    )

cloud_html = f'''
<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="UTF-8">
<style>
body {{
    margin: 0;
    background: white;
    font-family: {BANGLA_FONT_FAMILY};
}}
h1 {{
    text-align: center;
    font-size: 34px;
    margin: 28px 0;
}}
.cloud {{
    width: 1200px;
    min-height: 720px;
    padding: 35px;
    box-sizing: border-box;
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
    align-content: center;
    flex-wrap: wrap;
}}
</style>
</head>
<body>
<h1>Bangla Healthcare Query Frequency Cloud</h1>
<div class="cloud">
{''.join(word_spans)}
</div>
</body>
</html>
'''

CLOUD_HTML_PATH = (
    FIXED_HTML_DIR /
    'figure_frequency_cloud_fixed.html'
)

CLOUD_HTML_PATH.write_text(
    cloud_html,
    encoding='utf-8'
)

print(
    "Bangla-safe frequency cloud HTML saved:"
)

print(CLOUD_HTML_PATH)

# ============================================================
# BANGLA-SAFE LIME FIGURE
# ============================================================

LIME_CSV_PATH = (
    BACKUP_DIR /
    'explainability' /
    'lime_feature_weights.csv'
)

if LIME_CSV_PATH.exists():

    lime_df = pd.read_csv(
        LIME_CSV_PATH
    )

    feature_column = next(
        column
        for column in lime_df.columns
        if column.lower() in [
            'word_or_phrase',
            'feature',
            'word',
            'token'
        ]
    )

    weight_column = next(
        column
        for column in lime_df.columns
        if column.lower() in [
            'importance_weight',
            'importance',
            'weight',
            'value'
        ]
    )

    predicted_class = (
        lime_df[
            'predicted_class'
        ].iloc[0]
        if 'predicted_class'
        in lime_df.columns
        else 'Prediction'
    )

    lime_plot_df = (
        lime_df[
            [
                feature_column,
                weight_column
            ]
        ]
        .dropna()
        .sort_values(
            weight_column
        )
        .tail(15)
    )

    lime_colors = [
        '#198754'
        if value >= 0
        else '#DC3545'
        for value in lime_plot_df[
            weight_column
        ]
    ]

    lime_figure = go.Figure(
        go.Bar(
            x=lime_plot_df[
                weight_column
            ],
            y=lime_plot_df[
                feature_column
            ],
            orientation='h',
            marker_color=lime_colors,
            text=[
                f'{value:.4f}'
                for value
                in lime_plot_df[
                    weight_column
                ]
            ],
            textposition='outside'
        )
    )

    lime_figure.add_vline(
        x=0,
        line_color='black',
        line_width=1
    )

    apply_bangla_layout(
        lime_figure,
        f'LIME Explanation: {predicted_class}',
        width=1200,
        height=800
    )

    lime_figure.update_xaxes(
        title='Contribution to Prediction'
    )

    lime_figure.update_yaxes(
        title='Word or Phrase'
    )

    lime_figure.show()

    save_plotly_figure(
        lime_figure,
        'figure_lime_bangla_fixed.png',
        width=1200,
        height=800
    )

else:

    print(
        "LIME CSV not found:",
        LIME_CSV_PATH
    )

# ============================================================
# BANGLA-SAFE SHAP FIGURE
# ============================================================

if all(
    variable in globals()
    for variable in [
        'shap_values_for_plot',
        'feature_names',
        'class_names'
    ]
):

    clean_feature_names = np.array([
        str(feature)
        .replace('word__', '')
        .replace('char__', '')
        for feature in feature_names
    ])

    if isinstance(
        shap_values_for_plot,
        list
    ):

        shap_by_class = [
            np.asarray(values)
            for values
            in shap_values_for_plot
        ]

    elif isinstance(
        shap_values_for_plot,
        np.ndarray
    ) and shap_values_for_plot.ndim == 3:

        shap_by_class = [
            shap_values_for_plot[
                :,
                :,
                class_index
            ]
            for class_index in range(
                shap_values_for_plot.shape[2]
            )
        ]

    else:

        shap_array = np.asarray(
            shap_values_for_plot.values
        )

        shap_by_class = [
            shap_array[
                :,
                :,
                class_index
            ]
            for class_index in range(
                shap_array.shape[2]
            )
        ]

    mean_absolute_shap = (
        np.column_stack([
            np.abs(values).mean(
                axis=0
            )
            for values in shap_by_class
        ])
    )

    total_importance = (
        mean_absolute_shap.sum(
            axis=1
        )
    )

    top_indices = np.argsort(
        total_importance
    )[-20:]

    top_features = (
        clean_feature_names[
            top_indices
        ]
    )

    shap_colors = [
        '#1685E5',
        '#B94BD3',
        '#FF0051',
        '#00A02B'
    ]

    shap_figure = go.Figure()

    for class_index, class_name in enumerate(
        class_names
    ):

        shap_figure.add_trace(
            go.Bar(
                name=class_name,
                x=mean_absolute_shap[
                    top_indices,
                    class_index
                ],
                y=top_features,
                orientation='h',
                marker_color=shap_colors[
                    class_index
                ]
            )
        )

    apply_bangla_layout(
        shap_figure,
        'Global SHAP Feature Importance — Logistic Regression',
        width=1400,
        height=950
    )

    shap_figure.update_layout(
        barmode='stack',
        legend_title='Severity Class'
    )

    shap_figure.update_xaxes(
        title='Mean Absolute SHAP Value'
    )

    shap_figure.update_yaxes(
        title='Feature'
    )

    shap_figure.show()

    save_plotly_figure(
        shap_figure,
        'figure_shap_bangla_fixed.png',
        width=1400,
        height=950
    )

else:

    print(
        "SHAP values are not currently in memory."
    )

    print(
        "Rerun the corrected SHAP calculation cell, "
        "then rerun this cell."
    )

# ============================================================
# COLLECT SAFE FIGURES, VERIFY AND CREATE FINAL ZIP
# ============================================================

old_figure_dir = (
    BACKUP_DIR /
    'report_assets' /
    'figures'
)

# These figures use English/numeric labels and do not need
# Bengali shaping. Copy them if they already exist.
english_safe_patterns = [
    'figure_01_class_distribution.png',
    'figure_02_word_length_distribution.png',
    'figure_03_character_length_boxplot.png',
    'figure_04_script_composition.png',
    'figure_05_dataset_split_sizes.png',
    'figure_10_model_macro_f1_comparison.png',
    'figure_11_multi_metric_model_comparison.png',
    'figure_12_transformer_tuning_comparison.png',
    'figure_13_score_bn_confusion_matrices.png',
    'figure_14_score_bn_roc_curves.png',
    'figure_15_cross_script_robustness.png',
    'figure_16_score_bn_training_history.png',
    'figure_17_deep_model_training_curves.png',
    'figure_18_score_bn_error_summary.png'
]

for filename in english_safe_patterns:

    source = (
        old_figure_dir /
        filename
    )

    destination = (
        FIXED_FIGURE_DIR /
        filename
    )

    if source.exists():

        shutil.copy2(
            source,
            destination
        )

        print(
            "Copied English-safe figure:",
            filename
        )

# ------------------------------------------------------------
# Inventory
# ------------------------------------------------------------

inventory_rows = []

for asset_type, directory in [
    ('Figure PNG', FIXED_FIGURE_DIR),
    ('Table PNG', FIXED_TABLE_PNG_DIR),
    ('Table CSV', FIXED_TABLE_CSV_DIR),
    ('Interactive HTML', FIXED_HTML_DIR)
]:

    for file_path in directory.glob('*'):

        if file_path.is_file():

            inventory_rows.append({
                'Type': asset_type,
                'Filename': file_path.name,
                'Size KB': round(
                    file_path.stat().st_size
                    /
                    1024,
                    2
                ),
                'Path': str(file_path)
            })

inventory_df = pd.DataFrame(
    inventory_rows
).sort_values(
    [
        'Type',
        'Filename'
    ]
)

inventory_df.to_csv(
    FIXED_REPORT_DIR /
    'fixed_asset_inventory.csv',
    index=False,
    encoding='utf-8-sig'
)

display(inventory_df)

# ------------------------------------------------------------
# Save failures
# ------------------------------------------------------------

failure_df = pd.DataFrame(
    failed_assets
)

failure_df.to_csv(
    FIXED_REPORT_DIR /
    'asset_generation_failures.csv',
    index=False,
    encoding='utf-8-sig'
)

print(
    "\nGenerated PNG figures:",
    len(
        list(
            FIXED_FIGURE_DIR.glob(
                '*.png'
            )
        )
    )
)

print(
    "Generated PNG tables:",
    len(
        list(
            FIXED_TABLE_PNG_DIR.glob(
                '*.png'
            )
        )
    )
)

print(
    "Generated HTML files:",
    len(
        list(
            FIXED_HTML_DIR.glob(
                '*.html'
            )
        )
    )
)

print(
    "Failures:",
    len(failed_assets)
)

# ------------------------------------------------------------
# Create final ZIP
# ------------------------------------------------------------

EXPORT_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Exports'
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

temporary_zip = shutil.make_archive(
    '/content/SCORE_BN_BANGLA_FIXED_REPORT_ASSETS',
    'zip',
    root_dir=FIXED_REPORT_DIR
)

final_zip = (
    EXPORT_DIR /
    'SCORE_BN_BANGLA_FIXED_REPORT_ASSETS.zip'
)

shutil.copy2(
    temporary_zip,
    final_zip
)

print(
    "\nFinal fixed ZIP:"
)

print(final_zip)

print(
    "ZIP size:",
    round(
        final_zip.stat().st_size
        /
        1024**2,
        2
    ),
    "MB"
)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 1
# Installation, imports and folders
# CPU is sufficient
# ============================================================

!pip -q install -U regex plotly kaleido playwright shap lime
!playwright install chromium

import os
import re
import gc
import json
import shutil
import random
import warnings
import subprocess
import unicodedata

from pathlib import Path
from collections import Counter
from itertools import combinations

import regex
import joblib
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

from google.colab import drive

warnings.filterwarnings("ignore")

# Mount Google Drive if it is not already mounted
if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Existing project directory
# ------------------------------------------------------------

BACKUP_DIR = Path(
    "/content/drive/MyDrive/SCORE_BN_Checkpoints"
)

assert BACKUP_DIR.exists(), (
    f"Project folder not found: {BACKUP_DIR}"
)

# ------------------------------------------------------------
# New clean corrected-results directory
# ------------------------------------------------------------

FINAL_DIR = (
    BACKUP_DIR /
    "final_corrected_assets"
)

FIGURE_HTML_DIR = FINAL_DIR / "figures_html"
FIGURE_PNG_DIR = FINAL_DIR / "figures_png"
FIGURE_SVG_DIR = FINAL_DIR / "figures_svg"
TABLE_DIR = FINAL_DIR / "tables"
INVALID_ARCHIVE_DIR = FINAL_DIR / "invalid_old_assets_archive"
METADATA_DIR = FINAL_DIR / "metadata"

for directory in [
    FINAL_DIR,
    FIGURE_HTML_DIR,
    FIGURE_PNG_DIR,
    FIGURE_SVG_DIR,
    TABLE_DIR,
    INVALID_ARCHIVE_DIR,
    METADATA_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# ------------------------------------------------------------
# Load saved dataset splits
# ------------------------------------------------------------

train_df = pd.read_csv(
    BACKUP_DIR / "train.csv"
)

val_df = pd.read_csv(
    BACKUP_DIR / "validation.csv"
)

test_df = pd.read_csv(
    BACKUP_DIR / "test.csv"
)

for frame in [
    train_df,
    val_df,
    test_df
]:
    frame["text"] = (
        frame["text"]
        .fillna("")
        .astype(str)
    )

    frame["label"] = (
        frame["label"]
        .astype(int)
    )

# Combine only for descriptive EDA
df_corrected = pd.concat(
    [
        train_df.assign(split="Training"),
        val_df.assign(split="Validation"),
        test_df.assign(split="Test")
    ],
    ignore_index=True
)

LABEL2ID = {
    "General Query": 0,
    "Routine": 1,
    "Urgent": 2,
    "Emergency": 3
}

ID2LABEL = {
    value: key
    for key, value in LABEL2ID.items()
}

CLASS_NAMES = [
    ID2LABEL[index]
    for index in range(4)
]

print("Everything loaded successfully.")
print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))
print("Total clean rows:", len(df_corrected))
print("Corrected output folder:", FINAL_DIR)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 2
# Bangla-safe normalization and tokenization
# ============================================================

# A basic stopword collection for descriptive word analysis.
# These words are not removed from transformer inputs.
BANGLA_STOPWORDS = {
    "আমি", "আমার", "আমাকে", "আমরা", "আমাদের",
    "তুমি", "তোমার", "আপনি", "আপনার",
    "সে", "তার", "তারা", "তাদের",
    "এই", "ঐ", "ওই", "এটা", "ওটা", "সেটা",
    "এটি", "ওটি", "সেটি",
    "একটি", "একটা", "কোনো", "কোন",
    "কি", "কী", "কেন", "কিভাবে",
    "এবং", "আর", "বা", "কিন্তু",
    "যে", "যদি", "তাহলে", "তবে",
    "এর", "ও", "তে", "কে", "থেকে",
    "জন্য", "সাথে", "মধ্যে", "উপর",
    "নিচে", "দিকে", "পরে", "আগে",
    "হয়", "হয়", "হচ্ছে", "হয়ে", "হয়ে",
    "হবে", "ছিল", "আছে", "ছিলো",
    "না", "নেই", "খুব", "একটু",
    "করে", "করতে", "করলে", "করার",
    "লাগে", "লাগছে", "গেছে",
    "ধরে", "মতো", "হঠাৎ"
}

# Bengali letters and their dependent signs remain together.
BANGLA_WORD_PATTERN = regex.compile(
    r"[\p{Script=Bengali}\p{M}]+"
)

LATIN_WORD_PATTERN = regex.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)


def normalize_bangla_text(text):
    """
    Unicode-normalize text without deleting meaningful Bangla
    characters or vowel signs.
    """

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    text = re.sub(
        r"@[A-Za-z0-9_]+",
        " USER ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def bangla_safe_tokenize(
    text,
    remove_stopwords=False,
    minimum_length=1
):
    """
    Extract complete Bengali-script words while retaining
    Bengali combining marks.
    """

    text = normalize_bangla_text(
        text
    )

    bangla_tokens = (
        BANGLA_WORD_PATTERN.findall(
            text
        )
    )

    latin_tokens = (
        LATIN_WORD_PATTERN.findall(
            text
        )
    )

    tokens = (
        bangla_tokens
        +
        latin_tokens
    )

    cleaned_tokens = []

    for token in tokens:

        token = token.strip()

        # Require at least one real alphabetic character
        has_letter = any(
            unicodedata.category(character).startswith("L")
            for character in token
        )

        if not has_letter:
            continue

        if len(token) < minimum_length:
            continue

        if (
            remove_stopwords
            and token in BANGLA_STOPWORDS
        ):
            continue

        cleaned_tokens.append(
            token
        )

    return cleaned_tokens


# ------------------------------------------------------------
# Tokenize the complete clean dataset
# ------------------------------------------------------------

df_corrected["normalized_text"] = (
    df_corrected["text"]
    .map(normalize_bangla_text)
)

df_corrected["safe_tokens"] = (
    df_corrected["normalized_text"]
    .map(
        lambda text: bangla_safe_tokenize(
            text,
            remove_stopwords=False
        )
    )
)

df_corrected["content_tokens"] = (
    df_corrected["normalized_text"]
    .map(
        lambda text: bangla_safe_tokenize(
            text,
            remove_stopwords=True
        )
    )
)

# ------------------------------------------------------------
# Validate the tokenizer
# ------------------------------------------------------------

sample_texts = df_corrected["text"].sample(
    n=min(10, len(df_corrected)),
    random_state=SEED
)

tokenizer_check = pd.DataFrame({
    "Original Text": sample_texts.values,
    "Bangla-safe Tokens": [
        " | ".join(
            bangla_safe_tokenize(text)
        )
        for text in sample_texts
    ]
})

display(tokenizer_check)

tokenizer_check.to_csv(
    TABLE_DIR /
    "table_tokenizer_quality_check.csv",
    index=False,
    encoding="utf-8-sig"
)

# Detect suspicious one-character fragments
all_safe_tokens = [
    token
    for token_list in df_corrected["safe_tokens"]
    for token in token_list
]

single_character_tokens = [
    token
    for token in all_safe_tokens
    if len(token) == 1
]

print("Total tokens:", len(all_safe_tokens))
print("Vocabulary size:", len(set(all_safe_tokens)))
print(
    "Single-character token percentage:",
    round(
        100 * len(single_character_tokens)
        / max(len(all_safe_tokens), 1),
        3
    ),
    "%"
)

print("\nExample corrected tokens:")
print(all_safe_tokens[:50])

# ============================================================
# CORRECTED FINAL ASSETS — CELL 3
# Correct word, n-gram and co-occurrence calculations
# ============================================================

def count_ngrams(
    tokenized_documents,
    n,
    top_k=25
):
    """
    Calculate n-grams directly from already-tokenized Bengali
    words. This avoids sklearn's incorrect default Bangla
    token boundary behaviour.
    """

    counter = Counter()

    for tokens in tokenized_documents:

        if len(tokens) < n:
            continue

        document_ngrams = zip(
            *[
                tokens[position:]
                for position in range(n)
            ]
        )

        counter.update(
            " ".join(ngram)
            for ngram in document_ngrams
        )

    result = pd.DataFrame(
        counter.most_common(top_k),
        columns=[
            "N-gram",
            "Frequency"
        ]
    )

    return result


# ------------------------------------------------------------
# Word frequencies
# ------------------------------------------------------------

content_word_counter = Counter(
    token
    for token_list in df_corrected["content_tokens"]
    for token in token_list
)

stopword_counter = Counter(
    token
    for token_list in df_corrected["safe_tokens"]
    for token in token_list
    if token in BANGLA_STOPWORDS
)

top_content_words = pd.DataFrame(
    content_word_counter.most_common(25),
    columns=[
        "Word",
        "Frequency"
    ]
)

top_stopwords = pd.DataFrame(
    stopword_counter.most_common(25),
    columns=[
        "Stopword",
        "Frequency"
    ]
)

# ------------------------------------------------------------
# Bangla-safe n-grams
# Use non-stopword tokens for more meaningful descriptive EDA
# ------------------------------------------------------------

unigram_df = count_ngrams(
    df_corrected["content_tokens"],
    n=1,
    top_k=25
)

bigram_df = count_ngrams(
    df_corrected["content_tokens"],
    n=2,
    top_k=25
)

trigram_df = count_ngrams(
    df_corrected["content_tokens"],
    n=3,
    top_k=25
)

# ------------------------------------------------------------
# Document-level co-occurrence
# ------------------------------------------------------------

TOP_COOCCURRENCE_WORDS = [
    word
    for word, frequency
    in content_word_counter.most_common(15)
]

cooccurrence_counter = Counter()

for token_list in df_corrected["content_tokens"]:

    # Count a word once per document for document-level
    # co-occurrence
    document_words = set(
        token
        for token in token_list
        if token in TOP_COOCCURRENCE_WORDS
    )

    for first_word, second_word in combinations(
        sorted(document_words),
        2
    ):
        cooccurrence_counter[
            (
                first_word,
                second_word
            )
        ] += 1

cooccurrence_matrix = pd.DataFrame(
    0,
    index=TOP_COOCCURRENCE_WORDS,
    columns=TOP_COOCCURRENCE_WORDS,
    dtype=int
)

for (
    first_word,
    second_word
), frequency in cooccurrence_counter.items():

    cooccurrence_matrix.loc[
        first_word,
        second_word
    ] = frequency

    cooccurrence_matrix.loc[
        second_word,
        first_word
    ] = frequency

np.fill_diagonal(
    cooccurrence_matrix.values,
    0
)

# Ensure that the matrix is not incorrectly all zero
assert cooccurrence_matrix.values.sum() > 0, (
    "Co-occurrence matrix is unexpectedly all zero."
)

# ------------------------------------------------------------
# Save numerical tables
# ------------------------------------------------------------

tables_to_save = {
    "table_top_content_words.csv":
        top_content_words,

    "table_top_stopwords.csv":
        top_stopwords,

    "table_safe_unigrams.csv":
        unigram_df,

    "table_safe_bigrams.csv":
        bigram_df,

    "table_safe_trigrams.csv":
        trigram_df,

    "table_safe_cooccurrence_matrix.csv":
        cooccurrence_matrix.reset_index(
            names="Word"
        )
}

for filename, table in tables_to_save.items():

    table.to_csv(
        TABLE_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Saved:",
        TABLE_DIR / filename
    )

print(
    "\nCo-occurrence matrix sum:",
    int(cooccurrence_matrix.values.sum())
)

display(top_content_words)
display(bigram_df)
display(trigram_df)
display(cooccurrence_matrix)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 4
# Build browser-renderable Plotly figures
# ============================================================

BANGLA_FONT_FAMILY = (
    "Noto Sans Bengali, "
    "Nirmala UI, "
    "Vrinda, "
    "Arial Unicode MS, "
    "sans-serif"
)

CLASS_COLORS = {
    "General Query": "#1685E5",
    "Routine": "#B94BD3",
    "Urgent": "#FF0051",
    "Emergency": "#00A02B"
}


def apply_browser_layout(
    figure,
    title,
    width=1200,
    height=850
):
    """
    Apply a browser-compatible font stack. The browser performs
    Bengali shaping instead of Matplotlib.
    """

    figure.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center"
        },
        width=width,
        height=height,
        template="plotly_white",
        font={
            "family": BANGLA_FONT_FAMILY,
            "size": 17,
            "color": "#202124"
        },
        margin={
            "l": 230,
            "r": 80,
            "t": 100,
            "b": 100
        }
    )

    figure.update_xaxes(
        tickfont={
            "family": BANGLA_FONT_FAMILY,
            "size": 15
        },
        title_font={
            "family": BANGLA_FONT_FAMILY,
            "size": 17
        }
    )

    figure.update_yaxes(
        tickfont={
            "family": BANGLA_FONT_FAMILY,
            "size": 16
        },
        title_font={
            "family": BANGLA_FONT_FAMILY,
            "size": 17
        }
    )

    return figure


def save_plot_html(
    figure,
    filename
):
    """
    Save self-contained browser-renderable HTML.
    """

    output_path = (
        FIGURE_HTML_DIR /
        f"{filename}.html"
    )

    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displayModeBar": False,
            "responsive": False
        }
    )

    print(
        "HTML created:",
        output_path.name
    )

    return output_path


def create_frequency_figure(
    table,
    label_column,
    title,
    color,
    filename
):

    plot_table = table.sort_values(
        "Frequency",
        ascending=True
    )

    figure = go.Figure(
        go.Bar(
            x=plot_table["Frequency"],
            y=plot_table[label_column],
            orientation="h",
            marker_color=color,
            text=plot_table["Frequency"],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Frequency: %{x}"
                "<extra></extra>"
            )
        )
    )

    apply_browser_layout(
        figure,
        title,
        width=1250,
        height=900
    )

    figure.update_xaxes(
        title="Frequency"
    )

    figure.update_yaxes(
        title=label_column
    )

    save_plot_html(
        figure,
        filename
    )


# ------------------------------------------------------------
# Word and stopword figures
# ------------------------------------------------------------

create_frequency_figure(
    top_content_words,
    "Word",
    "Most Frequent Bangla Content Words",
    "#2A9D8F",
    "figure_01_bangla_content_words"
)

create_frequency_figure(
    top_stopwords,
    "Stopword",
    "Most Frequent Bangla Stopwords",
    "#457B9D",
    "figure_02_bangla_stopwords"
)

# ------------------------------------------------------------
# N-gram figures
# ------------------------------------------------------------

create_frequency_figure(
    unigram_df,
    "N-gram",
    "Top Bangla-safe Unigrams",
    "#6A4C93",
    "figure_03_bangla_safe_unigrams"
)

create_frequency_figure(
    bigram_df,
    "N-gram",
    "Top Bangla-safe Bigrams",
    "#E76F51",
    "figure_04_bangla_safe_bigrams"
)

create_frequency_figure(
    trigram_df,
    "N-gram",
    "Top Bangla-safe Trigrams",
    "#F4A261",
    "figure_05_bangla_safe_trigrams"
)

# ------------------------------------------------------------
# Correct co-occurrence heatmap
# ------------------------------------------------------------

heatmap_figure = go.Figure(
    go.Heatmap(
        z=cooccurrence_matrix.values,
        x=cooccurrence_matrix.columns.tolist(),
        y=cooccurrence_matrix.index.tolist(),
        colorscale="Blues",
        text=cooccurrence_matrix.values,
        texttemplate="%{text}",
        hovertemplate=(
            "%{y} + %{x}<br>"
            "Co-occurring documents: %{z}"
            "<extra></extra>"
        ),
        colorbar={
            "title": "Documents"
        }
    )
)

apply_browser_layout(
    heatmap_figure,
    "Bangla Word Co-occurrence Matrix",
    width=1250,
    height=1100
)

heatmap_figure.update_xaxes(
    tickangle=-45
)

save_plot_html(
    heatmap_figure,
    "figure_06_bangla_cooccurrence"
)

print(
    "\nAll corrected EDA HTML figures created."
)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 5
# Browser-rendered LIME
# ============================================================

possible_lime_files = [
    BACKUP_DIR /
    "explainability" /
    "lime_feature_weights.csv",

    BACKUP_DIR /
    "final_export" /
    "tables" /
    "lime_values.csv"
]

LIME_CSV_PATH = next(
    (
        path
        for path in possible_lime_files
        if path.exists()
    ),
    None
)

assert LIME_CSV_PATH is not None, (
    "Saved LIME feature-weight CSV was not found."
)

lime_df = pd.read_csv(
    LIME_CSV_PATH
)

print("LIME source:", LIME_CSV_PATH)
print("LIME columns:", lime_df.columns.tolist())

feature_candidates = [
    "word_or_phrase",
    "feature",
    "word",
    "token"
]

weight_candidates = [
    "importance_weight",
    "weight",
    "importance",
    "contribution"
]

feature_column = next(
    (
        column
        for column in lime_df.columns
        if column.lower() in feature_candidates
    ),
    None
)

weight_column = next(
    (
        column
        for column in lime_df.columns
        if column.lower() in weight_candidates
    ),
    None
)

assert feature_column is not None, (
    "Could not identify the LIME feature column."
)

assert weight_column is not None, (
    "Could not identify the LIME weight column."
)

lime_plot_df = (
    lime_df[
        [
            feature_column,
            weight_column
        ]
    ]
    .dropna()
    .copy()
)

lime_plot_df[feature_column] = (
    lime_plot_df[feature_column]
    .astype(str)
)

lime_plot_df[weight_column] = pd.to_numeric(
    lime_plot_df[weight_column],
    errors="coerce"
)

lime_plot_df = (
    lime_plot_df
    .dropna()
    .sort_values(
        weight_column,
        ascending=True
    )
    .tail(15)
)

lime_colors = [
    "#198754"
    if value >= 0
    else "#DC3545"
    for value in lime_plot_df[weight_column]
]

lime_figure = go.Figure(
    go.Bar(
        x=lime_plot_df[weight_column],
        y=lime_plot_df[feature_column],
        orientation="h",
        marker_color=lime_colors,
        text=[
            f"{value:.4f}"
            for value in lime_plot_df[
                weight_column
            ]
        ],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "LIME contribution: %{x:.4f}"
            "<extra></extra>"
        )
    )
)

apply_browser_layout(
    lime_figure,
    "LIME Explanation for a Bangla Healthcare Query",
    width=1300,
    height=850
)

lime_figure.update_xaxes(
    title="Contribution to Predicted Class",
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor="black"
)

lime_figure.update_yaxes(
    title="Word or Phrase"
)

save_plot_html(
    lime_figure,
    "figure_07_lime_browser_rendered"
)

lime_plot_df.to_csv(
    TABLE_DIR /
    "table_lime_explanation.csv",
    index=False,
    encoding="utf-8-sig"
)

display(lime_plot_df)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 6
# Bangla-safe word-level Logistic Regression and SHAP
# CPU is sufficient
# ============================================================

def identity_tokenizer(text):
    return bangla_safe_tokenize(
        text,
        remove_stopwords=False
    )


# Fit on training only
safe_word_vectorizer = TfidfVectorizer(
    tokenizer=identity_tokenizer,
    token_pattern=None,
    preprocessor=None,
    lowercase=False,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)

X_train_safe = safe_word_vectorizer.fit_transform(
    train_df["text"]
)

X_test_safe = safe_word_vectorizer.transform(
    test_df["text"]
)

safe_lr_model = LogisticRegression(
    max_iter=2500,
    class_weight="balanced",
    random_state=SEED
)

safe_lr_model.fit(
    X_train_safe,
    train_df["label"].values
)

safe_lr_probabilities = safe_lr_model.predict_proba(
    X_test_safe
)

safe_lr_predictions = safe_lr_probabilities.argmax(
    axis=1
)

safe_lr_precision, safe_lr_recall, safe_lr_f1, _ = (
    precision_recall_fscore_support(
        test_df["label"].values,
        safe_lr_predictions,
        average="macro",
        zero_division=0
    )
)

safe_lr_metrics = {
    "model":
        "Bangla-safe Word Logistic Regression (XAI)",

    "accuracy":
        accuracy_score(
            test_df["label"].values,
            safe_lr_predictions
        ),

    "macro_precision":
        safe_lr_precision,

    "macro_recall":
        safe_lr_recall,

    "macro_f1":
        safe_lr_f1,

    "roc_auc_ovr":
        roc_auc_score(
            test_df["label"].values,
            safe_lr_probabilities,
            multi_class="ovr",
            average="macro"
        )
}

print("Bangla-safe LR metrics:")
display(
    pd.DataFrame(
        [safe_lr_metrics]
    )
)

# Save the explainability model
joblib.dump(
    {
        "vectorizer": safe_word_vectorizer,
        "model": safe_lr_model
    },
    FINAL_DIR /
    "bangla_safe_xai_logistic_regression.joblib"
)

# ------------------------------------------------------------
# SHAP calculation
# ------------------------------------------------------------

BACKGROUND_SIZE = min(
    200,
    X_train_safe.shape[0]
)

EXPLANATION_SIZE = min(
    100,
    X_test_safe.shape[0]
)

background_indices = np.random.RandomState(
    SEED
).choice(
    X_train_safe.shape[0],
    size=BACKGROUND_SIZE,
    replace=False
)

explanation_indices = np.arange(
    EXPLANATION_SIZE
)

X_background = X_train_safe[
    background_indices
]

X_explain = X_test_safe[
    explanation_indices
]

shap_explainer = shap.LinearExplainer(
    safe_lr_model,
    X_background
)

shap_values = shap_explainer.shap_values(
    X_explain
)

feature_names = np.asarray(
    safe_word_vectorizer.get_feature_names_out()
)

# Convert SHAP results into one matrix per class
if isinstance(shap_values, list):

    shap_by_class = [
        np.asarray(values)
        for values in shap_values
    ]

else:

    shap_array = np.asarray(
        shap_values
    )

    if shap_array.ndim == 3:

        shap_by_class = [
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

        # Fallback for binary-like return formats
        shap_by_class = [
            shap_array
        ]

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

mean_absolute_shap = np.column_stack([
    np.abs(class_values).mean(
        axis=0
    )
    for class_values in shap_by_class
])

total_feature_importance = (
    mean_absolute_shap.sum(
        axis=1
    )
)

TOP_SHAP_FEATURES = 20

top_shap_indices = np.argsort(
    total_feature_importance
)[
    -TOP_SHAP_FEATURES:
]

top_shap_features = feature_names[
    top_shap_indices
]

# ------------------------------------------------------------
# Save SHAP numerical table
# ------------------------------------------------------------

shap_table = pd.DataFrame({
    "Feature": top_shap_features,
    "Total Mean Absolute SHAP":
        total_feature_importance[
            top_shap_indices
        ]
})

for class_index in range(
    mean_absolute_shap.shape[1]
):

    class_name = (
        CLASS_NAMES[class_index]
        if class_index < len(CLASS_NAMES)
        else f"Class {class_index}"
    )

    shap_table[
        f"{class_name} Mean Absolute SHAP"
    ] = mean_absolute_shap[
        top_shap_indices,
        class_index
    ]

shap_table = shap_table.sort_values(
    "Total Mean Absolute SHAP",
    ascending=False
)

shap_table.to_csv(
    TABLE_DIR /
    "table_shap_bangla_safe_features.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Browser-renderable SHAP figure
# ------------------------------------------------------------

shap_figure = go.Figure()

shap_colors = [
    "#1685E5",
    "#B94BD3",
    "#FF0051",
    "#00A02B"
]

for class_index in range(
    mean_absolute_shap.shape[1]
):

    class_name = (
        CLASS_NAMES[class_index]
        if class_index < len(CLASS_NAMES)
        else f"Class {class_index}"
    )

    shap_figure.add_trace(
        go.Bar(
            name=class_name,
            x=mean_absolute_shap[
                top_shap_indices,
                class_index
            ],
            y=top_shap_features,
            orientation="h",
            marker_color=shap_colors[
                class_index %
                len(shap_colors)
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Class: {class_name}<br>"
                "Mean |SHAP|: %{x:.6f}"
                "<extra></extra>"
            )
        )
    )

apply_browser_layout(
    shap_figure,
    "Global SHAP Importance — Bangla-safe Logistic Regression",
    width=1400,
    height=1000
)

shap_figure.update_layout(
    barmode="stack",
    legend_title="Severity Class"
)

shap_figure.update_xaxes(
    title="Mean Absolute SHAP Value"
)

shap_figure.update_yaxes(
    title="Bangla Word or Phrase"
)

save_plot_html(
    shap_figure,
    "figure_08_shap_browser_rendered"
)

display(shap_table.head(20))

# Free unnecessary memory
del X_background
del X_explain
gc.collect()

# ============================================================
# CORRECTED FINAL ASSETS — CELL 7
# Browser-render every Plotly HTML into PNG and SVG
# ============================================================

renderer_code = r'''
import sys
import base64
import urllib.parse

from pathlib import Path
from playwright.sync_api import sync_playwright


html_directory = Path(sys.argv[1])
png_directory = Path(sys.argv[2])
svg_directory = Path(sys.argv[3])

png_directory.mkdir(
    parents=True,
    exist_ok=True
)

svg_directory.mkdir(
    parents=True,
    exist_ok=True
)

html_files = sorted(
    html_directory.glob("*.html")
)

print(
    f"Found {len(html_files)} HTML figures."
)

failures = []

with sync_playwright() as playwright:

    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--font-render-hinting=none"
        ]
    )

    page = browser.new_page(
        viewport={
            "width": 1800,
            "height": 1400
        },
        device_scale_factor=2
    )

    for html_path in html_files:

        try:

            page.goto(
                html_path.resolve().as_uri(),
                wait_until="networkidle",
                timeout=120000
            )

            page.wait_for_selector(
                ".plotly-graph-div",
                timeout=120000
            )

            page.wait_for_timeout(
                1500
            )

            graph = page.locator(
                ".plotly-graph-div"
            ).first

            png_path = (
                png_directory /
                f"{html_path.stem}.png"
            )

            graph.screenshot(
                path=str(png_path),
                type="png"
            )

            svg_data_url = page.evaluate(
                """
                async () => {
                    const graph = document.querySelector(
                        '.plotly-graph-div'
                    );

                    return await Plotly.toImage(
                        graph,
                        {
                            format: 'svg'
                        }
                    );
                }
                """
            )

            header, encoded_data = (
                svg_data_url.split(
                    ",",
                    1
                )
            )

            if ";base64" in header:

                svg_bytes = base64.b64decode(
                    encoded_data
                )

                svg_text = svg_bytes.decode(
                    "utf-8"
                )

            else:

                svg_text = urllib.parse.unquote(
                    encoded_data
                )

            svg_path = (
                svg_directory /
                f"{html_path.stem}.svg"
            )

            svg_path.write_text(
                svg_text,
                encoding="utf-8"
            )

            print(
                "Rendered:",
                html_path.stem
            )

        except Exception as error:

            failures.append(
                {
                    "file": html_path.name,
                    "error": str(error)
                }
            )

            print(
                "FAILED:",
                html_path.name,
                error
            )

    browser.close()

failure_path = (
    html_directory.parent /
    "metadata" /
    "browser_rendering_failures.json"
)

failure_path.write_text(
    __import__("json").dumps(
        failures,
        indent=2
    ),
    encoding="utf-8"
)

print(
    f"Rendering finished with "
    f"{len(failures)} failure(s)."
)
'''

renderer_path = Path(
    "/content/render_plotly_browser.py"
)

renderer_path.write_text(
    renderer_code,
    encoding="utf-8"
)

render_process = subprocess.run(
    [
        "python",
        str(renderer_path),
        str(FIGURE_HTML_DIR),
        str(FIGURE_PNG_DIR),
        str(FIGURE_SVG_DIR)
    ],
    capture_output=True,
    text=True,
    timeout=1200
)

print(render_process.stdout)

if render_process.stderr:
    print(
        "Renderer messages:\n",
        render_process.stderr[-4000:]
    )

if render_process.returncode != 0:
    raise RuntimeError(
        "Browser rendering failed. "
        "Check the messages printed above."
    )

png_files = sorted(
    FIGURE_PNG_DIR.glob("*.png")
)

svg_files = sorted(
    FIGURE_SVG_DIR.glob("*.svg")
)

print("\nPNG files created:", len(png_files))
print("SVG files created:", len(svg_files))

assert len(png_files) > 0, (
    "No PNG figures were created."
)

assert len(svg_files) > 0, (
    "No SVG figures were created."
)

print("\nBrowser-rendered files:")
for path in png_files:
    print(path.name)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 8
# Rebuild complete model comparison
# ============================================================

comparison_rows = []


def add_result(
    model,
    stage,
    accuracy=np.nan,
    macro_precision=np.nan,
    macro_recall=np.nan,
    macro_f1=np.nan,
    roc_auc_ovr=np.nan,
    notes=""
):

    comparison_rows.append({
        "model": model,
        "stage": stage,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc_ovr": roc_auc_ovr,
        "notes": notes
    })


def clean_classical(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    text = re.sub(
        r"@[A-Za-z0-9_]+",
        " USER ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip().lower()

    return text


def evaluate_probabilities(
    model_name,
    stage,
    true_labels,
    probabilities,
    notes=""
):

    probabilities = np.asarray(
        probabilities
    )

    predictions = probabilities.argmax(
        axis=1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    try:

        auc = roc_auc_score(
            true_labels,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        auc = np.nan

    add_result(
        model=model_name,
        stage=stage,
        accuracy=accuracy_score(
            true_labels,
            predictions
        ),
        macro_precision=precision,
        macro_recall=recall,
        macro_f1=f1,
        roc_auc_ovr=auc,
        notes=notes
    )


y_test = test_df["label"].values

# ------------------------------------------------------------
# 1. Classical baseline models
# ------------------------------------------------------------

FEATURE_PATH = (
    BACKUP_DIR /
    "tfidf_features.joblib"
)

CLASSICAL_PATH = (
    BACKUP_DIR /
    "classical_models.joblib"
)

if (
    FEATURE_PATH.exists()
    and CLASSICAL_PATH.exists()
):

    try:

        classical_features = joblib.load(
            FEATURE_PATH
        )

        classical_models = joblib.load(
            CLASSICAL_PATH
        )

        X_test_classical = (
            classical_features.transform(
                test_df["text"].map(
                    clean_classical
                )
            )
        )

        for model_name, model in classical_models.items():

            try:

                probabilities = (
                    model.predict_proba(
                        X_test_classical
                    )
                )

                evaluate_probabilities(
                    model_name=model_name,
                    stage="Classical baseline",
                    true_labels=y_test,
                    probabilities=probabilities
                )

                print(
                    "Evaluated:",
                    model_name
                )

            except Exception as error:

                print(
                    "Could not evaluate",
                    model_name,
                    error
                )

    except Exception as error:

        print(
            "Classical model loading failed:",
            error
        )

# ------------------------------------------------------------
# 2. Tuned Linear SVM
# ------------------------------------------------------------

TUNED_SVM_PATH = (
    BACKUP_DIR /
    "tuned_linear_svm.joblib"
)

if (
    TUNED_SVM_PATH.exists()
    and "X_test_classical" in globals()
):

    try:

        tuned_svm = joblib.load(
            TUNED_SVM_PATH
        )

        tuned_svm_probabilities = (
            tuned_svm.predict_proba(
                X_test_classical
            )
        )

        evaluate_probabilities(
            "Tuned Linear SVM",
            "Tuned",
            y_test,
            tuned_svm_probabilities
        )

    except Exception as error:

        print(
            "Tuned SVM evaluation failed:",
            error
        )

# ------------------------------------------------------------
# 3. Tuned XGBoost
# ------------------------------------------------------------

TUNED_XGB_PATH = (
    BACKUP_DIR /
    "tuned_xgboost.joblib"
)

if (
    TUNED_XGB_PATH.exists()
    and "X_test_classical" in globals()
):

    try:

        tuned_xgb = joblib.load(
            TUNED_XGB_PATH
        )

        tuned_xgb_probabilities = (
            tuned_xgb.predict_proba(
                X_test_classical
            )
        )

        evaluate_probabilities(
            "Tuned XGBoost",
            "Tuned",
            y_test,
            tuned_xgb_probabilities
        )

    except Exception as error:

        print(
            "Tuned XGBoost evaluation failed:",
            error
        )

# ------------------------------------------------------------
# 4. CNN, BiLSTM and BiGRU
# ------------------------------------------------------------

deep_model_names = [
    "CNN",
    "BiLSTM",
    "BiGRU"
]

for model_name in deep_model_names:

    probability_candidates = [
        BACKUP_DIR /
        f"{model_name}_probabilities.npy",

        BACKUP_DIR /
        "final_export" /
        "arrays" /
        f"{model_name}_probabilities.npy"
    ]

    probability_path = next(
        (
            path
            for path in probability_candidates
            if path.exists()
        ),
        None
    )

    if probability_path is not None:

        probabilities = np.load(
            probability_path
        )

        evaluate_probabilities(
            model_name,
            "Deep learning",
            y_test,
            probabilities
        )

# ------------------------------------------------------------
# 5. BanglaBERT
# ------------------------------------------------------------

BERT_METRICS_PATH = (
    BACKUP_DIR /
    "banglabert_baseline" /
    "test_metrics.json"
)

if BERT_METRICS_PATH.exists():

    with open(
        BERT_METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        bert_metrics = json.load(
            file
        )

    def metric_value(
        dictionary,
        *possible_keys
    ):

        for key in possible_keys:

            if key in dictionary:
                return dictionary[key]

        return np.nan

    add_result(
        model="BanglaBERT",
        stage="Transformer baseline",

        accuracy=metric_value(
            bert_metrics,
            "eval_accuracy",
            "accuracy"
        ),

        macro_precision=metric_value(
            bert_metrics,
            "eval_macro_precision",
            "macro_precision"
        ),

        macro_recall=metric_value(
            bert_metrics,
            "eval_macro_recall",
            "macro_recall"
        ),

        macro_f1=metric_value(
            bert_metrics,
            "eval_macro_f1",
            "macro_f1"
        ),

        roc_auc_ovr=metric_value(
            bert_metrics,
            "eval_roc_auc_ovr",
            "roc_auc_ovr"
        ),

        notes=(
            "Memory-safe BanglaBERT baseline; "
            "previously omitted from final comparison."
        )
    )

    print(
        "BanglaBERT metrics added."
    )

else:

    print(
        "WARNING: BanglaBERT metrics JSON not found:",
        BERT_METRICS_PATH
    )

# ------------------------------------------------------------
# 6. SCORE-BN
# ------------------------------------------------------------

SCORE_METRICS_PATH = (
    BACKUP_DIR /
    "score_bn" /
    "test_metrics.json"
)

if SCORE_METRICS_PATH.exists():

    with open(
        SCORE_METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        score_metrics = json.load(
            file
        )

    add_result(
        model="SCORE-BN",
        stage="Proposed model",

        accuracy=score_metrics.get(
            "accuracy",
            np.nan
        ),

        macro_precision=score_metrics.get(
            "macro_precision",
            np.nan
        ),

        macro_recall=score_metrics.get(
            "macro_recall",
            np.nan
        ),

        macro_f1=score_metrics.get(
            "macro_f1",
            np.nan
        ),

        roc_auc_ovr=score_metrics.get(
            "roc_auc_ovr",
            np.nan
        ),

        notes=(
            "Ordinal and cross-script-consistent "
            "proposed model."
        )
    )

# ------------------------------------------------------------
# 7. Bangla-safe XAI Logistic Regression
# ------------------------------------------------------------

add_result(
    model=safe_lr_metrics["model"],
    stage="Explainability model",
    accuracy=safe_lr_metrics["accuracy"],
    macro_precision=safe_lr_metrics[
        "macro_precision"
    ],
    macro_recall=safe_lr_metrics[
        "macro_recall"
    ],
    macro_f1=safe_lr_metrics[
        "macro_f1"
    ],
    roc_auc_ovr=safe_lr_metrics[
        "roc_auc_ovr"
    ],
    notes=(
        "Word-level Bangla-safe model trained only "
        "for interpretable SHAP analysis."
    )
)

# ------------------------------------------------------------
# Final table
# ------------------------------------------------------------

final_comparison = pd.DataFrame(
    comparison_rows
)

# Remove repeated model-stage rows if any
final_comparison = (
    final_comparison
    .drop_duplicates(
        subset=[
            "model",
            "stage"
        ],
        keep="last"
    )
    .sort_values(
        "macro_f1",
        ascending=False,
        na_position="last"
    )
    .reset_index(drop=True)
)

numeric_columns = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "roc_auc_ovr"
]

final_comparison[
    numeric_columns
] = final_comparison[
    numeric_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)

display(
    final_comparison.style.format({
        column: "{:.4f}"
        for column in numeric_columns
    })
)

FINAL_COMPARISON_PATH = (
    TABLE_DIR /
    "table_final_model_comparison_corrected.csv"
)

final_comparison.to_csv(
    FINAL_COMPARISON_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nCorrected comparison saved to:",
    FINAL_COMPARISON_PATH
)

if (
    final_comparison[
        "model"
    ].eq("BanglaBERT").any()
):

    print(
        "VERIFIED: BanglaBERT is included."
    )

else:

    print(
        "WARNING: BanglaBERT is still missing."
    )

# ============================================================
# CORRECTED FINAL ASSETS — CELL 9
# Correct model-comparison figure
# ============================================================

comparison_plot_df = (
    final_comparison
    .dropna(
        subset=[
            "macro_f1"
        ]
    )
    .sort_values(
        "macro_f1",
        ascending=True
    )
)

comparison_colors = [
    "#D62828"
    if model == "BanglaBERT"
    else "#6A4C93"
    if model == "SCORE-BN"
    else "#457B9D"
    for model in comparison_plot_df[
        "model"
    ]
]

comparison_figure = go.Figure(
    go.Bar(
        x=comparison_plot_df[
            "macro_f1"
        ],
        y=comparison_plot_df[
            "model"
        ],
        orientation="h",
        marker_color=comparison_colors,
        text=[
            f"{value:.4f}"
            for value in comparison_plot_df[
                "macro_f1"
            ]
        ],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Macro-F1: %{x:.4f}"
            "<extra></extra>"
        )
    )
)

apply_browser_layout(
    comparison_figure,
    "Final Test-set Macro-F1 Comparison",
    width=1300,
    height=max(
        750,
        55 * len(comparison_plot_df)
    )
)

comparison_figure.update_xaxes(
    title="Macro-F1",
    range=[
        max(
            0,
            comparison_plot_df[
                "macro_f1"
            ].min() - 0.03
        ),
        min(
            1,
            comparison_plot_df[
                "macro_f1"
            ].max() + 0.03
        )
    ]
)

comparison_figure.update_yaxes(
    title="Model"
)

save_plot_html(
    comparison_figure,
    "figure_09_final_model_comparison"
)

print(
    "Comparison HTML created. "
    "Rerun Cell 7 once to render its PNG and SVG."
)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 10
# Identify and archive invalid/broken old assets
# ============================================================

invalid_file_candidates = [
    # Original broken or misleading Bangla figures
    BACKUP_DIR /
    "explainability" /
    "lime_explanation_final.png",

    BACKUP_DIR /
    "explainability" /
    "lime_example_bangla_fixed.png",

    BACKUP_DIR /
    "explainability" /
    "shap_global_summary.png",

    BACKUP_DIR /
    "report_assets" /
    "figures" /
    "figure_06_top_words.png",

    BACKUP_DIR /
    "report_assets" /
    "figures" /
    "figure_07_ngram_analysis.png",

    BACKUP_DIR /
    "report_assets" /
    "figures" /
    "figure_08_wordcloud.png",

    BACKUP_DIR /
    "report_assets" /
    "figures" /
    "figure_09_cooccurrence_heatmap.png"
]

invalid_records = []

for source_path in invalid_file_candidates:

    if not source_path.exists():
        continue

    destination_path = (
        INVALID_ARCHIVE_DIR /
        source_path.name
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    filename_lower = (
        source_path.name.lower()
    )

    if "cooccurrence" in filename_lower:

        reason = (
            "Invalid: previous matrix was all zero because "
            "of incompatible Bangla tokenization."
        )

    elif (
        "ngram" in filename_lower
        or "word" in filename_lower
    ):

        reason = (
            "Invalid: previous sklearn token pattern split "
            "Bangla words around combining marks."
        )

    elif (
        "lime" in filename_lower
        or "shap" in filename_lower
    ):

        reason = (
            "Invalid for final report: Matplotlib did not "
            "perform Bengali text shaping correctly."
        )

    else:

        reason = (
            "Superseded by browser-rendered corrected asset."
        )

    invalid_records.append({
        "Original File": str(source_path),
        "Archived Copy": str(destination_path),
        "Status": "DO NOT USE",
        "Reason": reason
    })

invalid_manifest = pd.DataFrame(
    invalid_records
)

INVALID_MANIFEST_PATH = (
    INVALID_ARCHIVE_DIR /
    "INVALID_OLD_ASSETS_DO_NOT_USE.csv"
)

invalid_manifest.to_csv(
    INVALID_MANIFEST_PATH,
    index=False,
    encoding="utf-8-sig"
)

readme_text = """
INVALID / SUPERSEDED ASSETS — DO NOT USE IN THE REPORT

The files in this directory were copied here only for audit and
traceability.

Reasons include:
1. Incorrect Bangla token boundaries.
2. All-zero co-occurrence matrix.
3. Broken Bengali glyph shaping in Matplotlib.
4. Replacement by browser-rendered corrected versions.

Use only the files inside:
- figures_png
- figures_svg
- figures_html
- tables
"""

(
    INVALID_ARCHIVE_DIR /
    "README_DO_NOT_USE.txt"
).write_text(
    readme_text.strip(),
    encoding="utf-8"
)

display(invalid_manifest)

print(
    "\nInvalid-assets manifest:",
    INVALID_MANIFEST_PATH
)

print(
    "No original files were deleted."
)

# ============================================================
# CORRECTED FINAL ASSETS — CELL 11
# Verify and export everything
# ============================================================

# Rerun browser renderer so any recently generated comparison
# HTML is also converted to PNG and SVG.
render_process = subprocess.run(
    [
        "python",
        str(renderer_path),
        str(FIGURE_HTML_DIR),
        str(FIGURE_PNG_DIR),
        str(FIGURE_SVG_DIR)
    ],
    capture_output=True,
    text=True,
    timeout=1200
)

print(render_process.stdout)

if render_process.returncode != 0:
    print(render_process.stderr)
    raise RuntimeError(
        "Final browser rendering failed."
    )

# ------------------------------------------------------------
# Build inventory
# ------------------------------------------------------------

inventory_rows = []

for asset_type, directory, pattern in [
    (
        "Browser-rendered PNG",
        FIGURE_PNG_DIR,
        "*.png"
    ),
    (
        "Browser-rendered SVG",
        FIGURE_SVG_DIR,
        "*.svg"
    ),
    (
        "Interactive HTML",
        FIGURE_HTML_DIR,
        "*.html"
    ),
    (
        "Numerical Table",
        TABLE_DIR,
        "*.csv"
    )
]:

    for path in sorted(
        directory.glob(pattern)
    ):

        inventory_rows.append({
            "Asset Type": asset_type,
            "Filename": path.name,
            "Size KB": round(
                path.stat().st_size /
                1024,
                2
            ),
            "Path": str(path)
        })

inventory_df = pd.DataFrame(
    inventory_rows
)

inventory_df.to_csv(
    METADATA_DIR /
    "corrected_asset_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

display(inventory_df)

# ------------------------------------------------------------
# Automated checks
# ------------------------------------------------------------

required_png_stems = [
    "figure_01_bangla_content_words",
    "figure_02_bangla_stopwords",
    "figure_03_bangla_safe_unigrams",
    "figure_04_bangla_safe_bigrams",
    "figure_05_bangla_safe_trigrams",
    "figure_06_bangla_cooccurrence",
    "figure_07_lime_browser_rendered",
    "figure_08_shap_browser_rendered",
    "figure_09_final_model_comparison"
]

verification_rows = []

for stem in required_png_stems:

    png_path = (
        FIGURE_PNG_DIR /
        f        f"{stem}.png"
    )

    svg_path = (
        FIGURE_SVG_DIR /
        f"{stem}.svg"
    )

    verification_rows.append({
        "Asset": stem,
        "PNG Exists": png_path.exists(),
        "PNG Size KB": (
            round(
                png_path.stat().st_size /
                1024,
                2
            )
            if png_path.exists()
            else 0
        ),
        "SVG Exists": svg_path.exists(),
        "SVG Size KB": (
            round(
                svg_path.stat().st_size /
                1024,
                2
            )
            if svg_path.exists()
            else 0
        )
    })

verification_df = pd.DataFrame(
    verification_rows
)

display(verification_df)

verification_df.to_csv(
    METADATA_DIR /
    "final_asset_verification.csv",
    index=False,
    encoding="utf-8-sig"
)

# Ensure every required file exists
assert verification_df[
    "PNG Exists"
].all(), (
    "At least one required PNG is missing."
)

assert verification_df[
    "SVG Exists"
].all(), (
    "At least one required SVG is missing."
)

assert final_comparison[
    "model"
].eq("BanglaBERT").any(), (
    "BanglaBERT is missing from the final comparison."
)

assert cooccurrence_matrix.values.sum() > 0, (
    "Co-occurrence matrix is invalid."
)

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary = {
    "clean_rows":
        int(len(df_corrected)),

    "training_rows":
        int(len(train_df)),

    "validation_rows":
        int(len(val_df)),

    "test_rows":
        int(len(test_df)),

    "bangla_safe_vocabulary_size":
        int(len(set(all_safe_tokens))),

    "cooccurrence_matrix_sum":
        int(cooccurrence_matrix.values.sum()),

    "png_figures":
        int(
            len(
                list(
                    FIGURE_PNG_DIR.glob(
                        "*.png"
                    )
                )
            )
        ),

    "svg_figures":
        int(
            len(
                list(
                    FIGURE_SVG_DIR.glob(
                        "*.svg"
                    )
                )
            )
        ),

    "html_figures":
        int(
            len(
                list(
                    FIGURE_HTML_DIR.glob(
                        "*.html"
                    )
                )
            )
        ),

    "csv_tables":
        int(
            len(
                list(
                    TABLE_DIR.glob(
                        "*.csv"
                    )
                )
            )
        ),

    "banglabert_in_comparison":
        bool(
            final_comparison[
                "model"
            ].eq(
                "BanglaBERT"
            ).any()
        )
}

with open(
    METADATA_DIR /
    "final_correction_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False
    )

# ------------------------------------------------------------
# Create ZIP in Google Drive
# ------------------------------------------------------------

EXPORT_DIR = Path(
    "/content/drive/MyDrive/SCORE_BN_Exports"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

temporary_zip = shutil.make_archive(
    "/content/SCORE_BN_FINAL_CORRECTED_ASSETS",
    "zip",
    root_dir=FINAL_DIR
)

FINAL_ZIP_PATH = (
    EXPORT_DIR /
    "SCORE_BN_FINAL_CORRECTED_ASSETS.zip"
)

shutil.copy2(
    temporary_zip,
    FINAL_ZIP_PATH
)

print("\n" + "=" * 70)
print("CORRECTION AND EXPORT COMPLETED")
print("=" * 70)

print(
    "Corrected assets folder:",
    FINAL_DIR
)

print(
    "Final ZIP:",
    FINAL_ZIP_PATH
)

print(
    "ZIP size:",
    round(
        FINAL_ZIP_PATH.stat().st_size /
        1024**2,
        2
    ),
    "MB"
)

print("\nSummary:")
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
)

print(
    "\nOnly use assets from final_corrected_assets "
    "in your report."
)

