## 1. Environment setup — CPU

Run this once. A runtime restart is usually unnecessary.
"""

!pip -q install -U pandas numpy scikit-learn seaborn matplotlib wordcloud openpyxl \
  imbalanced-learn xgboost optuna shap lime rapidfuzz indic-transliteration \
  transformers datasets accelerate evaluate sentencepiece captum streamlit pyngrok

import os, re, json, random, warnings, glob, zipfile, shutil, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROJECT_DIR = Path('/content/score_bn')
for folder in ['data/raw','data/processed','data/splits','models','results','figures','app']:
    (PROJECT_DIR/folder).mkdir(parents=True, exist_ok=True)
print('Project folder:', PROJECT_DIR)

