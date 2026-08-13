"""## 17. Transformer baseline — T4 GPU required

Default: `csebuetnlp/banglabert`. If loading fails, use `xlm-roberta-base`.
"""

import gc
import tensorflow as tf
import torch

tf.keras.backend.clear_session()

# Remove large TensorFlow sequence tensors if they exist
for variable_name in [
    'Xtr_seq',
    'Xva_seq',
    'Xte_seq',
    'model',
    'deep_models'
]:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")

import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_NAME = 'csebuetnlp/banglabert'

BERT_SAVE_DIR = (
    BACKUP_DIR / 'banglabert_baseline'
)

BERT_SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "Device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ------------------------------------------------------------
# Convert pandas DataFrames to Hugging Face datasets
# ------------------------------------------------------------

def make_hf_dataset(frame):

    required_columns = [
        'text',
        'label'
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    temporary_frame = (
        frame[['text', 'label']]
        .copy()
        .reset_index(drop=True)
    )

    temporary_frame['text'] = (
        temporary_frame['text']
        .fillna('')
        .astype(str)
    )

    temporary_frame['label'] = (
        temporary_frame['label']
        .astype(int)
    )

    dataset = Dataset.from_pandas(
        temporary_frame,
        preserve_index=False
    )

    dataset = dataset.map(
        lambda batch: tokenizer(
            batch['text'],
            truncation=True,
            max_length=160
        ),
        batched=True,
        desc='Tokenizing'
    )

    return dataset


train_ds = make_hf_dataset(train_df)
val_ds = make_hf_dataset(val_df)
test_ds = make_hf_dataset(test_df)

print("Training samples:", len(train_ds))
print("Validation samples:", len(val_ds))
print("Test samples:", len(test_ds))

# Dynamic padding saves GPU memory
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------

def compute_metrics_transformer(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1
    ).numpy()

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average='macro',
            zero_division=0
        )
    )

    try:
        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        auc = float('nan')

    return {
        'accuracy': accuracy_score(
            labels,
            predictions
        ),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'roc_auc_ovr': auc
    }

# ------------------------------------------------------------
# Load BanglaBERT
# ------------------------------------------------------------

baseline_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
)

# The warning about newly initialized classifier weights is normal.
# The classification layer will be learned from this dataset.

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=str(
        BERT_SAVE_DIR / 'checkpoints'
    ),

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    num_train_epochs=5,

    weight_decay=0.01,
    warmup_ratio=0.1,

    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=25,

    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,

    # Use T4 mixed-precision training
    fp16=torch.cuda.is_available(),

    report_to='none',
    seed=SEED,

    save_total_limit=1
)

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

trainer = Trainer(
    model=baseline_model,
    args=training_args,

    train_dataset=train_ds,
    eval_dataset=val_ds,

    processing_class=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics_transformer,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

train_output = trainer.train()

print("\nTraining completed.")

# ------------------------------------------------------------
# Evaluate on untouched test set
# ------------------------------------------------------------

test_metrics = trainer.evaluate(
    eval_dataset=test_ds
)

print("\nBanglaBERT test results:")

display(
    pd.DataFrame(
        [test_metrics]
    )
)

# ------------------------------------------------------------
# Save model, tokenizer and results to Google Drive
# ------------------------------------------------------------

FINAL_MODEL_DIR = (
    BERT_SAVE_DIR / 'final_model'
)

trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)

with open(
    BERT_SAVE_DIR / 'test_metrics.json',
    'w'
) as file:
    json.dump(
        {
            key: float(value)
            if isinstance(
                value,
                (float, int, np.floating, np.integer)
            )
            else value
            for key, value in test_metrics.items()
        },
        file,
        indent=2
    )

trainer.save_state()

print("\nModel saved to:")
print(FINAL_MODEL_DIR)

import inspect
import json
import numpy as np
import pandas as pd
import torch

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# ------------------------------------------------------------
# Build arguments compatible with the installed Transformers
# version
# ------------------------------------------------------------

supported_parameters = inspect.signature(
    TrainingArguments.__init__
).parameters

training_settings = {
    'output_dir': str(
        BERT_SAVE_DIR / 'checkpoints'
    ),

    'learning_rate': 2e-5,

    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 32,

    'num_train_epochs': 5,

    'weight_decay': 0.01,

    'save_strategy': 'epoch',
    'logging_strategy': 'steps',
    'logging_steps': 25,

    'load_best_model_at_end': True,
    'metric_for_best_model': 'macro_f1',
    'greater_is_better': True,

    'fp16': torch.cuda.is_available(),

    'report_to': 'none',
    'seed': SEED,

    'save_total_limit': 1
}

# Different Transformers versions use different names.
if 'eval_strategy' in supported_parameters:

    training_settings[
        'eval_strategy'
    ] = 'epoch'

elif 'evaluation_strategy' in supported_parameters:

    training_settings[
        'evaluation_strategy'
    ] = 'epoch'

else:

    raise RuntimeError(
        "This Transformers version does not provide an "
        "evaluation-strategy argument."
    )

# Add warm-up only when the installed version supports it.
if 'warmup_ratio' in supported_parameters:

    training_settings[
        'warmup_ratio'
    ] = 0.1

elif 'warmup_steps' in supported_parameters:

    # Approximately 10% warm-up is not essential here.
    # A fixed value keeps this version compatible.
    training_settings[
        'warmup_steps'
    ] = 20

# Keep only settings supported by the installed version
training_settings = {
    name: value
    for name, value in training_settings.items()
    if name in supported_parameters
}

print("Training arguments being used:")

for name, value in training_settings.items():
    print(f"{name}: {value}")

training_args = TrainingArguments(
    **training_settings
)

# ------------------------------------------------------------
# Create Trainer
# ------------------------------------------------------------

trainer_parameters = inspect.signature(
    Trainer.__init__
).parameters

trainer_settings = {
    'model': baseline_model,
    'args': training_args,

    'train_dataset': train_ds,
    'eval_dataset': val_ds,

    'data_collator': data_collator,

    'compute_metrics': compute_metrics_transformer,

    'callbacks': [
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
}

# Compatibility between Transformers versions
if 'processing_class' in trainer_parameters:

    trainer_settings[
        'processing_class'
    ] = tokenizer

elif 'tokenizer' in trainer_parameters:

    trainer_settings[
        'tokenizer'
    ] = tokenizer

trainer = Trainer(
    **trainer_settings
)

# ------------------------------------------------------------
# Train BanglaBERT
# ------------------------------------------------------------

print("\nStarting BanglaBERT training...")

train_output = trainer.train()

print("\nBanglaBERT training completed.")

# ------------------------------------------------------------
# Evaluate using the untouched test dataset
# ------------------------------------------------------------

test_metrics = trainer.evaluate(
    eval_dataset=test_ds
)

print("\nBanglaBERT test results:")

display(
    pd.DataFrame(
        [test_metrics]
    )
)

# ------------------------------------------------------------
# Generate and save test predictions
# ------------------------------------------------------------

prediction_output = trainer.predict(
    test_ds
)

test_logits = prediction_output.predictions

test_probabilities = torch.softmax(
    torch.tensor(test_logits),
    dim=-1
).numpy()

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

np.save(
    BERT_SAVE_DIR / 'test_probabilities.npy',
    test_probabilities
)

np.save(
    BERT_SAVE_DIR / 'test_predictions.npy',
    test_predictions
)

# ------------------------------------------------------------
# Save model and tokenizer
# ------------------------------------------------------------

FINAL_MODEL_DIR = (
    BERT_SAVE_DIR / 'final_model'
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)

# Convert NumPy values before saving JSON
serializable_metrics = {}

for key, value in test_metrics.items():

    if isinstance(
        value,
        (
            np.floating,
            np.integer
        )
    ):
        serializable_metrics[key] = value.item()

    elif isinstance(
        value,
        (
            float,
            int,
            str,
            bool
        )
    ):
        serializable_metrics[key] = value

with open(
    BERT_SAVE_DIR / 'test_metrics.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        serializable_metrics,
        file,
        indent=2
    )

trainer.save_state()

print("\nBanglaBERT model saved successfully to:")
print(FINAL_MODEL_DIR)

from pathlib import Path

CHECKPOINT_DIR = Path(
    '/Users/fatehahossainanushka/SCORE_BN/checkpoints'
)

print("Files saved in Google Drive:")

for file_path in sorted(CHECKPOINT_DIR.rglob('*')):
    if file_path.is_file():
        print(file_path.relative_to(CHECKPOINT_DIR))

!pip -q install -U transformers datasets accelerate evaluate sentencepiece

import gc
import json
import inspect
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# from google.colab import drive

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# Mount Google Drive
# drive.mount('/content/drive')

# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()

print(
    "Device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

# Google Drive folders
BACKUP_DIR = Path(
    '/Users/fatehahossainanushka/SCORE_BN/checkpoints'
)

BERT_SAVE_DIR = (
    BACKUP_DIR / 'banglabert_baseline'
)

BERT_SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Reload saved dataset splits
train_df = pd.read_csv(
    BACKUP_DIR / 'train.csv'
)

val_df = pd.read_csv(
    BACKUP_DIR / 'validation.csv'
)

test_df = pd.read_csv(
    BACKUP_DIR / 'test.csv'
)

# Make sure labels are integers
for frame in [
    train_df,
    val_df,
    test_df
]:
    frame['text'] = (
        frame['text']
        .fillna('')
        .astype(str)
    )

    frame['label'] = (
        frame['label']
        .astype(int)
    )

LABEL2ID = {
    'General Query': 0,
    'Routine': 1,
    'Urgent': 2,
    'Emergency': 3
}

ID2LABEL = {
    0: 'General Query',
    1: 'Routine',
    2: 'Urgent',
    3: 'Emergency'
}

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print("\nFree GPU memory before BanglaBERT:")

free_memory, total_memory = torch.cuda.mem_get_info()

print(
    "Free:",
    round(free_memory / 1024**3, 2),
    "GB"
)

print(
    "Total:",
    round(total_memory / 1024**3, 2),
    "GB"
)

MODEL_NAME = 'csebuetnlp/banglabert'
MAX_LENGTH = 128

# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ------------------------------------------------------------
# Prepare Hugging Face datasets
# ------------------------------------------------------------

def make_hf_dataset(frame):

    temporary_frame = (
        frame[['text', 'label']]
        .copy()
        .reset_index(drop=True)
    )

    dataset = Dataset.from_pandas(
        temporary_frame,
        preserve_index=False
    )

    dataset = dataset.map(
        lambda batch: tokenizer(
            batch['text'],
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True,
        desc='Tokenizing'
    )

    return dataset


train_ds = make_hf_dataset(train_df)
val_ds = make_hf_dataset(val_df)
test_ds = make_hf_dataset(test_df)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print("Training samples:", len(train_ds))
print("Validation samples:", len(val_ds))
print("Test samples:", len(test_ds))

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def compute_metrics_transformer(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1
    ).numpy()

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average='macro',
            zero_division=0
        )
    )

    try:
        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        auc = float('nan')

    return {
        'accuracy': accuracy_score(
            labels,
            predictions
        ),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'roc_auc_ovr': auc
    }

# ------------------------------------------------------------
# Load BanglaBERT
# ------------------------------------------------------------

baseline_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
)

# Saves additional memory during training
baseline_model.gradient_checkpointing_enable()

# ------------------------------------------------------------
# Version-compatible training arguments
# ------------------------------------------------------------

supported_parameters = inspect.signature(
    TrainingArguments.__init__
).parameters

training_settings = {
    'output_dir': str(
        BERT_SAVE_DIR / 'checkpoints'
    ),

    'learning_rate': 2e-5,

    # Memory-safe T4 settings
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,

    # Effective training batch size = 4 × 4 = 16
    'gradient_accumulation_steps': 4,

    'num_train_epochs': 5,
    'weight_decay': 0.01,

    'save_strategy': 'epoch',
    'logging_strategy': 'steps',
    'logging_steps': 25,

    'load_best_model_at_end': True,
    'metric_for_best_model': 'macro_f1',
    'greater_is_better': True,

    'fp16': True,

    'report_to': 'none',
    'seed': SEED,
    'save_total_limit': 1
}

if 'eval_strategy' in supported_parameters:

    training_settings[
        'eval_strategy'
    ] = 'epoch'

else:

    training_settings[
        'evaluation_strategy'
    ] = 'epoch'

if 'warmup_steps' in supported_parameters:

    training_settings[
        'warmup_steps'
    ] = 20

training_settings = {
    key: value
    for key, value in training_settings.items()
    if key in supported_parameters
}

training_args = TrainingArguments(
    **training_settings
)

# ------------------------------------------------------------
# Version-compatible Trainer
# ------------------------------------------------------------

trainer_settings = {
    'model': baseline_model,
    'args': training_args,

    'train_dataset': train_ds,
    'eval_dataset': val_ds,

    'data_collator': data_collator,

    'compute_metrics': compute_metrics_transformer,

    'callbacks': [
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
}

trainer_parameters = inspect.signature(
    Trainer.__init__
).parameters

if 'processing_class' in trainer_parameters:

    trainer_settings[
        'processing_class'
    ] = tokenizer

elif 'tokenizer' in trainer_parameters:

    trainer_settings[
        'tokenizer'
    ] = tokenizer

trainer = Trainer(
    **trainer_settings
)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("Starting memory-safe BanglaBERT training...")

trainer.train()

print("Training completed.")

# ------------------------------------------------------------
# Test evaluation
# ------------------------------------------------------------

test_metrics = trainer.evaluate(
    test_ds
)

display(
    pd.DataFrame(
        [test_metrics]
    )
)

# ------------------------------------------------------------
# Save to Google Drive
# ------------------------------------------------------------

FINAL_MODEL_DIR = (
    BERT_SAVE_DIR / 'final_model'
)

trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)

serializable_metrics = {
    key: (
        value.item()
        if isinstance(
            value,
            (
                np.integer,
                np.floating
            )
        )
        else value
    )
    for key, value in test_metrics.items()
}

with open(
    BERT_SAVE_DIR / 'test_metrics.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        serializable_metrics,
        file,
        indent=2
    )

print("BanglaBERT saved to:")
print(FINAL_MODEL_DIR)

