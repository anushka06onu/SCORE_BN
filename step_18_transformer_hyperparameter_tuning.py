"""## 18. Transformer hyperparameter tuning — T4 GPU

For the course, run three trials. Increase trials later for the conference extension.
"""

import gc
import torch

if 'baseline_model' in globals():
    del baseline_model

if 'trainer' in globals():
    del trainer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

free_memory, total_memory = torch.cuda.mem_get_info()

print(
    "Free GPU memory:",
    round(free_memory / 1024**3, 2),
    "GB"
)

import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("SEED defined:", SEED)

required_variables = [
    'SEED',
    'BACKUP_DIR',
    'MODEL_NAME',
    'ID2LABEL',
    'LABEL2ID',
    'train_ds',
    'val_ds',
    'tokenizer',
    'data_collator',
    'compute_metrics_transformer'
]

for variable in required_variables:
    print(
        variable,
        "available"
        if variable in globals()
        else "MISSING"
    )

import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Basic settings
# ------------------------------------------------------------

SEED = 42
MODEL_NAME = 'csebuetnlp/banglabert'
MAX_LENGTH = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 2. Google Drive paths
# ------------------------------------------------------------

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

assert BACKUP_DIR.exists(), (
    f"Checkpoint folder not found: {BACKUP_DIR}"
)

# ------------------------------------------------------------
# 3. Reload saved dataset splits
# ------------------------------------------------------------

train_df = pd.read_csv(
    BACKUP_DIR / 'train.csv'
)

val_df = pd.read_csv(
    BACKUP_DIR / 'validation.csv'
)

test_df = pd.read_csv(
    BACKUP_DIR / 'test.csv'
)

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

print("Training DataFrame:", train_df.shape)
print("Validation DataFrame:", val_df.shape)
print("Test DataFrame:", test_df.shape)

# ------------------------------------------------------------
# 4. Label mappings
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 5. Reload tokenizer
# ------------------------------------------------------------

BASELINE_FINAL_DIR = (
    BACKUP_DIR /
    'banglabert_baseline' /
    'final_model'
)

if BASELINE_FINAL_DIR.exists():

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASELINE_FINAL_DIR)
    )

    print(
        "Tokenizer loaded from saved baseline model."
    )

else:

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        "Tokenizer loaded from Hugging Face."
    )

# ------------------------------------------------------------
# 6. Recreate Hugging Face datasets
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

# ------------------------------------------------------------
# 7. Dynamic-padding collator
# ------------------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# Keep the old notebook variable name available as well
collator = data_collator

# ------------------------------------------------------------
# 8. Transformer evaluation function
# ------------------------------------------------------------

def compute_metrics_transformer(
    eval_prediction
):

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
# 9. Final verification
# ------------------------------------------------------------

print("\nHugging Face datasets recreated:")

print("Training samples:", len(train_ds))
print("Validation samples:", len(val_ds))
print("Test samples:", len(test_ds))

required_variables = [
    'SEED',
    'BACKUP_DIR',
    'MODEL_NAME',
    'ID2LABEL',
    'LABEL2ID',
    'train_ds',
    'val_ds',
    'test_ds',
    'tokenizer',
    'data_collator',
    'compute_metrics_transformer'
]

print("\nRequired-variable check:")

for variable in required_variables:

    print(
        variable,
        "available"
        if variable in globals()
        else "MISSING"
    )

# ============================================================
# SECTION 18: MEMORY-SAFE BANGLABERT HYPERPARAMETER TUNING
# ============================================================

import gc
import json
import inspect
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ------------------------------------------------------------
# 1. Check required variables
# ------------------------------------------------------------

required_variables = [
    'SEED',
    'BACKUP_DIR',
    'MODEL_NAME',
    'LABEL2ID',
    'ID2LABEL',
    'train_ds',
    'val_ds',
    'tokenizer',
    'data_collator',
    'compute_metrics_transformer'
]

missing_variables = [
    variable
    for variable in required_variables
    if variable not in globals()
]

if missing_variables:
    raise NameError(
        f"Missing variables: {missing_variables}. "
        "Run the prerequisite/reloading cell first."
    )

print("All prerequisite variables are available.")

# ------------------------------------------------------------
# 2. Clear previously loaded models
# ------------------------------------------------------------

for variable_name in [
    'trainer',
    'baseline_model',
    'tuning_trainer',
    'tune_trainer',
    'score_model',
    'optimizer',
    'scaler'
]:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device_name = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else 'CPU'
)

print("Device:", device_name)

if torch.cuda.is_available():
    free_memory, total_memory = torch.cuda.mem_get_info()

    print(
        "Free GPU memory:",
        round(free_memory / 1024**3, 2),
        "GB out of",
        round(total_memory / 1024**3, 2),
        "GB"
    )

# ------------------------------------------------------------
# 3. Tuning output folders
# ------------------------------------------------------------

TRANSFORMER_TUNING_DIR = (
    BACKUP_DIR / 'transformer_tuning'
)

TRIAL_OUTPUT_DIR = (
    TRANSFORMER_TUNING_DIR / 'temporary_trials'
)

BEST_MODEL_DIR = (
    TRANSFORMER_TUNING_DIR / 'best_tuned_model'
)

TRANSFORMER_TUNING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRIAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 4. Three hyperparameter configurations
# ------------------------------------------------------------

# Three controlled trials are sufficient for the course
# hyperparameter-tuning requirement.

trial_configurations = [
    {
        'trial': 1,
        'learning_rate': 1e-5,
        'weight_decay': 0.01
    },
    {
        'trial': 2,
        'learning_rate': 2e-5,
        'weight_decay': 0.05
    },
    {
        'trial': 3,
        'learning_rate': 3e-5,
        'weight_decay': 0.01
    }
]

print("\nTransformer configurations:")

display(pd.DataFrame(trial_configurations))

# ------------------------------------------------------------
# 5. Version-compatible argument helper
# ------------------------------------------------------------

supported_training_parameters = (
    inspect.signature(
        TrainingArguments.__init__
    ).parameters
)

trainer_parameters = (
    inspect.signature(
        Trainer.__init__
    ).parameters
)

def create_training_arguments(
    trial_number,
    learning_rate,
    weight_decay
):

    settings = {
        'output_dir': str(
            TRIAL_OUTPUT_DIR /
            f'trial_{trial_number}'
        ),

        'learning_rate': learning_rate,
        'weight_decay': weight_decay,

        # Memory-safe physical batch sizes
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 8,

        # Effective training batch size = 4 × 4 = 16
        'gradient_accumulation_steps': 4,

        # Two epochs for each tuning trial
        'num_train_epochs': 2,

        # No model checkpoints during trial search
        'save_strategy': 'no',

        'logging_strategy': 'epoch',

        'fp16': torch.cuda.is_available(),

        'report_to': 'none',
        'seed': SEED,

        # Avoid retaining unnecessary prediction columns
        'remove_unused_columns': True
    }

    if (
        'eval_strategy'
        in supported_training_parameters
    ):
        settings['eval_strategy'] = 'epoch'

    elif (
        'evaluation_strategy'
        in supported_training_parameters
    ):
        settings['evaluation_strategy'] = 'epoch'

    if (
        'warmup_steps'
        in supported_training_parameters
    ):
        settings['warmup_steps'] = 20

    settings = {
        key: value
        for key, value in settings.items()
        if key in supported_training_parameters
    }

    return TrainingArguments(**settings)

# ------------------------------------------------------------
# 6. Train the three trials sequentially
# ------------------------------------------------------------

tuning_results = []
best_macro_f1 = -1.0
best_configuration = None

for configuration in trial_configurations:

    trial_number = configuration['trial']
    learning_rate = configuration['learning_rate']
    weight_decay = configuration['weight_decay']

    print("\n" + "=" * 70)
    print(f"STARTING TRANSFORMER TRIAL {trial_number}/3")
    print("Learning rate:", learning_rate)
    print("Weight decay:", weight_decay)
    print("=" * 70)

    # Clear memory before creating each new model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Every trial starts from the same pretrained BanglaBERT
    trial_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    )

    # Reduce activation-memory consumption
    if hasattr(
        trial_model,
        'gradient_checkpointing_enable'
    ):
        trial_model.gradient_checkpointing_enable()

    training_args = create_training_arguments(
        trial_number=trial_number,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    trainer_settings = {
        'model': trial_model,
        'args': training_args,
        'train_dataset': train_ds,
        'eval_dataset': val_ds,
        'data_collator': data_collator,
        'compute_metrics': compute_metrics_transformer
    }

    # Compatibility across Transformers versions
    if 'processing_class' in trainer_parameters:
        trainer_settings[
            'processing_class'
        ] = tokenizer

    elif 'tokenizer' in trainer_parameters:
        trainer_settings[
            'tokenizer'
        ] = tokenizer

    trial_trainer = Trainer(
        **trainer_settings
    )

    start_time = time.time()

    # Train the current trial
    trial_trainer.train()

    training_minutes = (
        time.time() - start_time
    ) / 60

    # Evaluate on validation only during tuning
    validation_metrics = (
        trial_trainer.evaluate(
            eval_dataset=val_ds
        )
    )

    validation_macro_f1 = float(
        validation_metrics[
            'eval_macro_f1'
        ]
    )

    result = {
        'trial': trial_number,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'epochs': 2,
        'physical_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'effective_batch_size': 16,
        'validation_accuracy': float(
            validation_metrics.get(
                'eval_accuracy',
                np.nan
            )
        ),
        'validation_macro_precision': float(
            validation_metrics.get(
                'eval_macro_precision',
                np.nan
            )
        ),
        'validation_macro_recall': float(
            validation_metrics.get(
                'eval_macro_recall',
                np.nan
            )
        ),
        'validation_macro_f1': (
            validation_macro_f1
        ),
        'validation_roc_auc_ovr': float(
            validation_metrics.get(
                'eval_roc_auc_ovr',
                np.nan
            )
        ),
        'training_minutes': (
            training_minutes
        )
    }

    tuning_results.append(result)

    print(
        f"\nTrial {trial_number} "
        f"validation macro-F1: "
        f"{validation_macro_f1:.4f}"
    )

    # Save results after every trial
    tuning_results_df = pd.DataFrame(
        tuning_results
    )

    tuning_results_df.to_csv(
        TRANSFORMER_TUNING_DIR /
        'all_trial_results.csv',
        index=False
    )

    with open(
        TRANSFORMER_TUNING_DIR /
        f'trial_{trial_number}_metrics.json',
        'w',
        encoding='utf-8'
    ) as file:
        json.dump(
            result,
            file,
            indent=2
        )

    # Preserve the best model found so far
    if validation_macro_f1 > best_macro_f1:

        best_macro_f1 = validation_macro_f1

        best_configuration = {
            'trial': trial_number,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }

        # Overwrite the previous best model
        BEST_MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        trial_trainer.save_model(
            str(BEST_MODEL_DIR)
        )

        tokenizer.save_pretrained(
            str(BEST_MODEL_DIR)
        )

        print(
            "This is the best trial so far. "
            "Model saved."
        )

    # Remove the current trial from GPU memory
    del trial_trainer
    del trial_model
    del training_args

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        free_memory, _ = (
            torch.cuda.mem_get_info()
        )

        print(
            "Free GPU memory after trial:",
            round(
                free_memory / 1024**3,
                2
            ),
            "GB"
        )

# ------------------------------------------------------------
# 7. Save best hyperparameters
# ------------------------------------------------------------

best_hyperparameter_summary = {
    'best_objective_macro_f1': (
        best_macro_f1
    ),
    'best_hyperparameters': {
        'learning_rate': (
            best_configuration[
                'learning_rate'
            ]
        ),
        'weight_decay': (
            best_configuration[
                'weight_decay'
            ]
        )
    },
    'best_trial': (
        best_configuration['trial']
    ),
    'number_of_trials': 3,
    'epochs_per_trial': 2,
    'physical_batch_size': 4,
    'gradient_accumulation_steps': 4,
    'effective_batch_size': 16
}

BEST_PARAMETERS_FILE = (
    TRANSFORMER_TUNING_DIR /
    'best_hyperparameters.json'
)

with open(
    BEST_PARAMETERS_FILE,
    'w',
    encoding='utf-8'
) as file:
    json.dump(
        best_hyperparameter_summary,
        file,
        indent=2
    )

# ------------------------------------------------------------
# 8. Display final tuning results
# ------------------------------------------------------------

tuning_results_df = (
    pd.DataFrame(tuning_results)
    .sort_values(
        'validation_macro_f1',
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("TRANSFORMER TUNING COMPLETED")
print("=" * 70)

display(tuning_results_df)

print("\nBest configuration:")
print(
    json.dumps(
        best_hyperparameter_summary,
        indent=2
    )
)

print("\nTuning file exists:")
print(BEST_PARAMETERS_FILE.exists())

print("\nTuning file saved to:")
print(BEST_PARAMETERS_FILE)

print("\nBest tuned model saved to:")
print(BEST_MODEL_DIR)

from pathlib import Path
import json

BEST_PARAMETERS_FILE = (
    BACKUP_DIR
    / 'transformer_tuning'
    / 'best_hyperparameters.json'
)

print("Tuning file exists:", BEST_PARAMETERS_FILE.exists())
print("Location:", BEST_PARAMETERS_FILE)

if BEST_PARAMETERS_FILE.exists():
    with open(
        BEST_PARAMETERS_FILE,
        'r',
        encoding='utf-8'
    ) as file:
        tuning_result = json.load(file)

    print("\nSaved tuning result:")
    print(json.dumps(tuning_result, indent=2))

