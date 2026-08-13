"""## 20. SCORE-BN project-stage implementation — T4 GPU

The model uses:

- ordinary four-class classification;
- ordinal expected-severity penalty;
- Jensen-Shannon consistency between original and Romanized views;
- asymmetric under-prioritisation penalty.

This is the proposed component requiring an ablation study.
"""

# Commented out IPython magic to ensure Python compatibility.
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from torch.utils.data import (
    Dataset as TorchDataset,
    DataLoader
)

from transformers import AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 2. Release transformer-tuning models from GPU
# ------------------------------------------------------------

for variable_name in [
    'tuning_trainer',
    'tune_trainer',
    'trainer',
    'baseline_model'
]:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print(
    "Device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

if torch.cuda.is_available():

    free_memory, total_memory = (
        torch.cuda.mem_get_info()
    )

    print(
        "Free GPU memory:",
        round(
            free_memory / 1024**3,
            2
        ),
        "GB out of",
        round(
            total_memory / 1024**3,
            2
        ),
        "GB"
    )

# ------------------------------------------------------------
# 3. Output directory
# ------------------------------------------------------------

SCORE_BN_DIR = (
    BACKUP_DIR /
    'score_bn'
)

SCORE_BN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 4. Load best transformer hyperparameters
# ------------------------------------------------------------

BEST_PARAMETERS_FILE = (
    BACKUP_DIR /
    'transformer_tuning' /
    'best_hyperparameters.json'
)

learning_rate = 2e-5
weight_decay = 0.01

if BEST_PARAMETERS_FILE.exists():

    with open(
        BEST_PARAMETERS_FILE,
        'r',
        encoding='utf-8'
    ) as file:

        tuning_information = json.load(file)

    best_parameters = (
        tuning_information.get(
            'best_hyperparameters',
            {}
        )
    )

    learning_rate = float(
        best_parameters.get(
            'learning_rate',
            learning_rate
        )
    )

    weight_decay = float(
        best_parameters.get(
            'weight_decay',
            weight_decay
        )
    )

    print(
        "Using tuned learning rate:",
        learning_rate
    )

    print(
        "Using tuned weight decay:",
        weight_decay
    )

else:

    print(
        "Tuning file not found. "
        "Using default hyperparameters."
    )

# ------------------------------------------------------------
# 5. Training configuration
# ------------------------------------------------------------

MAX_LENGTH = 128

# Each sample contains two texts, so keep the physical
# batch size small.
PHYSICAL_BATCH_SIZE = 2

# Effective paired batch size = 2 × 4 = 8
GRADIENT_ACCUMULATION_STEPS = 4

NUMBER_OF_EPOCHS = 3

LAMBDA_ORDINAL = 0.5
LAMBDA_CONSISTENCY = 0.5
LAMBDA_UNDER = 0.3

# ------------------------------------------------------------
# 6. Paired dataset
# ------------------------------------------------------------

class PairedDataset(TorchDataset):

    def __init__(
        self,
        frame,
        tokenizer,
        max_length=128
    ):

        self.frame = (
            frame
            .copy()
            .reset_index(drop=True)
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.frame)

    def __getitem__(self, index):

        row = self.frame.iloc[index]

        original_encoding = self.tokenizer(
            str(row['text']),
            truncation=True,
            max_length=self.max_length
        )

        romanized_encoding = self.tokenizer(
            str(row['view_text']),
            truncation=True,
            max_length=self.max_length
        )

        return {
            'original': original_encoding,
            'view': romanized_encoding,
            'label': int(row['label'])
        }


def paired_collate(batch):

    original_batch = tokenizer.pad(
        [
            item['original']
            for item in batch
        ],
        return_tensors='pt'
    )

    view_batch = tokenizer.pad(
        [
            item['view']
            for item in batch
        ],
        return_tensors='pt'
    )

    labels = torch.tensor(
        [
            item['label']
            for item in batch
        ],
        dtype=torch.long
    )

    return {
        'original': original_batch,
        'view': view_batch,
        'labels': labels
    }

# ------------------------------------------------------------
# 7. SCORE-BN model
# ------------------------------------------------------------

class ScoreBN(nn.Module):

    def __init__(
        self,
        model_name,
        number_of_labels=4,
        dropout=0.2
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        # Reduce GPU-memory use
        if hasattr(
            self.encoder,
            'gradient_checkpointing_enable'
        ):
            self.encoder.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            number_of_labels
        )

    def encode_one(self, encoded_batch):

        encoder_output = self.encoder(
            **encoded_batch
        )

        cls_representation = (
            encoder_output
            .last_hidden_state[:, 0, :]
        )

        cls_representation = (
            self.dropout(
                cls_representation
            )
        )

        return self.classifier(
            cls_representation
        )

    def forward(
        self,
        original_batch,
        view_batch=None
    ):

        original_logits = self.encode_one(
            original_batch
        )

        view_logits = None

        if view_batch is not None:

            view_logits = self.encode_one(
                view_batch
            )

        return (
            original_logits,
            view_logits
        )

# ------------------------------------------------------------
# 8. SCORE-BN loss
# ------------------------------------------------------------

def score_bn_loss(
    original_logits,
    view_logits,
    labels,
    lambda_ordinal=0.5,
    lambda_consistency=0.5,
    lambda_under=0.3
):

    # Standard four-class classification loss
    classification_loss = (
        F.cross_entropy(
            original_logits,
            labels
        )
        +
        F.cross_entropy(
            view_logits,
            labels
        )
    ) / 2

    original_probabilities = F.softmax(
        original_logits,
        dim=-1
    )

    view_probabilities = F.softmax(
        view_logits,
        dim=-1
    )

    # --------------------------------------------------------
    # Ordinal cumulative-target loss
    # --------------------------------------------------------

    thresholds = torch.arange(
        3,
        device=labels.device
    ).unsqueeze(0)

    ordinal_targets = (
        labels.unsqueeze(1)
        > thresholds
    ).float()

    original_cumulative = torch.stack(
        [
            original_probabilities[
                :, 1:
            ].sum(dim=-1),

            original_probabilities[
                :, 2:
            ].sum(dim=-1),

            original_probabilities[
                :, 3:
            ].sum(dim=-1)
        ],
        dim=1
    ).clamp(
        1e-6,
        1 - 1e-6
    )

    view_cumulative = torch.stack(
        [
            view_probabilities[
                :, 1:
            ].sum(dim=-1),

            view_probabilities[
                :, 2:
            ].sum(dim=-1),

            view_probabilities[
                :, 3:
            ].sum(dim=-1)
        ],
        dim=1
    ).clamp(
        1e-6,
        1 - 1e-6
    )

    ordinal_loss = (
        F.binary_cross_entropy(
            original_cumulative,
            ordinal_targets
        )
        +
        F.binary_cross_entropy(
            view_cumulative,
            ordinal_targets
        )
    ) / 2

    # --------------------------------------------------------
    # Jensen-Shannon consistency loss
    # --------------------------------------------------------

    midpoint = (
        original_probabilities
        +
        view_probabilities
    ) / 2

    midpoint = midpoint.clamp(
        min=1e-8
    )

    consistency_loss = 0.5 * (
        F.kl_div(
            midpoint.log(),
            original_probabilities,
            reduction='batchmean'
        )
        +
        F.kl_div(
            midpoint.log(),
            view_probabilities,
            reduction='batchmean'
        )
    )

    # --------------------------------------------------------
    # Asymmetric under-prioritisation loss
    # --------------------------------------------------------

    severity_values = torch.arange(
        4,
        device=labels.device,
        dtype=torch.float32
    )

    original_expected_severity = (
        original_probabilities
        *
        severity_values
    ).sum(dim=-1)

    view_expected_severity = (
        view_probabilities
        *
        severity_values
    ).sum(dim=-1)

    under_prioritisation_loss = (
        F.relu(
            labels.float()
            -
            original_expected_severity
        ).pow(2)
        +
        F.relu(
            labels.float()
            -
            view_expected_severity
        ).pow(2)
    ).mean() / 2

    total_loss = (
        classification_loss
        +
        lambda_ordinal
        * ordinal_loss
        +
        lambda_consistency
        * consistency_loss
        +
        lambda_under
        * under_prioritisation_loss
    )

    components = {
        'classification': (
            classification_loss
            .detach()
            .item()
        ),

        'ordinal': (
            ordinal_loss
            .detach()
            .item()
        ),

        'consistency': (
            consistency_loss
            .detach()
            .item()
        ),

        'under': (
            under_prioritisation_loss
            .detach()
            .item()
        )
    }

    return total_loss, components

# ------------------------------------------------------------
# 9. DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    PairedDataset(
        paired_train,
        tokenizer,
        MAX_LENGTH
    ),

    batch_size=PHYSICAL_BATCH_SIZE,
    shuffle=True,
    collate_fn=paired_collate,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    PairedDataset(
        paired_validation,
        tokenizer,
        MAX_LENGTH
    ),

    batch_size=4,
    shuffle=False,
    collate_fn=paired_collate,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ------------------------------------------------------------
# 10. Initialize model and optimizer
# ------------------------------------------------------------

score_model = ScoreBN(
    MODEL_NAME
).to(device)

optimizer = torch.optim.AdamW(
    score_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Mixed precision reduces T4 memory usage
scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)

# ------------------------------------------------------------
# 11. Validation function
# ------------------------------------------------------------

def validate_score_bn(
    model,
    validation_loader
):

    model.eval()

    validation_losses = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch in validation_loader:

            original = {
                key: value.to(device)
                for key, value
                in batch[
                    'original'
                ].items()
            }

            view = {
                key: value.to(device)
                for key, value
                in batch[
                    'view'
                ].items()
            }

            labels = batch[
                'labels'
            ].to(device)

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                original_logits, view_logits = (
                    model(
                        original,
                        view
                    )
                )

                loss, _ = score_bn_loss(
                    original_logits,
                    view_logits,
                    labels,
                    LAMBDA_ORDINAL,
                    LAMBDA_CONSISTENCY,
                    LAMBDA_UNDER
                )

            validation_losses.append(
                loss.item()
            )

            predictions = (
                original_logits
                .argmax(dim=-1)
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='macro',
            zero_division=0
        )
    )

    return {
        'validation_loss': float(
            np.mean(
                validation_losses
            )
        ),

        'validation_accuracy': (
            accuracy_score(
                all_labels,
                all_predictions
            )
        ),

        'validation_precision': precision,
        'validation_recall': recall,
        'validation_macro_f1': macro_f1
    }

# ------------------------------------------------------------
# 12. Train SCORE-BN
# ------------------------------------------------------------

training_history = []

best_validation_f1 = -1.0

BEST_MODEL_PATH = (
    SCORE_BN_DIR /
    'best_score_bn.pt'
)

optimizer.zero_grad(
    set_to_none=True
)

training_start_time = time.time()

for epoch in range(
    NUMBER_OF_EPOCHS
):

    score_model.train()

    running_loss = 0.0
    completed_batches = 0

    for step, batch in enumerate(
        train_loader
    ):

        original = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in batch[
                'original'
            ].items()
        }

        view = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in batch[
                'view'
            ].items()
        }

        labels = batch[
            'labels'
        ].to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            original_logits, view_logits = (
                score_model(
                    original,
                    view
                )
            )

            loss, loss_components = (
                score_bn_loss(
                    original_logits,
                    view_logits,
                    labels,
                    LAMBDA_ORDINAL,
                    LAMBDA_CONSISTENCY,
                    LAMBDA_UNDER
                )
            )

            scaled_loss = (
                loss
                /
                GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(
            scaled_loss
        ).backward()

        should_update = (
            (
                step + 1
            )
#             % GRADIENT_ACCUMULATION_STEPS
            == 0
            or
            (
                step + 1
            )
            == len(train_loader)
        )

        if should_update:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                score_model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += loss.item()
        completed_batches += 1

        if (
            step + 1
        ) % 200 == 0:

            print(
                f"Epoch {epoch + 1}, "
                f"batch {step + 1}/"
                f"{len(train_loader)}, "
                f"loss "
                f"{running_loss / completed_batches:.4f}"
            )

    validation_metrics = (
        validate_score_bn(
            score_model,
            validation_loader
        )
    )

    epoch_result = {
        'epoch': epoch + 1,
        'training_loss': (
            running_loss
            /
            completed_batches
        ),
        **validation_metrics
    }

    training_history.append(
        epoch_result
    )

    print(
        f"\nEpoch {epoch + 1}: "
        f"training loss="
        f"{epoch_result['training_loss']:.4f}, "
        f"validation loss="
        f"{validation_metrics['validation_loss']:.4f}, "
        f"validation macro-F1="
        f"{validation_metrics['validation_macro_f1']:.4f}"
    )

    # Save the latest model after every epoch
    torch.save(
        {
            'epoch': epoch + 1,
            'model_state_dict': (
                score_model.state_dict()
            ),
            'optimizer_state_dict': (
                optimizer.state_dict()
            ),
            'validation_metrics': (
                validation_metrics
            )
        },
        SCORE_BN_DIR /
        'latest_score_bn_checkpoint.pt'
    )

    # Save the model with the best validation macro-F1
    if (
        validation_metrics[
            'validation_macro_f1'
        ]
        >
        best_validation_f1
    ):

        best_validation_f1 = (
            validation_metrics[
                'validation_macro_f1'
            ]
        )

        torch.save(
            score_model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "Best SCORE-BN model updated."
        )

    pd.DataFrame(
        training_history
    ).to_csv(
        SCORE_BN_DIR /
        'training_history.csv',
        index=False
    )

training_minutes = (
    time.time()
    -
    training_start_time
) / 60

# ------------------------------------------------------------
# 13. Save configuration
# ------------------------------------------------------------

configuration = {
    'base_model': MODEL_NAME,
    'max_length': MAX_LENGTH,
    'physical_batch_size': (
        PHYSICAL_BATCH_SIZE
    ),
    'gradient_accumulation_steps': (
        GRADIENT_ACCUMULATION_STEPS
    ),
    'effective_paired_batch_size': (
        PHYSICAL_BATCH_SIZE
        *
        GRADIENT_ACCUMULATION_STEPS
    ),
    'epochs': NUMBER_OF_EPOCHS,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'lambda_ordinal': LAMBDA_ORDINAL,
    'lambda_consistency': (
        LAMBDA_CONSISTENCY
    ),
    'lambda_under': LAMBDA_UNDER,
    'best_validation_macro_f1': (
        best_validation_f1
    ),
    'training_minutes': (
        training_minutes
    )
}

with open(
    SCORE_BN_DIR /
    'configuration.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        configuration,
        file,
        indent=2
    )

tokenizer.save_pretrained(
    SCORE_BN_DIR /
    'tokenizer'
)

print("\nSCORE-BN training completed.")
print(
    "Best validation macro-F1:",
    round(
        best_validation_f1,
        4
    )
)
print(
    "Training time:",
    round(
        training_minutes,
        2
    ),
    "minutes"
)
print(
    "Best model saved to:",
    BEST_MODEL_PATH
)

import torch
import torch.nn.functional as F

def score_bn_loss(
    original_logits,
    view_logits,
    labels,
    lambda_ordinal=0.5,
    lambda_consistency=0.5,
    lambda_under=0.3
):

    # Perform loss calculations in float32.
    original_logits = original_logits.float()
    view_logits = view_logits.float()

    # Standard classification loss
    classification_loss = (
        F.cross_entropy(
            original_logits,
            labels
        )
        +
        F.cross_entropy(
            view_logits,
            labels
        )
    ) / 2.0

    original_probabilities = F.softmax(
        original_logits,
        dim=-1
    )

    view_probabilities = F.softmax(
        view_logits,
        dim=-1
    )

    # --------------------------------------------------------
    # Ordinal loss
    # --------------------------------------------------------

    thresholds = torch.arange(
        3,
        device=labels.device
    ).unsqueeze(0)

    ordinal_targets = (
        labels.unsqueeze(1) > thresholds
    ).float()

    original_cumulative = torch.stack(
        [
            original_probabilities[:, 1:].sum(dim=-1),
            original_probabilities[:, 2:].sum(dim=-1),
            original_probabilities[:, 3:].sum(dim=-1)
        ],
        dim=1
    ).clamp(1e-6, 1.0 - 1e-6)

    view_cumulative = torch.stack(
        [
            view_probabilities[:, 1:].sum(dim=-1),
            view_probabilities[:, 2:].sum(dim=-1),
            view_probabilities[:, 3:].sum(dim=-1)
        ],
        dim=1
    ).clamp(1e-6, 1.0 - 1e-6)

    # Manual BCE avoids the mixed-precision autocast restriction.
    original_ordinal_loss = -(
        ordinal_targets
        * torch.log(original_cumulative)
        +
        (1.0 - ordinal_targets)
        * torch.log(1.0 - original_cumulative)
    ).mean()

    view_ordinal_loss = -(
        ordinal_targets
        * torch.log(view_cumulative)
        +
        (1.0 - ordinal_targets)
        * torch.log(1.0 - view_cumulative)
    ).mean()

    ordinal_loss = (
        original_ordinal_loss
        +
        view_ordinal_loss
    ) / 2.0

    # --------------------------------------------------------
    # Jensen-Shannon consistency loss
    # --------------------------------------------------------

    midpoint = (
        original_probabilities
        +
        view_probabilities
    ) / 2.0

    original_probabilities_safe = (
        original_probabilities.clamp(min=1e-8)
    )

    view_probabilities_safe = (
        view_probabilities.clamp(min=1e-8)
    )

    midpoint_safe = midpoint.clamp(min=1e-8)

    consistency_loss = 0.5 * (
        (
            original_probabilities_safe
            *
            (
                original_probabilities_safe.log()
                -
                midpoint_safe.log()
            )
        ).sum(dim=-1).mean()
        +
        (
            view_probabilities_safe
            *
            (
                view_probabilities_safe.log()
                -
                midpoint_safe.log()
            )
        ).sum(dim=-1).mean()
    )

    # --------------------------------------------------------
    # Under-prioritisation loss
    # --------------------------------------------------------

    severity_values = torch.arange(
        4,
        device=labels.device,
        dtype=torch.float32
    )

    original_expected_severity = (
        original_probabilities
        *
        severity_values
    ).sum(dim=-1)

    view_expected_severity = (
        view_probabilities
        *
        severity_values
    ).sum(dim=-1)

    original_under_loss = F.relu(
        labels.float()
        -
        original_expected_severity
    ).pow(2)

    view_under_loss = F.relu(
        labels.float()
        -
        view_expected_severity
    ).pow(2)

    under_prioritisation_loss = (
        original_under_loss.mean()
        +
        view_under_loss.mean()
    ) / 2.0

    # --------------------------------------------------------
    # Total SCORE-BN loss
    # --------------------------------------------------------

    total_loss = (
        classification_loss
        +
        lambda_ordinal * ordinal_loss
        +
        lambda_consistency * consistency_loss
        +
        lambda_under * under_prioritisation_loss
    )

    components = {
        'classification': (
            classification_loss.detach().item()
        ),
        'ordinal': (
            ordinal_loss.detach().item()
        ),
        'consistency': (
            consistency_loss.detach().item()
        ),
        'under': (
            under_prioritisation_loss.detach().item()
        )
    }

    return total_loss, components

print("Corrected SCORE-BN loss function is active.")
# The tuning stage selected these values:
learning_rate = 3e-5
weight_decay = 0.01

optimizer = torch.optim.AdamW(
    score_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Current recommended GradScaler syntax
scaler = torch.amp.GradScaler(
    'cuda',
    enabled=torch.cuda.is_available()
)

optimizer.zero_grad(set_to_none=True)

print("Optimizer and scaler reset.")
print("Learning rate:", learning_rate)
print("Weight decay:", weight_decay)

import inspect

loss_source = inspect.getsource(
    score_bn_loss
)

print(
    "Old unsafe BCE still present:",
    "F.binary_cross_entropy(" in loss_source
)

# Commented out IPython magic to ensure Python compatibility.
# ============================================================
# CONTINUE SCORE-BN TRAINING AFTER FIXING THE LOSS
# ============================================================

import time
import json
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

NUMBER_OF_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4

LAMBDA_ORDINAL = 0.5
LAMBDA_CONSISTENCY = 0.5
LAMBDA_UNDER = 0.3

SCORE_BN_DIR = (
    BACKUP_DIR / 'score_bn'
)

SCORE_BN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    SCORE_BN_DIR /
    'best_score_bn.pt'
)

LATEST_CHECKPOINT_PATH = (
    SCORE_BN_DIR /
    'latest_score_bn_checkpoint.pt'
)

# ------------------------------------------------------------
# Confirm required objects exist
# ------------------------------------------------------------

required_objects = [
    'score_model',
    'optimizer',
    'scaler',
    'train_loader',
    'validation_loader',
    'validate_score_bn',
    'score_bn_loss',
    'device'
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise NameError(
        f"Missing objects: {missing_objects}"
    )

print("Everything is ready.")
print("Starting SCORE-BN training...")

# ------------------------------------------------------------
# Reset training records
# ------------------------------------------------------------

training_history = []
best_validation_f1 = -1.0

optimizer.zero_grad(
    set_to_none=True
)

training_start_time = time.time()

# ------------------------------------------------------------
# Train for three epochs
# ------------------------------------------------------------

for epoch in range(NUMBER_OF_EPOCHS):

    score_model.train()

    running_loss = 0.0
    completed_batches = 0

    print(
        "\n" + "=" * 60
    )

    print(
        f"Starting epoch {epoch + 1}/"
        f"{NUMBER_OF_EPOCHS}"
    )

    print(
        "=" * 60
    )

    for step, batch in enumerate(
        train_loader
    ):

        original = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in batch['original'].items()
        }

        view = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in batch['view'].items()
        }

        labels = batch['labels'].to(
            device,
            non_blocking=True
        )

        # Mixed-precision forward pass
        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            original_logits, view_logits = (
                score_model(
                    original,
                    view
                )
            )

            loss, loss_components = (
                score_bn_loss(
                    original_logits,
                    view_logits,
                    labels,
                    LAMBDA_ORDINAL,
                    LAMBDA_CONSISTENCY,
                    LAMBDA_UNDER
                )
            )

            accumulated_loss = (
                loss /
                GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(
            accumulated_loss
        ).backward()

        should_update = (
            (step + 1)
#             % GRADIENT_ACCUMULATION_STEPS
            == 0
            or
            (step + 1)
            == len(train_loader)
        )

        if should_update:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                score_model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += loss.item()
        completed_batches += 1

        # Show progress every 200 batches
        if (step + 1) % 200 == 0:

            print(
                f"Epoch {epoch + 1}, "
                f"batch {step + 1}/"
                f"{len(train_loader)}, "
                f"average loss: "
                f"{running_loss / completed_batches:.4f}"
            )

    # --------------------------------------------------------
    # Validation after the epoch
    # --------------------------------------------------------

    validation_metrics = (
        validate_score_bn(
            score_model,
            validation_loader
        )
    )

    epoch_result = {
        'epoch': epoch + 1,
        'training_loss': (
            running_loss /
            completed_batches
        ),
        **validation_metrics
    }

    training_history.append(
        epoch_result
    )

    print(
        f"\nEpoch {epoch + 1} completed"
    )

    print(
        "Training loss:",
        round(
            epoch_result[
                'training_loss'
            ],
            4
        )
    )

    print(
        "Validation loss:",
        round(
            validation_metrics[
                'validation_loss'
            ],
            4
        )
    )

    print(
        "Validation macro-F1:",
        round(
            validation_metrics[
                'validation_macro_f1'
            ],
            4
        )
    )

    # --------------------------------------------------------
    # Save the latest checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            'epoch': epoch + 1,
            'model_state_dict': (
                score_model.state_dict()
            ),
            'optimizer_state_dict': (
                optimizer.state_dict()
            ),
            'validation_metrics': (
                validation_metrics
            )
        },
        LATEST_CHECKPOINT_PATH
    )

    # Save the best model
    if (
        validation_metrics[
            'validation_macro_f1'
        ]
        >
        best_validation_f1
    ):

        best_validation_f1 = (
            validation_metrics[
                'validation_macro_f1'
            ]
        )

        torch.save(
            score_model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "Best SCORE-BN model saved."
        )

    # Save history after every epoch
    pd.DataFrame(
        training_history
    ).to_csv(
        SCORE_BN_DIR /
        'training_history.csv',
        index=False
    )

# ------------------------------------------------------------
# Save final information
# ------------------------------------------------------------

training_minutes = (
    time.time()
    -
    training_start_time
) / 60

configuration = {
    'base_model': MODEL_NAME,
    'epochs': NUMBER_OF_EPOCHS,
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'physical_batch_size': 2,
    'gradient_accumulation_steps': (
        GRADIENT_ACCUMULATION_STEPS
    ),
    'effective_batch_size': 8,
    'lambda_ordinal': (
        LAMBDA_ORDINAL
    ),
    'lambda_consistency': (
        LAMBDA_CONSISTENCY
    ),
    'lambda_under': (
        LAMBDA_UNDER
    ),
    'best_validation_macro_f1': (
        best_validation_f1
    ),
    'training_minutes': (
        training_minutes
    )
}

with open(
    SCORE_BN_DIR /
    'configuration.json',
    'w',
    encoding='utf-8'
) as file:

    json.dump(
        configuration,
        file,
        indent=2
    )

tokenizer.save_pretrained(
    SCORE_BN_DIR /
    'tokenizer'
)

print(
    "\n" + "=" * 60
)

print(
    "SCORE-BN TRAINING COMPLETED"
)

print(
    "=" * 60
)

print(
    "Best validation macro-F1:",
    round(
        best_validation_f1,
        4
    )
)

print(
    "Training time:",
    round(
        training_minutes,
        2
    ),
    "minutes"
)

print(
    "Best model saved to:",
    BEST_MODEL_PATH
)

from pathlib import Path

tuning_files = list(
    BACKUP_DIR.rglob(
        'best_hyperparameters.json'
    )
)

print(
    "Number of tuning files found:",
    len(tuning_files)
)

for file_path in tuning_files:
    print(file_path)

import gc
import torch

for variable_name in [
    'score_model',
    'optimizer',
    'scaler',
    'train_loader',
    'validation_loader',
    'tuning_trainer'
]:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_memory, total_memory = torch.cuda.mem_get_info()

    print(
        "Free GPU memory:",
        round(free_memory / 1024**3, 2),
        "GB out of",
        round(total_memory / 1024**3, 2),
        "GB"
    )

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

missing_variables = []

for variable_name in required_variables:
    available = variable_name in globals()

    print(
        variable_name,
        "available" if available else "MISSING"
    )

    if not available:
        missing_variables.append(variable_name)

print("\nMissing variables:", missing_variables)

