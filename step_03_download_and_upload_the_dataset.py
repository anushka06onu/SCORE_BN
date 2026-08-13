"""## 3. Download and upload the dataset — CPU

Official page: **https://data.mendeley.com/datasets/37z8kgx79m/1**

Download the dataset from that page. Then run the next cell and upload the downloaded `.csv`, `.xlsx`, or `.zip` file.
"""

from google.colab import files
uploaded = files.upload()

raw_dir = PROJECT_DIR/'data/raw'
for filename, content in uploaded.items():
    destination = raw_dir/filename
    with open(destination, 'wb') as f:
        f.write(content)
    if destination.suffix.lower() == '.zip':
        with zipfile.ZipFile(destination) as zf:
            zf.extractall(raw_dir/'extracted')
print('Uploaded files:')
for p in raw_dir.rglob('*'):
    if p.is_file(): print('-', p)

from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

print("Checkpoint folder exists:", BACKUP_DIR.exists())

if BACKUP_DIR.exists():
    print("\nAvailable files:")

    for file_path in sorted(BACKUP_DIR.rglob('*')):
        if file_path.is_file():
            print(file_path.relative_to(BACKUP_DIR))

import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

BACKUP_DIR = Path(
    '/content/drive/MyDrive/SCORE_BN_Checkpoints'
)

CHECKPOINTS_DIR = (
    BACKUP_DIR /
    'banglabert_baseline' /
    'checkpoints'
)

# Find valid checkpoints containing trained model weights
valid_checkpoints = [
    checkpoint
    for checkpoint in CHECKPOINTS_DIR.glob(
        'checkpoint-*'
    )
    if (
        checkpoint /
        'model.safetensors'
    ).exists()
]

if not valid_checkpoints:
    raise FileNotFoundError(
        "No complete BanglaBERT checkpoint was found."
    )

# Arrange checkpoints according to training step
valid_checkpoints = sorted(
    valid_checkpoints,
    key=lambda path: int(
        path.name.split('-')[-1]
    )
)

latest_checkpoint = valid_checkpoints[-1]

print("Latest complete checkpoint:")
print(latest_checkpoint)

# Try to identify the best validation checkpoint
trainer_state_file = (
    latest_checkpoint /
    'trainer_state.json'
)

selected_checkpoint = latest_checkpoint

if trainer_state_file.exists():

    with open(
        trainer_state_file,
        'r',
        encoding='utf-8'
    ) as file:
        trainer_state = json.load(file)

    recorded_best = trainer_state.get(
        'best_model_checkpoint'
    )

    if recorded_best:

        best_checkpoint_name = Path(
            recorded_best
        ).name

        possible_best_checkpoint = (
            CHECKPOINTS_DIR /
            best_checkpoint_name
        )

        if (
            possible_best_checkpoint /
            'model.safetensors'
        ).exists():

            selected_checkpoint = (
                possible_best_checkpoint
            )

print("\nSelected baseline checkpoint:")
print(selected_checkpoint)

# Load the trained model and tokenizer
baseline_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        str(selected_checkpoint)
    )
)

tokenizer = AutoTokenizer.from_pretrained(
    str(selected_checkpoint)
)

# Save it as the official baseline model
FINAL_MODEL_DIR = (
    BACKUP_DIR /
    'banglabert_baseline' /
    'final_model'
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

baseline_model.save_pretrained(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)

print("\nFinal BanglaBERT baseline saved to:")
print(FINAL_MODEL_DIR)

print(
    "\nModel file exists:",
    (
        FINAL_MODEL_DIR /
        'model.safetensors'
    ).exists()
)

import gc
import torch

del baseline_model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("GPU memory cleared for transformer tuning.")

test_file = (
    BACKUP_DIR /
    'second_account_storage_test.txt'
)

test_file.write_text(
    'This file was created using the second account.',
    encoding='utf-8'
)

print("Test file created:")
print(test_file)

