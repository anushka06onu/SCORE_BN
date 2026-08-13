"""## 19. Create cross-script paired views — CPU

These are deterministic meaning-preserving transformations of real training queries. They are augmentation views, not a synthetic replacement dataset. Inspect a sample manually before training.
"""

# Install the transliteration package if necessary
!pip -q install indic-transliteration

import pandas as pd

from pathlib import Path
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ------------------------------------------------------------
# 1. Output directory
# ------------------------------------------------------------

PAIRED_DATA_DIR = (
    BACKUP_DIR /
    'paired_data'
)

PAIRED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 2. Bangla-to-Romanized transformation
# ------------------------------------------------------------

def romanize_bangla(text):

    text = str(text)

    try:

        romanized_text = transliterate(
            text,
            sanscript.BENGALI,
            sanscript.ITRANS
        )

        return romanized_text

    except Exception:

        # Preserve the original input if transformation fails
        return text

# ------------------------------------------------------------
# 3. Create paired training data
# ------------------------------------------------------------

paired_train = (
    train_df[
        [
            'text',
            'label'
        ]
    ]
    .copy()
    .reset_index(drop=True)
)

paired_train[
    'view_text'
] = paired_train[
    'text'
].map(
    romanize_bangla
)

# ------------------------------------------------------------
# 4. Create paired validation data
# ------------------------------------------------------------

paired_validation = (
    val_df[
        [
            'text',
            'label'
        ]
    ]
    .copy()
    .reset_index(drop=True)
)

paired_validation[
    'view_text'
] = paired_validation[
    'text'
].map(
    romanize_bangla
)

# ------------------------------------------------------------
# 5. Create paired test data
# ------------------------------------------------------------

paired_test = (
    test_df[
        [
            'text',
            'label'
        ]
    ]
    .copy()
    .reset_index(drop=True)
)

paired_test[
    'view_text'
] = paired_test[
    'text'
].map(
    romanize_bangla
)

# ------------------------------------------------------------
# 6. Basic quality checks
# ------------------------------------------------------------

def paired_view_summary(
    paired_frame,
    split_name
):

    unchanged = (
        paired_frame['text']
        ==
        paired_frame['view_text']
    ).sum()

    empty_views = (
        paired_frame['view_text']
        .fillna('')
        .str.strip()
        .eq('')
    ).sum()

    print(
        f"{split_name}: "
        f"{len(paired_frame)} samples, "
        f"{unchanged} unchanged, "
        f"{empty_views} empty views"
    )


paired_view_summary(
    paired_train,
    'Training'
)

paired_view_summary(
    paired_validation,
    'Validation'
)

paired_view_summary(
    paired_test,
    'Test'
)

# ------------------------------------------------------------
# 7. Display examples for manual inspection
# ------------------------------------------------------------

print("\nSample training pairs:")

display(
    paired_train.sample(
        n=min(
            10,
            len(paired_train)
        ),
        random_state=SEED
    )
)

# ------------------------------------------------------------
# 8. Save all paired data to Google Drive
# ------------------------------------------------------------

paired_train.to_csv(
    PAIRED_DATA_DIR /
    'paired_train_views.csv',
    index=False,
    encoding='utf-8-sig'
)

paired_validation.to_csv(
    PAIRED_DATA_DIR /
    'paired_validation_views.csv',
    index=False,
    encoding='utf-8-sig'
)

paired_test.to_csv(
    PAIRED_DATA_DIR /
    'paired_test_views.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\nPaired datasets saved to:")
print(PAIRED_DATA_DIR)

