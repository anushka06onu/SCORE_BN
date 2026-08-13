"""## 16. Text-CNN, BiLSTM and BiGRU — T4 recommended

These are three deep-learning families required for the course comparison.
"""

import gc
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# Reproducibility
tf.keras.utils.set_random_seed(SEED)

print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MAX_TOKENS = 30000
MAX_LEN = 128
EMBED_DIM = 128
BATCH_SIZE = 64
MAX_EPOCHS = 10

# ------------------------------------------------------------
# Text vectorization
# ------------------------------------------------------------

vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    standardize=None,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

# Convert all text to normal NumPy string arrays
train_text_array = X_train_text.astype(str).to_numpy()
validation_text_array = X_val_text.astype(str).to_numpy()
test_text_array = X_test_text.astype(str).to_numpy()

# Learn vocabulary using training data only
vectorizer.adapt(train_text_array)

# Convert text into integer sequences
Xtr_seq = vectorizer(train_text_array)
Xva_seq = vectorizer(validation_text_array)
Xte_seq = vectorizer(test_text_array)

print("Training shape:", Xtr_seq.shape)
print("Validation shape:", Xva_seq.shape)
print("Test shape:", Xte_seq.shape)
print("Vocabulary size:", len(vectorizer.get_vocabulary()))

# Save the learned vocabulary
vocabulary = vectorizer.get_vocabulary()

with open(
    BACKUP_DIR / 'text_vectorizer_vocabulary.txt',
    'w',
    encoding='utf-8'
) as file:
    for token in vocabulary:
        file.write(str(token) + '\n')

# ------------------------------------------------------------
# Deep-learning model builder
# ------------------------------------------------------------

def make_deep_model(kind):

    inputs = layers.Input(
        shape=(MAX_LEN,),
        dtype='int64',
        name='token_ids'
    )

    # CNN does not support the embedding mask in this architecture.
    use_mask = kind != 'CNN'

    x = layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=EMBED_DIM,
        mask_zero=use_mask,
        name='embedding'
    )(inputs)

    if kind == 'CNN':

        x = layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu',
            name='convolution'
        )(x)

        x = layers.GlobalMaxPooling1D(
            name='global_max_pooling'
        )(x)

    elif kind == 'BiLSTM':

        x = layers.Bidirectional(
            layers.LSTM(
                units=64,
                dropout=0.20
            ),
            name='bidirectional_lstm'
        )(x)

    elif kind == 'BiGRU':

        x = layers.Bidirectional(
            layers.GRU(
                units=64,
                dropout=0.20
            ),
            name='bidirectional_gru'
        )(x)

    else:
        raise ValueError(
            "Model kind must be CNN, BiLSTM, or BiGRU."
        )

    x = layers.Dropout(
        rate=0.35,
        name='dropout'
    )(x)

    outputs = layers.Dense(
        units=4,
        activation='softmax',
        name='severity_output'
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=kind
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------------------
# Train CNN, BiLSTM and BiGRU
# ------------------------------------------------------------

deep_results = []
deep_models = {}
training_histories = {}

model_names = [
    'CNN',
    'BiLSTM',
    'BiGRU'
]

for kind in model_names:

    print("\n" + "=" * 60)
    print("Training model:", kind)
    print("=" * 60)

    # Release memory from the previous model
    tf.keras.backend.clear_session()
    gc.collect()

    model = make_deep_model(kind)

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                BACKUP_DIR / f'best_{kind}.keras'
            ),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()

    history = model.fit(
        Xtr_seq,
        y_train,
        validation_data=(
            Xva_seq,
            y_val
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    training_minutes = (
        time.time() - start_time
    ) / 60

    # Generate test probabilities and predictions
    probabilities = model.predict(
        Xte_seq,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    predictions = probabilities.argmax(axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average='macro',
            zero_division=0
        )
    )

    result = {
        'model': kind,
        'accuracy': accuracy_score(
            y_test,
            predictions
        ),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'roc_auc_ovr': roc_auc_score(
            y_test,
            probabilities,
            multi_class='ovr'
        ),
        'training_minutes': training_minutes,
        'epochs_completed': len(
            history.history['loss']
        )
    }

    deep_results.append(result)

    # Keep the current model in memory
    deep_models[kind] = model

    # Keep training history
    training_histories[kind] = history.history

    # Save final model
    model.save(
        BACKUP_DIR / f'{kind}_final.keras'
    )

    # Save predictions for later evaluation
    np.save(
        BACKUP_DIR / f'{kind}_probabilities.npy',
        probabilities
    )

    np.save(
        BACKUP_DIR / f'{kind}_predictions.npy',
        predictions
    )

    # Save history
    pd.DataFrame(
        history.history
    ).to_csv(
        BACKUP_DIR / f'{kind}_training_history.csv',
        index=False
    )

    # Save updated results after every model
    pd.DataFrame(
        deep_results
    ).to_csv(
        BACKUP_DIR / 'deep_model_results.csv',
        index=False
    )

    print("\nCompleted:", kind)
    print("Macro-F1:", round(f1, 4))
    print(
        "Training time:",
        round(training_minutes, 2),
        "minutes"
    )
    print("Saved to Google Drive.")


# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

deep_results_df = pd.DataFrame(
    deep_results
).sort_values(
    by='macro_f1',
    ascending=False
)

display(deep_results_df)

