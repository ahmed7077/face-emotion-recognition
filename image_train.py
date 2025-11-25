# ====================================================
# TRAINING SCRIPT (VS Code Compatible)
# EfficientNetB0, memory-safe TF-only parsing
# ====================================================

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ----------------------------------------------------
# PATHS (EDIT THESE FOR YOUR LOCAL MACHINE)
# ----------------------------------------------------
CSV_PATH = r"C:\Users\Ahmed\FER2013\fer2013.csv"
MODEL_SAVE_PATH = r"C:\Users\Ahmed\FER2013\efficientnetb0_fer_best.h5"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS_HEAD = 12
EPOCHS_FINETUNE = 12

assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"

# ----------------------------------------------------
# Load CSV
# ----------------------------------------------------
df = pd.read_csv(CSV_PATH)
train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
val_df   = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
test_df  = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

# ----------------------------------------------------
# TF-only image parser
# ----------------------------------------------------
def tf_parse_pixel_string(pixel_str):
    parts = tf.strings.split(pixel_str, sep=' ')
    nums = tf.strings.to_number(parts, out_type=tf.int32)
    nums = tf.reshape(nums, (48, 48))
    img_gray = tf.expand_dims(tf.cast(nums, tf.uint8), axis=-1)
    img_rgb = tf.image.grayscale_to_rgb(img_gray)
    img_rgb = tf.cast(img_rgb, tf.float32)
    img_resized = tf.image.resize(img_rgb, [IMG_SIZE, IMG_SIZE])
    img_pre = efficientnet.preprocess_input(img_resized)
    return img_pre

# ----------------------------------------------------
# Dataset builder
# ----------------------------------------------------
def make_dataset(df_meta, shuffle=False, repeat=False):
    pixel_series = df_meta["pixels"].astype(str).values
    labels = df_meta["emotion"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((pixel_series, labels))

    if shuffle:
        ds = ds.shuffle(10000, seed=SEED)

    ds = ds.map(lambda p, l: (tf_parse_pixel_string(p), l),
                num_parallel_calls=AUTOTUNE)
    if repeat:
        ds = ds.repeat()

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, shuffle=True, repeat=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

STEPS_PER_EPOCH = math.ceil(len(train_df) / BATCH_SIZE)
VAL_STEPS = math.ceil(len(val_df) / BATCH_SIZE)

# ----------------------------------------------------
# Model
# ----------------------------------------------------
base = EfficientNetB0(include_top=False, weights='imagenet',
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='swish')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(7, activation='softmax')(x)

model = models.Model(inp, out)
model.compile(optimizer=optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True)
]

# ----------------------------------------------------
# Train Head
# ----------------------------------------------------
history_head = model.fit(
    train_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

# ----------------------------------------------------
# Fine-tuning
# ----------------------------------------------------
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_ds,
    epochs=EPOCHS_FINETUNE,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------
test_steps = math.ceil(len(test_df) / BATCH_SIZE)
loss, acc = model.evaluate(test_ds, steps=test_steps)
print("Test Accuracy:", acc)
print("Saved model to:", MODEL_SAVE_PATH)
