"""
Zenith Sky Condition Monitoring — V2 Model Training Pipeline
Trains an improved EfficientNetB0 cloud-cover classifier (0–8 oktas)
engineered for cross-camera and daytime-to-nighttime generalization.
"""

import os
import json
import random
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Ensure local cache directories for sandboxed execution
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))

import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_ROOT = Path("data/GOA-ALLSKY/ALLSKY-GOA")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODEL_SAVE_PATH = MODEL_DIR / "zenith_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 9
CLASS_NAMES = [str(index) for index in range(NUM_CLASSES)]
NIGHT_VALIDATION_FRACTION = 0.2
NIGHT_REPEAT_COUNT = 7


def augment_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply domain-invariance data augmentations:
    - Random horizontal & vertical flips (zenith-view rotation invariance)
    - Brightness jitter to simulate moonlit, twilight, and dark night conditions
    - Contrast jitter to simulate haze and varying cloud optical depth
    - Saturation jitter and random grayscale simulation to decouple cloud coverage
      from daylight blue/white chromaticity
    """
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)

    # Random brightness shift (delta up to 40/255)
    images = tf.image.random_brightness(images, max_delta=40.0)

    # Random contrast variation
    images = tf.image.random_contrast(images, lower=0.6, upper=1.4)

    # Random saturation variation
    images = tf.image.random_saturation(images, lower=0.2, upper=1.6)

    # Desaturation / Grayscale injection: independently vary each image.
    gray = tf.image.rgb_to_grayscale(images)
    gray_3ch = tf.concat([gray, gray, gray], axis=-1)
    gray_mask = tf.random.uniform([tf.shape(images)[0], 1, 1, 1]) < 0.30
    images = tf.where(gray_mask, gray_3ch, images)

    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


def split_adaptation_files(
    directory: Path,
    validation_fraction: float = NIGHT_VALIDATION_FRACTION,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Create a deterministic, class-stratified holdout from the C005 night set."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    training_paths = []
    training_labels = []
    validation_paths = []
    validation_labels = []
    random_generator = random.Random(SEED)

    for label, class_name in enumerate(CLASS_NAMES):
        class_directory = directory / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Night adaptation class directory not found: {class_directory}")

        image_paths = sorted(
            path for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(image_paths) < 2:
            raise ValueError(
                f"At least two images are required to split night adaptation class {class_name}: "
                f"{class_directory}"
            )

        random_generator.shuffle(image_paths)
        validation_count = max(1, round(len(image_paths) * validation_fraction))
        validation_paths.extend(image_paths[:validation_count])
        validation_labels.extend([label] * validation_count)
        training_paths.extend(image_paths[validation_count:])
        training_labels.extend([label] * (len(image_paths) - validation_count))

    return training_paths, training_labels, validation_paths, validation_labels


def load_image_paths(
    image_paths: List[Path],
    labels: List[int],
    batch_size: Optional[int],
    shuffle: bool,
):
    """Build a TensorFlow dataset from a deterministic set of image paths."""
    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in image_paths], labels)
    )

    def decode_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        return image, label

    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


class BestValidationWeights(tf.keras.callbacks.Callback):
    """Keep the lowest-loss model weights observed during a training phase."""

    def __init__(self):
        super().__init__()
        self.best_loss = float("inf")
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss")
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = float(val_loss)
            self.best_weights = self.model.get_weights()


def prepare_datasets(batch_size: int = BATCH_SIZE):
    """
    Assemble the training, validation, and test datasets:
    - Training: C013 daytime images and oversampled C005 nighttime adaptation images
    - Primary validation: class-stratified held-out C005 nighttime images
    - Reference validation: C013 night/day images from val/
    - Test: held-out nighttime images from unseen camera C009 (never used here)
    """
    print(f"Loading datasets from: {DATA_ROOT} ...")

    train_day_dir = DATA_ROOT / "train"
    train_night_dir = DATA_ROOT / "adaptation_sets" / "moonless_train_data"
    val_reference_night_dir = DATA_ROOT / "val" / "night"
    val_day_dir = DATA_ROOT / "val" / "day"

    # 1. Load raw daytime training set
    ds_train_day = tf.keras.utils.image_dataset_from_directory(
        train_day_dir,
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    # Hold out data from the C005 camera so checkpoint selection is not based
    # solely on C013 images, as in the previous pipeline.
    (
        night_train_paths,
        night_train_labels,
        night_val_paths,
        night_val_labels,
    ) = split_adaptation_files(train_night_dir)
    ds_train_night = load_image_paths(
        night_train_paths,
        night_train_labels,
        batch_size=None,
        shuffle=True,
    )
    ds_train_night_repeated = ds_train_night.repeat(NIGHT_REPEAT_COUNT)

    ds_val_night = load_image_paths(
        night_val_paths,
        night_val_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    # Combine daytime + nighttime adaptation data
    ds_train_combined = ds_train_day.concatenate(ds_train_night_repeated)
    ds_train = (
        ds_train_combined
        .shuffle(buffer_size=3000, seed=SEED)
        .batch(batch_size)
        .map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Keep C013 validation as a separate reference; it is not the primary
    # checkpointing set because it shares a camera with the daytime training set.
    ds_val_reference_night = tf.keras.utils.image_dataset_from_directory(
        val_reference_night_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)

    ds_val_day = tf.keras.utils.image_dataset_from_directory(
        val_day_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)

    training_metadata = {
        "day_training_samples": sum(1 for path in train_day_dir.glob("*/*") if path.is_file()),
        "night_training_images": len(night_train_paths),
        "night_validation_images": len(night_val_paths),
        "night_repeats_per_epoch": NIGHT_REPEAT_COUNT,
    }
    return ds_train, ds_val_night, ds_val_reference_night, ds_val_day, training_metadata


def build_model(num_classes: int = NUM_CLASSES) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build EfficientNetB0 architecture with integrated preprocessing and a regularized head.
    """
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="sky_image_input")

    # Native EfficientNetB0 normalization [-1, 1] (channel-wise scaling to ImageNet standards)
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # Transfer learning backbone
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # Feature extraction
    features = base_model(preprocessed, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="gap")(features)
    normed = tf.keras.layers.BatchNormalization(name="head_bn")(pooled)
    dense1 = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="head_dense",
    )(normed)
    dropped = tf.keras.layers.Dropout(0.4, seed=SEED, name="head_dropout")(dense1)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="okta_output")(dropped)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Zenith_EfficientNetB0_V2")
    return model, base_model


def train():
    """Execute complete two-stage transfer learning pipeline."""
    start_time = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("      ZENITH V2 MODEL TRAINING — 0 to 8 OKTAS            ")
    print("=========================================================")

    (
        train_ds,
        val_night_ds,
        val_reference_night_ds,
        val_day_ds,
        training_metadata,
    ) = prepare_datasets(batch_size=BATCH_SIZE)

    model, base_model = build_model(num_classes=NUM_CLASSES)
    print(f"\nModel initialized: {model.name}")
    print(f"Total Parameters: {model.count_params():,}")

    # -----------------------------------------------------------
    # STAGE 1: Train Classification Head (Backbone Frozen)
    # -----------------------------------------------------------
    print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    stage1_epochs = 6
    stage1_best = BestValidationWeights()
    history1 = model.fit(
        train_ds,
        epochs=stage1_epochs,
        validation_data=val_night_ds,
        callbacks=[stage1_best],
        shuffle=False,
        verbose=1,
    )
    if stage1_best.best_weights is None:
        raise RuntimeError("Stage 1 did not produce validation loss for checkpoint selection.")
    model.set_weights(stage1_best.best_weights)

    # -----------------------------------------------------------
    # STAGE 2: Controlled Fine-Tuning (Top Blocks Unfrozen)
    # -----------------------------------------------------------
    print("\n--- STAGE 2: Controlled Fine-Tuning (Top EfficientNet Blocks) ---")
    base_model.trainable = True

    # Freeze all layers except the top 35 layers (high-level EfficientNet blocks)
    fine_tune_at = len(base_model.layers) - 35
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Freeze all BatchNormalization layers to preserve moving statistics
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    print(f"Fine-tuning from base layer {fine_tune_at} / {len(base_model.layers)}")
    print(f"Trainable parameters in Stage 2: {trainable_count:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    stage2_epochs = 6
    history2 = model.fit(
        train_ds,
        epochs=stage1_epochs + stage2_epochs,
        initial_epoch=stage1_epochs,
        validation_data=val_night_ds,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    stage2_best_loss = min(history2.history["val_loss"])
    selected_checkpoint_stage = "stage2"
    if stage1_best.best_loss <= stage2_best_loss:
        model.set_weights(stage1_best.best_weights)
        selected_checkpoint_stage = "stage1"
        print(
            "Stage 1 achieved the lower C005 validation loss "
            f"({stage1_best.best_loss:.4f} vs. {stage2_best_loss:.4f}); "
            "retaining Stage 1 weights."
        )

    # Ensure best model is saved
    model.save(str(MODEL_SAVE_PATH))
    print(f"\n[SUCCESS] Best V2 Model successfully saved to: {MODEL_SAVE_PATH}")

    # Evaluate on the held-out C005 night set and retain C013 as a reference.
    val_night_eval = model.evaluate(val_night_ds, verbose=0)
    val_reference_night_eval = model.evaluate(val_reference_night_ds, verbose=0)
    val_day_eval = model.evaluate(val_day_ds, verbose=0)
    print(
        f"Final Validation (C005 night) Accuracy: {val_night_eval[1] * 100:.2f}% "
        f"| Loss: {val_night_eval[0]:.4f}"
    )
    print(
        f"Reference Validation (C013 night) Accuracy: {val_reference_night_eval[1] * 100:.2f}% "
        f"| Loss: {val_reference_night_eval[0]:.4f}"
    )
    print(f"Reference Validation (C013 day) Accuracy: {val_day_eval[1] * 100:.2f}% | Loss: {val_day_eval[0]:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal Training Duration: {total_time / 60:.2f} minutes")

    # Save training history summary
    training_summary = {
        "architecture": "EfficientNetB0 Transfer Learning (V2)",
        "num_classes": NUM_CLASSES,
        **training_metadata,
        "training_samples_per_epoch": (
            training_metadata["day_training_samples"]
            + training_metadata["night_training_images"] * NIGHT_REPEAT_COUNT
        ),
        "batch_size": BATCH_SIZE,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "duration_minutes": round(total_time / 60, 2),
        "checkpoint_validation_camera": "C005 night holdout",
        "reference_validation_camera": "C013 night/day",
        "selected_checkpoint_stage": selected_checkpoint_stage,
        "selected_checkpoint_val_loss": min(stage1_best.best_loss, stage2_best_loss),
        "stage1_val_accuracy": [float(v) for v in history1.history.get("val_accuracy", [])],
        "stage2_val_accuracy": [float(v) for v in history2.history.get("val_accuracy", [])],
        "final_val_c005_night_accuracy": float(val_night_eval[1]),
        "final_val_c005_night_loss": float(val_night_eval[0]),
        "reference_val_c013_night_accuracy": float(val_reference_night_eval[1]),
        "reference_val_c013_day_accuracy": float(val_day_eval[1]),
        "reference_val_c013_day_loss": float(val_day_eval[0]),
    }
    with open(REPORTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    return model


if __name__ == "__main__":
    train()
