"""Train a jewelry image classifier with transfer learning.

Expected dataset layout:

dataset/
  Rings/
  Bracelets/
  Necklaces/
  Earrings/

The script saves:
  models/jewelry_classifier.keras
  models/best_jewelry_classifier.keras
  models/class_names.json
  models/training_config.json
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models


DEFAULT_SEED = 123
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train jewelry category classifier")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--model-dir", default="models", help="Output model directory")
    parser.add_argument(
        "--backbone",
        choices=["efficientnetb0", "mobilenetv2"],
        default="efficientnetb0",
        help="Transfer-learning backbone",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Square image size")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25, help="Frozen-backbone epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-layers", type=int, default=35)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError("Dataset must contain at least two class folders")

    empty = [path.name for path in class_dirs if not any(path.iterdir())]
    if empty:
        raise ValueError(f"These class folders are empty: {', '.join(empty)}")


def load_datasets(args):
    image_size = (args.image_size, args.image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = list(train_ds.class_names)
    train_ds = train_ds.cache().shuffle(1000, seed=args.seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_backbone(backbone_name: str, image_size: int):
    input_shape = (image_size, image_size, 3)
    if backbone_name == "mobilenetv2":
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    return tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )


def build_classifier(args, class_count: int):
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(args.image_size, args.image_size, 3), name="image")
    x = data_augmentation(inputs)

    if args.backbone == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)

    backbone = build_backbone(args.backbone, args.image_size)
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(class_count, activation="softmax", name="category")(x)

    model = models.Model(inputs, outputs, name=f"jewelry_{args.backbone}_classifier")
    return model, backbone


def compile_model(model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_callbacks(model_dir: Path, phase: str):
    checkpoint_path = model_dir / "best_jewelry_classifier.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(model_dir / f"training_{phase}.csv"),
    ]


def fine_tune(model, backbone, args, train_ds, val_ds, model_dir: Path):
    if args.fine_tune_epochs <= 0:
        return None

    backbone.trainable = True

    if args.fine_tune_layers > 0:
        for layer in backbone.layers[:-args.fine_tune_layers]:
            layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, args.fine_tune_learning_rate)
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=build_callbacks(model_dir, "fine_tune"),
    )


def evaluate_model(model, val_ds, class_names):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    print("\nValidation Classification Report")
    print("=" * 40)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion Matrix")
    print("=" * 40)
    print(confusion_matrix(y_true, y_pred))


def save_metadata(model_dir: Path, args, class_names):
    with open(model_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    metadata = {
        "backbone": args.backbone,
        "image_size": args.image_size,
        "class_count": len(class_names),
        "class_names": class_names,
        "format": "keras",
        "feature_embedding_model": "MobileNetV2",
        "feature_vector_size": 1280,
    }
    with open(model_dir / "training_config.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    args = parse_args()
    configure_reproducibility(args.seed)

    dataset_dir = Path(args.dataset)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    validate_dataset(dataset_dir)

    train_ds, val_ds, class_names = load_datasets(args)
    print(f"Classes ({len(class_names)}): {class_names}")

    save_metadata(model_dir, args, class_names)

    model, backbone = build_classifier(args, len(class_names))
    compile_model(model, args.learning_rate)
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=build_callbacks(model_dir, "frozen"),
    )

    fine_tune(model, backbone, args, train_ds, val_ds, model_dir)
    evaluate_model(model, val_ds, class_names)

    output_path = model_dir / "jewelry_classifier.keras"
    model.save(output_path)
    print(f"\nModel saved to {output_path}")
    print(f"Best checkpoint saved to {model_dir / 'best_jewelry_classifier.keras'}")
    print(f"Class names saved to {model_dir / 'class_names.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
