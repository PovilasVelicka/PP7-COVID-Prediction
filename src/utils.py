import os
import json
from pathlib import Path
from typing import Tuple, Dict
from collections import defaultdict
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import src.config as config


def get_image_generators(
    dataset_path: Path,
    classes: list[str],
    img_size: int = 224,
    batch_size: int = 32,
    rescale: float = 1. / 255
) -> Tuple:
    """
    Create ImageDataGenerators for training and validation datasets.

    Returns:
        Tuple: (train_generator, val_generator)
    """
    train_gen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        channel_shift_range=10.0,
        fill_mode='constant',
        cval=0.0
    ).flow_from_directory(
        directory=dataset_path / "train",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=classes
    )

    val_gen = ImageDataGenerator(
        rescale=rescale
    ).flow_from_directory(
        directory=dataset_path / "val",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=classes
    )

    return train_gen, val_gen


# def get_image_generators(
#     dataset_path: Path,
#     classes: list[str],
#     img_size: int = 224,
#     batch_size: int = 32,
#     rescale: float = 1. / 255
# ) -> Tuple:
#     """
#     Create ImageDataGenerators for training and validation datasets.
#
#     Returns:
#         Tuple: (train_generator, val_generator)
#     """
#     train_gen = ImageDataGenerator(
#         rescale=rescale,
#         horizontal_flip=True,
#         rotation_range=7
#     ).flow_from_directory(
#         directory=dataset_path / "train",
#         target_size=(img_size, img_size),
#         batch_size=batch_size,
#         class_mode='binary',
#         classes=classes
#     )
#
#     val_gen = ImageDataGenerator(
#         rescale=rescale
#     ).flow_from_directory(
#         directory=dataset_path / "val",
#         target_size=(img_size, img_size),
#         batch_size=batch_size,
#         class_mode='binary',
#         classes=classes
#     )
#
#     return train_gen, val_gen


def get_test_generator(
    dataset_path: Path,
    classes: list[str],
    img_size: int = 224,
    batch_size: int = 32,
    rescale: float = 1. / 255
):
    """
    Create ImageDataGenerator for test set (without augmentation).
    """
    return ImageDataGenerator(rescale=rescale).flow_from_directory(
        directory=dataset_path / "test",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        classes=classes
    )


def compute_class_weights(generator) -> Dict[int, float]:
    """
    Compute class weights to handle class imbalance.

    Args:
        generator: training generator with .classes attribute

    Returns:
        dict: index -> weight
    """
    labels = generator.classes
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(weights))


def load_class_mapping(json_path: str) -> Dict[int, str]:
    """
    Load class index mapping and invert it to index → label.

    Args:
        json_path (str): path to class_indices.json

    Returns:
        dict: index -> label
    """
    with open(json_path, "r") as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}


def save_class_mapping(class_indices: dict, output_path: str):
    """
    Save class index mapping (e.g., {"non_covid": 0, "covid": 1}) to JSON.

    Args:
        class_indices: dict from ImageDataGenerator.class_indices
        output_path: path to save JSON (e.g., class_indices.json)
    """
    with open(output_path, "w") as f:
        json.dump(class_indices, f)


def verify_dataset_structure(subset: str = "train"):
    """
    Verifies dataset folder structure, image counts, and class indices.

    Args:
        subset (str): One of "train", "val", or "test"
    """
    subset_path = config.DATASET_SPLIT_DIR / subset
    print(f"\n📁 Checking directory: {subset_path}")

    # Check for missing class folders
    missing_classes = [cls for cls in config.CLASS_LIST if not (subset_path / cls).exists()]
    if missing_classes:
        print(f"❌ Missing class folders: {missing_classes}")
    else:
        print("✅ All class folders are present.")

    # Count images per class
    image_counts = defaultdict(int)
    for cls in config.CLASS_LIST:
        class_dir = subset_path / cls
        if class_dir.exists():
            count = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            image_counts[cls] = count

    print("📊 Image count per class:")
    for cls, count in image_counts.items():
        print(f"  - {cls}: {count} images")

    # Create generator and check class indices
    generator = ImageDataGenerator(rescale=config.RESCALE).flow_from_directory(
        directory=subset_path,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        classes=config.CLASS_LIST
    )

    print("🔢 Class indices:", generator.class_indices)
    print("📦 Total images in generator:", generator.samples)
    print("📈 Class distribution in generator:", dict(zip(config.CLASS_LIST, np.bincount(generator.classes))))
