from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from typing import Dict
from src.config import (DATASET_SPLIT_DIR, DATASET_DIR, CLASS_LIST)
from src.utils import verify_dataset_structure

IMAGES_DIR: Path = DATASET_DIR / "images"
META_CSV: Path = DATASET_DIR / "metadata.csv"

DATASET_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# === Load metadata CSV ===
df = pd.read_csv(META_CSV)

# === Quick inspection ===
print("🔍 Label distribution (finding):")
print(df['finding'].value_counts())
print("\n🔍 Modality:")
print(df['modality'].value_counts())
print("\n🔍 Views:")
print(df['view'].value_counts())

# === Filter relevant data ===
df = df[df['modality'] == 'X-ray']
df = df[df['view'].isin(['PA', 'AP'])]  # only frontal views

# Exclude unusable or unknown labels
df = df[~df['finding'].isin(['todo', 'Unknown'])]

# === Build full image path and ensure file exists ===
df['filename'] = df['filename'].astype(str)
df['path'] = df['filename'].apply(lambda x: str((IMAGES_DIR / x).resolve()))
df = df[df['path'].apply(lambda x: Path(x).exists())]

# === Map to binary class: covid vs non_covid ===
df['label'] = df['finding'].apply(
    lambda f: 'covid' if f == 'Pneumonia/Viral/COVID-19' else 'non_covid'
)

# === Stratified split into train, val, test ===
train_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df['label'], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df['label'], random_state=42
)

splits: Dict[str, pd.DataFrame] = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}


def copy_images(split_name: str, split_df: pd.DataFrame) -> None:
    """
    Copy images into structured folders for ImageDataGenerator.

    Args:
        split_name (str): One of 'train', 'val', 'test'
        split_df (pd.DataFrame): Subset of dataframe for that split
    """
    # Create output directories for each class
    for label in CLASS_LIST:
        out_dir = DATASET_SPLIT_DIR / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for _, row in split_df.iterrows():
        label = row['label']
        src = Path(row['path'])
        dst = DATASET_SPLIT_DIR / split_name / label / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


# === Process each split
for split_name, split_df in splits.items():
    copy_images(split_name, split_df)
    print(f"📁 {split_name}: {len(split_df)} images copied.")

print("✅ Done. Directory structure ready for ImageDataGenerator.")
verify_dataset_structure("train")
verify_dataset_structure("val")
verify_dataset_structure("test")