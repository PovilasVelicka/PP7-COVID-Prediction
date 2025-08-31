from pathlib import Path
from typing import List

# === Image Parameters ===
IMG_SIZE: int = 224                     # Image input size (e.g. 224x224)
BATCH_SIZE: int = 32                    # Batch size for training and testing
RESCALE: float = 1. / 255               # Rescale factor for pixel normalization
EPOCHS: int = 50                        # Default number of training epochs
FINETUNE_EPOCHS: int = 2                # Default number of finetune epochs

# === Class Mapping (IMPORTANT: order must be consistent across all scripts)
CLASS_LIST: List[str] = ["non_covid", "covid"]

# === Paths (adjust if project structure changes)
ROOT_DIR: Path = Path(__file__).parent.parent.resolve()
DATASET_DIR: Path = ROOT_DIR / "dataset"            # contains images/ and metadata.csv
DATASET_SPLIT_DIR: Path = DATASET_DIR / "split"     # Contains train/, val/, test/

# === Model and class mapping files
MODEL_PATH: str = "covid_classifier_resnet50.keras"
MODEL_FINETUNED_PATH: str = "covid_classifier_resnet50_finetuned.keras"
CLASS_MAP_PATH: str = "class_indices.json"
