import os
from typing import Union
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from utils import get_test_generator, load_class_mapping
from config import (
    IMG_SIZE, BATCH_SIZE, RESCALE,
    CLASS_LIST, DATASET_SPLIT_DIR,
    CLASS_MAP_PATH
)


def evaluate_model(model_path: Union[str, os.PathLike]) -> None:
    """
    Evaluate a trained model on the test dataset.

    Args:
        model_path (str or Path): Path to the saved .h5 model file
    """
    print(f"\n🔍 Evaluating model: {model_path}")

    # # === Load class index mapping
    # index_to_class = load_class_mapping(CLASS_MAP_PATH)
    # ordered_classes = [index_to_class[i] for i in sorted(index_to_class)]
    # print(f"✅ Ordered class labels: {ordered_classes}")

    # === Load model
    model = load_model(model_path)
    print(f"✅ Model loaded: {model_path}")

    # === Create test generator
    test_gen = get_test_generator(
        dataset_path=DATASET_SPLIT_DIR,
        classes=CLASS_LIST,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        rescale=RESCALE
    )

    # === Predict
    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # === Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LIST, zero_division=0))

    # === Confusion matrix (no hardcoded labels)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:\n{cm}")

    # Find actual indices
    covid_index = CLASS_LIST.index("covid")
    non_covid_index = CLASS_LIST.index("non_covid")

    # Assign confusion matrix elements
    tp = cm[covid_index, covid_index]
    fn = cm[covid_index, non_covid_index]
    tn = cm[non_covid_index, non_covid_index]
    fp = cm[non_covid_index, covid_index]

    # Sensitivity and specificity
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"✅ Sensitivity (Recall COVID): {sensitivity:.3f}")
    print(f"✅ Specificity (True Negative Rate non-COVID): {specificity:.3f}")


# === Optional CLI use
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❗ Please provide the model path.\nUsage: python evaluate_model.py model.h5")
        sys.exit(1)
    evaluate_model(sys.argv[1])
