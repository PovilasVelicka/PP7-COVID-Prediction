import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from src.utils import load_class_mapping

# === Parameters
MODEL_PATH = "covid_classifier_resnet50.keras"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224

# === Load model
model = load_model(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# === Load class mapping
index_to_label = load_class_mapping(CLASS_INDICES_PATH)
print(f"✅ Class mapping: {index_to_label}")

# === Preprocess image
def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Prediction
def predict(img_path: str) -> dict:
    img_tensor = preprocess_image(img_path)
    prob = model.predict(img_tensor)[0][0]
    pred_index = int(prob > 0.5)
    label = index_to_label[pred_index]
    confidence = round(float(prob if pred_index == 1 else 1 - prob), 4)
    return {
        "prediction": label,
        "confidence": confidence
    }

# === CLI Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❗ Please provide path to an image. Example:\npython app.py covid.png")
        exit(1)
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"❗ File not found: {image_path}")
        exit(1)
    result = predict(image_path)
    print("\n🔎 Result:")
    print(f"  ➤ Class      : {result['prediction']}")
    print(f"  ➤ Confidence : {result['confidence'] * 100:.1f}%")
