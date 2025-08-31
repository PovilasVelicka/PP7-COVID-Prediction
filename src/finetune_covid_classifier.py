import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from evalute_model import evaluate_model
from utils import (
    get_image_generators,
    compute_class_weights
)
from config import (
    IMG_SIZE, BATCH_SIZE, RESCALE, EPOCHS,
    CLASS_LIST, DATASET_SPLIT_DIR,
    MODEL_PATH
)

# === Load data
train_gen, val_gen = get_image_generators(
    dataset_path=DATASET_SPLIT_DIR,
    classes=CLASS_LIST,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    rescale=RESCALE
)

# === Load previously trained MobileNetV2 model
model = load_model(MODEL_PATH)
print(f"✅ Loaded model for fine-tuning: {MODEL_PATH}")

# === Unfreeze all layers for fine-tuning
for layer in model.layers:
    layer.trainable = True

# === Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# === Callbacks
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# === Compute class weights again (optional)
class_weights = compute_class_weights(train_gen)

# === Fine-tune
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

print(f"✅ Fine-tuned model saved to: {MODEL_PATH}")
evaluate_model(MODEL_PATH)
