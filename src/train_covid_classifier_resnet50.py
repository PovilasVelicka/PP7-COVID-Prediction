import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from evalute_model import evaluate_model

from utils import (
    get_image_generators,
    get_test_generator,
    compute_class_weights,
    save_class_mapping
)

from config import (
    IMG_SIZE, BATCH_SIZE, RESCALE, EPOCHS,
    CLASS_LIST, DATASET_SPLIT_DIR,
    MODEL_PATH, CLASS_MAP_PATH
)

# === Data generators
train_gen, val_gen = get_image_generators(
    dataset_path=DATASET_SPLIT_DIR,
    classes=CLASS_LIST,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    rescale=RESCALE
)

print(train_gen.class_indices)
import collections
print(collections.Counter(train_gen.classes))

# Save mapping for app.py and others
save_class_mapping(train_gen.class_indices, CLASS_MAP_PATH)

# === Build model
base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True

for layer in base_model.layers[:len(base_model.layers) // 3 * 2]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
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
        patience=3,
        min_lr=5e-6,
        verbose=1
    )
]


# === Compute class weights
class_weights = compute_class_weights(train_gen)


# === Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

print(f"✅ Model trained and saved to: {MODEL_PATH}")

evaluate_model(MODEL_PATH)
