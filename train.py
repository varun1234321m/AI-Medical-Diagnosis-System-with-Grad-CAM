import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

# ==========================================
# IMAGE + TRAINING SETTINGS
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# ==========================================
# DATA AUGMENTATION
# ==========================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_gen = ImageDataGenerator(
    rescale=1./255
)

# ==========================================
# LOAD DATASET
# ==========================================
train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==========================================
# DATASET INFO
# ==========================================
print("Class Labels:", train_data.class_indices)
print("Training Samples:", train_data.samples)
print("Validation Samples:", val_data.samples)

# ==========================================
# CLASS WEIGHTS (Fix Imbalance)
# ==========================================
counter = Counter(train_data.classes)
max_count = max(counter.values())

class_weights = {
    0: max_count / counter[0],
    1: max_count / counter[1]
}

print("Class Weights:", class_weights)

# ==========================================
# BUILD MODEL
# ==========================================
inputs = Input(shape=(224, 224, 3))

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)

# Freeze all layers first
for layer in base_model.layers:
    layer.trainable = False

# Fine-tune only last 10 layers
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)

# Prevent overfitting
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# ==========================================
# COMPILE MODEL
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# CALLBACKS
# ==========================================

# Stop training if validation loss worsens
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Reduce learning rate automatically
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ==========================================
# TRAIN MODEL
# ==========================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ==========================================
# SAVE MODEL
# ==========================================
model.save("model.keras")

print("\n Training completed successfully!")
print(" Model saved as model.keras")