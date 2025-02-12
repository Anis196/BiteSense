import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import cv2
import os

# Enable mixed precision for faster training
set_global_policy('mixed_float16')

# Step 1: Check GPU Availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

# Step 2: Load & Preprocess the Dataset
dataset_path = "C:/Users/shaik/Documents/snk_cnn/dataset"
batch_size = 32
image_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Step 3: Load or Train Model
MODEL_PATH = "resnet50_snake_bite_classifier.h5"
if os.path.exists(MODEL_PATH):
    print("✅ Loading pre-trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("🚀 Training model (first-time only)...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the Model
    history = model.fit(train_generator, epochs=10, validation_data=val_generator)

    # Fine-Tune the Last Layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    history_fine = model.fit(train_generator, epochs=5, validation_data=val_generator)

    # Save the model after training
    model.save(MODEL_PATH)
    print("✅ Model trained and saved.")

# Step 4: Evaluate the Model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Step 5: Make Predictions on New Images
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}. Check the file path!")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (ResNet50 expects RGB)
    img = cv2.resize(img, (224, 224))
    
    # Apply correct preprocessing
    img = img.astype(np.float32)
    img = img / 255.0  # Keep normalization consistent with training
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print(f"Predicted: Poisonous 🐍 (Confidence: {prediction:.2f})")
    else:
        print(f"Predicted: Non-Poisonous ✅ (Confidence: {1-prediction:.2f})")


predict_image(r"C:\Users\shaik\Downloads\istockphoto-2114848838-612x612.jpg", model)
