import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'dataset/' # This should contain 'orca_vocal' and 'ambient_noise' folders

# 2. Data Preparation (Automatically splits into training and testing)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 20% of images are saved for the "Final Exam"
    rotation_range=5,     # Subtle "Augmentation" to make the model tougher
    width_shift_range=0.1
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 3. The Model (Transfer Learning with MobileNetV2)
# We use a pre-trained "base" and add our own "Orca-specific" layers on top
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the "genius" layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Prevents the model from "cheating" (overfitting)
    layers.Dense(1, activation='sigmoid') #  Orca or No Orca?
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Training
print("Starting AI Training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# 5. Save the Results
model.save('orca_safe_model.h5')
print("Model saved as orca_safe_model.h5")

# 6. Plot Accuracy for Science Fair Board
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Figure 4: Model Learning Curve')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('learning_curve.png')
plt.show()
