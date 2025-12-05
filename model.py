import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Define dataset paths
train_dir = r"C:\Users\HP\Desktop\SIGN_LANGUAGE_WITH_SENTENCE_CONSTRUCTION\dataset\dataset"
# You can add validation and test directories similarly

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # MobileNetV2 input size
    batch_size=32,
    class_mode='categorical'
)

# Load MobileNetV2 model with pre-trained weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Create the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, steps_per_epoch=train_generator.samples // train_generator.batch_size)

# Save the model
model.save("asl_mobilenet_model.h5")

# Plot accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Confusion Matrix
# You need to use a validation set or test set to calculate confusion matrix. 
# Assuming you have a validation set or test set, we can generate the confusion matrix.
# Let's use the validation data generator for this:

# Assuming validation data is similarly loaded with a validation data generator:
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'C:/Users/HP/Desktop/SIGN_LANGUAGE_WITH_SENTENCE_CONSTRUCTION/dataset/dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(validation_generator)

# Predict labels
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = validation_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
cm_df = pd.DataFrame(cm, index=train_generator.class_indices.keys(), columns=train_generator.class_indices.keys())

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
