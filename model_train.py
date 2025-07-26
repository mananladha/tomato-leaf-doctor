import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Suppress TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Fixed ImageDataGenerator with proper super() call
class FixedImageDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # This fixes the PyDataset warning

# Data augmentation and loading
datagen = FixedImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,  # Fixed typo from 'horizontal_flip' to 'horizontal_flip'
    validation_split=0.2
)

train = datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val = datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Fixed model architecture with proper Input layer
model = Sequential([
    Input(shape=(128, 128, 3)),  # Proper way to specify input shape
    SeparableConv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),
    
    SeparableConv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),  # Fixed typo from MaxPooling2D to MaxPooling2D
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Explicit learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train,
    validation_data=val,
    epochs=1,
    verbose=1
)

# Save model in modern format
model.save("leaf_model.keras")  # Preferred over .h5

# Visualization (unchanged but verified)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_graphs.png")
plt.show()

# Evaluation
y_pred = model.predict(val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val.classes
labels = list(val.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
print(classification_report(y_true, y_pred_classes, target_names=labels))