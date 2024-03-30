import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = '/Users/tanekanz/CEPP-2/AugCEPP7'  # Dataset PATH
batch_size = 32
image_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # 80:20 split
)

print(f"Training {'MobileNetV2'}...")

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(21, activation='softmax')  # N classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save model history callback
history_callback = tf.keras.callbacks.CSVLogger('/Users/tanekanz/CEPP-2/model_history/mobilenetv2_history.csv')  # PATH

#stop = EarlyStopping(monitor='val_loss', patience=4, mode='min', restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[history_callback]
    #callbacks=[stop]
)

#model_save_path = '/Users/tanekanz/CEPP-2/model/mobilenetv2/mobilenetv2_model(try_version).h5' # Save model PATH
#model.save(model_save_path)
#print(f"Model {'MobileNetV2'} saved to {model_save_path}")

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

multilabel_confusion = multilabel_confusion_matrix(y_true, y_pred)

f1_scores = np.zeros(21)  # Assuming 21 classes

for i in range(21):
    confusion_matrix = multilabel_confusion[i]
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_scores[i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"F1 scores for each class: ")
print(f1_scores)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    multilabel_confusion,  # Using the full multilabel confusion matrix
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=range(21),  # Labels for classes on x-axis
    yticklabels=range(21),  # Labels for classes on y-axis
    cbar=True,
    square=True
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Multilabel Confusion Matrix')
plt.show()


plt.figure(figsize=(12, 6))

# Plot training accuracy for each model
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()