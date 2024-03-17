import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data_dir = '/Users/tanekanz/CEPP-2/AugCEPP'  # Dataset PATH
batch_size = 32
image_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # 80:20 split
)

print(f"Training {'MobileNetV3Large'}...")

base_model = MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
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
#history_callback = tf.keras.callbacks.CSVLogger('/Users/tanekanz/CEPP-2/model/training_history.csv')  # PATH

#stop = EarlyStopping(monitor='val_loss', patience=4, mode='min', restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    #callbacks=[history_callback]
    #callbacks=[stop]
)

model_save_path = '/Users/tanekanz/CEPP-2/model/compare/mobilenetv3_b128e10.h5' # Save model PATH
model.save(model_save_path)
print(f"Model {'MobileNetV3Large'} saved to {model_save_path}")  # Save model PATH

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()