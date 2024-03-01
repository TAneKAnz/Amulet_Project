import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming you have a dataset with three classes and the data is organized in directories
# such as 'class1', 'class2', and 'class3' in a parent directory 'data'

data_dir = '/Users/tanekanz/CEPP-2/DATASET_ALL/dataset1/train'  # Update with the correct 'dataset' path
batch_size = 128
image_size = (224, 224)

# Create a data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # 80:20 split
)

# Load the MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom layers on top of the base model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(21, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create train and test generators using flow_from_directory
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # specify this is the training set
)

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # specify this is the validation set
)

# Save model history callback
history_callback = tf.keras.callbacks.CSVLogger('/Users/tanekanz/CEPP-2/model/training_history.csv')  # Update with the correct path

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # You can adjust the number of epochs
    validation_data=test_generator,
    callbacks=[history_callback]
)

# Save the entire model
model.save('/Users/tanekanz/CEPP-2/model/mobilenetv2_model_origin.h5')  # Update with the correct path

# Plot training history
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
