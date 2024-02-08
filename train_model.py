import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Check if GPU is available
if tf.config.experimental.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("No GPU detected. Using CPU.")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Define your CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Optional dropout for regularization
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define image dimensions and number of classes
input_shape = (128, 128, 3)  # adjust the input size according to your images

# Get the list of class names dynamically from the folder names
class_names = sorted(os.listdir('/Users/tanekanz/CEPP-2/dataset/train'))

# Get the number of classes
num_classes = len(class_names)

# Create the model
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Set up data augmentation and normalization
datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0,
                             horizontal_flip=False,
                             validation_split=0.2)

# Set up data generators for training and validation
batch_size = 32
train_generator = datagen.flow_from_directory('/Users/tanekanz/CEPP-2/dataset/train',
                                              target_size=(128, 128),
                                              batch_size=batch_size,
                                              class_mode='sparse',
                                              subset='training')

validation_generator = datagen.flow_from_directory('/Users/tanekanz/CEPP-2/dataset/validation',
                                                   target_size=(128, 128),
                                                   batch_size=batch_size,
                                                   class_mode='sparse',
                                                   subset='validation')

# Calculate steps_per_epoch for training
steps_per_epoch_train = len(train_generator)

# Calculate steps_per_epoch for validation
steps_per_epoch_validation = len(validation_generator)

# Set up ModelCheckpoint callback to save the model only after training finishes
checkpoint_path = '/Users/tanekanz/CEPP-2/model/model_checkpoint.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True, save_freq='epoch')

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=steps_per_epoch_validation,
                    callbacks=[checkpoint_callback])

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.save('your_model.h5')
