import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models
from keras.optimizers import Adam
import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = '/Users/tanekanz/DL/augmented2'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Check if the base directory exists, and create it if not
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Check and create the subdirectories if they do not exist
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

dataset_dir = '/Users/tanekanz/DL/pokemon2'
pokemons = os.listdir(dataset_dir)

for pokemon in pokemons:
    # Skip non-directory entries (like .DS_Store)
    if not os.path.isdir(os.path.join(dataset_dir, pokemon)):
        continue

    pokemon_dir = os.path.join(dataset_dir, pokemon)
    train_pokemon_dir = os.path.join(train_dir, pokemon)
    validation_pokemon_dir = os.path.join(validation_dir, pokemon)
    test_pokemon_dir = os.path.join(test_dir, pokemon)
    if not os.path.exists(train_pokemon_dir):
        os.mkdir(train_pokemon_dir)
    if not os.path.exists(validation_pokemon_dir):
        os.mkdir(validation_pokemon_dir)
    if not os.path.exists(test_pokemon_dir):
        os.mkdir(test_pokemon_dir)

    train_images, temp_images = train_test_split(os.listdir(pokemon_dir), test_size=0.2, random_state=42)
    validation_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    for image in train_images:
        src = os.path.join(pokemon_dir, image)
        dst = os.path.join(train_pokemon_dir, image)
        shutil.copyfile(src, dst)

    for image in validation_images:
        src = os.path.join(pokemon_dir, image)
        dst = os.path.join(validation_pokemon_dir, image)
        shutil.copyfile(src, dst)

    for image in test_images:
        src = os.path.join(pokemon_dir, image)
        dst = os.path.join(test_pokemon_dir, image)
        shutil.copyfile(src, dst)

# Split dataset into Train, Validation & Test sets
size = 224   # Resize all images to (size,size)
bs = 32      # Batch size

#validation_split=0.15,
# Data augmentation
train_data_gen = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, zoom_range=0.1, shear_range=0.1, brightness_range=[0.8,1.2], preprocessing_function=preprocess_input)

train_data = train_data_gen.flow_from_directory(base_dir + '/train', class_mode='categorical', target_size=(size,size), color_mode='rgb', batch_size=bs, seed=42)

validation_data = train_data_gen.flow_from_directory(base_dir + '/validation', class_mode='categorical', target_size=(size,size), color_mode='rgb', batch_size=bs, seed=42)

test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = test_data_gen.flow_from_directory(base_dir + "/test", class_mode='categorical', target_size=(size,size), color_mode='rgb', shuffle=False)

shape = train_data.image_shape                 # Shape of train images (height,width,channels)
print(shape)
k = train_data.num_classes                     # Total number of labels or classes
train_samples = train_data.samples             # Total number of images in train set
validation_samples = validation_data.samples   # total number of images in validation set

input = Input(shape=shape)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # Basemodel is InceptionV3 with pretrained weights trained on imagenet dataset
for layer in base_model.layers:
    layer.trainable = False                                                                   # Freeze the weights in all layers of the CNN

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(19, activation='softmax'))  # Set num_classes based on your dataset

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize callbacks
stop = EarlyStopping(monitor='val_loss', patience=4, mode='min', restore_best_weights=True)                                             # Stops training early to prevent overfitting
checkpoint = ModelCheckpoint(filepath='{val_loss:.4f}-weights-{epoch:02d}.hdf5', monitor='val_loss', mode='min', save_best_only=True)   # Saves the model for every best val_loss

# Model Summary
model.summary()

# Train the model
ep = 100                      # Number of epochs
spe = train_samples//bs       # Steps per epoch
vs = validation_samples//bs   # Validation steps

r = model.fit(train_data, validation_data=validation_data, steps_per_epoch=spe, validation_steps=vs, epochs=ep, callbacks=[stop,checkpoint])

# Evaluate the model
model.evaluate(validation_data)

# save model
model.save('/Users/tanekanz/DL/model/mnetv2_pokemon_final.h5')

# Plot training history
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# Predictions on the test data
pred = model.predict(test_data).argmax(axis=1)
labels = list(train_data.class_indices.keys())

# Visualize random predictions on the test data
rand = np.random.randint(low=0, high=test_data.samples, size=5)

for n in rand:
    true_index = test_data.classes[n]
    predicted_index = pred[n]
    img = cv2.imread(test_data.filepaths[n])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('True label={}  Predicted label={}'.format(labels[true_index], labels[predicted_index]))
    plt.show()