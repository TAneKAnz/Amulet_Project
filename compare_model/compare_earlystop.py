import tensorflow as tf
from tensorflow.keras.applications import DenseNet201, MobileNetV3Large, InceptionV3, ResNet50, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data_dir = '/Users/tanekanz/CEPP-2/AugCEPP'
batch_size = 128

# Specify the model names
model_names = [ "MobileNetV3", "DenseNet-201", "InceptionV3", "ResNet-50", "Xception"]

# Define a list to store the models and their training histories
models = []

# Specify the input shape based on the selected models
input_shape_dict = {
    "DenseNet-201": (224, 224, 3),
    "MobileNetV3": (224, 224, 3),
    "InceptionV3": (299, 299, 3),  # InceptionV3 requires (299, 299) input size
    "ResNet-34": (224, 224, 3),
    "Xception": (299, 299, 3)  # Xception requires (299, 299) input size
}

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # 80:20 split
)

# Iterate through each model
for model_name in model_names:
    print(f"Training {model_name}...")

    # Create the base model
    if model_name == "DenseNet-201":
        base_model = DenseNet201(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "MobileNetV3":
        base_model = MobileNetV3Large(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "InceptionV3":
        base_model = InceptionV3(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "ResNet-50":
        base_model = ResNet50(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "Xception":
        base_model = Xception(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Build the model on top of the base model
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(21, activation='softmax')  # N classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create data generators
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape_dict[model_name][:2],  # Adjust target size
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape_dict[model_name][:2],  # Adjust target size
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Train the model
    stop = EarlyStopping(monitor='val_loss', patience=4, mode='min', restore_best_weights=True)
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=test_generator,
        callbacks=[stop]
    )

    # Save the model and history in the list
    models.append((model, history))

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training accuracy for each model
plt.subplot(1, 2, 1)
for model_name, (model, history) in zip(model_names, models):
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()

# Plot validation accuracy for each model
plt.subplot(1, 2, 2)
for model_name, (model, history) in zip(model_names, models):
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()

plt.tight_layout()
plt.show()
