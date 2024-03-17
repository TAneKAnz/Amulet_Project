import tensorflow as tf
from tensorflow.keras.applications import DenseNet201, MobileNetV3Large, InceptionV3, ResNet50, Xception, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data_dir = '/Users/tanekanz/CEPP-2/AugCEPP'
test_dir = '/Users/tanekanz/CEPP-2/AugTest/Augt'
batch_size = 32

# Specify the model names
model_names = ["ResNet-50", "InceptionV3", "Xception", "DenseNet-201", "MobileNetV3", "MobileNetV2"]

# Define a list to store the models and their training histories
models = []

# Specify the input shape based on the selected models
input_shape_dict = {
    "DenseNet-201": (224, 224, 3),
    "MobileNetV3": (224, 224, 3),
    "InceptionV3": (299, 299, 3),  # InceptionV3 requires (299, 299) input size
    "ResNet-50": (224, 224, 3),
    "Xception": (299, 299, 3),  # Xception requires (299, 299) input size
    "MobileNetV2": (224, 224, 3),
}

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # 80:20 split
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# Iterate through each model
for model_name in model_names:
    print(f"Training {model_name}...")

    # Create the base model
    if model_name == "ResNet-50":
        base_model = ResNet50(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "InceptionV3":
        base_model = InceptionV3(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "Xception":
        base_model = Xception(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "DenseNet-201":
        base_model = DenseNet201(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "MobileNetV3":
        base_model = MobileNetV3Large(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')
    elif model_name == "MobileNetV2":
        base_model = MobileNetV2(input_shape=input_shape_dict[model_name], include_top=False, weights='imagenet')


    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Build the model on top of the base model
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(21, activation='softmax')  # N classes
    ])

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create data generators
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape_dict[model_name][:2],  # Adjust target size
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape_dict[model_name][:2],  # Adjust target size
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape_dict[model_name][:2],  # Adjust target size
        class_mode='categorical',
    )

    # Train the model
    #stop = EarlyStopping(monitor='val_loss', patience=4, mode='min', restore_best_weights=True)
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        #callbacks=[stop]
    )

    model_save_path = f'/Users/tanekanz/CEPP-2/model/compare/{model_name.lower()}_b32e10(2).h5'
    model.save(model_save_path)
    print(f"Model {model_name} saved to {model_save_path}")

    # Save the model and history in the list
    models.append((model, history))

    print("-"*90)
    print()
    pred = model.predict(test_generator).argmax(axis=1)
    print(classification_report(test_generator.classes, pred))
    print()
    print("-"*90)
    print()

# Plot the results
plt.figure(figsize=(12, 12))

# Plot training accuracy for each model
plt.subplot(2, 2, 1)
for model_name, (model, history) in zip(model_names, models):
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()

# Plot validation accuracy for each model
plt.subplot(2, 2, 2)
for model_name, (model, history) in zip(model_names, models):
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()

# Plot training loss for each model
plt.subplot(2, 2, 3)
for model_name, (model, history) in zip(model_names, models):
    plt.plot(history.history['loss'], label=f'{model_name} Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# Plot validation loss for each model
plt.subplot(2, 2, 4)
for model_name, (model, history) in zip(model_names, models):
    plt.plot(history.history['val_loss'], label=f'{model_name} Validation loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()

plt.tight_layout()
plt.show()