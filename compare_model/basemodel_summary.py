import tensorflow as tf
from tensorflow.keras.applications import DenseNet201, MobileNetV3Large, InceptionV3, ResNet50, Xception, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data_dir = '/Users/tanekanz/CEPP-2/AugCEPP'

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

# Iterate through each model
for model_name in model_names:
    print(f"Basemodel : {model_name}")

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

    # Build the model on top of the base model
    model = tf.keras.models.Sequential(base_model)

    model.summary()

    print("")

