import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.models import load_model
import requests
import json
import os

# Load model
model_path = '/Users/tanekanz/CEPP-2/model/mobilenetv2_model_origin.h5'
model = load_model(model_path)

# Fetching labels from ImageNet
response = requests.get('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
imgnet_map = response.json()
imgnet_map = {v[1]: k for k, v in imgnet_map.items()}

# Load class mapping from JSON file
with open('./class_names.json', 'r') as f:
    class_mapping = json.load(f)

# Folder containing images for testing
images_folder = '/Users/tanekanz/CEPP-2/Test/t'

# Iterate over each image in the folder
for filename in os.listdir(images_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(images_folder, filename)

        # Loading and preprocessing image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Make predictions
        predictions = model.predict(img)

        # Adjust decoding based on the loaded class mapping
        top_indices = np.argsort(predictions[0])[::-1][:5]

        # Display top 5 predictions
        print(f"\nPredictions for image: {filename}")
        for i, idx in enumerate(top_indices):
            if 0 <= idx < len(class_mapping):
                label = class_mapping[idx]
            else:
                label = f'Class_{idx}'  # Default to 'Class_idx' if index is out of bounds
            score = predictions[0][idx]
            print(f"{i + 1}: {label} ({score:.2f})")

            # Optionally, you can save the predictions or display them as needed

        # Display the image
        plt.imshow(tf.keras.preprocessing.image.load_img(image_path))
        plt.show()
