#import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.models import load_model
import requests
import json

model_path = '/Users/tanekanz/CEPP-2/model/mobilenetv2_model_origin.h5'  # Update with the correct path model
model = load_model(model_path)

print(model.summary)

#loading and preprocessing cat image
IMAGE_PATH = './nangpraya_praykhu (1).jpg'
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)
#view the image
plt.imshow(img/255.)

#fetching labels from Imagenet
response = requests.get('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
imgnet_map = response.json()
imgnet_map = {v[1]:k for k, v in imgnet_map.items()}

#make model predictions
img = preprocess_input(img)
predictions = model.predict(np.array([img]))

# Load class mapping from JSON file
with open('./class_names.json', 'r') as f:
    class_mapping = json.load(f)

# Adjust decoding based on the loaded class mapping
top_indices = np.argsort(predictions[0])[::-1][:5]

# Display top 5 predictions
for i, idx in enumerate(top_indices):
    if 0 <= idx < len(class_mapping):
        label = class_mapping[idx]
    else:
        label = f'Class_{idx}'  # Default to 'Class_idx' if index is out of bounds
    score = predictions[0][idx]
    print(f"{i + 1}: {label} ({score:.2f})")

plt.show()
