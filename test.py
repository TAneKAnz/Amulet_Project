import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

# Load your trained model
model_path = '/Users/tanekanz/CEPP-2/model/mobilenetv2_model_grey_notcallback.h5'  # Update with the correct path model
model = load_model(model_path)

# Path to the folder containing images for prediction
test_dataset_path = '/Users/tanekanz/CEPP-2/Test_GREY/t'  # Update with the correct path [origin, grey]

# Load the class names used during training
class_names_file = 'class_names.json'  # Update with the correct path
with open(class_names_file, 'r') as f:
    class_names = json.load(f)

# List all image files in the folder
image_files = [f for f in os.listdir(test_dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define a function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) #fix target_size (128, 224) for each model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    return img_array

# Initialize variables for accuracy calculation
total_images = len(image_files)
correct_predictions = 0
mismatched_pairs = []

# Loop through each image file and make predictions
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(test_dataset_path, image_file)

    # Preprocess the image for prediction
    img_array = preprocess_image(image_path)

    # Get predictions
    predictions = model.predict(img_array)

    # Map predicted class index to class name
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]

    # Get the ground truth class (extracted from the folder name)
    true_class_name = image_file  # Adjust this based on your folder naming convention

    # Check if the prediction is a substring of the true class name or vice versa
    if predicted_class_name in true_class_name or true_class_name in predicted_class_name:
        correct_predictions += 1
    else:
        mismatched_pairs.append((predicted_class_name, true_class_name))

    print(f"Image: {image_file}, Predicted Class: {predicted_class_name}, True Class: {true_class_name}")

# Print mismatched pairs
print("\nMismatched Pairs:")
for pair in mismatched_pairs:
    print(f"{pair[0]} ---- {pair[1]}")

# Calculate accuracy
accuracy = correct_predictions / total_images
print(f"\nUse model : {os.path.basename(model_path)}")
print(f"Total Correct Predictions: {correct_predictions}/{total_images}")
print(f"Accuracy: {accuracy * 100:.2f}%")
