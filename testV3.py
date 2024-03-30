import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

# Load your trained model
model_path = '/Users/tanekanz/CEPP-2/model/InceptionV3/InceptionV3_model.h5'  # Update with the correct path model
model = load_model(model_path)

# Parent folder containing subfolders with images for prediction
parent_folder_path = '/Users/tanekanz/CEPP-2/CEPP'  # Update with the correct path [origin, grey]

class_path = '/Users/tanekanz/CEPP-2/CEPP'

# Get the subfolder names as class names
class_names = sorted(os.listdir(class_path))[1:] # Get the class names

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0
mismatched_pairs = []

# Define a function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))  # fix target_size (128, 224) for each model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array /= 255.0  # Normalize the image

    return img_array

# Iterate over each subfolder
for subfolder in class_names:
    subfolder_path = os.path.join(parent_folder_path, subfolder)

    # Skip if it's not a directory
    if not os.path.isdir(subfolder_path):
        continue

    # List all image files in the subfolder
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Update the total number of images
    total_images += len(image_files)

    # Loop through each image file and make predictions
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(subfolder_path, image_file)

        # Preprocess the image for prediction
        img_array = preprocess_image(image_path)

        # Get predictions
        predictions = model.predict(img_array)

        # Map predicted class index to class name
        predicted_class_index = np.argmax(predictions[0])
        # Map predicted class index to class name
        if predicted_class_index < len(class_names):
            predicted_class_name = class_names[predicted_class_index]
        else:
            predicted_class_name = "Unknown"


        # Get the ground truth class (extracted from the folder name)
        true_class_name = subfolder  # Adjust this based on your folder naming convention

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
print(f"\nUse model : {os.path.basename(model_path)}")
print(f"Total Correct Predictions: {correct_predictions}/{total_images}")

if total_images > 0:
    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy * 100:.2f}%")
else:
    print("No images found for prediction, cannot calculate accuracy.")