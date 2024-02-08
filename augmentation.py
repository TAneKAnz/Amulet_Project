import os
import cv2
import numpy as np

# Function to create a new folder if it does not exist
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to perform image augmentation for all folders in the specified directory
def augment_images_for_all_folders(main_folder, output_folder_prefix='Aug', num_augmentations=5):
    # Set the output folder path for AugCEPP
    augcepp_output_folder = os.path.join(os.path.dirname(main_folder), 'AugCEPP')  # Updated output folder path

    # List all folders in the main directory
    folders = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

    # Iterate through each folder
    for folder in folders:
        # Set the input and output folder paths
        input_folder_path = os.path.join(main_folder, folder)

        # Adjusted output_folder_name to avoid duplication
        output_folder_name = f'{output_folder_prefix}{folder}' if not folder.startswith(output_folder_prefix) else folder
        output_folder_path = os.path.join(augcepp_output_folder, output_folder_name)  # Updated output folder path

        # Perform image augmentation for the current folder
        augment_images(input_folder_path, output_folder_path, num_augmentations)

# Function to perform image augmentation
def augment_images(input_folder, output_folder, num_augmentations=5):
    # Create an output folder if it does not exist
    create_folder(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file in the input folder
    for file in files:
        # Construct the full path of the input file
        input_file_path = os.path.join(input_folder, file)

        # Read the image using OpenCV
        original_image = cv2.imread(input_file_path)

        # Check if the image is None
        if original_image is None:
            print(f"Skipping {file} as it could not be read.")
            continue

        # Generate augmented versions using each method
        for method in ['original', 'flip', 'rotate', 'color', 'skew', 'noise']:
            for i in range(num_augmentations):
                augmented_image = apply_augmentation(original_image, method)

                # Write the augmented image with noise
                output_file_name = f"{method}_{os.path.splitext(file)[0]}_{i+1}.jpg"
                cv2.imwrite(os.path.join(output_folder, output_file_name), augmented_image)

def apply_augmentation(image, method):
    # Method: original (no augmentation)
    if method == 'original':
        return image

    # Method: flip horizontally
    elif method == 'flip':
        return cv2.flip(image, 1)

    # Method: rotate
    elif method == 'rotate':
        rotation_angle = np.random.uniform(-30, 30)
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Method: brightness, contrast, saturation
    elif method == 'color':
        # Adjust brightness
        brightness_factor = np.random.uniform(0.5, 1.5)
        augmented_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

        # Adjust contrast
        contrast_factor = np.random.uniform(0.7, 1.3)
        augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast_factor, beta=0)

        # Adjust saturation
        hsv_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV)
        saturation_factor = np.random.uniform(0.5, 1.5)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return augmented_image

    # Method: skew
    elif method == 'skew':
        skew_matrix = np.float32([[1, 0.5, 0], [0, 1, 0]])
        return cv2.warpAffine(image, skew_matrix, (image.shape[1], image.shape[0]))

    # Method: noise
    elif method == 'noise':
        noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

# Set the main folder path
main_folder = '/Users/tanekanz/CEPP-2/CEPP'

# Perform image augmentation for all folders in the specified directory
augment_images_for_all_folders(main_folder)
