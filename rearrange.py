import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to your original dataset
original_dataset_path = '/Users/tanekanz/CEPP-2/AugCEPP'  # Adjust this path to the location of your 'dataset' folder

# Set the path to create the new dataset structure
new_dataset_path = '/Users/tanekanz/CEPP-2'  # Adjust this path to the desired location

# Create the 'dataset' folder
dataset_path = os.path.join(new_dataset_path, 'dataset')
os.makedirs(dataset_path, exist_ok=True)

# Create train and validation directories within 'dataset'
train_path = os.path.join(dataset_path, 'train')
validation_path = os.path.join(dataset_path, 'validation')

os.makedirs(train_path, exist_ok=True)
os.makedirs(validation_path, exist_ok=True)

# List all 'Aug' folders
aug_folders = [folder for folder in os.listdir(original_dataset_path) if folder.startswith('Aug')]

# Loop through each 'Aug' folder
for aug_folder in aug_folders:
    aug_folder_path = os.path.join(original_dataset_path, aug_folder)

    # Extract class name from folder name
    class_name = aug_folder[3:]  # Assuming the class name follows 'Aug' in the folder name

    # Create corresponding 'train' and 'validation' folders
    train_class_path = os.path.join(train_path, f'train{class_name}')
    validation_class_path = os.path.join(validation_path, f'valid{class_name}')

    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(validation_class_path, exist_ok=True)

    # List all images in the 'Aug' folder
    images = [img for img in os.listdir(aug_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Split the images into train and validation sets
    train_images, validation_images = train_test_split(images, test_size=0.2, random_state=42)

    # Move images to the respective directories
    for img in train_images:
        src = os.path.join(aug_folder_path, img)
        dest = os.path.join(train_class_path, img)
        shutil.copy(src, dest)

    for img in validation_images:
        src = os.path.join(aug_folder_path, img)
        dest = os.path.join(validation_class_path, img)
        shutil.copy(src, dest)

print("Dataset rearranged successfully.")
