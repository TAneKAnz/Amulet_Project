import os
import cv2

def convert_folder_to_grayscale(input_folder, output_folder):
    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions as needed
            input_path = os.path.join(input_folder, filename)

            # Read the color image
            color_image = cv2.imread(input_path)

            # Convert to grayscale
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Save the grayscale image to the output folder
            output_path = os.path.join(output_folder, filename)
            #print(os.path.basename(output_path), end="=>")
            cv2.imwrite(output_path, grayscale_image)

if __name__ == "__main__":
    # Set the paths to your input and output folders
    input_root_folder = "/Users/tanekanz/CEPP-2/Test"
    output_root_folder = "/Users/tanekanz/CEPP-2/Test/Test_GREY"

    # Iterate through each subfolder in the root input folder
    for subfolder in os.listdir(input_root_folder):
        subfolder_path = os.path.join(input_root_folder, subfolder)

        # Check if the item is a directory
        if os.path.isdir(subfolder_path):
            # Convert images in the subfolder to grayscale
            output_subfolder = os.path.join(output_root_folder, subfolder)
            convert_folder_to_grayscale(subfolder_path, output_subfolder)
            print('converting : %s to grayscale' % output_subfolder)
            
print('Finished converting!!')
