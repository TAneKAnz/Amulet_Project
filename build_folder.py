import os

def create_folder(base_name, path="."):
    folder_name = os.path.join(path, base_name)
    counter = 1

    while os.path.exists(folder_name):
        counter += 1
        folder_name = os.path.join(path, f"{base_name}{counter}")

    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")

# Example usage:
base_folder_name = "A"
target_path = "/Users/tanekanz/CEPP-2/folder"  # Replace this with the desired path
create_folder(base_folder_name, target_path)
