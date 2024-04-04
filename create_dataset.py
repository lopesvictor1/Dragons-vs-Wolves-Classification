import os
import sys
import cv2
import numpy as np

def load_images_from_folder(folder, width, height):
    """
    Load images and corresponding labels from a folder structure where each subdirectory represents a class.

    Parameters:
        folder (str): Path to the root folder containing subdirectories for each class.

    Returns:
        numpy.ndarray: Array of images loaded from the folder.
        numpy.ndarray: Array of labels corresponding to the images.

    Example:
        images, labels = load_images_from_folder("path/to/your/dataset")
    """
    images = []
    labels = []
    class_folders = sorted(os.listdir(folder))  # List all subdirectories (classes)
    for class_label, class_name in enumerate(class_folders):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".jpeg"):
                    image = cv2.imread(img_path)
                    # Resize or preprocess image if needed
                    resized_image = cv2.resize(image, (width, height))
                    images.append(resized_image)
                    labels.append(class_label)
    return np.array(images), np.array(labels)



if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) != 2:
        print(len(sys.argv))
        print("Usage: python create_dataset.py <image width> <image height>")
        sys.exit(1)
    
    width = int(sys.argv[0])
    height = int(sys.argv[1])
    
    # Example usage:
    folder = os.getcwd()
    images, labels = load_images_from_folder(folder, width, height)

    # Save images and labels to numpy arrays
    np.save("images.npy", images)
    np.save("labels.npy", labels)

    print("Images shape: ", images.shape)
    print("Labels shape: ", labels.shape)

