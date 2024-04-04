import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates

def rotate_image(image):
    """
    Rotates the image by a random angle within a specified range.

    Parameters:
        image (numpy.ndarray): Input image.
        angle_range (tuple): Range of angles for random rotation (default: (-30, 30)).

    Returns:
        numpy.ndarray: Rotated image.
    """
    
    angle = np.random.randint(-30, 30)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def flip_image(image):
    """
    Horizontally flips the image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Horizontally flipped image.
    """
    return cv2.flip(image, 1)

def adjust_brightness_contrast(image):
    """
    Adjusts the brightness and contrast of the image.

    Parameters:
        image (numpy.ndarray): Input image.
        alpha (float): Contrast adjustment factor (default: 1.5).
        beta (int): Brightness adjustment value (default: 10).

    Returns:
        numpy.ndarray: Image with adjusted brightness and contrast.
    """
    alpha = 1.5  
    beta = 10    
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds gaussian noise to the image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the normal distribution (Standard: 0).
        sigma (float): Standard deaviation of the normal distribution (Standard: 25).
        
    Returns:
        numpy.ndarray: Imagem com ruído gaussiano adicionado.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_elastic_distortion(image, alpha=200, sigma=10):
    """
    Aplica distorção elástica à imagem.
    
    Parameters:
        image (numpy.ndarray): Imagem de entrada.
        alpha (int): Parâmetro de deformação elástica (padrão: 200).
        sigma (int): Desvio padrão do filtro gaussiano para suavização (padrão: 10).
        
    Returns:
        numpy.ndarray: Imagem com distorção elástica aplicada.
    """
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(image.shape)


# Folder of the intended augmented images
directory = "/home/lopesvictor/git/Beans-DataAugmentation/dragons"

# List that will store augmented images
augmented_images = []

# List that will store augmented images names
image_name = []

# Loop over every image in the folder
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        new_filename = filename[:-4]
        print(f' Utilizing image: {filename}...')
        
        # Load the image
        image = cv2.imread(os.path.join(directory, filename))
        
        # Apply the Data Augmentation functions
        print('Rotating the image...')
        rotated_image = rotate_image(image)
        print('Inverting the image...')
        flipped_image = flip_image(image)
        print('Ajusting brightness contrast of the image...')
        adjusted_image = adjust_brightness_contrast(image)
        print('Adding gaussian noise to the image...')
        noisy_image = add_gaussian_noise(image)
        print('Adding elastic distortion to the image...')
        distorted_image = add_elastic_distortion(image)
        
        # Adding the generated images to the list
        augmented_images.extend([rotated_image, flipped_image, adjusted_image, noisy_image, distorted_image])
        # Adding the generated images names to the list
        image_name.extend([new_filename+'rotated', new_filename+'flipped', new_filename+'adjusted', new_filename+'noisy', new_filename+'distorted'])

# Saving the generated images
output_directory = "/home/lopesvictor/git/Beans-DataAugmentation/dragons"
for i, image in enumerate(augmented_images):
    cv2.imwrite(os.path.join(output_directory, f"augmented_{image_name[i]}.jpg"), image)
