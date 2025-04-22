from skimage.color import rgb2lab
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
# from utils import set_seed
# set_seed(42)
random.seed(42)


def check_distributions(image_paths, sample_size=1000, size=(256, 256)):
    """
    Check the distribution of the L channel in the LAB color space for a set of images.

    Args:
        image_paths (list): List of paths to the images.
    """
    
    l_values = []
    a_values, b_values = [], []

    if sample_size:
        image_paths = random.sample(image_paths, sample_size)

    for i, path in enumerate(image_paths[:sample_size]):
        img = Image.open(path).convert("RGB").resize(size)
        lab = rgb2lab(np.array(img))
        l_channel = lab[:, :, 0]
        a_values.append(lab[:, :, 1].flatten())
        b_values.append(lab[:, :, 2].flatten())
        l_values.append(l_channel.flatten())

        if i % 100 == 0:
            print(f"Processed {i} images")

    l_values = np.concatenate(l_values)
    # plt.hist(l_values, bins=50, color='gray')
    # plt.title("Distribution of Luminance (L channel)")
    # plt.xlabel("L value")
    # plt.ylabel("Frequency")
    # plt.show()

    a_values = np.concatenate(a_values)
    b_values = np.concatenate(b_values)
    # plt.hexbin(a_values, b_values, gridsize=100, cmap='inferno')
    # plt.title("ab Channel Distribution")
    # plt.xlabel("a values")
    # plt.ylabel("b values")
    # plt.colorbar(label="Pixel Count")
    # plt.show()

    chroma = np.sqrt(a_values**2 + b_values**2)
    plt.hist(chroma, bins=100, color='navy')
    plt.title("Chroma Magnitude Distribution")
    plt.xlabel("Chroma (Color Intensity)")
    plt.ylabel("Pixel Count")
    plt.show()

    plt.hexbin(l_values, chroma, gridsize=80, cmap='inferno')
    plt.xlabel("Luminance (L)")
    plt.ylabel("Chroma")
    plt.title("Chroma vs Luminance")
    plt.colorbar(label="Pixel Count")
    plt.show()


if __name__ == "__main__":
    # Example usage
    image_dir = "image_reconstruction/data/train"
    sample_size = 1000
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    check_distributions(image_paths, sample_size=sample_size)