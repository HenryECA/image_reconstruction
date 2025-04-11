import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def show_image(original, pred):
    """
    Show the original, grayscale, and predicted images.
    """

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(pred)
    ax[1].set_title("Predicted Image")
    ax[1].axis("off")

    plt.show()
    
def save_model(model, path, name):
    """
    Save the model to the specified path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), f"{path}/{name}.pth")
    print(f"Model saved at {path}/{name}.pth")
    

def load_model(model, path, name):
    """
    Load the model from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path {path} does not exist.")

    model.load_state_dict(torch.load(f"{path}/{name}.pth"))

    return model


def seed_all(seed: int = 42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    
def MAE(pred, target):
    """
    Mean Absolute Error (MAE) between predicted and target images.
    """

    return torch.mean(torch.abs(pred - target))

def MSE(pred, target):
    """
    Mean Squared Error (MSE) between predicted and target images.
    """

    return torch.mean((pred - target) ** 2)
