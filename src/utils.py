import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json

def normalize_for_vgg(img):
    # Imagenet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std

def show_image(original, pred, save_path=None):
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

    if save_path:
        plt.savefig(os.path.join(save_path))
        print(f"Prediction saved at {save_path}")

    plt.show()

    
def save_model(model, path, name, parameters=None):
    """
    Save the model to the specified path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), f"{path}/{name}.pth")
    print(f"Model saved at {path}/{name}.pth")

    # Save a json file with the model parameters
    if parameters is not None:
        with open(f"{path}/{name}_params.json", "w") as f:
            json.dump(parameters, f)
        print(f"Model parameters saved at {path}/{name}_params.json")

    

def load_model(model, path, name, parameters=False):
    """
    Load the model from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path {path} does not exist.")

    model.load_state_dict(torch.load(f"{path}/{name}.pth"))

    if parameters:
        with open(f"{path}/{name}_params.json", "r") as f:
            params = json.load(f)
        print(f"Model parameters loaded from {path}/{name}_params.json")
        return model, params

    return model, None


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
