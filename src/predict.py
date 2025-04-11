from PIL import Image
import os
import torch
from torchvision import transforms

def predict_image(model, image, size, device):
    """
    Predict the image using the model.
    """
    # Read the image
    image = Image.open(image).convert("RGB")

    original_image = image.copy()   

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    # Predict the image
    with torch.no_grad():
        model.eval()
        pred = model(image)
        pred = pred.squeeze(0).cpu()
        pred = transforms.ToPILImage()(pred)
        pred = pred.resize(original_image.size)

    return original_image, pred

    