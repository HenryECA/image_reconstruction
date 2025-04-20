from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torch.nn.functional as F
import cv2

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


def predict_image_zhang(model, image_path, size, device, pts_in_hull_path="pts_in_hull.npy", temperature=0.38):
    """
    Perform inference with Zhang model and reconstruct color image.
    Returns: original RGB image, predicted RGB image
    """
    # Load color bins
    pts_in_hull = np.load(pts_in_hull_path)  # (313, 2)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_resized = transforms.Resize(size)(image)

    # Grayscale input
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    input_l = transform(image_resized).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Also get original L for reconstruction
    image_np = np.array(image_resized)
    lab = rgb2lab(image_np)
    L_orig = lab[:, :, 0] / 100.0  # Normalize to [0, 1]
    L_orig_tensor = torch.from_numpy(L_orig).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_l)  # (1, 313, h, w)
        output = F.softmax(output / temperature, dim=1)  # apply temperature
        output = output[0].cpu().numpy()  # (313, h, w)

    # Convert output probabilities to ab channels using soft annealed mean
    ab = np.tensordot(output.transpose(1, 2, 0), pts_in_hull, axes=([2], [0]))  # (h, w, 2)

    # Upsample ab to match L_orig size
    ab = cv2.resize(ab, size, interpolation=cv2.INTER_CUBIC)  # (H, W, 2)
    L_resized = L_orig_tensor.squeeze().cpu().numpy()  # (H, W)

    # Stack L and ab → Lab image
    lab_pred = np.concatenate((L_resized[:, :, np.newaxis] * 100, ab), axis=2)  # (H, W, 3)

    # Convert to RGB
    rgb_pred = lab2rgb(lab_pred)
    rgb_pred = (rgb_pred * 255).astype(np.uint8)

    # Resize to original size
    rgb_pred = cv2.resize(rgb_pred, original_size, interpolation=cv2.INTER_CUBIC)  # (original_H, original_W, 3)

    return image, Image.fromarray(rgb_pred)

    