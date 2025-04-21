import torch
import torch.nn.functional as F

def validate(model, dataloader, device, l1_criterion, perceptual_criterion=None, lambda_val=0.0):
    """
    Evaluate the model on the validation set.
    Returns the average loss per sample.
    Combines L1 loss with perceptual loss if lambda_val > 0 and perceptual_criterion is provided.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if outputs.shape != targets.shape:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)

            l1_loss = l1_criterion(outputs, targets)

            if lambda_val > 0 and perceptual_criterion is not None:
                from src.utils import normalize_for_vgg  # in case not already imported
                perceptual_loss = perceptual_criterion(normalize_for_vgg(outputs), normalize_for_vgg(targets))
                loss = l1_loss + lambda_val * perceptual_loss
            else:
                loss = l1_loss

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples



def validate_zhang(model, dataloader, device, criterion):
    """
    Evaluate the model on the validation set for Zhang-style colorization.
    - Inputs: grayscale (B, 1, H, W)
    - Targets: class indices (B, H, W)
    - Outputs: logits (B, 313, h, w)
    Returns average loss per sample.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)               # (B, 1, H, W)
            targets = targets.to(device).long()      # (B, H, W)

            outputs = model(inputs)                  # (B, 313, h, w)

            # Resize targets to match model output shape
            if outputs.shape[2:] != targets.shape[1:]:
                targets = F.interpolate(targets.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()

            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples