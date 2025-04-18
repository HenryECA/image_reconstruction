import torch
import torch.nn.functional as F

def validate(model, dataloader, device, criterion):
    """
    Evaluate the model on the validation set (used during training).
    Returns the average loss per sample.
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
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

