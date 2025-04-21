from src.evaluate import validate, validate_zhang
import time
from tqdm import tqdm
import torch.nn.functional as F
from src.zhang_model import ZhangColorizationNet
from src.perceptual_loss import PerceptualLoss
from src.utils import normalize_for_vgg

def train_step(model, optimizer, l1_criterion, perceptual_criterion, dataloader, device, lambda_val=0.0):
    """
    Perform a single training step (epoch).
    If λ > 0 and perceptual_criterion is provided, combine L1 and perceptual loss.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training Batches", leave=False)

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if outputs.shape != targets.shape:
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)

        l1_loss = l1_criterion(outputs, targets)

        if lambda_val > 0 and perceptual_criterion is not None:
            perceptual_loss = perceptual_criterion(normalize_for_vgg(outputs), normalize_for_vgg(targets))
            loss = l1_loss + lambda_val * perceptual_loss
        else:
            loss = l1_loss

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / total_samples


def train_step_zhang(model, optimizer, criterion, dataloader, device):
    """
    Perform a single training step (epoch) for Zhang-style colorization model.
    - Inputs: grayscale images (B, 1, H, W)
    - Targets: class indices (B, H, W) for 313 bins
    - Outputs: logits (B, 313, h, w)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training Batches", leave=False)

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)               # shape: (B, 1, H, W)
        targets = targets.to(device).long()      # shape: (B, H, W)

        optimizer.zero_grad()
        outputs = model(inputs)                  # shape: (B, 313, h, w)

        # If the output is smaller (e.g., 64x64), resize the targets to match it
        if outputs.shape[2:] != targets.shape[1:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()

        loss = criterion(outputs, targets)       # CrossEntropyLoss expects (B, C, H, W) + (B, H, W)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / total_samples



def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, lambda_val=None, scheduler=None, device=None):
    train_history = []
    val_history = []

    if lambda_val is not None and lambda_val > 0:
        perceptual_criterion = PerceptualLoss().to(device)
    else:
        perceptual_criterion = None

    for epoch in range(num_epochs):
        t_0 = time.time()
        
        if isinstance(model, ZhangColorizationNet):
            train_loss = train_step_zhang(model, optimizer, criterion, train_dataloader, device)
            val_loss = validate_zhang(model, val_dataloader, device, criterion)
        else:
            train_loss = train_step(model, optimizer, criterion, perceptual_criterion, train_dataloader, device, lambda_val)
            val_loss = validate(model, val_dataloader, device, criterion, perceptual_criterion, lambda_val)

        train_history.append(train_loss)
        val_history.append(val_loss)

        t_1 = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {t_1 - t_0:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(val_loss)

    return model, train_history, val_history
