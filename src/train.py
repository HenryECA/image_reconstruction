from src.evaluate import validate, validate_zhang
import time
from tqdm import tqdm
import torch.nn.functional as F
from src.zhang_model import ZhangColorizationNet

def train_step(model, optimizer, criterion, dataloader, device):
    """
    Perform a single training step (epoch), with tqdm showing progress per batch.
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

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / total_samples

import torch.nn.functional as F
from tqdm import tqdm

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



def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, scheduler=None, device=None):
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        t_0 = time.time()
        
        if isinstance(model, ZhangColorizationNet):
            train_loss = train_step_zhang(model, optimizer, criterion, train_dataloader, device)
            val_loss = validate_zhang(model, val_dataloader, device, criterion)
        else:
            train_loss = train_step(model, optimizer, criterion, train_dataloader, device)
            val_loss = validate(model, val_dataloader, device, criterion)

        train_history.append(train_loss)
        val_history.append(val_loss)

        t_1 = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {t_1 - t_0:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(val_loss)

    return model, train_history, val_history
