from src.evaluate import validate
import time
from tqdm import tqdm
import torch.nn.functional as F

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


def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, scheduler=None, device=None):
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        t_0 = time.time()
        
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
