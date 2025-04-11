from src.evaluate import validate
import time

def train_step(model, optimizer, criterion, dataloader, device):
    """
    Perform a single training step (epoch), correctly weighted by batch size.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples



def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, scheduler=None, device=None):
    """
    Train the model.
    """
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        t_0 = time.time()
        train_loss = train_step(model, optimizer, criterion, train_dataloader, device)
        val_loss = validate(model, val_dataloader, device, criterion)

        train_history.append(train_loss)
        val_history.append(val_loss)

        t_1 = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs} in {t_1 - t_0:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(val_loss)

    return model, train_history, val_history
