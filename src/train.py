import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    """
    Train the given model.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on ('cpu' or 'cuda').
        num_epochs: Number of epochs to train.

    Returns:
        model: Trained model.
        train_losses: List of training losses per epoch.
    """
    model.to(device)
    model.train()
    
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, leave=False)
        
        for inputs, targets, domain_labels in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs} Training Loss: {epoch_loss:.4f}')
    
    return model, train_losses
