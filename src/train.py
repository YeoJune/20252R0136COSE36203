"""
Training loop for poker models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


def train_epoch(model, train_loader, criterion, optimizer, device='cuda'):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in tqdm(train_loader, desc='Training', leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_multimodal_epoch(model, train_loader, criterion, optimizer, device='cuda'):
    """
    Train multimodal model for one epoch
    
    Args:
        model: Multimodal PyTorch model
        train_loader: DataLoader returning (game_features, text_embeddings, labels)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for game_batch, text_batch, y_batch in tqdm(train_loader, desc='Training', leave=False):
        game_batch = game_batch.to(device)
        text_batch = text_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(game_batch, text_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        avg_loss: Average loss
        accuracy: Test accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Evaluating', leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def evaluate_multimodal(model, test_loader, criterion, device='cuda'):
    """
    Evaluate multimodal model on test set
    
    Args:
        model: Multimodal PyTorch model
        test_loader: DataLoader returning (game_features, text_embeddings, labels)
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        avg_loss: Average loss
        accuracy: Test accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for game_batch, text_batch, y_batch in tqdm(test_loader, desc='Evaluating', leave=False):
            game_batch = game_batch.to(device)
            text_batch = text_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(game_batch, text_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model,
    train_loader,
    test_loader,
    n_epochs=10,
    learning_rate=0.0005,
    class_weights=None,
    device='cuda',
    checkpoint_dir='checkpoints',
    model_name='poker_model',
    multimodal=False,
    config=None,
    pca=None
):
    """
    Full training loop with evaluation and checkpointing
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        n_epochs: Number of epochs
        learning_rate: Learning rate
        class_weights: Tensor of class weights for imbalanced data
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        model_name: Name for checkpoint files
        multimodal: Whether using multimodal model
        
    Returns:
        history: Dict containing training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss and optimizer
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train
        if multimodal:
            train_loss, train_acc = train_multimodal_epoch(
                model, train_loader, criterion, optimizer, device
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
        
        # Evaluate
        if multimodal:
            test_loss, test_acc, _, _ = evaluate_multimodal(
                model, test_loader, criterion, device
            )
        else:
            test_loss, test_acc, _, _ = evaluate(
                model, test_loader, criterion, device
            )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"{model_name}_epoch_{epoch+1}.pt"
        )
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
        }
        
        # Add config and pca if provided
        if config is not None:
            checkpoint['config'] = config
        if pca is not None:
            checkpoint['pca'] = pca
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ New best model saved (Test Acc: {test_acc:.2f}%)")
    
    print(f"\n✓ Training complete!")
    print(f"  Best test accuracy: {best_test_acc:.2f}%")
    
    return history


def load_model_from_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model weights from checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from {checkpoint_path}")
    if 'test_acc' in checkpoint:
        print(f"  Test Acc: {checkpoint['test_acc']:.2f}%")
    
    return model
