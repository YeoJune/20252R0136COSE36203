"""
Step 2: Train Baseline Model
Train MLP model on game state features only
"""
import sys
sys.path.append('src')

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from dataset import load_processed_data, PokerDataset
from models import get_model
from train import train_model, evaluate
from evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_training_history,
    plot_class_distribution,
    save_evaluation_report
)

# Configuration
CONFIG = {
    'batch_size': 256,
    'learning_rate': 0.0002,
    'n_epochs': 30,
    'hidden_dims': [2048, 1024],
    'dropout': 0.2,
    'test_size': 0.2,
    'random_seed': 42,
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs'
}

def main():
    print("="*60)
    print("STEP 2: TRAIN BASELINE MODEL")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seed
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Load processed data
    print("\n1. Loading processed data...")
    features, labels = load_processed_data('data/processed')
    
    # Apply PCA to game features for fair comparison with multimodal
    print("\n2. Applying PCA to game features...")
    print(f"  Original dimension: {features.shape[1]}")
    print(f"  Target dimension: 256 (same as multimodal)")
    
    from sklearn.decomposition import PCA
    
    # Split first, then fit PCA on train only
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features, labels,
        test_size=CONFIG['test_size'],
        stratify=labels,
        random_state=CONFIG['random_seed']
    )
    
    pca = PCA(n_components=256)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)
    
    print(f"  ✓ Reduced to 256 dims")
    print(f"  ✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Show class distribution
    print("\n3. Visualizing class distribution...")
    plot_class_distribution(labels, save_path='outputs/class_distribution.png')
    
    # Create dataloaders (data already split above)
    print("\n4. Creating dataloaders...")
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.FloatTensor(class_weights)
    
    train_dataset = PokerDataset(X_train, y_train)
    test_dataset = PokerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("\n5. Creating baseline model...")
    model = get_model(
        model_type='baseline',
        device=device,
        input_dim=256,  # Changed from 377 to 256
        hidden_dims=CONFIG['hidden_dims'],
        output_dim=6,
        dropout=CONFIG['dropout']
    )
    
    # Train
    print("\n6. Training model...")
    
    # Prepare config for checkpoint
    model_config = {
        'input_dim': 256,
        'hidden_dims': CONFIG['hidden_dims'],
        'output_dim': 6,
        'dropout': CONFIG['dropout']
    }
    
    history = train_model(
        model,
        train_loader,
        test_loader,
        n_epochs=CONFIG['n_epochs'],
        learning_rate=CONFIG['learning_rate'],
        class_weights=class_weights,
        device=device,
        checkpoint_dir=CONFIG['checkpoint_dir'],
        model_name='baseline',
        multimodal=False,
        config=model_config,
        pca=pca
    )
    
    # Plot training history
    print("\n7. Plotting training history...")
    plot_training_history(history, save_path='outputs/baseline_training_history.png')
    
    # Final evaluation
    print("\n8. Final evaluation on test set...")
    from train import evaluate
    
    # Load best model
    best_checkpoint = f"{CONFIG['checkpoint_dir']}/baseline_best.pt"
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loss, test_acc, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Detailed metrics
    metrics = evaluate_model(predictions, true_labels, verbose=True)
    
    # Confusion matrix
    print("\n9. Plotting confusion matrix...")
    plot_confusion_matrix(predictions, true_labels, save_path='outputs/baseline_confusion_matrix.png')
    
    # Save report
    print("\n10. Saving evaluation report...")
    save_evaluation_report(
        predictions, true_labels, metrics,
        output_dir=CONFIG['output_dir'],
        experiment_name='baseline'
    )
    
    print("\n" + "="*60)
    print("✓ BASELINE TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Test Accuracy: {checkpoint['test_acc']:.2f}%")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

if __name__ == '__main__':
    main()