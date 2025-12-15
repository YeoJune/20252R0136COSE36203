"""
Step 4: Train Multimodal Model
Train model combining game state features and text embeddings
"""
import sys
sys.path.append('src')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dataset import load_processed_data, MultimodalPokerDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from models import get_model
from train import train_model
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    save_evaluation_report,
    compare_models
)
from generate_text import load_dialogues

# Configuration
CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0001,
    'n_epochs': 15,
    'game_hidden_dims': [1024, 1024, 1024],
    'fusion_hidden_dims': [384, 192],
    'dropout': 0.2,
    'test_size': 0.2,
    'random_seed': 42,
    'text_model': 'distilbert-base-uncased',
    'text_dim': 256,
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs',
    'dialogue_file': 'data/text/dialogues.jsonl'
}

def compute_text_embeddings(dialogues, model_name='distilbert-base-uncased', batch_size=64, device='cuda'):
    """
    Pre-compute text embeddings using DistilBERT
    
    Args:
        dialogues: List of dialogue strings
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        device: Device to use
        
    Returns:
        embeddings: np.ndarray of shape (N, 768)
    """
    print(f"Computing text embeddings using {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dialogues), batch_size), desc='Encoding text'):
            batch_texts = dialogues[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Encode
            outputs = model(**encoded)
            
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"✓ Computed {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    return embeddings

def main():
    print("="*60)
    print("STEP 4: TRAIN MULTIMODAL MODEL")
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
    
    # Load dialogues
    print("\n2. Loading dialogues...")
    dialogues, dialogue_labels = load_dialogues(CONFIG['dialogue_file'])
    
    # Verify alignment
    assert len(dialogues) == len(labels), "Dialogue count mismatch!"
    assert np.all(dialogue_labels == labels), "Dialogue labels mismatch!"
    print(f"  ✓ Loaded {len(dialogues)} dialogues (aligned with features)")
    
    # Split data FIRST (before any fitting)
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test, dial_train, dial_test = train_test_split(
        features, labels, dialogues,
        test_size=CONFIG['test_size'],
        stratify=labels,
        random_state=CONFIG['random_seed']
    )
    
    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    
    # Compute text embeddings separately for train and test
    print("\n4. Computing text embeddings...")
    print("  ⚠️  Computing embeddings separately to prevent data leakage")
    
    text_train = compute_text_embeddings(
        dial_train,
        model_name=CONFIG['text_model'],
        batch_size=CONFIG['batch_size'],
        device=device
    )
    
    text_test = compute_text_embeddings(
        dial_test,
        model_name=CONFIG['text_model'],
        batch_size=CONFIG['batch_size'],
        device=device
    )
    
    # Project to desired dimension (fit PCA on train only)
    from sklearn.decomposition import PCA
    if text_train.shape[1] != CONFIG['text_dim']:
        print(f"\n5. Projecting embeddings from {text_train.shape[1]} to {CONFIG['text_dim']} dims...")
        print(f"  ⚠️  Fitting PCA on TRAIN data only")
        pca = PCA(n_components=CONFIG['text_dim'])
        text_train = pca.fit_transform(text_train)  # Fit on train only!
        text_test = pca.transform(text_test)  # Transform test using train PCA
        print(f"  ✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Compute class weights
    print("\n6. Computing class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.FloatTensor(class_weights)
    
    # Create datasets
    train_dataset = MultimodalPokerDataset(X_train, y_train, None, text_train)
    test_dataset = MultimodalPokerDataset(X_test, y_test, None, text_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Create model
    print("\n7. Creating multimodal model...")
    model = get_model(
        model_type='multimodal',
        device=device,
        game_input_dim=377,
        text_input_dim=CONFIG['text_dim'],
        game_hidden_dims=CONFIG['game_hidden_dims'],
        fusion_hidden_dims=CONFIG['fusion_hidden_dims'],
        output_dim=6,
        dropout=CONFIG['dropout']
    )
    
    # Train
    print("\n8. Training model...")
    history = train_model(
        model,
        train_loader,
        test_loader,
        n_epochs=CONFIG['n_epochs'],
        learning_rate=CONFIG['learning_rate'],
        class_weights=class_weights,
        device=device,
        checkpoint_dir=CONFIG['checkpoint_dir'],
        model_name='multimodal',
        multimodal=True
    )
    
    # Plot training history
    print("\n9. Plotting training history...")
    plot_training_history(history, save_path='outputs/multimodal_training_history.png')
    
    # Final evaluation
    print("\n10. Final evaluation on test set...")
    from train import evaluate_multimodal
    
    # Load best model
    best_checkpoint = f"{CONFIG['checkpoint_dir']}/multimodal_best.pt"
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loss, test_acc, predictions, true_labels = evaluate_multimodal(
        model, test_loader, criterion, device
    )
    
    # Detailed metrics
    metrics = evaluate_model(predictions, true_labels, verbose=True)
    
    # Confusion matrix
    print("\n11. Plotting confusion matrix...")
    plot_confusion_matrix(predictions, true_labels, save_path='outputs/multimodal_confusion_matrix.png')
    
    # Save report
    print("\n12. Saving evaluation report...")
    save_evaluation_report(
        predictions, true_labels, metrics,
        output_dir=CONFIG['output_dir'],
        experiment_name='multimodal'
    )
    
    # Compare with baseline
    print("\n13. Comparing with baseline...")
    # Load baseline results (if available)
    baseline_report = 'outputs/baseline_report.txt'
    if os.path.exists(baseline_report):
        # Parse baseline metrics (simplified)
        # In practice, save metrics to pickle for easier comparison
        print("  Note: For detailed comparison, check individual reports")
    
    print("\n" + "="*60)
    print("✓ MULTIMODAL TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Test Accuracy: {checkpoint['test_acc']:.2f}%")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

if __name__ == '__main__':
    import os
    main()