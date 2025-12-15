"""
Neural network models for poker decision-making
"""
import torch
import torch.nn as nn


class PokerMLP(nn.Module):
    """
    Baseline MLP model for poker action prediction (game state only)
    
    Architecture:
        Input (377) -> Hidden layers with ReLU + Dropout -> Output (6)
    """
    
    def __init__(self, input_dim=377, hidden_dims=[1024, 512, 256], output_dim=6, dropout=0.2):
        """
        Args:
            input_dim: Input feature dimension (default 377)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of action classes (default 6)
            dropout: Dropout probability
        """
        super(PokerMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 377)
            
        Returns:
            logits: Tensor of shape (batch_size, 6)
        """
        return self.network(x)


class MultimodalPokerModel(nn.Module):
    """
    Multimodal model combining game state features and text embeddings
    
    Architecture:
        Game State -> MLP Encoder -> [512]
                                            -> Concatenate -> [768] -> MLP -> [6]
        Text -> BERT Encoder -> [256]
    """
    
    def __init__(
        self, 
        game_input_dim=377,
        text_input_dim=256,
        game_hidden_dims=[512, 256],
        fusion_hidden_dims=[384, 192],
        output_dim=6,
        dropout=0.2
    ):
        """
        Args:
            game_input_dim: Game state feature dimension (default 377)
            text_input_dim: Text embedding dimension (default 256 for DistilBERT)
            game_hidden_dims: Hidden dimensions for game state encoder
            fusion_hidden_dims: Hidden dimensions for fusion layers
            output_dim: Number of action classes (default 6)
            dropout: Dropout probability
        """
        super(MultimodalPokerModel, self).__init__()
        
        # Game state encoder
        game_layers = []
        prev_dim = game_input_dim
        for hidden_dim in game_hidden_dims:
            game_layers.append(nn.Linear(prev_dim, hidden_dim))
            game_layers.append(nn.ReLU())
            game_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.game_encoder = nn.Sequential(*game_layers)
        
        # Fusion layers
        fusion_input_dim = game_hidden_dims[-1] + text_input_dim
        fusion_layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        fusion_layers.append(nn.Linear(prev_dim, output_dim))
        self.fusion_network = nn.Sequential(*fusion_layers)
    
    def forward(self, game_features, text_embeddings):
        """
        Args:
            game_features: Tensor of shape (batch_size, 377)
            text_embeddings: Tensor of shape (batch_size, 256)
            
        Returns:
            logits: Tensor of shape (batch_size, 6)
        """
        game_encoded = self.game_encoder(game_features)
        fused = torch.cat([game_encoded, text_embeddings], dim=1)
        logits = self.fusion_network(fused)
        return logits


class TextEncoder(nn.Module):
    """
    Wrapper for pre-trained text encoder (e.g., DistilBERT)
    """
    
    def __init__(self, model_name='distilbert-base-uncased', output_dim=256, freeze_encoder=False):
        """
        Args:
            model_name: HuggingFace model name
            output_dim: Output embedding dimension
            freeze_encoder: Whether to freeze the encoder weights
        """
        super(TextEncoder, self).__init__()
        
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Project to desired output dimension
        encoder_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_dim, output_dim)
        
    def forward(self, texts):
        """
        Args:
            texts: List of strings (batch_size,)
            
        Returns:
            embeddings: Tensor of shape (batch_size, output_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.encoder.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Encode
        outputs = self.encoder(**encoded)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project
        embeddings = self.projection(cls_embedding)
        
        return embeddings


def get_model(model_type='baseline', device='cuda', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'baseline' or 'multimodal'
        device: Device to place model on
        **kwargs: Additional arguments for model initialization
        
    Returns:
        model: Initialized model on specified device
    """
    if model_type == 'baseline':
        model = PokerMLP(**kwargs)
    elif model_type == 'multimodal':
        model = MultimodalPokerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Created {model_type} model")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✓ Loaded checkpoint from {path}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss


# ============================================================================
# Reinforcement Learning Models (PPO)
# ============================================================================

class SimpleActorCritic(nn.Module):
    """
    Simple Actor-Critic network for PPO (baseline - game state only)
    Intentionally kept simple to prevent overfitting
    
    Architecture:
        Input (256) -> Hidden -> Actor head (6) + Critic head (1)
    """
    
    def __init__(self, input_dim=256, hidden_dim=256, n_actions=6, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            n_actions: Number of actions (default 6)
            dropout: Dropout probability
        """
        super(SimpleActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
            
        Returns:
            action_logits: Tensor of shape (batch_size, n_actions)
            value: Tensor of shape (batch_size, 1)
        """
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, x, deterministic=False):
        """
        Sample action from policy
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
            deterministic: If True, return argmax action
            
        Returns:
            action: Tensor of shape (batch_size,)
            log_prob: Tensor of shape (batch_size,)
            value: Tensor of shape (batch_size,)
        """
        action_logits, value = self.forward(x)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=1)
            log_prob = torch.log_softmax(action_logits, dim=1).gather(1, action.unsqueeze(1)).squeeze(1)
        else:
            # Sample from categorical distribution
            probs = torch.softmax(action_logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(1)
    
    def evaluate_actions(self, x, actions):
        """
        Evaluate given actions
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
            actions: Tensor of shape (batch_size,)
            
        Returns:
            log_probs: Tensor of shape (batch_size,)
            values: Tensor of shape (batch_size,)
            entropy: Tensor of shape (batch_size,)
        """
        action_logits, values = self.forward(x)
        
        probs = torch.softmax(action_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(1), entropy


class MultimodalActorCritic(nn.Module):
    """
    Multimodal Actor-Critic network for PPO (game state + text)
    
    Architecture:
        Game State (256) -> Encoder -> [128]
                                            -> Concat -> [384] -> Actor + Critic
        Text (256) -> Encoder -> [128]
    """
    
    def __init__(
        self, 
        game_input_dim=256,
        text_input_dim=256, 
        hidden_dim=128,
        fusion_dim=256,
        n_actions=6, 
        dropout=0.1
    ):
        """
        Args:
            game_input_dim: Game state feature dimension
            text_input_dim: Text embedding dimension
            hidden_dim: Hidden dimension for each encoder
            fusion_dim: Fusion layer dimension
            n_actions: Number of actions
            dropout: Dropout probability
        """
        super(MultimodalActorCritic, self).__init__()
        
        # Game state encoder
        self.game_encoder = nn.Sequential(
            nn.Linear(game_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        fusion_input_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(fusion_dim, n_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(fusion_dim, 1)
    
    def forward(self, game_features, text_embeddings):
        """
        Args:
            game_features: Tensor of shape (batch_size, game_input_dim)
            text_embeddings: Tensor of shape (batch_size, text_input_dim)
            
        Returns:
            action_logits: Tensor of shape (batch_size, n_actions)
            value: Tensor of shape (batch_size, 1)
        """
        game_encoded = self.game_encoder(game_features)
        text_encoded = self.text_encoder(text_embeddings)
        
        fused = torch.cat([game_encoded, text_encoded], dim=1)
        features = self.fusion(fused)
        
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(self, game_features, text_embeddings, deterministic=False):
        """
        Sample action from policy
        
        Returns:
            action: Tensor of shape (batch_size,)
            log_prob: Tensor of shape (batch_size,)
            value: Tensor of shape (batch_size,)
        """
        action_logits, value = self.forward(game_features, text_embeddings)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=1)
            log_prob = torch.log_softmax(action_logits, dim=1).gather(1, action.unsqueeze(1)).squeeze(1)
        else:
            probs = torch.softmax(action_logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(1)
    
    def evaluate_actions(self, game_features, text_embeddings, actions):
        """
        Evaluate given actions
        
        Returns:
            log_probs: Tensor of shape (batch_size,)
            values: Tensor of shape (batch_size,)
            entropy: Tensor of shape (batch_size,)
        """
        action_logits, values = self.forward(game_features, text_embeddings)
        
        probs = torch.softmax(action_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(1), entropy
