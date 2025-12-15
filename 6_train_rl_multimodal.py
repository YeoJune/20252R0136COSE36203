"""
Step 7: Train RL Multimodal Model (PPO)
Train Actor-Critic model using PPO on game state features + dialogue
Focus on profit maximization with dialogue context
"""
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import json

from dataset import load_processed_data
from models import MultimodalActorCritic
from generate_text import load_dialogues

# Configuration
CONFIG = {
    'batch_size': 256,
    'learning_rate': 0.0003,
    'n_episodes': 1000,
    'max_steps_per_episode': 50,
    'ppo_epochs': 4,
    'clip_epsilon': 0.2,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'hidden_dim': 128,
    'fusion_dim': 256,
    'dropout': 0.1,
    'random_seed': 42,
    'text_model': 'distilbert-base-uncased',
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs',
    'dialogue_file': 'data/text/dialogues.jsonl',
    'save_freq': 50,
}


class MultimodalPokerRLEnvironment:
    """
    Multimodal Poker RL Environment with game state + text
    """
    
    def __init__(self, features, labels, text_embeddings, rng_seed=42):
        """
        Args:
            features: np.ndarray of shape (N, feature_dim)
            labels: np.ndarray of shape (N,) - expert actions
            text_embeddings: np.ndarray of shape (N, text_dim)
        """
        self.features = features
        self.labels = labels
        self.text_embeddings = text_embeddings
        self.n_samples = len(features)
        self.rng = np.random.RandomState(rng_seed)
        
        # Current episode state
        self.current_idx = None
        self.episode_step = 0
        self.episode_profit = 0.0
        
        # Action costs and rewards (same as baseline)
        self.action_costs = {
            0: -10.0, 1: -2.0, 2: -5.0, 3: -8.0, 4: -12.0, 5: -20.0,
        }
        self.action_rewards = {
            0: 0.0, 1: 15.0, 2: 20.0, 3: 25.0, 4: 30.0, 5: 50.0,
        }
    
    def reset(self):
        """Start a new episode"""
        self.current_idx = self.rng.randint(0, self.n_samples)
        self.episode_step = 0
        self.episode_profit = 0.0
        
        state = self.features[self.current_idx]
        text_emb = self.text_embeddings[self.current_idx]
        return state, text_emb
    
    def step(self, action):
        """
        Take action and get reward
        
        Returns:
            next_state: (game_state, text_embedding)
            reward: float
            done: bool
            info: dict
        """
        self.episode_step += 1
        
        # Get expert action
        expert_action = int(self.labels[self.current_idx])
        
        # Calculate reward (same logic as baseline)
        reward = self.action_costs[action]
        
        if action == expert_action:
            reward += 10.0
        
        action_quality = 1.0 if action == expert_action else 0.5
        reward += self.action_rewards[action] * action_quality
        
        reward += self.rng.normal(0, 2.0)
        
        self.episode_profit += reward
        
        # Episode termination
        done = (self.episode_step >= CONFIG['max_steps_per_episode']) or (self.rng.random() < 0.1)
        
        # Get next state
        if not done:
            self.current_idx = self.rng.randint(0, self.n_samples)
        
        next_state = self.features[self.current_idx]
        next_text_emb = self.text_embeddings[self.current_idx]
        
        info = {
            'episode_profit': self.episode_profit,
            'expert_action': expert_action,
            'action_match': action == expert_action
        }
        
        return (next_state, next_text_emb), reward, done, info


class MultimodalPPOTrainer:
    """PPO Trainer for Multimodal Actor-Critic model"""
    
    def __init__(self, model, optimizer, config, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Training history
        self.history = {
            'episode_rewards': [],
            'episode_profits': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'expert_match_rate': []
        }
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.config['gamma'] * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return np.array(advantages), np.array(returns)
    
    def update(self, game_states, text_embeddings, actions, old_log_probs, returns, advantages):
        """PPO update step for multimodal model"""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.config['ppo_epochs']):
            # Evaluate actions with current policy
            log_probs, values, entropy = self.model.evaluate_actions(game_states, text_embeddings, actions)
            
            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.config['value_loss_coef'] * value_loss + self.config['entropy_coef'] * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.mean().item()
    
    def train_episode(self, env):
        """Train for one episode"""
        game_states, text_embeddings = [], []
        actions, log_probs, rewards, dones, values = [], [], [], [], []
        
        state, text_emb = env.reset()
        episode_reward = 0
        episode_matches = 0
        episode_steps = 0
        
        while True:
            # Convert to tensors
            game_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            text_tensor = torch.FloatTensor(text_emb).unsqueeze(0).to(self.device)
            
            # Get action
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(game_tensor, text_tensor)
            
            action = action.item()
            log_prob = log_prob.item()
            value = value.item()
            
            # Take action
            (next_state, next_text_emb), reward, done, info = env.step(action)
            
            # Store transition
            game_states.append(state)
            text_embeddings.append(text_emb)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            episode_reward += reward
            episode_matches += int(info['action_match'])
            episode_steps += 1
            
            state = next_state
            text_emb = next_text_emb
            
            if done:
                break
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        game_tensor = torch.FloatTensor(np.array(game_states)).to(self.device)
        text_tensor = torch.FloatTensor(np.array(text_embeddings)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Update policy
        policy_loss, value_loss, entropy = self.update(
            game_tensor, text_tensor, actions_tensor, old_log_probs_tensor, 
            returns_tensor, advantages_tensor
        )
        
        # Record metrics
        expert_match_rate = episode_matches / episode_steps
        
        return {
            'episode_reward': episode_reward,
            'episode_profit': info['episode_profit'],
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'expert_match_rate': expert_match_rate,
            'episode_steps': episode_steps
        }


def compute_text_embeddings(dialogues, model_name='distilbert-base-uncased', batch_size=64, device='cuda'):
    """Pre-compute text embeddings"""
    print(f"Computing text embeddings using {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dialogues), batch_size), desc='Encoding text'):
            batch_texts = dialogues[i:i+batch_size]
            
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Apply PCA to reduce to 256 dimensions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=256)
    embeddings_pca = pca.fit_transform(embeddings)
    
    print(f"✓ Computed {len(embeddings_pca)} embeddings of dimension {embeddings_pca.shape[1]}")
    print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return embeddings_pca, pca


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Episode rewards
    ax = axes[0, 0]
    ax.plot(history['episode_rewards'], alpha=0.6, label='Raw')
    ax.plot(np.convolve(history['episode_rewards'], np.ones(20)/20, mode='valid'), 
            label='Moving Avg (20)', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Episode profits
    ax = axes[0, 1]
    ax.plot(history['episode_profits'], alpha=0.6, label='Raw')
    ax.plot(np.convolve(history['episode_profits'], np.ones(20)/20, mode='valid'), 
            label='Moving Avg (20)', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Profit')
    ax.set_title('Episode Profits (Big Blinds)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Expert match rate
    ax = axes[0, 2]
    ax.plot(history['expert_match_rate'], alpha=0.6, label='Raw')
    ax.plot(np.convolve(history['expert_match_rate'], np.ones(20)/20, mode='valid'), 
            label='Moving Avg (20)', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Match Rate')
    ax.set_title('Expert Action Match Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Policy loss
    ax = axes[1, 0]
    ax.plot(history['policy_losses'], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.grid(alpha=0.3)
    
    # Value loss
    ax = axes[1, 1]
    ax.plot(history['value_losses'], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss')
    ax.grid(alpha=0.3)
    
    # Entropy
    ax = axes[1, 2]
    ax.plot(history['entropies'], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("STEP 7: TRAIN RL MULTIMODAL MODEL (PPO)")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seed
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Create directories
    Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)
    
    # Load processed data
    print("\n1. Loading processed data...")
    features, labels = load_processed_data('data/processed')
    print(f"Loaded {len(features)} samples")
    
    # Apply PCA to game features
    print("\n2. Applying PCA to game features...")
    print(f"  Original dimension: {features.shape[1]}")
    print(f"  Target dimension: 256")
    
    pca_game = PCA(n_components=256)
    features_pca = pca_game.fit_transform(features)
    print(f"  ✓ Reduced to 256 dims")
    print(f"  ✓ Explained variance: {pca_game.explained_variance_ratio_.sum():.2%}")
    
    # Save PCA for inference
    import pickle
    with open('checkpoints/rl_multimodal_game_pca.pkl', 'wb') as f:
        pickle.dump(pca_game, f)
    print("  ✓ Saved game PCA model")
    
    # Load dialogues
    print("\n3. Loading dialogues...")
    dialogue_data = load_dialogues(CONFIG['dialogue_file'])
    dialogues = [d['dialogue'] for d in dialogue_data]
    print(f"Loaded {len(dialogues)} dialogues")
    
    # Compute text embeddings
    print("\n4. Computing text embeddings...")
    text_embeddings, pca_text = compute_text_embeddings(
        dialogues, 
        model_name=CONFIG['text_model'],
        batch_size=64,
        device=device
    )
    
    # Save text PCA for inference
    with open('checkpoints/rl_multimodal_text_pca.pkl', 'wb') as f:
        pickle.dump(pca_text, f)
    print("  ✓ Saved text PCA model")
    
    # Create environment
    print("\n5. Creating multimodal RL environment...")
    env = MultimodalPokerRLEnvironment(
        features_pca, 
        labels, 
        text_embeddings, 
        rng_seed=CONFIG['random_seed']
    )
    print(f"Environment ready with {env.n_samples} states")
    
    # Create model
    print("\n6. Creating Multimodal Actor-Critic model...")
    model = MultimodalActorCritic(
        game_input_dim=256,
        text_input_dim=256,
        hidden_dim=CONFIG['hidden_dim'],
        fusion_dim=CONFIG['fusion_dim'],
        n_actions=6,
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"  Architecture: Multimodal (Game[256] + Text[256] -> {CONFIG['fusion_dim']} -> 6 actions + 1 value)")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Create trainer
    trainer = MultimodalPPOTrainer(model, optimizer, CONFIG, device)
    
    # Training loop
    print("\n7. Training with PPO...")
    print(f"Training for {CONFIG['n_episodes']} episodes")
    
    best_avg_profit = -float('inf')
    
    for episode in tqdm(range(CONFIG['n_episodes']), desc='Training'):
        # Train one episode
        metrics = trainer.train_episode(env)
        
        # Record history
        trainer.history['episode_rewards'].append(metrics['episode_reward'])
        trainer.history['episode_profits'].append(metrics['episode_profit'])
        trainer.history['policy_losses'].append(metrics['policy_loss'])
        trainer.history['value_losses'].append(metrics['value_loss'])
        trainer.history['entropies'].append(metrics['entropy'])
        trainer.history['expert_match_rate'].append(metrics['expert_match_rate'])
        
        # Print progress
        if (episode + 1) % 50 == 0:
            recent_profits = trainer.history['episode_profits'][-50:]
            recent_rewards = trainer.history['episode_rewards'][-50:]
            recent_match = trainer.history['expert_match_rate'][-50:]
            
            avg_profit = np.mean(recent_profits)
            avg_reward = np.mean(recent_rewards)
            avg_match = np.mean(recent_match)
            
            print(f"\nEpisode {episode+1}/{CONFIG['n_episodes']}")
            print(f"  Avg Profit (50 ep): {avg_profit:+.2f} BB")
            print(f"  Avg Reward (50 ep): {avg_reward:+.2f}")
            print(f"  Expert Match Rate: {avg_match:.2%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            
            # Save best model
            if avg_profit > best_avg_profit:
                best_avg_profit = avg_profit
                checkpoint_path = Path(CONFIG['checkpoint_dir']) / 'rl_multimodal_best.pt'
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_profit': avg_profit,
                    'config': CONFIG
                }, checkpoint_path)
                print(f"  ✓ Saved best model (profit: {avg_profit:+.2f} BB)")
        
        # Save checkpoint periodically
        if (episode + 1) % CONFIG['save_freq'] == 0:
            checkpoint_path = Path(CONFIG['checkpoint_dir']) / f'rl_multimodal_ep{episode+1}.pt'
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': trainer.history,
                'config': CONFIG
            }, checkpoint_path)
    
    # Save final model
    print("\n8. Saving final model...")
    final_path = Path(CONFIG['checkpoint_dir']) / 'rl_multimodal_final.pt'
    torch.save({
        'episode': CONFIG['n_episodes'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': trainer.history,
        'config': CONFIG
    }, final_path)
    print(f"✓ Saved final model: {final_path}")
    
    # Plot training history
    print("\n9. Plotting training history...")
    plot_path = Path(CONFIG['output_dir']) / 'rl_multimodal_training.png'
    plot_training_history(trainer.history, plot_path)
    
    # Save training history
    history_path = Path(CONFIG['output_dir']) / 'rl_multimodal_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"✓ Saved training history: {history_path}")
    
    # Print final statistics
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    final_profits = trainer.history['episode_profits'][-100:]
    final_match = trainer.history['expert_match_rate'][-100:]
    print(f"Final 100 episodes:")
    print(f"  Avg Profit: {np.mean(final_profits):+.2f} ± {np.std(final_profits):.2f} BB")
    print(f"  Best Avg Profit: {best_avg_profit:+.2f} BB")
    print(f"  Expert Match Rate: {np.mean(final_match):.2%}")
    print(f"\nModel saved: checkpoints/rl_multimodal_best.pt")
    print("="*60)


if __name__ == '__main__':
    main()
