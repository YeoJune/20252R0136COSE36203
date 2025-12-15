"""
Step 6: Train RL Baseline Model (PPO)
Train simple Actor-Critic model using PPO on game state features only
Focus on profit maximization with simple architecture to prevent overfitting
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
import matplotlib.pyplot as plt
import json

from dataset import load_processed_data
from models import SimpleActorCritic

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
    'hidden_dim': 256,
    'dropout': 0.1,
    'random_seed': 42,
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs',
    'save_freq': 50,
}


class PokerRLEnvironment:
    """
    Simplified Poker RL Environment for training
    Uses real game states from dataset with simulated opponents
    """
    
    def __init__(self, features, labels, rng_seed=42):
        """
        Args:
            features: np.ndarray of shape (N, feature_dim)
            labels: np.ndarray of shape (N,) - expert actions
        """
        self.features = features
        self.labels = labels
        self.n_samples = len(features)
        self.rng = np.random.RandomState(rng_seed)
        
        # Current episode state
        self.current_idx = None
        self.episode_step = 0
        self.episode_profit = 0.0
        
        # Action costs (big blinds)
        self.action_costs = {
            0: -10.0,   # fold (lose blinds/bets)
            1: -2.0,    # check/call (pay to see cards)
            2: -5.0,    # raise_small
            3: -8.0,    # raise_medium
            4: -12.0,   # raise_large
            5: -20.0,   # all_in
        }
        
        # Action rewards (expected value based on expert action)
        self.action_rewards = {
            0: 0.0,     # fold (no upside)
            1: 15.0,    # check/call (moderate)
            2: 20.0,    # raise_small
            3: 25.0,    # raise_medium
            4: 30.0,    # raise_large
            5: 50.0,    # all_in (high risk/reward)
        }
    
    def reset(self):
        """Start a new episode"""
        self.current_idx = self.rng.randint(0, self.n_samples)
        self.episode_step = 0
        self.episode_profit = 0.0
        
        state = self.features[self.current_idx]
        return state
    
    def step(self, action):
        """
        Take action and get reward
        
        Args:
            action: int in [0, 5]
            
        Returns:
            next_state: np.ndarray
            reward: float
            done: bool
            info: dict
        """
        self.episode_step += 1
        
        # Get expert action
        expert_action = int(self.labels[self.current_idx])
        
        # Calculate reward
        # 1. Base cost for taking action
        reward = self.action_costs[action]
        
        # 2. Bonus for matching expert (imitation learning component)
        if action == expert_action:
            reward += 10.0
        
        # 3. Expected value from action
        # Scale by how good the action is (expert action = best)
        action_quality = 1.0 if action == expert_action else 0.5
        reward += self.action_rewards[action] * action_quality
        
        # 4. Add noise to simulate variance
        reward += self.rng.normal(0, 2.0)
        
        self.episode_profit += reward
        
        # Episode ends after max steps or randomly
        done = (self.episode_step >= CONFIG['max_steps_per_episode']) or (self.rng.random() < 0.1)
        
        # Get next state
        if not done:
            self.current_idx = self.rng.randint(0, self.n_samples)
            next_state = self.features[self.current_idx]
        else:
            next_state = self.features[self.current_idx]
        
        info = {
            'episode_profit': self.episode_profit,
            'expert_action': expert_action,
            'action_match': action == expert_action
        }
        
        return next_state, reward, done, info


class PPOTrainer:
    """PPO Trainer for Actor-Critic model"""
    
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
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            advantages: np.ndarray
            returns: np.ndarray
        """
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
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO update step
        
        Args:
            states: Tensor (batch_size, feature_dim)
            actions: Tensor (batch_size,)
            old_log_probs: Tensor (batch_size,)
            returns: Tensor (batch_size,)
            advantages: Tensor (batch_size,)
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.config['ppo_epochs']):
            # Evaluate actions with current policy
            log_probs, values, entropy = self.model.evaluate_actions(states, actions)
            
            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy bonus (encourage exploration)
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
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        state = env.reset()
        episode_reward = 0
        episode_matches = 0
        episode_steps = 0
        
        while True:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(state_tensor)
            
            action = action.item()
            log_prob = log_prob.item()
            value = value.item()
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            episode_reward += reward
            episode_matches += int(info['action_match'])
            episode_steps += 1
            
            state = next_state
            
            if done:
                break
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Update policy
        policy_loss, value_loss, entropy = self.update(
            states_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor
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
    print("STEP 6: TRAIN RL BASELINE MODEL (PPO)")
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
    
    # Apply PCA to reduce dimensionality (same as supervised baseline)
    print("\n2. Applying PCA to game features...")
    print(f"  Original dimension: {features.shape[1]}")
    print(f"  Target dimension: 256")
    
    pca = PCA(n_components=256)
    features_pca = pca.fit_transform(features)
    print(f"  ✓ Reduced to 256 dims")
    print(f"  ✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Save PCA for inference
    import pickle
    with open('checkpoints/rl_baseline_pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    print("  ✓ Saved PCA model")
    
    # Create environment
    print("\n3. Creating RL environment...")
    env = PokerRLEnvironment(features_pca, labels, rng_seed=CONFIG['random_seed'])
    print(f"Environment ready with {env.n_samples} states")
    
    # Create model
    print("\n4. Creating Actor-Critic model...")
    model = SimpleActorCritic(
        input_dim=256,
        hidden_dim=CONFIG['hidden_dim'],
        n_actions=6,
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"  Architecture: Simple (256 -> {CONFIG['hidden_dim']} -> 6 actions + 1 value)")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Create trainer
    trainer = PPOTrainer(model, optimizer, CONFIG, device)
    
    # Training loop
    print("\n5. Training with PPO...")
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
                checkpoint_path = Path(CONFIG['checkpoint_dir']) / 'rl_baseline_best.pt'
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
            checkpoint_path = Path(CONFIG['checkpoint_dir']) / f'rl_baseline_ep{episode+1}.pt'
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': trainer.history,
                'config': CONFIG
            }, checkpoint_path)
    
    # Save final model
    print("\n6. Saving final model...")
    final_path = Path(CONFIG['checkpoint_dir']) / 'rl_baseline_final.pt'
    torch.save({
        'episode': CONFIG['n_episodes'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': trainer.history,
        'config': CONFIG
    }, final_path)
    print(f"✓ Saved final model: {final_path}")
    
    # Plot training history
    print("\n7. Plotting training history...")
    plot_path = Path(CONFIG['output_dir']) / 'rl_baseline_training.png'
    plot_training_history(trainer.history, plot_path)
    
    # Save training history
    history_path = Path(CONFIG['output_dir']) / 'rl_baseline_history.json'
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
    print(f"\nModel saved: checkpoints/rl_baseline_best.pt")
    print("="*60)


if __name__ == '__main__':
    main()
