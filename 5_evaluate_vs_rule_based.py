"""
Step 5: Evaluate Models Against Rule-Based Agent
Play simulated poker games between trained models and rule-based agent
"""
import sys
sys.path.append('src')

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from models import PokerMLP, MultimodalPokerModel, TextEncoder
from generate_text import features_to_state, create_dialogue_prompt

# Action labels
ACTION_NAMES = ['fold', 'check_call', 'raise_small', 'raise_medium', 'raise_large', 'all_in']


# ============================================================================
# Rule-Based Agent
# ============================================================================

class RuleBasedAgent:
    """
    Simple rule-based poker agent for evaluation
    Uses hand strength and pot odds to make decisions
    """
    
    def __init__(self, aggression_level=0.5, seed=42):
        """
        Args:
            aggression_level: 0-1, how aggressive the agent plays
            seed: Random seed for reproducibility
        """
        self.aggression_level = aggression_level
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, game_state):
        """
        Make decision based on simple rules
        
        Args:
            game_state: Dict with keys: hole_cards, board_cards, pot, stack, bet_to_call, etc.
            
        Returns:
            action_idx: 0-5 (fold, check_call, raise_small, raise_medium, raise_large, all_in)
        """
        hole_cards = game_state.get('hole_cards', [])
        board_cards = game_state.get('board_cards', [])
        bet_to_call = game_state.get('bet_to_call', 0)
        pot = game_state.get('pot', 100)
        stack = game_state.get('stack', 1000)
        street = game_state.get('street', 'preflop')
        
        # Calculate pot odds
        pot_odds = bet_to_call / (pot + bet_to_call) if (pot + bet_to_call) > 0 else 0
        
        # Estimate hand strength (very simplified)
        hand_strength = self._estimate_hand_strength(hole_cards, board_cards, street)
        
        # Decision thresholds (modified by aggression)
        fold_threshold = 0.2 + (1 - self.aggression_level) * 0.1
        call_threshold = 0.4
        raise_threshold = 0.6 + (self.aggression_level - 0.5) * 0.2
        
        # Decision logic
        if bet_to_call == 0:
            # Free action
            if hand_strength > raise_threshold:
                return self._choose_raise_size(hand_strength, pot, stack)
            else:
                return 1  # check
        else:
            # Must decide: fold, call, or raise
            if hand_strength < fold_threshold:
                # Weak hand
                if pot_odds < 0.3:
                    return 0  # fold
                else:
                    return 1  # call (pot odds justify)
            elif hand_strength < call_threshold:
                # Mediocre hand
                if bet_to_call > stack * 0.3:
                    return 0  # fold, too expensive
                else:
                    return 1  # call
            elif hand_strength < raise_threshold:
                # Good hand
                if self.rng.random() < 0.7:
                    return 1  # call
                else:
                    return self._choose_raise_size(hand_strength, pot, stack)
            else:
                # Strong hand
                if self.rng.random() < 0.3:
                    return 1  # slow play
                else:
                    return self._choose_raise_size(hand_strength, pot, stack)
    
    def _estimate_hand_strength(self, hole_cards, board_cards, street):
        """
        Estimate hand strength (0-1)
        Very simplified heuristic based on card ranks
        """
        if not hole_cards or len(hole_cards) < 2:
            return 0.3  # default mediocre
        
        # Extract ranks
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        
        hole_ranks = [rank_values.get(card[0], 0) for card in hole_cards[:2]]
        
        # Base strength from hole cards
        strength = sum(hole_ranks) / 28.0  # normalized by max (AA = 28)
        
        # Boost for pairs
        if len(set(hole_ranks)) == 1:
            strength += 0.2
        
        # Boost for high cards
        if max(hole_ranks) >= 12:  # Q or better
            strength += 0.1
        
        # Adjust by street (more conservative later)
        if street == 'preflop':
            pass
        elif street == 'flop':
            strength *= 0.9
        elif street == 'turn':
            strength *= 0.8
        elif street == 'river':
            strength *= 0.7
        
        # Add noise
        strength += self.rng.normal(0, 0.05)
        
        return np.clip(strength, 0, 1)
    
    def _choose_raise_size(self, hand_strength, pot, stack):
        """Choose raise size based on hand strength"""
        if hand_strength > 0.85 or stack < pot * 0.5:
            return 5  # all_in
        elif hand_strength > 0.75:
            return 4  # raise_large
        elif hand_strength > 0.65:
            return 3  # raise_medium
        else:
            return 2  # raise_small


# ============================================================================
# Game Simulator
# ============================================================================

class PokerGameSimulator:
    """
    Simulate simplified heads-up poker games
    Focus on decision-making evaluation, not full game logic
    """
    
    def __init__(self, starting_stack=1000, small_blind=5, big_blind=10, seed=42):
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = np.random.RandomState(seed)
    
    def generate_random_game_state(self):
        """Generate a random poker game state for testing"""
        # Generate random hole cards
        deck = []
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        for rank in ranks:
            for suit in suits:
                deck.append(rank + suit)
        
        self.rng.shuffle(deck)
        
        hole_cards = deck[:2]
        
        # Random street
        streets = ['preflop', 'flop', 'turn', 'river']
        street = self.rng.choice(streets)
        
        # Board cards based on street
        n_board = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[street]
        board_cards = deck[2:2+n_board] if n_board > 0 else []
        
        # Random pot, stack, bet
        pot = int(self.rng.uniform(50, 500))
        stack = int(self.rng.uniform(300, 2000))
        bet_to_call = int(self.rng.uniform(0, min(100, stack)))
        position = self.rng.randint(0, 6)
        
        state = {
            'hole_cards': hole_cards,
            'board_cards': board_cards,
            'street': street,
            'pot': pot,
            'stack': stack,
            'bet_to_call': bet_to_call,
            'position': position,
        }
        
        return state


# ============================================================================
# Model Wrappers
# ============================================================================

class BaselineModelAgent:
    """Wrapper for baseline MLP model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer model architecture from state_dict
        # Find all linear layer weights to determine architecture
        linear_layers = []
        idx = 0
        while f'network.{idx}.weight' in state_dict:
            linear_layers.append(state_dict[f'network.{idx}.weight'])
            idx += 3  # Skip ReLU and Dropout layers
        
        # First layer determines input dim
        input_dim = linear_layers[0].shape[1]
        
        # Middle layers determine hidden dims
        hidden_dims = [layer.shape[0] for layer in linear_layers[:-1]]
        
        # Get config from checkpoint
        model_config = checkpoint.get('config', {})
        dropout = model_config.get('dropout', 0.2)
        
        # Load PCA if it exists in checkpoint
        self.pca = checkpoint.get('pca', None)
        self.use_pca = (self.pca is not None)
        
        # Create model with inferred architecture
        self.model = PokerMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded baseline model from {model_path}")
        print(f"  Architecture: input={input_dim}, hidden={hidden_dims}")
        print(f"  Using PCA: {self.use_pca}")
    
    def state_to_features(self, game_state):
        """Convert game state dict to 377-dim feature vector"""
        features = []
        
        # Hole cards (104 dim)
        hole_cards_vec = np.zeros(104)
        for i, card in enumerate(game_state.get('hole_cards', [])[:2]):
            idx = self._card_to_index(card)
            if idx >= 0:
                hole_cards_vec[i * 52 + idx] = 1
        features.extend(hole_cards_vec)
        
        # Board cards (260 dim)
        board_cards_vec = np.zeros(260)
        for i, card in enumerate(game_state.get('board_cards', [])[:5]):
            idx = self._card_to_index(card)
            if idx >= 0:
                board_cards_vec[i * 52 + idx] = 1
        features.extend(board_cards_vec)
        
        # Position (6 dim)
        position_vec = np.zeros(6)
        pos = game_state.get('position', 0)
        if 0 <= pos < 6:
            position_vec[pos] = 1
        features.extend(position_vec)
        
        # Street (4 dim)
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_vec = np.zeros(4)
        street = game_state.get('street', 'preflop')
        if street in street_map:
            street_vec[street_map[street]] = 1
        features.extend(street_vec)
        
        # Numeric features (3 dim)
        pot_norm = game_state.get('pot', 100) / 10000.0
        stack_norm = game_state.get('stack', 1000) / 10000.0
        bet_norm = game_state.get('bet_to_call', 0) / 10000.0
        features.extend([pot_norm, stack_norm, bet_norm])
        
        return np.array(features, dtype=np.float32)
    
    def _card_to_index(self, card):
        """Convert card string to index (0-51)"""
        if len(card) != 2:
            return -1
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                    '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        
        if card[0] in rank_map and card[1] in suit_map:
            return rank_map[card[0]] * 4 + suit_map[card[1]]
        return -1
    
    def get_action(self, game_state):
        """Predict action for given game state"""
        features = self.state_to_features(game_state)
        
        # Apply PCA if used during training
        if self.use_pca:
            features = self.pca.transform(features.reshape(1, -1))[0]
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            action = torch.argmax(logits, dim=1).item()
        
        return action


class MultimodalModelAgent:
    """Wrapper for multimodal model with real-time dialogue generation"""
    
    def __init__(self, model_path, text_gen_model=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer architecture from checkpoint
        model_config = checkpoint.get('config', {})
        
        # Try to get from config first, otherwise infer from weights
        if 'game_input_dim' in model_config:
            game_input_dim = model_config['game_input_dim']
            text_input_dim = model_config['text_input_dim']
            game_hidden_dims = model_config['game_hidden_dims']
            fusion_hidden_dims = model_config['fusion_hidden_dims']
            dropout = model_config.get('dropout', 0.2)
        else:
            # Infer from weights
            game_input_dim = state_dict['game_encoder.0.weight'].shape[1]
            text_input_dim = 256  # Default for DistilBERT
            
            # Get game encoder hidden dims
            game_hidden_dims = []
            layer_idx = 0
            while f'game_encoder.{layer_idx}.weight' in state_dict:
                weight = state_dict[f'game_encoder.{layer_idx}.weight']
                game_hidden_dims.append(weight.shape[0])
                layer_idx += 3
            
            # Get fusion hidden dims
            fusion_hidden_dims = []
            layer_idx = 0
            while f'fusion_network.{layer_idx}.weight' in state_dict:
                weight = state_dict[f'fusion_network.{layer_idx}.weight']
                fusion_hidden_dims.append(weight.shape[0])
                layer_idx += 3
            fusion_hidden_dims = fusion_hidden_dims[:-1]  # Remove output layer
            
            dropout = 0.2
        
        self.model = MultimodalPokerModel(
            game_input_dim=game_input_dim,
            text_input_dim=text_input_dim,
            game_hidden_dims=game_hidden_dims,
            fusion_hidden_dims=fusion_hidden_dims,
            dropout=dropout
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Text encoder
        self.text_encoder = TextEncoder().to(self.device)
        self.text_encoder.eval()
        
        # Optional: LLM for real-time generation
        self.text_gen_model = text_gen_model
        self.use_llm = text_gen_model is not None
        
        print(f"Loaded multimodal model from {model_path}")
        print(f"  Architecture: game_input={game_input_dim}, text_input={text_input_dim}")
        print(f"  Game encoder: {game_hidden_dims}, Fusion: {fusion_hidden_dims}")
        print(f"  Real-time dialogue generation: {self.use_llm}")
    
    def state_to_features(self, game_state):
        """Convert game state dict to 377-dim feature vector"""
        # Same as BaselineModelAgent
        features = []
        
        # Hole cards (104 dim)
        hole_cards_vec = np.zeros(104)
        for i, card in enumerate(game_state.get('hole_cards', [])[:2]):
            idx = self._card_to_index(card)
            if idx >= 0:
                hole_cards_vec[i * 52 + idx] = 1
        features.extend(hole_cards_vec)
        
        # Board cards (260 dim)
        board_cards_vec = np.zeros(260)
        for i, card in enumerate(game_state.get('board_cards', [])[:5]):
            idx = self._card_to_index(card)
            if idx >= 0:
                board_cards_vec[i * 52 + idx] = 1
        features.extend(board_cards_vec)
        
        # Position (6 dim)
        position_vec = np.zeros(6)
        pos = game_state.get('position', 0)
        if 0 <= pos < 6:
            position_vec[pos] = 1
        features.extend(position_vec)
        
        # Street (4 dim)
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_vec = np.zeros(4)
        street = game_state.get('street', 'preflop')
        if street in street_map:
            street_vec[street_map[street]] = 1
        features.extend(street_vec)
        
        # Numeric features (3 dim)
        pot_norm = game_state.get('pot', 100) / 10000.0
        stack_norm = game_state.get('stack', 1000) / 10000.0
        bet_norm = game_state.get('bet_to_call', 0) / 10000.0
        features.extend([pot_norm, stack_norm, bet_norm])
        
        return np.array(features, dtype=np.float32)
    
    def _card_to_index(self, card):
        """Convert card string to index (0-51)"""
        if len(card) != 2:
            return -1
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                    '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        
        if card[0] in rank_map and card[1] in suit_map:
            return rank_map[card[0]] * 4 + suit_map[card[1]]
        return -1
    
    def generate_dialogue(self, game_state):
        """Generate dialogue for game state"""
        if self.use_llm:
            # Real-time generation using LLM
            prompt = create_dialogue_prompt(game_state)
            
            from transformers import pipeline
            generator = pipeline('text-generation', model=self.text_gen_model, device=0 if torch.cuda.is_available() else -1)
            
            result = generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
            dialogue = result[0]['generated_text'].replace(prompt, '').strip()
            
            return dialogue
        else:
            # Simple rule-based fallback
            street = game_state.get('street', 'preflop')
            bet = game_state.get('bet_to_call', 0)
            
            if bet == 0:
                return "Let's see what happens."
            elif bet > game_state.get('pot', 100):
                return "That's a big bet..."
            else:
                return f"Interesting {street}."
    
    def get_action(self, game_state):
        """Predict action for given game state"""
        features = self.state_to_features(game_state)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Generate dialogue
        dialogue = self.generate_dialogue(game_state)
        
        # Encode dialogue
        with torch.no_grad():
            text_emb = self.text_encoder([dialogue])
            
            # Predict
            logits = self.model(x, text_emb)
            action = torch.argmax(logits, dim=1).item()
        
        return action, dialogue


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_vs_rule_based(agent, opponent, n_games=1000, verbose=False):
    """
    Evaluate agent against rule-based opponent
    
    Args:
        agent: Model agent (BaselineModelAgent or MultimodalModelAgent)
        opponent: RuleBasedAgent
        n_games: Number of games to simulate
        verbose: Whether to print detailed logs
        
    Returns:
        results: Dict with evaluation metrics
    """
    simulator = PokerGameSimulator()
    
    agent_actions = []
    opponent_actions = []
    agreements = []
    
    dialogues = [] if isinstance(agent, MultimodalModelAgent) else None
    
    print(f"\nEvaluating against rule-based agent ({n_games} games)...")
    
    for i in tqdm(range(n_games)):
        # Generate random game state
        state = simulator.generate_random_game_state()
        
        # Get actions
        if isinstance(agent, MultimodalModelAgent):
            agent_action, dialogue = agent.get_action(state)
            if dialogues is not None:
                dialogues.append(dialogue)
        else:
            agent_action = agent.get_action(state)
        
        opponent_action = opponent.get_action(state)
        
        agent_actions.append(agent_action)
        opponent_actions.append(opponent_action)
        agreements.append(1 if agent_action == opponent_action else 0)
        
        if verbose and i < 5:
            print(f"\nGame {i+1}:")
            print(f"  State: {state['street']}, Pot: ${state['pot']}, Bet: ${state['bet_to_call']}")
            print(f"  Cards: {state['hole_cards']} | {state['board_cards']}")
            if dialogues and len(dialogues) > 0:
                print(f"  Dialogue: \"{dialogue}\"")
            print(f"  Agent: {ACTION_NAMES[agent_action]}")
            print(f"  Opponent: {ACTION_NAMES[opponent_action]}")
            print(f"  Agreement: {agreements[-1]}")
    
    # Compute metrics
    agent_actions = np.array(agent_actions)
    opponent_actions = np.array(opponent_actions)
    
    agreement_rate = np.mean(agreements) * 100
    
    # Action distribution
    agent_dist = np.bincount(agent_actions, minlength=6) / len(agent_actions)
    opponent_dist = np.bincount(opponent_actions, minlength=6) / len(opponent_actions)
    
    results = {
        'n_games': n_games,
        'agreement_rate': agreement_rate,
        'agent_action_distribution': agent_dist.tolist(),
        'opponent_action_distribution': opponent_dist.tolist(),
        'agent_actions': agent_actions.tolist(),
        'opponent_actions': opponent_actions.tolist(),
    }
    
    if dialogues:
        results['sample_dialogues'] = dialogues[:10]
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Games played: {n_games}")
    print(f"Agreement rate: {agreement_rate:.2f}%")
    print(f"\nAgent action distribution:")
    for i, (name, prob) in enumerate(zip(ACTION_NAMES, agent_dist)):
        print(f"  {name:15s}: {prob*100:5.2f}%")
    print(f"\nOpponent action distribution:")
    for i, (name, prob) in enumerate(zip(ACTION_NAMES, opponent_dist)):
        print(f"  {name:15s}: {prob*100:5.2f}%")
    
    if dialogues:
        print(f"\nSample dialogues:")
        for i, d in enumerate(dialogues[:5], 1):
            print(f"  {i}. \"{d}\"")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate models against rule-based agent')
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'multimodal'],
                      help='Model type to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--n_games', type=int, default=1000,
                      help='Number of games to simulate')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed logs')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    print(f"Loading {args.model_type} model...")
    if args.model_type == 'baseline':
        agent = BaselineModelAgent(args.model_path, device=args.device)
    else:
        agent = MultimodalModelAgent(args.model_path, device=args.device)
    
    # Initialize opponent
    opponent = RuleBasedAgent(aggression_level=0.5, seed=42)
    
    # Evaluate
    results = evaluate_vs_rule_based(
        agent=agent,
        opponent=opponent,
        n_games=args.n_games,
        verbose=args.verbose
    )
    
    # Save results
    output_file = Path(args.output_dir) / f'{args.model_type}_vs_rule_based.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
