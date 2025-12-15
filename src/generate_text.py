"""
Text generation for poker dialogues using LLM
"""
import torch
import numpy as np
from tqdm import tqdm
import json
import os


# ============================================================================
# State Reconstruction
# ============================================================================

def features_to_state(feature_vector):
    """
    Convert 377-dim feature vector back to game state
    
    Feature breakdown:
    - [0:104]: Hole cards (52*2 one-hot)
    - [104:364]: Board cards (52*5 one-hot)
    - [364:370]: Position (6 one-hot)
    - [370:374]: Street (4 one-hot)
    - [374:377]: pot, stack, bet_to_call (normalized)
    
    Returns:
        state dict with readable fields
    """
    # Decode hole cards
    hole_cards = []
    for i in range(2):
        card_vec = feature_vector[i*52:(i+1)*52]
        card_idx = np.argmax(card_vec)
        if card_vec[card_idx] > 0:
            hole_cards.append(index_to_card(card_idx))
    
    # Decode board cards
    board_cards = []
    for i in range(5):
        card_vec = feature_vector[104 + i*52:104 + (i+1)*52]
        card_idx = np.argmax(card_vec)
        if card_vec[card_idx] > 0:
            board_cards.append(index_to_card(card_idx))
    
    # Decode position
    position_vec = feature_vector[364:370]
    position = np.argmax(position_vec)
    
    # Decode street
    street_vec = feature_vector[370:374]
    streets = ['preflop', 'flop', 'turn', 'river']
    street = streets[np.argmax(street_vec)]
    
    # Denormalize numeric features
    pot = int(feature_vector[374] * 10000)
    stack = int(feature_vector[375] * 10000)
    bet_to_call = int(feature_vector[376] * 10000)
    
    return {
        'hole_cards': hole_cards,
        'board_cards': board_cards,
        'position': int(position),
        'street': street,
        'pot': pot,
        'stack': stack,
        'bet_to_call': bet_to_call
    }


def index_to_card(idx):
    """Convert 0-51 index to card string (e.g., 'Tc', 'As')"""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    rank = ranks[idx // 4]
    suit = suits[idx % 4]
    return rank + suit


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are a poker player. Say what you would say out loud during the game. Keep it very short (under 10 words). Examples:
"I'll call"
"Let's see the flop"
"Too rich for me"
"I'm all in"
Only output the dialogue, nothing else."""

def create_dialogue_prompt(state):
    """
    Create prompt for dialogue generation from game state ONLY
    
    Args:
        state: Game state dict
        
    Returns:
        prompt: String prompt for LLM
    """
    # Format cards
    hole_str = ' '.join(state['hole_cards']) if state['hole_cards'] else 'unknown'
    board_str = ' '.join(state['board_cards']) if state['board_cards'] else 'none yet'
    
    # Position names
    position_names = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
    position_str = position_names[state['position']] if state['position'] < 6 else f"P{state['position']}"
    
    prompt = f"""Position: {position_str}
Your cards: {hole_str}
Board: {board_str}
Street: {state['street']}
Pot: ${state['pot']}
Your stack: ${state['stack']}
Bet to call: ${state['bet_to_call']}

What would you say?"""
    
    return prompt


# ============================================================================
# vLLM-based Generation (Fast Batch Inference)
# ============================================================================

def generate_dialogues_vllm(
    states,
    action_labels,
    model_name='meta-llama/Llama-3.2-3B-Instruct',
    max_tokens=50,
    temperature=0.7,
    batch_size=32,
    output_file='data/text/dialogues.jsonl',
    use_cache=True,
    hf_token=None
):
    """
    Generate dialogues using vLLM for fast batch inference
    
    Args:
        states: List of game state dicts
        action_labels: List of action labels (for saving only)
        model_name: HuggingFace model name
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        batch_size: Batch size for generation
        output_file: Path to save generated dialogues
        use_cache: Whether to use cached results
        hf_token: HuggingFace token for gated models (optional)
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached dialogues from {output_file}")
        dialogues = []
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                dialogues.append(data['dialogue'])
        print(f"✓ Loaded {len(dialogues)} cached dialogues")
        return dialogues
    
    print(f"Generating dialogues using vLLM...")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(states)}")
    
    # Setup HuggingFace token if provided
    if hf_token is None:
        hf_token = os.environ.get('HF_TOKEN')
    
    if hf_token:
        print("  Using HuggingFace token for authentication")
    
    try:
        from vllm import LLM, SamplingParams
        
        # Initialize vLLM with token if available
        llm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': 1,
            'dtype': 'float16',
            'gpu_memory_utilization': 0.9,
            'max_model_len': 2048,
            'trust_remote_code': True
        }
        
        if hf_token:
            llm_kwargs['tokenizer_mode'] = 'auto'
            os.environ['HF_TOKEN'] = hf_token
        
        llm = LLM(**llm_kwargs)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        # Create prompts
        prompts = []
        for state in tqdm(states, desc='Creating prompts'):
            prompt = create_dialogue_prompt(state)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            prompts.append(full_prompt)
        
        # Generate in batches
        print("\nGenerating dialogues...")
        all_outputs = llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        dialogues = []
        for output in all_outputs:
            generated_text = output.outputs[0].text.strip()
            dialogues.append(generated_text)
        
        # Save to file
        with open(output_file, 'w') as f:
            for i, dialogue in enumerate(dialogues):
                data = {
                    'index': i,
                    'dialogue': dialogue,
                    'action': int(action_labels[i])
                }
                f.write(json.dumps(data) + '\n')
        
        print(f"✓ Generated {len(dialogues)} dialogues")
        print(f"✓ Saved to {output_file}")
        
        return dialogues
        
    except ImportError:
        print("vLLM not available. Falling back to HuggingFace transformers (slower)...")
        return generate_dialogues_hf(
            states, action_labels, model_name, max_tokens, 
            temperature, batch_size, output_file
        )


# ============================================================================
# HuggingFace Transformers-based Generation (Fallback)
# ============================================================================

def generate_dialogues_hf(
    states,
    action_labels,
    model_name='microsoft/Phi-3-mini-4k-instruct',
    max_tokens=50,
    temperature=0.7,
    batch_size=8,
    output_file='data/text/dialogues.jsonl',
    use_cache=True,
    hf_token=None
):
    """
    Generate dialogues using HuggingFace transformers (fallback if vLLM unavailable)
    
    Args:
        states: List of game state dicts
        action_labels: List of action labels (for saving only)
        model_name: HuggingFace model name
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        batch_size: Batch size for generation
        output_file: Path to save generated dialogues
        use_cache: Whether to use cached results
        hf_token: HuggingFace token for gated models
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached dialogues from {output_file}")
        dialogues = []
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                dialogues.append(data['dialogue'])
        print(f"✓ Loaded {len(dialogues)} cached dialogues")
        return dialogues
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Generating dialogues using HuggingFace transformers...")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(states)}")
    
    # Setup HuggingFace token if provided
    if hf_token is None:
        hf_token = os.environ.get('HF_TOKEN')
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    if device == 'cuda':
        print(f"  GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    tokenizer_kwargs = {'trust_remote_code': True}
    model_kwargs = {
        'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
        'trust_remote_code': True
    }
    
    if device == 'cuda':
        model_kwargs['device_map'] = 'auto'
        model_kwargs['low_cpu_mem_usage'] = True
    
    if hf_token:
        tokenizer_kwargs['token'] = hf_token
        model_kwargs['token'] = hf_token
    
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    if device == 'cuda' and 'device_map' not in model_kwargs:
        model = model.to(device)
        print(f"  ✓ Model moved to {device}")
    elif device == 'cuda':
        print(f"  ✓ Model loaded with device_map=auto")
    else:
        model = model.to(device)
        print(f"  ⚠️  Using CPU (this will be slow)")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dialogues = []
    
    # Generate in batches
    print("  Generating dialogues...")
    print("  Note: First batch may take longer due to model compilation")
    for batch_idx, i in enumerate(tqdm(range(0, len(states), batch_size), desc='Generating')):
        batch_states = states[i:i+batch_size]
        
        # Create prompts
        prompts = []
        for state in batch_states:
            prompt = create_dialogue_prompt(state)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            prompts.append(full_prompt)
        
        # Tokenize
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean
            generated_text = generated_text.strip()
            
            # Remove common meta phrases
            bad_phrases = [
                "Given the", "After checking", "might say:", "could be:", 
                "would say:", "Here's", "A realistic", "dialogue", 
                "In this situation", "Based on", "According to",
                "The player", "This player", "would likely",
                "could say", "might respond", "example:", "such as:",
                "for instance", "generate", "output", "say something like"
            ]
            
            for phrase in bad_phrases:
                if phrase.lower() in generated_text.lower():
                    generated_text = ""
                    break
            
            # If empty or too long, use simple fallback
            if not generated_text or len(generated_text.split()) > 15:
                fallbacks = [
                    "Let's see what happens.",
                    "I'll play this one.",
                    "Okay.",
                    "Let's go.",
                    "I'm in."
                ]
                generated_text = fallbacks[j % len(fallbacks)]
            else:
                # Take only first sentence
                if '.' in generated_text:
                    generated_text = generated_text.split('.')[0] + '.'
                if '\n' in generated_text:
                    generated_text = generated_text.split('\n')[0]
                
                # Remove quotes
                generated_text = generated_text.replace('"', '').replace("'", '')
            
            dialogues.append(generated_text)
    
    # Save to file
    with open(output_file, 'w') as f:
        for i, dialogue in enumerate(dialogues):
            data = {
                'index': i,
                'dialogue': dialogue,
                'action': int(action_labels[i])
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"✓ Generated {len(dialogues)} dialogues")
    print(f"✓ Saved to {output_file}")
    
    return dialogues


# ============================================================================
# Rule-based Fallback Templates
# ============================================================================

RULE_BASED_TEMPLATES = {
    0: ["I'm out this hand.", "Not for me.", "I'll pass.", "Too much for me.", "Next hand."],
    1: ["Let's see it.", "I'm in.", "Okay.", "Let's play.", "Sure."],
    2: ["Let me add a bit.", "Making it interesting.", "I'll bump it up.", "A little more.", "Let's raise the stakes."],
    3: ["Time to play.", "Let's see who's got it.", "I like my hand.", "Building this pot.", "Let's make it worth it."],
    4: ["Big move here.", "Let's play for real.", "This is my hand.", "Time to commit.", "I'm feeling good about this."],
    5: ["Everything goes in.", "Let's settle this.", "I'm committed.", "This is it.", "All my chips."]
}

def generate_dialogues_rule_based(action_labels, output_file='data/text/dialogues_rule_based.jsonl'):
    """
    Generate dialogues using simple rule-based templates (fast fallback)
    
    Args:
        action_labels: List of action labels
        output_file: Path to save generated dialogues
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    print(f"Generating dialogues using rule-based templates...")
    print(f"Total samples: {len(action_labels)}")
    
    dialogues = []
    rng = np.random.RandomState(42)
    
    for action_label in tqdm(action_labels, desc='Generating'):
        templates = RULE_BASED_TEMPLATES[action_label]
        dialogue = rng.choice(templates)
        dialogues.append(dialogue)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for i, dialogue in enumerate(dialogues):
            data = {
                'index': i,
                'dialogue': dialogue,
                'action': int(action_labels[i])
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"✓ Generated {len(dialogues)} dialogues")
    print(f"✓ Saved to {output_file}")
    
    # Also save to the default filename for compatibility
    if 'rule_based' in output_file:
        default_file = output_file.replace('_rule_based', '')
        with open(default_file, 'w') as f:
            for i, dialogue in enumerate(dialogues):
                data = {
                    'index': i,
                    'dialogue': dialogue,
                    'action': int(action_labels[i])
                }
                f.write(json.dumps(data) + '\n')
        print(f"✓ Also saved to {default_file} for compatibility")
    
    return dialogues


# ============================================================================
# Loading Utilities
# ============================================================================

def load_dialogues(input_file='data/text/dialogues.jsonl'):
    """
    Load generated dialogues from file
    
    Args:
        input_file: Path to dialogue file
        
    Returns:
        dialogues: List of dialogue strings
        actions: List of action labels
    """
    # Try primary file first
    if not os.path.exists(input_file):
        # Try rule-based fallback
        fallback_file = input_file.replace('.jsonl', '_rule_based.jsonl')
        if os.path.exists(fallback_file):
            print(f"Primary file not found, using fallback: {fallback_file}")
            input_file = fallback_file
        else:
            raise FileNotFoundError(f"Dialogue file not found: {input_file}")
    
    dialogues = []
    actions = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            dialogues.append(data['dialogue'])
            actions.append(data['action'])
    
    print(f"✓ Loaded {len(dialogues)} dialogues from {input_file}")
    return dialogues, actions