"""
Step 3: Generate Dialogues
Generate poker dialogues using LLM for each decision point
"""
import sys
sys.path.append('src')

import os
import numpy as np
import pickle
from dataset import load_processed_data
from generate_text import (
    generate_dialogues_vllm,
    generate_dialogues_rule_based,
    features_to_state
)

# Configuration
CONFIG = {
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
    'max_tokens': 50,
    'temperature': 0.7,
    'batch_size': 256,
    'output_file': 'data/text/dialogues.jsonl',
    'use_vllm': False,
    'use_transformers': True,
    'use_cache': True,
    'n_samples': None,
    'hf_token': None
}

def main():
    print("="*60)
    print("STEP 3: GENERATE DIALOGUES")
    print("="*60)
    
    # Load processed data
    print("\n1. Loading processed data...")
    features, labels = load_processed_data('data/processed')
    
    # Optionally limit samples for testing
    if CONFIG['n_samples'] is not None:
        features = features[:CONFIG['n_samples']]
        labels = labels[:CONFIG['n_samples']]
        print(f"  Using first {CONFIG['n_samples']} samples for testing")
    
    print(f"  Total samples: {len(labels)}")
    
    # Reconstruct game states from features
    print("\n2. Reconstructing game states from features...")
    states = [features_to_state(f) for f in features]
    print(f"  Created {len(states)} state representations")
    
    # Generate dialogues
    print("\n3. Generating dialogues...")
    
    if CONFIG.get('use_transformers', False):
        print("  Using HuggingFace Transformers...")
        try:
            from generate_text import generate_dialogues_hf
            dialogues = generate_dialogues_hf(
                states,
                labels,
                model_name=CONFIG['model_name'],
                max_tokens=CONFIG['max_tokens'],
                temperature=CONFIG['temperature'],
                batch_size=CONFIG['batch_size'],
                output_file=CONFIG['output_file']
            )
        except Exception as e:
            print(f"\n  Warning: Transformers generation failed: {e}")
            print("  Falling back to rule-based templates...")
            dialogues = generate_dialogues_rule_based(
                labels,
                output_file=CONFIG['output_file'].replace('.jsonl', '_rule_based.jsonl')
            )
    elif CONFIG['use_vllm']:
        print("  Using vLLM for fast batch generation...")
        try:
            dialogues = generate_dialogues_vllm(
                states,
                labels,
                model_name=CONFIG['model_name'],
                max_tokens=CONFIG['max_tokens'],
                temperature=CONFIG['temperature'],
                batch_size=CONFIG['batch_size'],
                output_file=CONFIG['output_file'],
                use_cache=CONFIG['use_cache'],
                hf_token=CONFIG.get('hf_token')
            )
        except Exception as e:
            print(f"\n  Warning: vLLM generation failed: {e}")
            print("  Falling back to rule-based templates...")
            dialogues = generate_dialogues_rule_based(
                labels,
                output_file=CONFIG['output_file'].replace('.jsonl', '_rule_based.jsonl')
            )
    else:
        print("  Using rule-based templates...")
        dialogues = generate_dialogues_rule_based(
            labels,
            output_file=CONFIG['output_file'].replace('.jsonl', '_rule_based.jsonl')
        )
    
    # Show examples
    print("\n4. Example dialogues:")
    action_names = ['fold', 'check_call', 'raise_small', 'raise_medium', 'raise_large', 'all_in']
    for i in range(min(10, len(dialogues))):
        print(f"\n  [{action_names[labels[i]]}] {dialogues[i]}")
    
    # Save metadata
    print("\n5. Saving metadata...")
    metadata = {
        'n_dialogues': len(dialogues),
        'model_name': CONFIG['model_name'],
        'max_tokens': CONFIG['max_tokens'],
        'temperature': CONFIG['temperature'],
        'method': 'vllm' if CONFIG['use_vllm'] else ('transformers' if CONFIG['use_transformers'] else 'rule_based')
    }
    
    # Determine actual output file used
    if CONFIG['use_vllm'] or CONFIG['use_transformers']:
        actual_output_file = CONFIG['output_file']
    else:
        actual_output_file = CONFIG['output_file'].replace('.jsonl', '_rule_based.jsonl')
    
    metadata_file = actual_output_file.replace('.jsonl', '_metadata.pkl')
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Saved metadata to {metadata_file}")
    
    print("\n" + "="*60)
    print("✓ DIALOGUE GENERATION COMPLETE")
    print("="*60)
    print(f"\nGenerated {len(dialogues)} dialogues")
    print(f"Saved to {actual_output_file}")

if __name__ == '__main__':
    main()