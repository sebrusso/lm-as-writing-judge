import sys
import os
# Add the parent directory to Python path for importing model_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import pandas as pd
from model_service import ModelService, ModelProvider
import random
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime
import json

# Define model configurations
MODELS = [
    # OpenAI models
    {"provider": ModelProvider.OPENAI, "model": "gpt-4o-mini", "name": "GPT-4-Mini"},
    {"provider": ModelProvider.OPENAI, "model": "4o", "name": "GPT-4"},
    {"provider": ModelProvider.OPENAI, "model": "o1", "name": "o1"},
    {"provider": ModelProvider.OPENAI, "model": "o1-mini", "name": "o1-mini"},
    {"provider": ModelProvider.OPENAI, "model": "o3-mini", "name": "o3-mini"},
    # DeepInfra models
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-reasoning", "name": "Qwen-Reasoning"},
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-base", "name": "Qwen-Base"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-33-70b", "name": "Llama-3.3-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-70b-instruct", "name": "Llama-3.1-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-8b-instruct", "name": "Llama-3.1-8B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "phi-4", "name": "Phi-4"},
    # DeepSeek models
    {"provider": ModelProvider.DEEPSEEK, "model": "r1", "name": "DeepSeek-Reasoner"},
    {"provider": ModelProvider.DEEPSEEK, "model": "v3", "name": "DeepSeek-Chat"}
]

def load_writingprompts_dataset():
    """Load the WritingPrompts pairwise test dataset from Hugging Face"""
    dataset = load_dataset("SAA-Lab/writingprompts-pairwise-test")
    return dataset['train']

def prepare_evaluation_pairs(
    dataset, 
    num_samples: int = 10, 
    num_permutations: int = 3,
    replace: bool = False,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare random samples from the dataset for evaluation with multiple permutations
    
    Args:
        dataset: The loaded dataset
        num_samples: Number of random samples to select
        num_permutations: Number of permutations for each sample (N=3 for statistical significance)
        replace: Whether to sample with replacement (True) or without replacement (False)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        List of dictionaries containing prompts and story pairs
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)
    
    # Validate sample size when sampling without replacement
    if not replace and num_samples > len(df):
        print(f"Warning: Requested {num_samples} samples but dataset only has {len(df)} items.")
        print("Reducing number of samples to dataset size.")
        num_samples = len(df)
    
    # Randomly sample rows
    sampled_rows = df.sample(n=num_samples, replace=replace)
    
    evaluation_pairs = []
    for _, row in sampled_rows.iterrows():
        # Create N permutations of each pair
        for perm_id in range(num_permutations):
            pair = {
                'prompt': row['prompt'],
                'story_a': row['chosen'],
                'story_b': row['rejected'],
                'original_chosen': 'a',  # Track which was originally chosen
                'upvotes_a': row['upvotes_chosen'],
                'upvotes_b': row['upvotes_rejected'],
                'permutation_id': perm_id,  # Track which permutation this is
                'sample_id': _  # Track the original sample ID
            }
            
            # Randomly shuffle the order of stories
            if random.random() < 0.5:
                pair['story_a'], pair['story_b'] = pair['story_b'], pair['story_a']
                pair['upvotes_a'], pair['upvotes_b'] = pair['upvotes_b'], pair['upvotes_a']
                pair['original_chosen'] = 'b'
                
            evaluation_pairs.append(pair)
    
    return evaluation_pairs

def save_prompt(prompt_data: Dict[str, Any], prompts_dir: str = 'prompts'):
    """
    Save a prompt and its metadata to a JSONL file
    
    Args:
        prompt_data: Dictionary containing prompt data and metadata
        prompts_dir: Directory to save prompts in
    """
    os.makedirs(prompts_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    prompts_file = os.path.join(prompts_dir, f'prompts_{timestamp}.jsonl')
    
    with open(prompts_file, 'a') as f:
        json.dump(prompt_data, f)
        f.write('\n')

def evaluate_story_pair(service: ModelService, model_config: Dict[str, Any], prompt: str, story_a: str, story_b: str) -> Dict[str, Any]:
    """
    Evaluate a pair of stories using the specified model
    
    Args:
        service: ModelService instance
        model_config: Dictionary containing model provider and name
        prompt: The writing prompt
        story_a: First story
        story_b: Second story
    
    Returns:
        Dictionary containing the model's evaluation
    """
    messages = [
        {"role": "system", "content": "You are a creative writing judge evaluating two stories based on a given prompt. "
                                    "Consider the following criteria:\n"
                                    "1. Creativity and originality\n"
                                    "2. Coherence and story structure\n"
                                    "3. Engagement and emotional impact\n"
                                    "4. Adherence to the prompt\n"
                                    "5. Writing quality and style"},
        {"role": "user", "content": f"Prompt: {prompt}\n\nStory A:\n{story_a}\n\nStory B:\n{story_b}\n\n"
                                   "Which story is better? First respond with a single letter 'A' or 'B', "
                                   "then provide your reasoning based on the evaluation criteria."}
    ]
    
    # Save prompt data
    prompt_data = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_config['name'],
        'model_provider': model_config['provider'].value,
        'model_id': model_config['model'],
        'messages': messages,
        'writing_prompt': prompt,
        'story_a': story_a,
        'story_b': story_b
    }
    save_prompt(prompt_data)
    
    try:
        response = service.generate_response(
            messages, 
            model_config["provider"],
            model_config["model"]
        )
        
        # Add model info to response
        response['model_name'] = model_config['name']
        return response
        
    except Exception as e:
        print(f"Error with {model_config['name']}: {str(e)}")
        return {'error': str(e), 'model_name': model_config['name']}

def calculate_agreement_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate agreement statistics between model judgments and human preferences"""
    stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'agreement_rate': 0.0,
        'std_dev': []
    })
    
    for result in results:
        model_name = result['model_name']
        stats[model_name]['total'] += 1
        
        # Extract model's choice (first character of answer)
        if 'answer' in result and result['answer']:
            model_choice = result['answer'][0].upper()
            if model_choice == result['original_chosen'].upper():
                stats[model_name]['correct'] += 1
            stats[model_name]['std_dev'].append(1 if model_choice == result['original_chosen'].upper() else 0)
    
    # Calculate agreement rates and standard deviations
    for model_name in stats:
        total = stats[model_name]['total']
        if total > 0:
            agreement_rate = stats[model_name]['correct'] / total
            stats[model_name]['agreement_rate'] = agreement_rate
            stats[model_name]['std_dev'] = np.std(stats[model_name]['std_dev']) / np.sqrt(total)
    
    return stats

def main():
    # Initialize model service
    service = ModelService()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_writingprompts_dataset()
    
    # Prepare evaluation pairs
    print("Preparing evaluation pairs...")
    eval_pairs = prepare_evaluation_pairs(
        dataset,
        num_samples=10,
        num_permutations=3,
        replace=False,  # Sample without replacement by default
        random_seed=42  # For reproducibility
    )
    
    print(f"Evaluating {len(eval_pairs)} total pairs ({len(eval_pairs)//3} unique samples, 3 permutations each)")
    
    # Store all results
    all_results = []
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate pairs with each model
    print("\nEvaluating story pairs...")
    for model_config in MODELS:
        print(f"\nUsing model: {model_config['name']}")
        
        for i, pair in enumerate(eval_pairs, 1):
            print(f"\nEvaluating pair {i} (permutation {pair['permutation_id'] + 1}/3):")
            print(f"Prompt: {pair['prompt'][:100]}...")
            
            result = evaluate_story_pair(service, model_config, pair['prompt'], pair['story_a'], pair['story_b'])
            
            # Add evaluation metadata
            result.update({
                'pair_id': i,
                'permutation_id': pair['permutation_id'],
                'original_chosen': pair['original_chosen'],
                'upvotes_chosen': pair['upvotes_a'] if pair['original_chosen'] == 'a' else pair['upvotes_b'],
                'upvotes_rejected': pair['upvotes_b'] if pair['original_chosen'] == 'a' else pair['upvotes_a']
            })
            
            all_results.append(result)
            
            # Print individual result
            print("\nModel evaluation:")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Answer: {result['answer']}")
                if result.get('reasoning'):
                    print(f"Reasoning: {result['reasoning']}")
            
            print(f"Ground truth: Story {pair['original_chosen'].upper()} was originally chosen")
            print(f"Upvotes - A: {pair['upvotes_a']}, B: {pair['upvotes_b']}")
            print("-" * 80)
    
    # Calculate and display statistics
    print("\nAgreement Statistics:")
    stats = calculate_agreement_stats(all_results)
    for model_name, model_stats in stats.items():
        print(f"\n{model_name}:")
        print(f"Agreement rate: {model_stats['agreement_rate']:.2%} ± {model_stats['std_dev']:.2%}")
        print(f"Total evaluations: {model_stats['total']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'evaluation_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'statistics': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'std_dev'} 
                          for k, v in stats.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main() 