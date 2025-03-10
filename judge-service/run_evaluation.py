import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import pandas as pd
import random
from typing import List, Dict, Any
from model_service import ModelService, ModelProvider
from judge_service import JudgeService
import time

def prepare_evaluation_pairs(
    dataset, 
    num_samples: int = 10, 
    num_permutations: int = 3,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Prepare story pairs for evaluation"""
    random.seed(random_seed)
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)
    sampled_rows = df.sample(n=min(num_samples, len(df)))
    
    evaluation_pairs = []
    for idx, row in sampled_rows.iterrows():
        # Create N permutations of each pair
        for perm_id in range(num_permutations):
            pair = {
                'pair_id': idx,
                'prompt': row['prompt'],
                'story_a': row['chosen'],
                'story_b': row['rejected'],
                'original_chosen': 'a',
                'upvotes_chosen': row['upvotes_chosen'],
                'upvotes_rejected': row['upvotes_rejected'],
                'permutation_id': perm_id
            }
            
            # Randomly shuffle order
            if random.random() < 0.5:
                pair['story_a'], pair['story_b'] = pair['story_b'], pair['story_a']
                pair['upvotes_chosen'], pair['upvotes_rejected'] = pair['upvotes_rejected'], pair['upvotes_chosen']
                pair['original_chosen'] = 'b'
            
            evaluation_pairs.append(pair)
    
    return evaluation_pairs

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("SAA-Lab/writingprompts-pairwise-test")
    
    # Prepare evaluation pairs
    print("Preparing evaluation pairs...")
    eval_pairs = prepare_evaluation_pairs(
        dataset['train'],
        num_samples=10,      # Use 10 samples
        num_permutations=3  # 3 permutations each for statistical significance
    )
    
    print(f"Prepared {len(eval_pairs)} pairs for evaluation")
    
    # Initialize judge service
    service = JudgeService()
    
    # Run evaluation
    print("\nStarting evaluation...")
    start_time = time.time()
    
    results = service.evaluate_pairs(eval_pairs)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print(f"\nEvaluation completed in {duration:.2f} seconds")
    print(f"Total evaluations: {len(results)}")
    
    # Calculate agreement rates
    agreements = {}
    for result in results:
        model = result['model_name']
        if model not in agreements:
            agreements[model] = {'correct': 0, 'total': 0}
            
        if 'error' not in result and 'answer' in result:
            agreements[model]['total'] += 1
            if result['answer'][0].upper() == result['original_chosen'].upper():
                agreements[model]['correct'] += 1
    
    # Print agreement rates
    print("\nAgreement rates:")
    for model, stats in agreements.items():
        if stats['total'] > 0:
            rate = stats['correct'] / stats['total']
            print(f"{model}: {rate:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results
    results_file = service.save_results(results)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main() 