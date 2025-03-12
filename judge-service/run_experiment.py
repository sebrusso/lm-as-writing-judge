import sys
import os
from datasets import load_dataset
from judge_service import JudgeService
import argparse

def prepare_evaluation_pairs(dataset, num_samples=1):
    """Prepare pairs for evaluation"""
    df = dataset['train'].to_pandas()
    samples = df.sample(n=num_samples)
    
    pairs = []
    for idx, row in samples.iterrows():
        pair = {
            'pair_id': idx,
            'prompt': row['prompt'],
            'story_a': row['chosen'],
            'story_b': row['rejected'],
            'original_chosen': 'a',
            'upvotes_chosen': row['upvotes_chosen'],
            'upvotes_rejected': row['upvotes_rejected'],
            'permutation_id': 0
        }
        pairs.append(pair)
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description='Run writing evaluation experiment')
    parser.add_argument('--num-samples', type=int, default=1,
                      help='Number of story pairs to evaluate')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory to save logs')
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("SAA-Lab/writingprompts-pairwise-test")
    
    # Prepare evaluation pairs
    print(f"Preparing {args.num_samples} evaluation pairs...")
    eval_pairs = prepare_evaluation_pairs(dataset, args.num_samples)
    
    # Initialize service
    print("\nInitializing judge service...")
    service = JudgeService(log_dir=args.log_dir)
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = service.evaluate_pairs(eval_pairs)
    
    # Save results
    service.save_results(results, args.results_dir)

if __name__ == "__main__":
    main() 