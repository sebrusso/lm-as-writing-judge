from typing import List, Dict, Any
import queue
from judge_worker import JudgeWorker
from model_service import ModelProvider
import json
import os
from datetime import datetime
from tqdm import tqdm
import logging
from pathlib import Path

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

class JudgeService:
    def __init__(self, models=MODELS, log_dir: str = 'logs'):
        self.models = models
        self.input_queues = {model['name']: queue.Queue() for model in models}
        self.output_queue = queue.Queue()
        self.workers = []
        self.progress_bars = {}
        self.log_dir = Path(log_dir)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup file handler for all responses
        self.responses_log = self.log_dir / f'responses_{timestamp}.jsonl'
        
        # Setup logging for general events
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / f'judge_service_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('JudgeService')
        
    def log_response(self, response: Dict[str, Any]):
        """Log a single response to the JSONL file"""
        with open(self.responses_log, 'a') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                **response
            }, f)
            f.write('\n')
            
    def start_workers(self):
        """Start worker threads for each model"""
        self.logger.info("Starting worker threads...")
        for model in self.models:
            worker = JudgeWorker(model, self.input_queues[model['name']], self.output_queue)
            worker.start()
            self.workers.append(worker)
        self.logger.info(f"Started {len(self.workers)} workers")
            
    def stop_workers(self):
        """Stop all workers"""
        self.logger.info("Stopping workers...")
        for model in self.models:
            self.input_queues[model['name']].put(None)  # Send poison pill
        for worker in self.workers:
            worker.join()
        self.logger.info("All workers stopped")
            
    def evaluate_pairs(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate story pairs using all models concurrently
        
        Args:
            evaluation_pairs: List of story pairs to evaluate
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting evaluation of {len(evaluation_pairs)} pairs with {len(self.models)} models")
        results = []
        total_tasks = len(evaluation_pairs) * len(self.models)
        
        # Create progress bars
        print("\nStarting evaluation...")
        main_pbar = tqdm(total=total_tasks, desc="Overall Progress", position=0)
        
        # Create progress bars for each model
        for i, model in enumerate(self.models, 1):
            self.progress_bars[model['name']] = tqdm(
                total=len(evaluation_pairs),
                desc=f"{model['name']:15}",
                position=i,
                leave=True
            )
        
        # Start workers
        self.start_workers()
        
        try:
            # Distribute tasks to workers
            for pair in evaluation_pairs:
                for model in self.models:
                    self.input_queues[model['name']].put(pair)
            
            # Collect results
            completed = 0
            while completed < total_tasks:
                result = self.output_queue.get()
                
                # Log the response
                self.log_response(result)
                results.append(result)
                
                # Update progress bars
                main_pbar.update(1)
                if 'model_name' in result:
                    self.progress_bars[result['model_name']].update(1)
                    
                    # Show any errors in the progress bar description
                    if 'error' in result:
                        error_msg = str(result['error'])[:50] + "..." if len(str(result['error'])) > 50 else str(result['error'])
                        self.progress_bars[result['model_name']].set_description(
                            f"{result['model_name']:15} - Error: {error_msg}"
                        )
                        self.logger.error(f"Error in {result['model_name']}: {result['error']}")
                
                completed += 1
                self.output_queue.task_done()
                
        finally:
            # Ensure workers are stopped
            self.stop_workers()
            
            # Close progress bars
            main_pbar.close()
            for pbar in self.progress_bars.values():
                pbar.close()
            
        self.logger.info(f"Evaluation completed. Total results: {len(results)}")
        return results
        
    def save_results(self, results: List[Dict[str, Any]], results_dir: str = 'results'):
        """Save evaluation results to file"""
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f'evaluation_results_{timestamp}.json'
        
        # Calculate summary statistics
        summary = {
            'total_evaluations': len(results),
            'evaluations_per_model': {},
            'agreement_rates': {},
            'average_confidence': {},
            'errors': {},
            'logs_location': {
                'responses_log': str(self.responses_log),
                'service_log': str(self.log_dir / f'judge_service_{timestamp}.log')
            }
        }
        
        for model in self.models:
            model_results = [r for r in results if r['model_name'] == model['name']]
            successful = [r for r in model_results if 'error' not in r]
            
            summary['evaluations_per_model'][model['name']] = len(model_results)
            summary['errors'][model['name']] = len(model_results) - len(successful)
            
            if successful:
                # Calculate agreement rate
                agreements = sum(1 for r in successful if r['choice'] == r['original_chosen'])
                summary['agreement_rates'][model['name']] = agreements / len(successful)
                
                # Calculate average confidence
                confidences = [r['confidence'] for r in successful if 'confidence' in r]
                if confidences:
                    summary['average_confidence'][model['name']] = sum(confidences) / len(confidences)
        
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2)
            
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Full responses log: {self.responses_log}")
        
        print(f"\nResults saved to: {results_file}")
        print(f"Full responses log: {self.responses_log}")
        print("\nSummary:")
        print(json.dumps(summary, indent=2))
            
        return results_file 