from typing import Dict, Any, List, Optional
import queue
import threading
from model_service import ModelService, ModelProvider
import json
import os
from datetime import datetime
import re
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

class JudgeWorker(threading.Thread):
    """
    A worker thread that evaluates creative writing responses using language models.
    
    This worker processes tasks from an input queue, where each task contains a pair
    of creative writing responses to be compared. The worker uses a language model
    to evaluate the responses and produces a structured output with the evaluation
    results.
    
    Attributes:
        model_config (Dict[str, Any]): Configuration for the language model
        input_queue (queue.Queue): Queue containing tasks to process
        output_queue (queue.Queue): Queue for storing evaluation results
        service (ModelService): Service for interacting with language models
        logger (logging.Logger): Logger instance for this worker
    """
    
    def __init__(self, model_config: Dict[str, Any], input_queue: queue.Queue, output_queue: queue.Queue):
        """
        Initialize the worker thread.
        
        Args:
            model_config: Configuration dictionary containing model settings
            input_queue: Queue containing evaluation tasks
            output_queue: Queue for storing evaluation results
        """
        super().__init__()
        self.model_config = model_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.service = ModelService()
        self.logger = logging.getLogger(f"JudgeWorker-{model_config['name']}")
        
    def parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse the LLM's response into a structured format with validation.
        
        The expected response format is:
        Preferred: [A or B]
        Reasoning: [detailed analysis]
        Confidence: [0-1 score]
        
        Args:
            text: Raw response text from the language model
            
        Returns:
            dict: Structured response with validation results
        """
        # Initialize default values
        result = {
            'choice': None,
            'reasoning': '',
            'confidence': 0.0,
            'raw_response': text,
            'is_valid': False
        }
        
        try:
            # Extract preferred choice
            preferred_match = re.search(r'Preferred:\s*([AB])', text, re.IGNORECASE)
            if preferred_match:
                result['choice'] = preferred_match.group(1).upper()
                
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=Confidence:|$)', text, re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
                
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*(0?\.\d+|1\.0|1|0)', text)
            if confidence_match:
                result['confidence'] = float(confidence_match.group(1))
            
            # Validate required fields
            result['is_valid'] = (
                result['choice'] is not None and 
                result['choice'] in ['A', 'B'] and
                result['reasoning'].strip() != '' and
                0 <= result['confidence'] <= 1
            )
            
            if not result['is_valid']:
                missing = []
                if result['choice'] not in ['A', 'B']:
                    missing.append('valid choice (A/B)')
                if not result['reasoning'].strip():
                    missing.append('reasoning')
                if not (0 <= result['confidence'] <= 1):
                    missing.append('valid confidence score')
                self.logger.warning(f"Invalid response - missing: {', '.join(missing)}")
                
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            result['error'] = str(e)
            
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_model_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Get response from model with retry logic.
        
        This method will retry failed API calls up to 3 times with exponential backoff.
        
        Args:
            messages: List of message dictionaries to send to the model
            
        Returns:
            dict: Model response
            
        Raises:
            Exception: If all retry attempts fail
        """
        try:
            response = self.service.generate_response(
                messages,
                self.model_config["provider"],
                self.model_config["model"]
            )
            
            # Handle missing answer field (specific to DeepSeek)
            if 'answer' not in response:
                if 'choices' in response and len(response['choices']) > 0:
                    if 'message' in response['choices'][0]:
                        response['answer'] = response['choices'][0]['message']['content']
                    elif 'text' in response['choices'][0]:
                        response['answer'] = response['choices'][0]['text']
                else:
                    raise ValueError("Response missing required 'answer' field")
                    
            return response
            
        except Exception as e:
            self.logger.error(f"API error: {str(e)}")
            raise  # Let retry decorator handle it
    
    def evaluate_story_pair(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a pair of creative writing responses.
        
        This method constructs appropriate prompts based on the model type,
        handles the evaluation process, and formats the results.
        
        Args:
            task: Dictionary containing the story pair and metadata
            
        Returns:
            dict: Evaluation results including choice, reasoning, and confidence
        """
        # Special handling for GPT-3.5 to enforce stricter format
        if self.model_config['name'] == 'GPT-3.5':
            messages = [
                {"role": "system", "content": 
                    "You are a creative writing judge evaluating two responses. "
                    "You MUST format your response EXACTLY as follows:\n"
                    "Preferred: [ONLY write A or B]\n"
                    "Reasoning: [your analysis]\n"
                    "Confidence: [a number between 0 and 1]\n\n"
                    "Any deviation from this format will result in failure."
                },
                {"role": "user", "content": 
                    f"Compare these responses to: {task['prompt']}\n\n"
                    f"Response A:\n{task['story_a']}\n\n"
                    f"Response B:\n{task['story_b']}\n\n"
                    "Analyze based on:\n"
                    "1. Imagery (vivid descriptions)\n"
                    "2. Tension (dramatic interest)\n"
                    "3. Pattern (structure)\n"
                    "4. Energy (engaging style)\n"
                    "5. Insight (meaningful ideas)\n\n"
                    "IMPORTANT: Your response MUST start with 'Preferred: ' followed by ONLY 'A' or 'B'.\n"
                    "Then 'Reasoning: ' followed by your analysis.\n"
                    "End with 'Confidence: ' and a number between 0 and 1."
                }
            ]
        else:
            messages = [
                {"role": "system", "content": 
                    "You are a creative writing judge evaluating the creativity of two responses to a prompt. "
                    "You will analyze them based on specific creative dimensions and provide a structured evaluation."
                },
                {"role": "user", "content": 
                    "Please compare the following two responses to the given input.\n"
                    "Think through this step by step:\n\n"
                    f"Input: {task['prompt']}\n\n"
                    f"Response A: {task['story_a']}\n\n"
                    f"Response B: {task['story_b']}\n\n"
                    "You are acting as a creative writer tasked with evaluating the creativity of two comments.\n"
                    "Compare the comments based on these dimensions:\n"
                    "- Imagery: vivid descriptions and sensory details\n"
                    "- Tension: dramatic interest and narrative conflict\n"
                    "- Pattern: structural elements and composition\n"
                    "- Energy: engaging style and dynamic writing\n"
                    "- Insight: meaningful ideas and depth of thought\n\n"
                    "Choose which comment (A or B) displays these creative elements more effectively overall.\n"
                    "In your analysis, provide:\n"
                    "- Your detailed reasoning for the preference, citing specific examples from each response\n"
                    "- Which response you prefer (A or B)\n"
                    "- Your confidence in this comparison (0-1)\n\n"
                    "Format your answer exactly as follows:\n"
                    "Preferred: [A or B]\n"
                    "Reasoning: [your step-by-step comparison]\n"
                    "Confidence: [confidence score]"
                }
            ]
        
        try:
            # Get response with retries
            response = self.get_model_response(messages)
            
            # Parse and validate the response
            parsed = self.parse_response(response['answer'])
            
            # If response is invalid, try one more time with a more explicit prompt
            if not parsed['is_valid']:
                self.logger.warning("Invalid response, retrying with explicit prompt")
                if self.model_config['name'] == 'GPT-3.5':
                    messages[0]['content'] = (
                        "CRITICAL: Your response MUST follow this EXACT format:\n\n"
                        "Preferred: [write ONLY A or B]\n"
                        "Reasoning: [your analysis]\n"
                        "Confidence: [number between 0-1]\n\n"
                        "DO NOT deviate from this format or include any other text."
                    )
                else:
                    messages[1]['content'] += "\n\nIMPORTANT: You MUST explicitly state your preference (A or B) and confidence score (0-1) in the exact format shown above."
                response = self.get_model_response(messages)
                parsed = self.parse_response(response['answer'])
            
            # Add metadata
            result = {
                'model_name': self.model_config['name'],
                'pair_id': task['pair_id'],
                'permutation_id': task['permutation_id'],
                'original_chosen': task['original_chosen'],
                'upvotes_chosen': task['upvotes_chosen'],
                'upvotes_rejected': task['upvotes_rejected'],
                'choice': parsed['choice'],
                'reasoning': parsed['reasoning'],
                'confidence': parsed['confidence'],
                'raw_response': parsed['raw_response'],
                'timestamp': datetime.now().isoformat(),
                'is_valid': parsed['is_valid']
            }
            
            if not parsed['is_valid']:
                result['error'] = 'Invalid response format'
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating story pair: {str(e)}")
            return {
                'error': str(e),
                'model_name': self.model_config['name'],
                'timestamp': datetime.now().isoformat(),
                'is_valid': False,
                **task  # Include original task data
            }
    
    def run(self):
        """
        Main worker thread loop.
        
        Continuously processes tasks from the input queue until a poison pill (None)
        is received. Each task is evaluated and the results are placed in the output
        queue.
        """
        while True:
            try:
                # Get task from queue with timeout
                task = self.input_queue.get(timeout=1)
                
                # Check for poison pill
                if task is None:
                    self.input_queue.task_done()
                    break
                    
                try:
                    # Evaluate story pair
                    result = self.evaluate_story_pair(task)
                    self.output_queue.put(result)
                finally:
                    # Mark task as done regardless of success/failure
                    self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in worker thread: {str(e)}")
                continue 