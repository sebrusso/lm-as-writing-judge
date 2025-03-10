# LM-as-judge Benchmark for Writing Preference Decisions

A framework for evaluating and comparing creative writing using language models as judges. This system supports multiple LLM providers and implements a robust evaluation pipeline for comparing pairs of creative responses.

## Project Structure

- `judge-service/`: Core evaluation service
  - `judge_worker.py`: Thread-based worker implementation for parallel evaluation
  - `judge_service.py`: Main service coordinating the evaluation process
  - `run_evaluation.py`: CLI tool to run evaluations
  - `results/`: Directory for storing evaluation results
  - `logs/`: Evaluation logs

- `model-service/`: LLM API integration layer
  - `model_service.py`: Unified interface for multiple LLM providers
  - `test_api_keys.py`: Utility to verify API key configuration
  - `gemini.py`: Google Gemini API integration (experimental)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install openai tenacity python-dotenv tqdm
   ```
3. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_key
   DEEPINFRA_API_KEY=your_deepinfra_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

## Usage 

1. Prepare your evaluation data in JSON format:
   ```json
   {
     "pairs": [
       {
         "pair_id": "1",
         "prompt": "Write a story about...",
         "story_a": "First response...",
         "story_b": "Second response...",
         "original_chosen": "A",
         "upvotes_chosen": 10,
         "upvotes_rejected": 5
       }
     ]
   }
   ```

2. Run the evaluation:
   ```bash
   python judge-service/run_evaluation.py \
     --input data.json \
     --output results/ \
     --models gpt4,deepseek-r1 \
     --num-workers 3
   ```

## Supported Models

- OpenAI
  - GPT-4-Mini (`gpt-4o-mini`)
  - GPT-4 (`4o`)
  - GPT-3.5 (`o1`)
  - GPT-3.5-Mini (`o1-mini`)
  - GPT-3-Mini (`o3-mini`)
  
- DeepInfra
  - Qwen Reasoning (`qwen-reasoning`)
  - Qwen Base (`qwen-base`)
  - Llama 3.3 70B (`llama-33-70b`)
  - Llama 3.1 70B (`llama-31-70b-instruct`)
  - Llama 3.1 8B (`llama-31-8b-instruct`)
  - Phi-4 (`phi-4`)

- DeepSeek
  - DeepSeek Reasoner (`r1`)
  - DeepSeek Chat (`v3`)

## Output Format

Results are saved in JSON format with the following structure:
```json
{
  "model_name": "gpt-4",
  "pair_id": "1",
  "choice": "A",
  "reasoning": "Detailed analysis...",
  "confidence": 0.85,
  "is_valid": true,
  "timestamp": "2024-03-07T12:00:00"
}
```

## Error Handling

The system implements robust error handling with:
- Automatic retries for API failures
- Response validation and reformatting
- Detailed logging
- Queue management for parallel processing

## Contributing

Contributions are welcome! Please ensure you:
1. Add tests for new features
2. Update documentation
3. Follow the existing code style
4. Handle errors appropriately 