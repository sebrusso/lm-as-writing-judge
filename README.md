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

- `analysis/`: Analysis tools for evaluation results
  - `analyzer.py`: Main analysis orchestration module
  - `metrics/`: Analysis metrics modules
    - `accuracy.py`: Model accuracy analysis
    - `length_analysis.py`: Story length impact analysis
    - `upvote_analysis.py`: Upvote impact analysis
    - `agreement.py`: Model agreement analysis
  - `visualization.py`: Visualization and plotting functions
  - `plots/`: Generated visualization plots
  - `logs/`: Analysis log files
  - `results/`: Analysis results in JSON format

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install openai tenacity python-dotenv tqdm pandas matplotlib seaborn numpy
   ```
3. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_key
   DEEPINFRA_API_KEY=your_deepinfra_key
   DEEPSEEK_API_KEY=your_deepseek_key
   GEMINI_API_KEY=your_gemini_key
   ```

## Usage 

### Running Evaluations

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

### Analyzing Results

1. Run the analyzer on evaluation results:
   ```bash
   python -m analysis.analyzer [results_directory]
   ```

   This will:
   - Calculate model accuracy metrics
   - Analyze the impact of upvote differences on model accuracy
   - Analyze the impact of length differences on model accuracy
   - Analyze model agreement
   - Generate visualizations in the `analysis/plots/` directory
   - Save analysis results in `analysis/results/` directory
   - Write logs to the `analysis/logs/` directory

## Supported Models

- OpenAI
  - GPT-4-Mini (`gpt-4o-mini`)
  - GPT-4 (`4o`)
  - o1(`o1`)
  - o1-mini (`o1-mini`)
  - o3-mini (`o3-mini`)
  
- DeepInfra
  - Qwen Reasoning (`qwen-reasoning`)
  - Qwen Base (`qwen-base`)
  - Llama 3.3 70B (`llama-33-70b`)
  - Llama 3.1 70B (`llama-31-70b-instruct`)
  - Llama 3.1 8B (`llama-31-8b-instruct`)
  - Microsoft Phi-4 (`phi-4`)

- DeepSeek
  - DeepSeek Reasoner (`r1`)
  - DeepSeek Chat (`v3`)
  
- Google (Experimental)
  - Gemini (`gemini`)

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

## Analysis Features

### Length Analysis

The system includes advanced analysis of how story length differences impact model accuracy:

- **Equal Frequency Binning**: Ensures each bin contains approximately the same number of samples for more balanced analysis
- **Outlier Handling**: Filters outliers using configurable Z-score thresholds
- **Flexible Bin Count**: Configurable number of bins for granular or coarse analysis
- **Sample Count Tracking**: Tracks sample counts per bin for reliability assessment
- **Enhanced Visualization**: Combined plots showing both accuracy and sample counts
- **Binning Strategy Comparison**: Compares equal frequency and equal width binning approaches

For more details, see [Length Analysis Improvements](length_analysis_improvements.md).

### Upvote Analysis

Analyzes how differences in upvotes between story pairs affect model accuracy:

- Bins story pairs by absolute upvote difference
- Calculates model accuracy within each bin
- Visualizes the relationship between upvote differences and model accuracy

### Model Agreement Analysis

Measures agreement between different models:

- Calculates pairwise agreement between models
- Computes Fleiss' Kappa for overall inter-model agreement
- Generates agreement heatmaps for visual comparison

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