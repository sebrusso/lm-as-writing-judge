# Prompt Optimization for Creative Writing Evaluation

This directory contains tools for optimizing system prompts for creative writing evaluation, specifically for judging pairwise comparisons of responses to writing prompts.

## Overview

The goal of this module is to automatically improve system prompts used for evaluating creative writing by:

1. Testing a prompt against a dataset of paired creative writing examples (where one is known to be better)
2. Analyzing the effectiveness of the prompt based on accuracy of predictions
3. Iteratively refining the prompt to improve judgment accuracy

## Key Components

- **`prompt_opt.py`**: Main script for optimizing prompts using TextGrad and GPT models
- **`test_hf_auth.py`**: Helper script for testing Hugging Face authentication
- **`optimized_prompt_*.txt`**: Output files containing the original and optimized prompts

## Features

- **Automatic Preference Extraction**: Robust multi-strategy parser for extracting model preferences
- **Format Compliance Tracking**: Monitoring how well the model follows output format instructions
- **Extraction Method Statistics**: Tracking which extraction methods are most effective
- **Incremental Prompt Improvement**: Automatically refines prompts to improve accuracy

## How to Use

### Prerequisites

Ensure you have the required dependencies installed:
```
pip install textgrad datasets huggingface-hub python-dotenv
```

You'll need a Hugging Face token for accessing the dataset. Set it as an environment variable:
```
export HUGGING_FACE_TOKEN=your_token_here
```

Or create a `.env` file in the project root with:
```
HUGGING_FACE_TOKEN=your_token_here
```

### Running the Optimizer

To start the prompt optimization process:

```bash
python prompt_opt.py
```

The script will:
1. Load examples from the WritingPrompts dataset
2. Test the initial prompt against these examples
3. Iteratively improve the prompt based on performance
4. Save the optimized prompt to `optimized_prompt_[timestamp].txt`

### Output

The optimizer saves a detailed output file containing:
- The original prompt
- The optimized prompt
- Usage instructions
- Training results for each example
- Statistics on extraction methods used

## Performance Metrics

The optimizer tracks two key metrics:
- **Accuracy**: How often the model correctly identifies the better response
- **Format Compliance**: How often the model follows the specified output format

## Troubleshooting

If you encounter issues with Hugging Face authentication, run:
```bash
python test_hf_auth.py
```

This will verify your token and access to the dataset.

## Dataset

This tool uses the [SAA-Lab/writingprompts-pairwise-train](https://huggingface.co/datasets/SAA-Lab/writingprompts-pairwise-train) dataset, which contains pairs of creative writing responses where one response is preferred over the other. 