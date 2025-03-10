# LM Judge Analysis Tools

This directory contains tools for analyzing results from language model judge experiments.

## Overview

The analysis script (`analyze_results.py`) provides comprehensive analysis of model performance, including:

- Model accuracy and confidence metrics
- Comparative analysis between models
- Impact of upvote differences on accuracy
- Impact of story length differences on accuracy
- Inter-model agreement analysis
- Visualization plots

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run analysis:
```bash
python analyze_results.py
```

## Output

The script generates:

1. Analysis results JSON file with:
   - Model accuracy metrics
   - Upvote impact analysis
   - Length impact analysis
   - Model agreement statistics
   - Fleiss' kappa for inter-model agreement

2. Visualization plots:
   - Model accuracy comparison
   - Upvote difference impact
   - Length difference impact
   - Model agreement heatmap

3. Detailed logs of the analysis process

## Directory Structure

```
analysis/
├── README.md
├── requirements.txt
├── analyze_results.py
├── plots/                    # Generated plots
│   └── YYYYMMDD_HHMMSS/     # Timestamped plot directories
└── analysis_results_*.json   # Analysis results files
```

## Analysis Details

### Model Accuracy
- Per-model accuracy rates
- Confidence scores
- Error rates
- Statistical significance tests

### Upvote Analysis
- Correlation between upvote differences and model accuracy
- Binned accuracy rates by upvote difference
- Statistical significance of upvote impact

### Length Analysis
- Impact of story length differences on model choices
- Correlation between length differences and accuracy
- Binned accuracy rates by length difference

### Model Agreement
- Pairwise agreement rates between models
- Fleiss' kappa for overall agreement
- Agreement heatmap visualization 