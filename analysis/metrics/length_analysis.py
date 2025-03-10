"""
Length impact analysis for model evaluations.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Callable

class LengthAnalyzer:
    """
    Analyzes the impact of length differences on model accuracy.
    """
    
    def __init__(self, data: pd.DataFrame, length_diff_func: Callable, logger: Optional[logging.Logger] = None):
        """
        Initialize the length analyzer.
        
        Args:
            data: DataFrame containing evaluation results
            length_diff_func: Function to calculate length difference
            logger: Logger instance
        """
        self.data = data
        self.length_diff_func = length_diff_func
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_length_impact(self, binning_strategy: str = 'equal_freq', n_bins: int = 10, 
                             filter_outliers: bool = False, outlier_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Analyze the impact of length differences on model accuracy.
        
        Args:
            binning_strategy: Strategy for creating bins ('equal_width', 'equal_freq')
            n_bins: Number of bins to create
            filter_outliers: Whether to filter outliers before binning
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Dictionary with length impact analysis results
        """
        # Calculate length differences
        self.data['length_diff'] = self.data.apply(self.length_diff_func, axis=1)
        
        # Count valid length differences
        valid_length_diffs = self.data['length_diff'].dropna()
        self.logger.info(f"Valid length differences: {len(valid_length_diffs)} out of {len(self.data)} rows")
        
        if len(valid_length_diffs) == 0:
            return {'error': 'No valid length differences found'}
        
        if len(valid_length_diffs) < n_bins:
            self.logger.warning(f"Only {len(valid_length_diffs)} valid length differences found, which is less than the {n_bins} required for binning")
            return {'error': f'Only {len(valid_length_diffs)} valid length differences found, need at least {n_bins} for binning'}
        
        # Filter outliers if requested
        if filter_outliers:
            z_scores = np.abs((valid_length_diffs - valid_length_diffs.mean()) / valid_length_diffs.std())
            outliers = z_scores > outlier_threshold
            
            if outliers.sum() > 0:
                self.logger.info(f"Filtering {outliers.sum()} outliers with z-score > {outlier_threshold}")
                self.data = self.data[~(self.data['length_diff'].notna() & outliers)]
                valid_length_diffs = self.data['length_diff'].dropna()
        
        # Create bins based on the selected strategy
        if binning_strategy == 'equal_width':
            # Equal-width bins (original approach)
            min_val = valid_length_diffs.min()
            max_val = valid_length_diffs.max()
            
            # If all values are the same, we can't create meaningful bins
            if min_val == max_val:
                self.logger.warning(f"All length differences are the same value ({min_val}), cannot create bins")
                return {'error': f'All length differences are the same value ({min_val}), cannot create bins'}
            
            # Create bin edges with linspace to ensure equal spacing
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            self.data['length_diff_bin'] = pd.cut(
                self.data['length_diff'], 
                bins=bin_edges,
                labels=False,
                include_lowest=True
            )
            
            # Get bin categories for reporting
            bin_categories = [f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]
            
        elif binning_strategy == 'equal_freq':
            # Equal-frequency bins (quantiles)
            try:
                bin_result = pd.qcut(valid_length_diffs, n_bins, retbins=True, duplicates='drop')
                self.data['length_diff_bin'] = pd.qcut(self.data['length_diff'], n_bins, labels=False, duplicates='drop')
                bin_edges = bin_result[1]
                bin_categories = [f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]
            except ValueError as e:
                self.logger.error(f"Error creating equal-frequency bins: {e}")
                # Fall back to equal-width bins
                self.logger.info("Falling back to equal-width bins")
                min_val = valid_length_diffs.min()
                max_val = valid_length_diffs.max()
                bin_edges = np.linspace(min_val, max_val, n_bins + 1)
                self.data['length_diff_bin'] = pd.cut(
                    self.data['length_diff'], 
                    bins=bin_edges,
                    labels=False,
                    include_lowest=True
                )
                bin_categories = [f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]
        else:
            self.logger.error(f"Unknown binning strategy: {binning_strategy}")
            return {'error': f'Unknown binning strategy: {binning_strategy}'}
        
        self.logger.debug(f"Bin categories: {bin_categories}")
        
        # Log unique bins for debugging
        unique_bins = self.data['length_diff_bin'].unique()
        self.logger.debug(f"Unique bins: {sorted([b for b in unique_bins if pd.notna(b)])}")
        
        # Calculate accuracy by bin for each model
        accuracy_by_bin = {}
        sample_counts_by_bin = {}
        
        for model in self.data['model_name'].unique():
            model_df = self.data[self.data['model_name'] == model]
            
            # Filter out rows with errors
            model_df = model_df[model_df['error'].isna()]
            
            # Normalize choice and original_chosen for comparison (case-insensitive)
            model_df['choice_norm'] = model_df['choice'].str.lower()
            model_df['original_chosen_norm'] = model_df['original_chosen'].str.lower()
            
            accuracies = []
            sample_counts = []
            
            for bin_idx in range(n_bins):
                bin_df = model_df[model_df['length_diff_bin'] == bin_idx]
                
                if len(bin_df) > 0:
                    # Calculate accuracy for this bin
                    accuracy = (bin_df['choice_norm'] == bin_df['original_chosen_norm']).mean()
                    accuracies.append(accuracy)
                    sample_counts.append(len(bin_df))
                else:
                    # No data for this bin
                    accuracies.append(None)
                    sample_counts.append(0)
            
            # Replace None values with 0 for backward compatibility
            accuracies_clean = [acc if acc is not None else 0 for acc in accuracies]
            
            accuracy_by_bin[model] = accuracies_clean
            sample_counts_by_bin[model] = sample_counts
        
        # Calculate correlation between length difference and accuracy
        length_diff_correlation = {}
        
        for model in self.data['model_name'].unique():
            model_df = self.data[self.data['model_name'] == model]
            
            # Skip if all length differences are the same
            if model_df['length_diff'].nunique() <= 1:
                self.logger.warning(f"Model {model} has uniform length differences, skipping correlation calculation")
                length_diff_correlation[model] = {
                    'correlation': np.nan,
                    'pvalue': np.nan
                }
                continue
                
            # Normalize choice and original_chosen for comparison
            model_df['choice_norm'] = model_df['choice'].str.lower()
            model_df['original_chosen_norm'] = model_df['original_chosen'].str.lower()
            
            # Calculate accuracy
            model_df['correct'] = (model_df['choice_norm'] == model_df['original_chosen_norm']).astype(int)
            
            # Calculate correlation
            try:
                from scipy.stats import pearsonr
                valid_df = model_df.dropna(subset=['length_diff', 'correct'])
                
                if len(valid_df) > 1:
                    correlation, pvalue = pearsonr(valid_df['length_diff'], valid_df['correct'])
                    length_diff_correlation[model] = {
                        'correlation': correlation,
                        'pvalue': pvalue
                    }
                else:
                    self.logger.warning(f"Not enough valid data points for model {model} to calculate correlation")
                    length_diff_correlation[model] = {
                        'correlation': np.nan,
                        'pvalue': np.nan
                    }
            except Exception as e:
                self.logger.error(f"Error calculating correlation for model {model}: {e}")
                length_diff_correlation[model] = {
                    'correlation': np.nan,
                    'pvalue': np.nan
                }
        
        # Create a combined visualization with accuracy and sample counts
        self._create_length_impact_plot(accuracy_by_bin, sample_counts_by_bin, bin_categories, binning_strategy)
        
        return {
            'accuracy_by_bin': accuracy_by_bin,
            'sample_counts_by_bin': sample_counts_by_bin,
            'bin_categories': bin_categories,
            'length_diff_correlation': length_diff_correlation,
            'binning_strategy': binning_strategy
        }
    
    def _create_length_impact_plot(self, accuracy_by_bin, sample_counts_by_bin, bin_categories, binning_strategy):
        """
        Create a combined plot showing both accuracy and sample counts.
        
        Args:
            accuracy_by_bin: Dictionary mapping model names to lists of accuracies by bin
            sample_counts_by_bin: Dictionary mapping model names to lists of sample counts by bin
            bin_categories: List of bin category labels
            binning_strategy: Strategy used for binning
        """
        try:
            # Create figure with two subplots
            plt.figure(figsize=(15, 12))
            
            # Plot accuracy by bin
            plt.subplot(2, 1, 1)
            
            for model, accuracies in accuracy_by_bin.items():
                plt.plot(range(len(accuracies)), accuracies, marker='o', label=model, linewidth=2, markersize=8)
            
            plt.xlabel('Absolute Length Difference Bins\n(Smallest → Largest)', fontsize=12)
            plt.ylabel('Model Accuracy', fontsize=12)
            plt.title(f'Model Accuracy by Absolute Length Difference\n({binning_strategy.replace("_", " ").title()} Binning)', fontsize=14, pad=20)
            
            # Add bin categories as x-tick labels
            plt.xticks(range(len(bin_categories)), [f'D{i+1}\n|Δ|={cat}' for i, cat in enumerate(bin_categories)], 
                      rotation=45, ha='right')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            # Plot sample counts by bin
            plt.subplot(2, 1, 2)
            
            x = np.arange(len(bin_categories))
            width = 0.8 / len(sample_counts_by_bin)
            
            for i, (model, counts) in enumerate(sample_counts_by_bin.items()):
                plt.bar(x + i * width - 0.4, counts, width, label=model)
            
            plt.xlabel('Absolute Length Difference Bins\n(Smallest → Largest)', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            
            # Add bin categories as x-tick labels
            plt.xticks(range(len(bin_categories)), [f'D{i+1}\n|Δ|={cat}' for i, cat in enumerate(bin_categories)], 
                      rotation=45, ha='right')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            plt.tight_layout()
            
            # Save plot
            from pathlib import Path
            plots_dir = Path('analysis/plots')
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            filename = plots_dir / f'model_accuracy_by_length_{binning_strategy}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved length impact plot to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error creating length impact plot: {e}") 