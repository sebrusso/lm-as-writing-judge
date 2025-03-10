"""
Upvote impact analysis for model evaluations.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class UpvoteAnalyzer:
    """
    Analyzes the impact of upvote differences on model accuracy.
    """
    
    def __init__(self, data: pd.DataFrame, logger: Optional[logging.Logger] = None):
        """
        Initialize the upvote analyzer.
        
        Args:
            data: DataFrame containing evaluation results
            logger: Logger instance
        """
        self.data = data
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_upvote_impact(self) -> Dict[str, Any]:
        """
        Analyze the impact of upvote differences on model accuracy.
        
        Returns:
            Dictionary with upvote impact analysis results
        """
        # Calculate upvote differences
        self.data['upvote_diff'] = self.data['upvotes_chosen'] - self.data['upvotes_rejected']
        
        # Calculate absolute upvote differences for binning
        self.data['abs_upvote_diff'] = self.data['upvote_diff'].abs()
        
        # Count valid upvote differences
        valid_upvote_diffs = self.data['upvote_diff'].dropna()
        self.logger.info(f"Valid upvote differences: {len(valid_upvote_diffs)} out of {len(self.data)} rows")
        
        if len(valid_upvote_diffs) == 0:
            return {'error': 'No valid upvote differences found'}
        
        if len(valid_upvote_diffs) < 10:
            self.logger.warning(f"Only {len(valid_upvote_diffs)} valid upvote differences found, which is less than the 10 required for decile analysis")
            return {'error': f'Only {len(valid_upvote_diffs)} valid upvote differences found, need at least 10 for decile analysis'}
        
        # Create 10 quantile-based bins for absolute upvote differences
        try:
            # Use quantile-based binning to ensure equal number of samples in each bin
            quantiles = pd.qcut(self.data['abs_upvote_diff'], 10, duplicates='drop')
            self.data['upvote_diff_bin'] = quantiles.cat.codes
            
            # Get the bin edges for reporting
            bin_edges = [round(edge) for edge in quantiles.cat.categories.map(lambda x: x.right).tolist()]
            bin_edges = [0] + bin_edges  # Add 0 as the first edge
            
            # Create formatted bin labels
            bin_categories = []
            for i in range(len(bin_edges) - 1):
                bin_categories.append(f"{bin_edges[i]}-{bin_edges[i+1]}")
            
            self.logger.debug(f"Bin categories: {bin_categories}")
            
            # Log unique deciles for debugging
            unique_deciles = self.data['upvote_diff_bin'].unique()
            self.logger.debug(f"Unique deciles: {sorted(unique_deciles)}")
            
        except Exception as e:
            self.logger.error(f"Error creating bins: {e}")
            return {'error': f'Error creating bins: {e}'}
        
        # Calculate accuracy by decile for each model
        accuracy_by_decile = {}
        
        for model in self.data['model_name'].unique():
            model_df = self.data[self.data['model_name'] == model]
            
            # Normalize choice and original_chosen for comparison (case-insensitive)
            model_df['choice_norm'] = model_df['choice'].str.lower()
            model_df['original_chosen_norm'] = model_df['original_chosen'].str.lower()
            
            accuracies = []
            
            for i in range(len(bin_categories)):
                bin_df = model_df[model_df['upvote_diff_bin'] == i]
                
                if len(bin_df) > 0:
                    # Calculate accuracy for this bin
                    accuracy = (bin_df['choice_norm'] == bin_df['original_chosen_norm']).mean()
                    accuracies.append(accuracy)
                else:
                    # No data for this bin
                    accuracies.append(0)
            
            # Ensure we have the right number of accuracies
            if len(accuracies) < len(bin_categories):
                accuracies.extend([0] * (len(bin_categories) - len(accuracies)))
                
            accuracy_by_decile[model] = accuracies
        
        # Calculate correlation between upvote difference and accuracy
        try:
            from scipy.stats import pearsonr
            
            # Normalize choice and original_chosen for comparison
            self.data['choice_norm'] = self.data['choice'].str.lower()
            self.data['original_chosen_norm'] = self.data['original_chosen'].str.lower()
            
            # Calculate accuracy
            self.data['correct'] = (self.data['choice_norm'] == self.data['original_chosen_norm']).astype(int)
            
            # Calculate correlation
            valid_df = self.data.dropna(subset=['upvote_diff', 'correct'])
            
            if len(valid_df) > 1 and valid_df['upvote_diff'].nunique() > 1:
                correlation, pvalue = pearsonr(valid_df['upvote_diff'], valid_df['correct'])
                upvote_diff_correlation = {
                    'correlation': correlation,
                    'pvalue': pvalue
                }
                self.logger.info(f"Upvote difference correlation: {correlation:.4f} (p={pvalue:.6f})")
            else:
                self.logger.warning("Not enough valid data points or uniform upvote differences, skipping correlation calculation")
                upvote_diff_correlation = {
                    'correlation': np.nan,
                    'pvalue': np.nan
                }
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            upvote_diff_correlation = {
                'correlation': np.nan,
                'pvalue': np.nan
            }
        
        return {
            'accuracy_by_decile': accuracy_by_decile,
            'decile_ranges': bin_categories,
            'upvote_diff_correlation': upvote_diff_correlation
        } 