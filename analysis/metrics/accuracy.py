"""
Model accuracy metrics and calculations.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

class AccuracyAnalyzer:
    """
    Analyzes model accuracy from evaluation results.
    """
    
    def __init__(self, data: pd.DataFrame, logger: Optional[logging.Logger] = None):
        """
        Initialize the accuracy analyzer.
        
        Args:
            data: DataFrame containing evaluation results
            logger: Logger instance
        """
        self.data = data
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_model_accuracy(self) -> pd.DataFrame:
        """
        Calculate the accuracy of each model.
        
        Returns:
            DataFrame with model accuracy metrics
        """
        model_accuracy = []
        
        for model in self.data['model_name'].unique():
            model_df = self.data[self.data['model_name'] == model]
            total_evals = len(model_df)
            errors = model_df['error'].notna().sum() if 'error' in model_df.columns else 0
            successful_evals = total_evals - errors
            
            if successful_evals > 0:
                # Filter out rows with errors
                valid_df = model_df[model_df['error'].isna()] if 'error' in model_df.columns else model_df
                
                # Normalize choice and original_chosen for comparison (case-insensitive)
                valid_df['choice_norm'] = valid_df['choice'].str.lower()
                valid_df['original_chosen_norm'] = valid_df['original_chosen'].str.lower()
                
                # Calculate accuracy - check if model's choice matches the original chosen response
                accuracy = (valid_df['choice_norm'] == valid_df['original_chosen_norm']).mean()
                
                # Log some examples for debugging
                self.logger.debug(f"Model {model} - Sample comparisons:")
                for i, (choice, original) in enumerate(zip(valid_df['choice_norm'].head(5), valid_df['original_chosen_norm'].head(5))):
                    self.logger.debug(f"  Row {i}: choice={choice}, original={original}, match={choice == original}")
                
                # Calculate average confidence
                confidence = valid_df['confidence'].mean() if 'confidence' in valid_df.columns else None
                
                model_accuracy.append({
                    'model': model,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'total_evaluations': total_evals,
                    'errors': errors,
                    'successful_evaluations': successful_evals
                })
                
                self.logger.debug(f"Model {model} stats:")
                self.logger.debug(f"Total evaluations: {total_evals}")
                self.logger.debug(f"Successful evaluations: {successful_evals}")
                self.logger.debug(f"Accuracy: {accuracy}")
            else:
                self.logger.warning(f"No successful evaluations for model {model}")
        
        return pd.DataFrame(model_accuracy)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive model statistics.
        
        Returns:
            Dictionary with model statistics
        """
        accuracy_df = self.calculate_model_accuracy()
        
        return {
            'model_accuracy': accuracy_df.to_dict(orient='records'),
            'total_evaluations': len(self.data),
            'models_evaluated': len(accuracy_df)
        } 