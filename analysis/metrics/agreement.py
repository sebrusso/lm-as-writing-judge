"""
Model agreement analysis module.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable

class AgreementAnalyzer:
    """
    Analyzes agreement between different models.
    """
    
    def __init__(self, data: pd.DataFrame, kappa_func: Callable, logger: Optional[logging.Logger] = None):
        """
        Initialize the agreement analyzer.
        
        Args:
            data: DataFrame containing evaluation results
            kappa_func: Function to calculate Fleiss' kappa
            logger: Logger instance
        """
        self.data = data
        self.kappa_func = kappa_func
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_model_agreement(self) -> Dict[str, Any]:
        """
        Analyze agreement between different models.
        
        Returns:
            Dictionary with model agreement analysis results
        """
        results = {}
        
        # Create agreement matrix
        models = self.data['model_name'].unique()
        agreement_matrix = pd.DataFrame(index=models, columns=models)
        
        for i, model1 in enumerate(models):
            for model2 in models:
                if model1 == model2:
                    continue
                    
                # Get overlapping evaluations
                model1_data = self.data[self.data['model_name'] == model1]
                model2_data = self.data[self.data['model_name'] == model2]
                
                # Filter out errors if error column exists
                if 'error' in self.data.columns:
                    model1_data = model1_data[~model1_data['error'].notna()]
                    model2_data = model2_data[~model2_data['error'].notna()]
                
                # Get common pairs
                common_pairs = pd.merge(
                    model1_data[['pair_id', 'permutation_id', 'choice']],
                    model2_data[['pair_id', 'permutation_id', 'choice']],
                    on=['pair_id', 'permutation_id'],
                    suffixes=('_1', '_2')
                )
                
                if len(common_pairs) > 0:
                    # Calculate agreement rate
                    agreement = (common_pairs['choice_1'].str.upper() == common_pairs['choice_2'].str.upper()).mean()
                    agreement_matrix.loc[model1, model2] = agreement
        
        results['agreement_matrix'] = agreement_matrix.to_dict()
        
        # Calculate Fleiss' kappa for overall agreement
        # Group by pair and get all model decisions
        if 'error' in self.data.columns:
            decisions_data = self.data[~self.data['error'].notna()]
        else:
            decisions_data = self.data
            
        pair_decisions = decisions_data.groupby(['pair_id', 'permutation_id'])['choice'].apply(list)
        
        if len(pair_decisions) > 0:
            # Convert to matrix format required for Fleiss' kappa
            decision_matrix = np.zeros((len(pair_decisions), 2))  # 2 categories: A and B
            for i, decisions in enumerate(pair_decisions):
                decision_matrix[i, 0] = decisions.count('A')
                decision_matrix[i, 1] = decisions.count('B')
            
            results['fleiss_kappa'] = self.kappa_func(decision_matrix)
        
        return results 