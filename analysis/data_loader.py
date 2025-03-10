"""
Data loading and preprocessing for the analysis module.
"""
import os
import json
import re
import glob
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

class DataLoader:
    """
    Handles loading and preprocessing data for analysis.
    """
    
    def __init__(self, results_dir: Union[str, Path], logger: Optional[logging.Logger] = None):
        """
        Initialize the data loader.
        
        Args:
            results_dir: Directory containing results files
            logger: Logger instance
        """
        self.results_dir = Path(results_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.results = None
    
    def load_results(self) -> pd.DataFrame:
        """
        Load results from the specified directory.
        
        Returns:
            DataFrame containing the loaded results
        """
        self.logger.info(f"Loading results from {self.results_dir}")
        
        # Find all JSON and JSONL files in the results directory
        json_files = glob.glob(os.path.join(self.results_dir, "evaluation_results_*.json"))
        jsonl_files = glob.glob(os.path.join(self.results_dir, "evaluation_results_*.jsonl"))
        
        all_files = json_files + jsonl_files
        if not all_files:
            self.logger.error(f"No result files found in {self.results_dir}")
            return pd.DataFrame()
        
        # Get the most recent file based on the timestamp in the filename
        all_files.sort(key=lambda x: os.path.basename(x).split('_')[2].split('.')[0], reverse=True)
        most_recent_file = all_files[0]
        self.logger.info(f"Loading results from {most_recent_file}")
        
        # Load the results
        if most_recent_file.endswith('.json'):
            with open(most_recent_file, 'r') as f:
                self.results = pd.DataFrame(json.load(f))
        else:  # JSONL file
            results = []
            with open(most_recent_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
            self.results = pd.DataFrame(results)
        
        self.logger.info(f"Loaded {len(self.results)} results")
        
        # Process the data
        self._extract_stories()
        self._log_data_stats()
        
        return self.results
    
    def _extract_stories(self):
        """Extract stories from the messages field."""
        # Add story_a and story_b columns
        self.results['story_a'] = None
        self.results['story_b'] = None
        
        # Extract stories from messages
        for idx, row in self.results.iterrows():
            try:
                if 'messages' in row and isinstance(row['messages'], list) and len(row['messages']) >= 2:
                    user_message = next((msg['content'] for msg in row['messages'] if msg['role'] == 'user'), None)
                    if user_message:
                        # Extract Response A
                        response_a_match = re.search(r'Response A:(.*?)(?=Response B:|$)', user_message, re.DOTALL)
                        # Extract Response B
                        response_b_match = re.search(r'Response B:(.*?)(?=You are acting as|$)', user_message, re.DOTALL)
                        
                        if response_a_match:
                            self.results.at[idx, 'story_a'] = response_a_match.group(1).strip()
                        if response_b_match:
                            self.results.at[idx, 'story_b'] = response_b_match.group(1).strip()
            except Exception as e:
                self.logger.error(f"Error extracting stories for row {idx}: {e}")
    
    def _log_data_stats(self):
        """Log statistics about the loaded data."""
        # Log DataFrame information
        self.logger.info(f"DataFrame shape: {self.results.shape}")
        self.logger.info(f"DataFrame columns: {list(self.results.columns)}")
        
        # Check for None values in story columns
        none_count_a = self.results['story_a'].isna().sum()
        none_count_b = self.results['story_b'].isna().sum()
        self.logger.info(f"NaN counts: story_a: {none_count_a}, story_b: {none_count_b}")
        
        # Check for empty strings
        empty_count_a = (self.results['story_a'] == '').sum()
        empty_count_b = (self.results['story_b'] == '').sum()
        self.logger.info(f"Empty string counts: story_a: {empty_count_a}, story_b: {empty_count_b}")
        
        # Count None values (different from NaN)
        none_literal_count_a = (self.results['story_a'].isna() | (self.results['story_a'] == None)).sum()
        none_literal_count_b = (self.results['story_b'].isna() | (self.results['story_b'] == None)).sum()
        self.logger.info(f"None counts: story_a: {none_literal_count_a}, story_b: {none_literal_count_b}")
        
        # Count non-null values
        non_null_count_a = (~self.results['story_a'].isna()).sum()
        non_null_count_b = (~self.results['story_b'].isna()).sum()
        self.logger.info(f"Non-null values: story_a: {non_null_count_a}, story_b: {non_null_count_b}")
        
        # Print a few sample rows with non-null stories
        non_null_indices = self.results[~self.results['story_a'].isna() & ~self.results['story_b'].isna()].index
        if len(non_null_indices) > 0:
            self.logger.info("Sample non-null story data:")
            for i, idx in enumerate(non_null_indices[:5]):  # Show up to 5 examples
                story_a = self.results.at[idx, 'story_a']
                story_b = self.results.at[idx, 'story_b']
                self.logger.info(f"Row {idx} (non-null):")
                self.logger.info(f"  story_a: {story_a[:100]}...")
                self.logger.info(f"  story_b: {story_b[:100]}...")
        else:
            self.logger.warning("No rows with non-null stories found")
    
    def get_length_diff(self, row):
        """
        Calculate the length difference between two stories.
        
        Args:
            row: DataFrame row containing story_a and story_b
            
        Returns:
            Absolute length difference or None if invalid
        """
        try:
            # Check if both stories exist
            if pd.isna(row['story_a']) or pd.isna(row['story_b']):
                self.logger.debug(f"Invalid story data - A: {not pd.isna(row['story_a'])}, B: {not pd.isna(row['story_b'])}")
                return None
            
            # Get the lengths
            story_a_length = len(row['story_a']) if isinstance(row['story_a'], str) else 0
            story_b_length = len(row['story_b']) if isinstance(row['story_b'], str) else 0
            
            # Log the lengths for debugging
            self.logger.debug(f"Story A length: {story_a_length}, Story B length: {story_b_length}")
            
            # Calculate the difference
            if story_a_length == 0 or story_b_length == 0:
                self.logger.debug("One or both stories have zero length")
                return None
                
            # Return the absolute difference
            return abs(story_a_length - story_b_length)
        except Exception as e:
            self.logger.error(f"Error calculating length difference: {e}")
            return None 