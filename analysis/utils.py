"""
Utility functions for the analysis module.
"""
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List

def setup_logging(analysis_dir: Union[str, Path], logger_name: str = 'ResultsAnalyzer') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        analysis_dir: Directory to save log files
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = analysis_dir / f'analysis_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(logger_name)

def find_results_directory() -> Optional[Path]:
    """
    Find the results directory by checking common locations.
    
    Returns:
        Path to the results directory if found, None otherwise
    """
    # Check common locations
    possible_locations = [
        Path('results'),
        Path('judge-service/results'),
        Path('../judge-service/results'),
        Path(os.path.dirname(os.path.dirname(__file__))) / 'judge-service' / 'results'
    ]
    
    for loc in possible_locations:
        if loc.exists() and loc.is_dir():
            return loc
    
    return None

def fleiss_kappa(ratings):
    """
    Calculate Fleiss' kappa for reliability of agreement.
    
    Args:
        ratings: Matrix of ratings (n_subjects x n_categories)
        
    Returns:
        Fleiss' kappa value
    """
    n_sub, n_cat = ratings.shape
    n_raters = float(ratings.sum(1)[0])
    n_rat = n_raters * n_sub
    
    # Calculate P(e)
    p_cat = ratings.sum(0) / n_rat
    Pe = (p_cat * p_cat).sum()
    
    # Calculate P(a)
    Pa = (ratings * ratings).sum(1)
    Pa = (Pa - n_raters) / (n_raters * (n_raters - 1))
    Pa = Pa.mean()
    
    # Calculate kappa
    kappa = (Pa - Pe) / (1 - Pe)
    
    return float(kappa) 