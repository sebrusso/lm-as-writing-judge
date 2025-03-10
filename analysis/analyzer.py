"""
Main analyzer module that orchestrates the analysis process.
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

import pandas as pd

from analysis.data_loader import DataLoader
from analysis.metrics.accuracy import AccuracyAnalyzer
from analysis.metrics.upvote_analysis import UpvoteAnalyzer
from analysis.metrics.length_analysis import LengthAnalyzer
from analysis.metrics.agreement import AgreementAnalyzer
from analysis.visualization import Visualizer
from analysis.utils import setup_logging, find_results_directory, fleiss_kappa

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Analyzer:
    """
    Main analyzer class that orchestrates the analysis process.
    """
    
    def __init__(self, results_dir: Optional[Union[str, Path]] = None, analysis_dir: Union[str, Path] = 'analysis'):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing results files. If None, will look in standard locations
            analysis_dir: Directory to save analysis outputs
        """
        # Find results directory if not specified
        if results_dir is None:
            results_dir = find_results_directory()
            if results_dir is None:
                raise ValueError(
                    "Could not find results directory. Please specify the path to the directory "
                    "containing evaluation_results_*.json files"
                )
        
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Create dedicated directories for logs and results
        self.logs_dir = self.analysis_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        self.results_output_dir = self.analysis_dir / 'results'
        self.results_output_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.analysis_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Setup logging to use the logs directory
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = None
        self.data = None
        self.accuracy_analyzer = None
        self.upvote_analyzer = None
        self.length_analyzer = None
        self.agreement_analyzer = None
        self.visualizer = None
    
    def _setup_logging(self, logger_name: str = 'ResultsAnalyzer') -> logging.Logger:
        """
        Setup logging configuration with logs going to the logs directory.
        
        Args:
            logger_name: Name of the logger
            
        Returns:
            Configured logger instance
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f'analysis_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(logger_name)
    
    def load_data(self):
        """
        Load and preprocess data.
        
        Returns:
            DataFrame containing the loaded data
        """
        self.logger.info("Loading data...")
        self.data_loader = DataLoader(self.results_dir, self.logger)
        self.data = self.data_loader.load_results()
        
        if self.data.empty:
            raise ValueError(
                f"No results found in {self.results_dir}. "
                "Please ensure there are evaluation_results_*.json files in this directory."
            )
        
        # Verify required columns exist
        required_columns = ['model_name', 'choice', 'original_chosen', 'upvotes_chosen', 'upvotes_rejected']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Results files missing required columns: {missing_columns}")
        
        self.logger.info(f"Successfully loaded {len(self.data)} evaluations from {self.results_dir}")
        
        # Initialize analyzers
        self.accuracy_analyzer = AccuracyAnalyzer(self.data, self.logger)
        self.upvote_analyzer = UpvoteAnalyzer(self.data, self.logger)
        self.length_analyzer = LengthAnalyzer(self.data, self.data_loader.get_length_diff, self.logger)
        self.agreement_analyzer = AgreementAnalyzer(self.data, fleiss_kappa, self.logger)
        self.visualizer = Visualizer(self.data, self.logger)
        
        return self.data
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis and return results.
        
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Starting analysis...")
        
        try:
            # Load data if not already loaded
            if self.data is None:
                self.load_data()
            
            # Run analyses
            model_accuracy = self.accuracy_analyzer.calculate_model_accuracy()
            upvote_analysis = self.upvote_analyzer.analyze_upvote_impact()
            
            # Use equal frequency binning for length analysis
            length_analysis = self.length_analyzer.analyze_length_impact(
                binning_strategy='equal_freq',  # Use equal frequency binning
                n_bins=10,                      # Use 10 bins
                filter_outliers=True,           # Filter outliers
                outlier_threshold=3.0           # Z-score threshold for outliers
            )
            
            # Also run with equal width binning for comparison
            length_analysis_equal_width = self.length_analyzer.analyze_length_impact(
                binning_strategy='equal_width',
                n_bins=10,
                filter_outliers=True,
                outlier_threshold=3.0
            )
            
            # Combine both analyses
            length_analysis['equal_width_analysis'] = length_analysis_equal_width
            
            agreement_analysis = self.agreement_analyzer.analyze_model_agreement()
            
            # Compile results
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_evaluations': len(self.data),
                'model_accuracy': model_accuracy.to_dict(orient='records'),
                'upvote_analysis': upvote_analysis,
                'length_analysis': length_analysis,
                'model_agreement': agreement_analysis
            }
            
            # Generate plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_plots_dir = self.plots_dir / timestamp
            self.visualizer.generate_plots(
                run_plots_dir, 
                upvote_analysis, 
                length_analysis, 
                model_accuracy, 
                agreement_analysis
            )
            
            # Save results to the results directory
            results_file = self.results_output_dir / f'analysis_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
            self.logger.info(f"Analysis results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise

def main():
    """Main entry point."""
    try:
        # Allow specifying results directory as command line argument
        results_dir = sys.argv[1] if len(sys.argv) > 1 else None
        
        analyzer = Analyzer(results_dir=results_dir)
        results = analyzer.run_analysis()
        
        print("\nAnalysis Summary:")
        print(json.dumps(results, indent=2, cls=NumpyEncoder))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 