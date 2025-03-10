import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import logging
from collections import defaultdict
import re
import glob
from scipy.stats import pointbiserialr

class ResultsAnalyzer:
    """
    Analyzes results from language model judge experiments.
    Provides comprehensive analysis of model performance, agreement, and various correlations.
    """
    
    def __init__(self, results_dir: str = None, analysis_dir: str = 'analysis'):
        """
        Initialize the analyzer with paths to results and analysis directories
        
        Args:
            results_dir: Directory containing results files. If None, will look in standard locations
            analysis_dir: Directory to save analysis outputs
        """
        # Try to find results directory if not specified
        if results_dir is None:
            # Check common locations
            possible_locations = [
                Path('results'),
                Path('judge-service/results'),
                Path('../judge-service/results'),
                Path(os.path.dirname(os.path.dirname(__file__))) / 'judge-service' / 'results'
            ]
            
            for loc in possible_locations:
                if loc.exists() and loc.is_dir():
                    results_dir = loc
                    break
            
            if results_dir is None:
                raise ValueError(
                    "Could not find results directory. Please specify the path to the directory "
                    "containing evaluation_results_*.json files"
                )
        
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load results
        self.results = []
        self.load_results()
        
        if not self.results:
            raise ValueError(
                f"No results found in {self.results_dir}. "
                "Please ensure there are evaluation_results_*.json files in this directory."
            )
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.results)
        
        # Verify required columns exist
        required_columns = ['model_name', 'choice', 'original_chosen', 'upvotes_chosen', 'upvotes_rejected']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Results files missing required columns: {missing_columns}")
        
        self.logger.info(f"Successfully loaded {len(self.df)} evaluations from {self.results_dir}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.analysis_dir / f'analysis_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ResultsAnalyzer')
        
    def get_length_diff(self, row):
        """Calculate the length difference between two stories."""
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
        
    def load_results(self, results_dir=None):
        """Load results from the specified directory."""
        if results_dir is None:
            results_dir = self.results_dir
        
        self.logger.info(f"Loading results from {results_dir}")
        
        # Find all JSON and JSONL files in the results directory
        json_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.json"))
        jsonl_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.jsonl"))
        
        all_files = json_files + jsonl_files
        if not all_files:
            self.logger.error(f"No result files found in {results_dir}")
            return
        
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
        
        return self.results
    
    def calculate_model_accuracy(self) -> pd.DataFrame:
        """Calculate the accuracy of each model."""
        model_accuracy = []
        
        for model in self.results['model_name'].unique():
            model_df = self.results[self.results['model_name'] == model]
            total_evals = len(model_df)
            errors = model_df['error'].notna().sum()
            successful_evals = total_evals - errors
            
            if successful_evals > 0:
                # Filter out rows with errors
                valid_df = model_df[model_df['error'].isna()]
                
                # Calculate accuracy
                accuracy = (valid_df['choice'] == valid_df['original_chosen']).mean()
                
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
            else:
                self.logger.warning(f"No successful evaluations for model {model}")
        
        return pd.DataFrame(model_accuracy)
    
    def analyze_upvote_impact(self):
        """Analyze the impact of upvote differences on model accuracy."""
        try:
            # Calculate upvote differences
            self.results['upvote_diff'] = self.results['upvotes_chosen'].astype(float) - self.results['upvotes_rejected'].astype(float)
            
            # Create bins for upvote differences
            upvote_diffs = self.results['upvote_diff'].dropna()
            
            if len(upvote_diffs) == 0:
                self.logger.error("No valid upvote differences found")
                return {"error": "No valid upvote differences found"}
            
            # Create deciles
            try:
                self.results['upvote_diff_decile'] = pd.qcut(upvote_diffs, 10, duplicates='drop')
                decile_ranges = [str(category) for category in self.results['upvote_diff_decile'].cat.categories]
            except Exception as e:
                self.logger.error(f"Error creating upvote difference deciles: {e}")
                return {"error": f"Error creating upvote difference deciles: {e}"}
            
            # Calculate accuracy by decile for each model
            accuracy_by_decile = {}
            
            for model in self.results['model_name'].unique():
                model_df = self.results[self.results['model_name'] == model].copy()
                
                # Filter out rows with errors
                model_df = model_df[model_df['error'].isna()]
                
                # Calculate accuracy by decile
                accuracy_by_decile[model] = []
                
                for decile in self.results['upvote_diff_decile'].cat.categories:
                    decile_rows = model_df[model_df['upvote_diff_decile'] == decile]
                    if len(decile_rows) > 0:
                        accuracy = (decile_rows['choice'] == decile_rows['original_chosen']).mean()
                        accuracy_by_decile[model].append(accuracy)
                    else:
                        accuracy_by_decile[model].append(None)
            
            # Calculate correlation between upvote difference and accuracy
            # Filter out rows with errors
            valid_rows = self.results[self.results['error'].isna()]
            
            if len(valid_rows) > 0:
                # Calculate correlation
                correct = (valid_rows['choice'] == valid_rows['original_chosen']).astype(int)
                correlation, pvalue = pointbiserialr(valid_rows['upvote_diff'], correct)
                
                correlation_results = {
                    "correlation": correlation,
                    "pvalue": pvalue
                }
                
                self.logger.info(f"Upvote difference correlation: {correlation:.4f} (p={pvalue:.6f})")
            else:
                self.logger.warning("No valid data for upvote correlation calculation")
                correlation_results = {"error": "No valid data"}
            
            # Create and save plot
            try:
                self._create_upvote_impact_plot(accuracy_by_decile, decile_ranges)
            except Exception as e:
                self.logger.warning(f"Cannot generate upvote analysis plot: {e}")
            
            return {
                "accuracy_by_decile": accuracy_by_decile,
                "decile_ranges": decile_ranges,
                "upvote_diff_correlation": correlation_results
            }
        
        except Exception as e:
            self.logger.error(f"Error in analyze_upvote_impact: {e}")
            return {"error": str(e)}
    
    def analyze_length_impact(self):
        """Analyze the impact of length differences on model accuracy."""
        try:
            # Calculate length differences
            self.results['length_diff'] = self.results.apply(self.get_length_diff, axis=1)
            
            # Count valid length differences
            valid_diffs = self.results['length_diff'].notna().sum()
            self.logger.info(f"Valid length differences: {valid_diffs} out of {len(self.results)}")
            
            # If no valid length differences, return error
            if valid_diffs == 0:
                self.logger.error("No valid length differences found")
                return {"error": "No valid length differences found", "valid_diffs": 0, "total_rows": len(self.results)}
            
            # Calculate min, max, mean of length differences
            length_diff_stats = self.results['length_diff'].describe()
            self.logger.info(f"Length diff stats: min={length_diff_stats['min']}, max={length_diff_stats['max']}, mean={length_diff_stats['mean']:.2f}")
            
            # If too few valid length differences, return error
            if valid_diffs < 10:
                self.logger.error(f"Too few valid length differences ({valid_diffs}) to perform analysis")
                return {"error": f"Too few valid length differences ({valid_diffs}) to perform analysis", 
                        "valid_diffs": valid_diffs, "total_rows": len(self.results)}
            
            # Determine number of bins (use fewer bins if we have fewer samples)
            n_bins = min(10, max(2, valid_diffs // 2))  # At least 2 bins, at most 10, aim for at least 2 samples per bin
            
            # Create bins for length differences
            length_diffs = self.results['length_diff'].dropna()
            
            # Check if all length differences are the same
            if length_diffs.nunique() <= 1:
                self.logger.error("All length differences are the same, cannot create bins")
                return {"error": "All length differences are the same, cannot create bins", 
                        "valid_diffs": valid_diffs, "total_rows": len(self.results)}
            
            # Create bins
            try:
                bins = pd.cut(length_diffs, bins=n_bins)
                self.results['length_diff_decile'] = pd.cut(self.results['length_diff'], bins=bins.categories)
                
                # Log unique deciles for debugging
                unique_deciles = self.results['length_diff_decile'].unique()
                self.logger.info(f"Unique deciles: {[str(d) for d in unique_deciles if not pd.isna(d)]}")
            except Exception as e:
                self.logger.error(f"Error creating bins: {e}")
                return {"error": f"Error creating bins: {e}", 
                        "valid_diffs": valid_diffs, "total_rows": len(self.results)}
            
            # Calculate accuracy by decile for each model
            accuracy_by_decile = {}
            decile_ranges = [str(category) for category in bins.categories]
            
            for model in self.results['model_name'].unique():
                model_df = self.results[self.results['model_name'] == model].copy()
                
                # Filter out rows with errors
                model_df = model_df[model_df['error'].isna()]
                
                # Calculate accuracy by decile
                accuracy_by_decile[model] = []
                
                for decile in bins.categories:
                    decile_rows = model_df[model_df['length_diff_decile'] == decile]
                    if len(decile_rows) > 0:
                        accuracy = (decile_rows['choice'] == decile_rows['original_chosen']).mean()
                        accuracy_by_decile[model].append(accuracy)
                    else:
                        accuracy_by_decile[model].append(None)
            
            # Calculate correlation between length difference and accuracy
            correlation_results = {}
            
            # Filter out rows with errors or missing length differences
            valid_rows = self.results[self.results['error'].isna() & self.results['length_diff'].notna()]
            
            if len(valid_rows) > 0:
                # Calculate range of length differences
                min_diff = valid_rows['length_diff'].min()
                max_diff = valid_rows['length_diff'].max()
                self.logger.info(f"Length difference range: {min_diff} to {max_diff}")
                
                # Calculate accuracy distribution
                accuracy = (valid_rows['choice'] == valid_rows['original_chosen']).mean()
                self.logger.info(f"Overall accuracy: {accuracy:.4f}")
                
                # Check if all length differences are the same
                if valid_rows['length_diff'].nunique() <= 1:
                    self.logger.warning("All length differences are the same, cannot calculate correlation")
                    correlation_results = {"error": "All length differences are the same"}
                else:
                    # Calculate point-biserial correlation
                    correct = (valid_rows['choice'] == valid_rows['original_chosen']).astype(int)
                    correlation, pvalue = pointbiserialr(valid_rows['length_diff'], correct)
                    
                    correlation_results = {
                        "correlation": correlation,
                        "pvalue": pvalue
                    }
                    
                    self.logger.info(f"Length difference correlation: {correlation:.4f} (p={pvalue:.6f})")
            else:
                self.logger.warning("No valid data for correlation calculation")
                correlation_results = {"error": "No valid data"}
            
            # Create and save plot
            try:
                self._create_length_impact_plot(accuracy_by_decile, decile_ranges)
            except Exception as e:
                self.logger.warning(f"Cannot generate length analysis plot: {e}")
            
            return {
                "accuracy_by_decile": accuracy_by_decile,
                "decile_ranges": decile_ranges,
                "length_diff_correlation": correlation_results
            }
        
        except Exception as e:
            self.logger.error(f"Error in analyze_length_impact: {e}")
            return {"error": str(e), "valid_diffs": 0, "total_rows": len(self.results)}
    
    def analyze_model_agreement(self) -> Dict[str, Any]:
        """Analyze agreement between different models"""
        results = {}
        
        # Create agreement matrix
        models = self.results['model_name'].unique()
        agreement_matrix = pd.DataFrame(index=models, columns=models)
        
        for i, model1 in enumerate(models):
            for model2 in models:
                if model1 == model2:
                    continue
                    
                # Get overlapping evaluations
                model1_data = self.results[self.results['model_name'] == model1]
                model2_data = self.results[self.results['model_name'] == model2]
                
                # Filter out errors if error column exists
                if 'error' in self.results.columns:
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
        if 'error' in self.results.columns:
            decisions_data = self.results[~self.results['error'].notna()]
        else:
            decisions_data = self.results
            
        pair_decisions = decisions_data.groupby(['pair_id', 'permutation_id'])['choice'].apply(list)
        
        if len(pair_decisions) > 0:
            # Convert to matrix format required for Fleiss' kappa
            decision_matrix = np.zeros((len(pair_decisions), 2))  # 2 categories: A and B
            for i, decisions in enumerate(pair_decisions):
                decision_matrix[i, 0] = decisions.count('A')
                decision_matrix[i, 1] = decisions.count('B')
            
            results['fleiss_kappa'] = self.fleiss_kappa(decision_matrix)
        
        return results
    
    @staticmethod
    def fleiss_kappa(ratings):
        """Calculate Fleiss' kappa for reliability of agreement"""
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
    
    def plot_model_accuracy_by_upvotes(self, results_dir: Path):
        """Generate line plot showing model accuracy across absolute upvote difference deciles"""
        plt.figure(figsize=(15, 8))
        
        upvote_analysis = self.analyze_upvote_impact()
        accuracy_by_decile = upvote_analysis['accuracy_by_decile']
        decile_ranges = upvote_analysis['decile_ranges']
        
        # Plot line for each model
        for model, accuracies in accuracy_by_decile.items():
            plt.plot(range(len(accuracies)), accuracies, marker='o', label=model, linewidth=2, markersize=8)
        
        plt.xlabel('Absolute Upvote Difference Deciles\n(Smallest → Largest)', fontsize=12)
        plt.ylabel('Model Accuracy', fontsize=12)
        plt.title('Model Accuracy by Absolute Upvote Difference', fontsize=14, pad=20)
        
        # Add decile ranges as x-tick labels
        plt.xticks(range(len(decile_ranges)), [f'D{i+1}\n|Δ|={r}' for i, r in enumerate(decile_ranges)], 
                  rotation=45, ha='right')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        # Save plot
        plots_dir = results_dir / 'model_accuracy_by_upvotes.png'
        plt.savefig(plots_dir, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model accuracy by upvotes plot saved to: {plots_dir}")

    def plot_model_accuracy_by_length(self, results_dir: Path):
        """Generate line plot showing model accuracy across absolute length difference deciles"""
        plt.figure(figsize=(15, 8))
        
        length_analysis = self.analyze_length_impact()
        if 'error' in length_analysis:
            self.logger.warning("Cannot generate length analysis plot: story text not found")
            return
            
        accuracy_by_decile = length_analysis['accuracy_by_decile']
        decile_ranges = length_analysis['decile_ranges']
        
        # Plot line for each model
        for model, accuracies in accuracy_by_decile.items():
            plt.plot(range(len(accuracies)), accuracies, marker='o', label=model, linewidth=2, markersize=8)
        
        plt.xlabel('Absolute Length Difference Deciles\n(Smallest → Largest)', fontsize=12)
        plt.ylabel('Model Accuracy', fontsize=12)
        plt.title('Model Accuracy by Absolute Length Difference', fontsize=14, pad=20)
        
        # Add decile ranges as x-tick labels
        plt.xticks(range(len(decile_ranges)), [f'D{i+1}\n|Δ|={r}' for i, r in enumerate(decile_ranges)], 
                  rotation=45, ha='right')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        # Save plot
        plots_dir = results_dir / 'model_accuracy_by_length.png'
        plt.savefig(plots_dir, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model accuracy by length plot saved to: {plots_dir}")

    def generate_plots(self):
        """Generate visualization plots for the analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = self.analysis_dir / 'plots' / timestamp
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        
        # Generate upvote analysis plot
        self.plot_model_accuracy_by_upvotes(plots_dir)
        
        # Generate length analysis plot
        self.plot_model_accuracy_by_length(plots_dir)
        
        # Model accuracy plot
        accuracies = self.calculate_model_accuracy()
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=accuracies, x='model', y='accuracy')
        plt.title('Model Accuracy Comparison', pad=20)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(accuracies['accuracy']):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(plots_dir / 'model_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model agreement heatmap
        agreement_data = self.analyze_model_agreement()
        if 'agreement_matrix' in agreement_data:
            plt.figure(figsize=(10, 8))
            agreement_df = pd.DataFrame(agreement_data['agreement_matrix'])
            mask = np.triu(np.ones_like(agreement_df, dtype=bool))
            
            sns.heatmap(
                agreement_df,
                annot=True,
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                mask=mask,
                fmt='.3f'
            )
            plt.title('Model Agreement Heatmap')
            plt.tight_layout()
            plt.savefig(plots_dir / 'model_agreement.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Plots saved to: {plots_dir}")
        
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete analysis and return results"""
        self.logger.info("Starting analysis...")
        
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_evaluations': len(self.df),
                'model_accuracy': self.calculate_model_accuracy().to_dict(orient='records'),
                'upvote_analysis': self.analyze_upvote_impact(),
                'length_analysis': self.analyze_length_impact(),
                'model_agreement': self.analyze_model_agreement()
            }
            
            # Generate plots
            self.generate_plots()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.analysis_dir / f'analysis_results_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Analysis results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise

    @property
    def models(self):
        """Get the unique models in the results."""
        if hasattr(self, 'results') and isinstance(self.results, pd.DataFrame) and 'model_name' in self.results.columns:
            return self.results['model_name'].unique()
        return []

def main():
    """Main entry point"""
    try:
        # Allow specifying results directory as command line argument
        results_dir = sys.argv[1] if len(sys.argv) > 1 else None
        
        analyzer = ResultsAnalyzer(results_dir=results_dir)
        results = analyzer.run_analysis()
        
        print("\nAnalysis Summary:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 