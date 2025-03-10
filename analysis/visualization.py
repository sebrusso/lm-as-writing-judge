"""
Visualization and plotting functions for analysis results.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

class Visualizer:
    """
    Handles visualization and plotting of analysis results.
    """
    
    def __init__(self, data: pd.DataFrame, logger: Optional[logging.Logger] = None):
        """
        Initialize the visualizer.
        
        Args:
            data: DataFrame containing evaluation results
            logger: Logger instance
        """
        self.data = data
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_plots(self, plots_dir: Path, upvote_analysis: Dict[str, Any], 
                      length_analysis: Dict[str, Any], model_accuracy: pd.DataFrame,
                      agreement_data: Dict[str, Any]):
        """
        Generate visualization plots for the analysis.
        
        Args:
            plots_dir: Directory to save plots
            upvote_analysis: Results from upvote impact analysis
            length_analysis: Results from length impact analysis
            model_accuracy: DataFrame with model accuracy metrics
            agreement_data: Results from model agreement analysis
        """
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        
        # Generate upvote analysis plot
        self._create_upvote_impact_plot(plots_dir, upvote_analysis)
        
        # Generate length analysis plot
        self._create_length_impact_plot(plots_dir, length_analysis)
        
        # Model accuracy plot
        self._create_model_accuracy_plot(plots_dir, model_accuracy)
        
        # Model agreement heatmap
        self._create_agreement_heatmap(plots_dir, agreement_data)
        
        self.logger.info(f"Plots saved to: {plots_dir}")
    
    def _create_upvote_impact_plot(self, plots_dir: Path, upvote_analysis: Dict[str, Any]):
        """
        Generate line plot showing model accuracy across absolute upvote difference deciles.
        
        Args:
            plots_dir: Directory to save plots
            upvote_analysis: Results from upvote impact analysis
        """
        if 'error' in upvote_analysis:
            self.logger.warning(f"Cannot generate upvote analysis plot: {upvote_analysis['error']}")
            return
            
        plt.figure(figsize=(15, 8))
        
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
        plot_path = plots_dir / 'model_accuracy_by_upvotes.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model accuracy by upvotes plot saved to: {plot_path}")
    
    def _create_length_impact_plot(self, plots_dir: Path, length_analysis: Dict[str, Any]):
        """
        Generate line plot showing model accuracy across length difference bins.
        
        Args:
            plots_dir: Directory to save plots
            length_analysis: Results from length impact analysis
        """
        if 'error' in length_analysis:
            self.logger.warning(f"Cannot generate length analysis plot: {length_analysis['error']}")
            return
        
        # Note: The LengthAnalyzer now creates its own plots, so we don't need to duplicate that here
        # However, we'll create a combined plot showing both accuracy and sample counts for comparison
        
        # Check if we're using the new format (with accuracy_by_bin) or old format (with accuracy_by_decile)
        if 'accuracy_by_bin' in length_analysis:
            accuracy_data = length_analysis['accuracy_by_bin']
            bin_categories = length_analysis['bin_categories']
            sample_counts = length_analysis.get('sample_counts_by_bin', {})
            binning_strategy = length_analysis.get('binning_strategy', 'unknown')
        else:
            # Fall back to old format for backward compatibility
            accuracy_data = length_analysis['accuracy_by_decile']
            bin_categories = length_analysis['decile_ranges']
            sample_counts = {}
            binning_strategy = 'equal_width'
            
        # Create a combined plot showing accuracy for both binning strategies if available
        if 'equal_width_analysis' in length_analysis and 'binning_strategy' in length_analysis:
            plt.figure(figsize=(15, 10))
            
            # Equal frequency binning
            plt.subplot(2, 1, 1)
            for model, accuracies in accuracy_data.items():
                plt.plot(range(len(accuracies)), accuracies, marker='o', label=model, linewidth=2, markersize=8)
            
            plt.xlabel('Absolute Length Difference Bins\n(Smallest → Largest)', fontsize=12)
            plt.ylabel('Model Accuracy', fontsize=12)
            plt.title(f'Model Accuracy by Absolute Length Difference\n(Equal Frequency Binning)', fontsize=14, pad=20)
            
            # Add bin categories as x-tick labels
            plt.xticks(range(len(bin_categories)), [f'B{i+1}\n|Δ|={r}' for i, r in enumerate(bin_categories)], 
                      rotation=45, ha='right')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            # Equal width binning
            plt.subplot(2, 1, 2)
            equal_width_data = length_analysis['equal_width_analysis']
            
            if 'accuracy_by_bin' in equal_width_data:
                ew_accuracy_data = equal_width_data['accuracy_by_bin']
                ew_bin_categories = equal_width_data['bin_categories']
                
                for model, accuracies in ew_accuracy_data.items():
                    plt.plot(range(len(accuracies)), accuracies, marker='o', label=model, linewidth=2, markersize=8)
                
                plt.xlabel('Absolute Length Difference Bins\n(Smallest → Largest)', fontsize=12)
                plt.ylabel('Model Accuracy', fontsize=12)
                plt.title('Model Accuracy by Absolute Length Difference\n(Equal Width Binning)', fontsize=14, pad=20)
                
                # Add bin categories as x-tick labels
                plt.xticks(range(len(ew_bin_categories)), [f'B{i+1}\n|Δ|={r}' for i, r in enumerate(ew_bin_categories)], 
                          rotation=45, ha='right')
                
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            plt.tight_layout()
            
            # Save comparison plot
            plot_path = plots_dir / 'model_accuracy_by_length_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Length binning comparison plot saved to: {plot_path}")
        
        # Create a simple plot for backward compatibility
        plt.figure(figsize=(15, 8))
        
        for model, accuracies in accuracy_data.items():
            plt.plot(range(len(accuracies)), accuracies, marker='o', label=model, linewidth=2, markersize=8)
        
        plt.xlabel('Absolute Length Difference Bins\n(Smallest → Largest)', fontsize=12)
        plt.ylabel('Model Accuracy', fontsize=12)
        plt.title(f'Model Accuracy by Absolute Length Difference', fontsize=14, pad=20)
        
        # Add bin categories as x-tick labels
        plt.xticks(range(len(bin_categories)), [f'B{i+1}\n|Δ|={r}' for i, r in enumerate(bin_categories)], 
                  rotation=45, ha='right')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        # Save plot
        plot_path = plots_dir / 'model_accuracy_by_length.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model accuracy by length plot saved to: {plot_path}")
    
    def _create_model_accuracy_plot(self, plots_dir: Path, model_accuracy: pd.DataFrame):
        """
        Generate bar plot showing model accuracy.
        
        Args:
            plots_dir: Directory to save plots
            model_accuracy: DataFrame with model accuracy metrics
        """
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=model_accuracy, x='model', y='accuracy')
        plt.title('Model Accuracy Comparison', pad=20)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(model_accuracy['accuracy']):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
            
        plt.tight_layout()
        
        # Save plot
        plot_path = plots_dir / 'model_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model accuracy plot saved to: {plot_path}")
    
    def _create_agreement_heatmap(self, plots_dir: Path, agreement_data: Dict[str, Any]):
        """
        Generate heatmap showing model agreement.
        
        Args:
            plots_dir: Directory to save plots
            agreement_data: Results from model agreement analysis
        """
        if 'agreement_matrix' not in agreement_data:
            self.logger.warning("Cannot generate agreement heatmap: agreement matrix not found")
            return
            
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
        
        # Save plot
        plot_path = plots_dir / 'model_agreement.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model agreement heatmap saved to: {plot_path}") 