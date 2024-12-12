import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

def run_qc_analysis(data, groups):
    """Run QC analysis for each group separately"""
    try:
        # Set seaborn style
        sns.set_style('whitegrid')
        
        logger.info(f"Running QC analysis on data with shape: {data.shape}")
        
        qc_results = {}
        
        # Analyze each group separately
        for group_name, group_samples in groups.items():
            if not group_samples:
                continue
                
            group_data = data[group_samples]
            logger.info(f"Analyzing {group_name} with {len(group_samples)} samples")
            
            # Calculate QC metrics
            missing_values = group_data.isnull().sum()
            mean_values = group_data.mean()
            std_values = group_data.std()
            
            # Calculate quality scores
            quality_scores = 1 - (missing_values / len(group_data))
            
            # Group metrics
            metrics = {
                'missing_values': int(group_data.isnull().sum().sum()),
                'total_samples': len(group_samples),
                'total_features': len(group_data.index),
                'value_range': {
                    'min': float(group_data.min().min()),
                    'max': float(group_data.max().max())
                },
                'sample_names': group_samples,
                'mean_values_per_sample': [float(x) for x in mean_values.values],
                'missing_values_per_sample': [int(x) for x in missing_values.values],
                'quality_scores': [float(x) for x in quality_scores.values]
            }
            
            # Generate plots for this group
            plots = {}
            
            try:
                # 1. Box plot
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=group_data)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of Expression Values - {group_name}')
                plt.tight_layout()
                plots['boxplot'] = _convert_plot_to_base64()
                plt.close()

                # 2. Density plot
                plt.figure(figsize=(12, 6))
                for column in group_data.columns:
                    sns.kdeplot(data=group_data[column].dropna(), label=column)
                plt.title(f'Density Plot - {group_name}')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plots['density'] = _convert_plot_to_base64()
                plt.close()

                # 3. Missing values heatmap
                plt.figure(figsize=(12, 6))
                sns.heatmap(group_data.isnull(), yticklabels=False, cbar=False)
                plt.title(f'Missing Values Heatmap - {group_name}')
                plt.tight_layout()
                plots['missing_values'] = _convert_plot_to_base64()
                plt.close()

                # 4. Sample correlation heatmap
                plt.figure(figsize=(12, 8))
                correlation_matrix = group_data.corr()
                sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
                plt.title(f'Sample Correlation Heatmap - {group_name}')
                plt.tight_layout()
                plots['correlation'] = _convert_plot_to_base64()
                plt.close()

            except Exception as e:
                logger.error(f"Error generating plots for {group_name}: {str(e)}")
                continue

            qc_results[group_name] = {
                'metrics': metrics,
                'plots': plots
            }

        logger.info("QC analysis completed successfully")
        return qc_results

    except Exception as e:
        logger.error(f"Error in QC analysis: {str(e)}")
        return None

def _convert_plot_to_base64():
    """Convert matplotlib plot to base64 string"""
    try:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting plot to base64: {str(e)}")
        return None