import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class ExpressionNormalizer:
    def __init__(self):
        self.normalized_data = None
        self.normalization_stats = {}
        self.normalization_plots = {}

    def normalize_data(self, expression_data, method='quantile'):
        """Normalize expression data using specified method"""
        try:
            logger.info(f"Normalizing data using {method} method...")
            
            if method == 'quantile':
                self.normalized_data = self._quantile_normalization(expression_data)
            elif method == 'log2':
                self.normalized_data = self._log2_normalization(expression_data)
            elif method == 'zscore':
                self.normalized_data = self._zscore_normalization(expression_data)
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            # Calculate normalization statistics
            self._calculate_normalization_stats(expression_data)
            
            # Generate normalization plots
            self._generate_normalization_plots(expression_data)
            
            logger.info("Normalization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            raise

    def _quantile_normalization(self, data):
        """Perform quantile normalization"""
        try:
            # Rank the values in each column
            rank_mean = data.stack().groupby(data.rank(method='first').stack().astype(int)).mean()
            return data.rank(method='min').stack().astype(int).map(rank_mean).unstack()
            
        except Exception as e:
            logger.error(f"Error in quantile normalization: {str(e)}")
            raise

    def _log2_normalization(self, data):
        """Perform log2 normalization"""
        try:
            # Add small constant to avoid log(0)
            return np.log2(data + 1)
            
        except Exception as e:
            logger.error(f"Error in log2 normalization: {str(e)}")
            raise

    def _zscore_normalization(self, data):
        """Perform Z-score normalization"""
        try:
            return pd.DataFrame(
                stats.zscore(data, axis=0),
                index=data.index,
                columns=data.columns
            )
            
        except Exception as e:
            logger.error(f"Error in Z-score normalization: {str(e)}")
            raise

    def _calculate_normalization_stats(self, original_data):
        """Calculate normalization statistics"""
        try:
            self.normalization_stats = {
                'original': {
                    'mean': float(original_data.mean().mean()),
                    'std': float(original_data.std().mean()),
                    'median': float(original_data.median().mean())
                },
                'normalized': {
                    'mean': float(self.normalized_data.mean().mean()),
                    'std': float(self.normalized_data.std().mean()),
                    'median': float(self.normalized_data.median().mean())
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating normalization stats: {str(e)}")
            raise

    def _generate_normalization_plots(self, original_data):
        """Generate normalization comparison plots"""
        try:
            # Distribution plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.boxplot(data=original_data)
            plt.title('Original Data Distribution')
            plt.xticks(rotation=90)
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=self.normalized_data)
            plt.title('Normalized Data Distribution')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            self.normalization_plots['distribution'] = self._get_plot_base64()
            plt.close()

            # Correlation heatmap
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.heatmap(original_data.corr(), cmap='coolwarm', center=0)
            plt.title('Original Data Correlation')
            
            plt.subplot(1, 2, 2)
            sns.heatmap(self.normalized_data.corr(), cmap='coolwarm', center=0)
            plt.title('Normalized Data Correlation')
            
            plt.tight_layout()
            self.normalization_plots['correlation'] = self._get_plot_base64()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating normalization plots: {str(e)}")
            raise

    def _get_plot_base64(self):
        """Convert current plot to base64 string"""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def get_serializable_stats(self):
        """Get JSON-serializable normalization statistics"""
        return self.normalization_stats 