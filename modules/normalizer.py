import numpy as np
import pandas as pd
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
        self.plots = {}
        self.supported_methods = ['quantile', 'zscore', 'log2']

    def normalize(self, data, method='quantile'):
        """Normalize expression data using specified method"""
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
            
            if data.empty:
                raise ValueError("Input data is empty")
            
            if method not in self.supported_methods:
                raise ValueError(f"Unsupported normalization method: {method}. Supported methods: {', '.join(self.supported_methods)}")
            
            # Check for non-numeric values
            if not data.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all().all():
                raise ValueError("Data contains non-numeric values")

            logger.info(f"Starting {method} normalization on data of shape {data.shape}")
            logger.info(f"Input data columns: {data.columns.tolist()}")
            logger.info(f"Input data index size: {len(data.index)}")
            logger.info(f"Input data stats - Mean: {data.mean().mean():.2f}, Std: {data.std().mean():.2f}")
            
            # Perform normalization based on method
            if method == 'quantile':
                normalized = self._quantile_normalization(data)
            elif method == 'zscore':
                normalized = self._zscore_normalization(data)
            elif method == 'log2':
                normalized = self._log2_transform(data)
            
            self.normalized_data = normalized
            
            logger.info(f"Normalization completed. Output shape: {normalized.shape}")
            logger.info(f"Output data columns: {normalized.columns.tolist()}")
            logger.info(f"Output data stats - Mean: {normalized.mean().mean():.2f}, Std: {normalized.std().mean():.2f}")
            
            # Generate plots
            self.generate_plots(data, normalized)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            raise

    def _quantile_normalization(self, data):
        """Perform quantile normalization"""
        try:
            logger.info("Starting quantile normalization")
            
            # Convert to numpy array for faster computation
            data_array = data.values
            
            # Calculate ranks for each column
            rank_mean = np.zeros_like(data_array)
            for i in range(data_array.shape[1]):
                rank_mean[:, i] = stats.rankdata(data_array[:, i], method='average')
            
            # Calculate means of ranks across rows
            sorted_data = np.sort(data_array, axis=0)
            mean_sorted = np.mean(sorted_data, axis=1)
            
            # Create normalized data
            normalized_data = np.zeros_like(data_array)
            for i in range(data_array.shape[1]):
                normalized_data[:, i] = np.interp(
                    rank_mean[:, i],
                    np.arange(1, len(mean_sorted) + 1),
                    mean_sorted
                )
            
            result = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)
            logger.info(f"Quantile normalization completed. Shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in quantile normalization: {str(e)}")
            raise

    def _zscore_normalization(self, data):
        """Perform Z-score normalization"""
        try:
            logger.info("Performing Z-score normalization")
            return (data - data.mean()) / data.std()
        except Exception as e:
            logger.error(f"Error in Z-score normalization: {str(e)}")
            raise

    def _log2_transform(self, data):
        """Perform log2 transformation"""
        try:
            logger.info("Performing log2 transformation")
            # Check for negative values
            if (data < 0).any().any():
                raise ValueError("Cannot perform log2 transformation on negative values")
            
            # Add small constant to avoid log(0)
            min_nonzero = data[data > 0].min().min()
            offset = min_nonzero / 10 if min_nonzero > 0 else 1
            return np.log2(data + offset)
        except Exception as e:
            logger.error(f"Error in log2 transformation: {str(e)}")
            raise

    def generate_plots(self, original_data, normalized_data):
        """Generate comparison plots before and after normalization"""
        try:
            plots = {}

            # 1. Box plot (standalone)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.boxplot(data=original_data)
            plt.title('Expression Distribution - Before')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Expression Level')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=normalized_data)
            plt.title('Expression Distribution - After')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Expression Level')
            
            plt.tight_layout()
            plots['boxplot'] = self._convert_plot_to_base64()

            # 2. Density plots comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            for col in original_data.columns:
                sns.kdeplot(data=original_data[col], label=col)
            plt.title('Density Distribution - Before')
            plt.xlabel('Expression Level')
            plt.ylabel('Density')
            
            plt.subplot(1, 2, 2)
            for col in normalized_data.columns:
                sns.kdeplot(data=normalized_data[col], label=col)
            plt.title('Density Distribution - After')
            plt.xlabel('Expression Level')
            plt.ylabel('Density')
            
            plt.tight_layout()
            plots['density_comparison'] = self._convert_plot_to_base64()

            # 3. QQ plots
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            stats.probplot(original_data.values.flatten(), dist="norm", plot=plt)
            plt.title('Q-Q Plot - Before')
            
            plt.subplot(1, 2, 2)
            stats.probplot(normalized_data.values.flatten(), dist="norm", plot=plt)
            plt.title('Q-Q Plot - After')
            
            plt.tight_layout()
            plots['qq_comparison'] = self._convert_plot_to_base64()

            # Store plots
            self.plots = plots
            logger.info("Generated all comparison plots successfully")
            return plots

        except Exception as e:
            logger.error(f"Error generating normalization plots: {str(e)}")
            raise

    def _convert_plot_to_base64(self):
        """Convert matplotlib plot to base64 string"""
        try:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            return base64.b64encode(image_png).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting plot to base64: {str(e)}")
            raise