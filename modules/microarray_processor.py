import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import quantile_transform, StandardScaler
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class MicroarrayProcessor:
    def __init__(self):
        self.is_microarray = False
        self.platform_type = None
        self.raw_data = None
        self.preprocessed_data = None
        self.normalized_data = None
        self.preprocessing_stats = {}
        self.normalization_stats = {}
        self.plots = {}

    def detect_data_type(self, expression_data, platform_info):
        """Detect if the data is from a microarray platform"""
        try:
            # Debug logging
            logger.info(f"Checking platform info: {platform_info}")
            
            # Check platform information for microarray keywords
            platform_keywords = ['affymetrix', 'illumina', 'agilent', 'probe', 'microarray']
            platform_text = ' '.join([str(v).lower() for v in platform_info.values()])
            
            logger.info(f"Platform text: {platform_text}")
            
            # Check if any microarray keywords are present
            self.is_microarray = any(keyword in platform_text for keyword in platform_keywords)
            
            if self.is_microarray:
                # Determine platform type
                if 'affymetrix' in platform_text:
                    self.platform_type = 'affymetrix'
                elif 'illumina' in platform_text:
                    self.platform_type = 'illumina'
                elif 'agilent' in platform_text:
                    self.platform_type = 'agilent'
                else:
                    self.platform_type = 'unknown'
                
                logger.info(f"Detected microarray data: {self.platform_type}")
                self.raw_data = expression_data.copy()
                return True
            
            logger.info("Data not detected as microarray")
            return False
            
        except Exception as e:
            logger.error(f"Error detecting data type: {str(e)}")
            raise

    def preprocess_data(self):
        """Preprocess microarray data"""
        try:
            if not self.is_microarray:
                raise ValueError("Data is not from microarray platform")

            logger.info(f"Starting preprocessing for {self.platform_type} platform")
            
            # Background correction
            self.preprocessed_data = self._background_correct(self.raw_data)
            
            # Log2 transformation
            self.preprocessed_data = np.log2(self.preprocessed_data + 1)
            
            # Filter low intensity probes
            self._filter_low_intensity_probes()
            
            # Generate QC plots
            self._generate_preprocessing_plots()
            
            logger.info("Preprocessing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _background_correct(self, data):
        """Perform background correction"""
        try:
            # Calculate background as the median of the lowest 5% intensities
            background = data.quantile(0.05, axis=1)
            
            # Subtract background from each probe
            corrected = data.sub(background, axis=0)
            
            # Set negative values to small positive number
            corrected[corrected < 0] = 0.0001
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in background correction: {str(e)}")
            raise

    def _filter_low_intensity_probes(self):
        """Filter out low intensity probes"""
        try:
            # Calculate mean intensity per probe
            mean_intensities = self.preprocessed_data.mean(axis=1)
            
            # Keep probes with mean intensity > 25th percentile
            intensity_threshold = mean_intensities.quantile(0.25)
            high_intensity_mask = mean_intensities > intensity_threshold
            
            self.preprocessed_data = self.preprocessed_data[high_intensity_mask]
            
            # Store filtering stats
            self.preprocessing_stats['filtered_probes'] = {
                'total_probes': len(mean_intensities),
                'retained_probes': len(self.preprocessed_data),
                'filtered_out': len(mean_intensities) - len(self.preprocessed_data)
            }
            
        except Exception as e:
            logger.error(f"Error filtering probes: {str(e)}")
            raise

    def normalize_data(self):
        """Perform quantile normalization"""
        try:
            if self.preprocessed_data is None:
                raise ValueError("No preprocessed data available")

            logger.info("Starting quantile normalization")
            
            # Store the original data for comparison
            data_for_norm = self.preprocessed_data.copy()
            
            # Perform quantile normalization
            normalized_values = quantile_transform(
                data_for_norm.T,
                output_distribution='normal',
                n_quantiles=min(len(data_for_norm), 1000),
                copy=True
            ).T
            
            # Convert back to DataFrame with original index and columns
            self.normalized_data = pd.DataFrame(
                normalized_values,
                index=data_for_norm.index,
                columns=data_for_norm.columns
            )
            
            # Generate normalization plots
            self._generate_normalization_plots()
            
            # Calculate normalization stats
            self._calculate_normalization_stats()
            
            logger.info("Normalization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            raise

    def _generate_preprocessing_plots(self):
        """Generate preprocessing QC plots"""
        try:
            # Intensity distribution plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.preprocessed_data)
            plt.xticks(rotation=90)
            plt.title('Preprocessed Data Intensity Distribution')
            self.plots['preprocessing_intensity'] = self._get_plot_base64()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating preprocessing plots: {str(e)}")
            raise

    def _generate_normalization_plots(self):
        """Generate normalization comparison plots"""
        try:
            # Create figure with larger size and better spacing
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot 1: Before normalization
            sns.boxplot(data=self.preprocessed_data, ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.set_title('Expression Distribution Before Normalization', pad=20)
            ax1.set_xlabel('Samples')
            ax1.set_ylabel('Expression Level')
            
            # Plot 2: After normalization
            sns.boxplot(data=self.normalized_data, ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_title('Expression Distribution After Normalization', pad=20)
            ax2.set_xlabel('Samples')
            ax2.set_ylabel('Expression Level')
            
            # Adjust layout
            plt.tight_layout(pad=3.0)
            
            # Save plot
            self.plots['normalization_comparison'] = self._get_plot_base64()
            plt.close()
            
            # Generate preprocessing plot separately
            plt.figure(figsize=(15, 6))
            sns.boxplot(data=self.preprocessed_data)
            plt.xticks(rotation=45, ha='right')
            plt.title('Expression Distribution After Preprocessing')
            plt.xlabel('Samples')
            plt.ylabel('Expression Level')
            plt.tight_layout()
            self.plots['preprocessing_intensity'] = self._get_plot_base64()
            plt.close()
            
            logger.info("Generated normalization plots successfully")
            
        except Exception as e:
            logger.error(f"Error generating normalization plots: {str(e)}")
            raise

    def _calculate_normalization_stats(self):
        """Calculate normalization statistics"""
        try:
            self.normalization_stats = {
                'before': {
                    'mean': float(self.preprocessed_data.mean().mean()),
                    'std': float(self.preprocessed_data.std().mean()),
                    'median': float(self.preprocessed_data.median().mean())
                },
                'after': {
                    'mean': float(self.normalized_data.mean().mean()),
                    'std': float(self.normalized_data.std().mean()),
                    'median': float(self.normalized_data.median().mean())
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating normalization stats: {str(e)}")
            raise

    def _get_plot_base64(self):
        """Convert current plot to base64 string"""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def get_serializable_stats(self):
        """Get JSON-serializable statistics"""
        return {
            'preprocessing': self.preprocessing_stats,
            'normalization': self.normalization_stats
        } 