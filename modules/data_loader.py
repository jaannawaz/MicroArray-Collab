import pandas as pd
import GEOparse
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class GEODataLoader:
    def __init__(self):
        self.expression_data = None
        self.metadata = None
        self.platform_info = None
        self.normalized_data = None
        self.sample_info = None
        self.is_data_loaded = False

    def load_from_geo(self, geo_id):
        """Load data from GEO database"""
        try:
            logger.info(f"Loading GEO dataset: {geo_id}")
            gse = GEOparse.get_GEO(geo=geo_id)
            
            self._extract_expression_data(gse)
            self._extract_metadata(gse)
            
            # Create sample info from metadata
            self.sample_info = pd.DataFrame({
                'sample_id': self.metadata['sample_id'],
                'title': self.metadata['title'],
                'source': self.metadata['source']
            }).set_index('sample_id').to_dict('index')
            
            self.is_data_loaded = True
            logger.info("Data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading GEO dataset: {str(e)}")
            raise

    def _extract_expression_data(self, gse):
        """Extract expression data from GEO dataset"""
        try:
            platform = list(gse.gpls.values())[0]
            first_gsm = list(gse.gsms.values())[0]
            
            expression_data = pd.DataFrame()
            for gsm_name, gsm in gse.gsms.items():
                if 'VALUE' in gsm.table.columns:
                    expression_data[gsm_name] = pd.to_numeric(gsm.table['VALUE'], errors='coerce')
            
            if 'ID_REF' in first_gsm.table.columns:
                expression_data.index = first_gsm.table['ID_REF']
            
            self.expression_data = expression_data
            logger.info(f"Extracted expression data with shape: {expression_data.shape}")
            
        except Exception as e:
            logger.error(f"Error extracting expression data: {str(e)}")
            raise

    def _extract_metadata(self, gse):
        """Extract metadata from GEO dataset"""
        try:
            self.metadata = pd.DataFrame({
                'sample_id': list(gse.gsms.keys()),
                'title': [gsm.metadata['title'][0] for gsm in gse.gsms.values()],
                'source': [gsm.metadata.get('source_name_ch1', ['N/A'])[0] for gsm in gse.gsms.values()]
            })
            
            platform = list(gse.gpls.values())[0]
            self.platform_info = {
                'title': platform.metadata['title'][0],
                'technology': platform.metadata.get('technology_type', [''])[0],
                'organism': platform.metadata.get('organism', ['Homo sapiens'])[0]
            }
            
            logger.info("Metadata extracted successfully")
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise

    def normalize_data(self, method='quantile'):
        """Normalize expression data"""
        try:
            if self.expression_data is None:
                raise ValueError("No expression data loaded")

            if method == 'quantile':
                self.normalized_data = self._quantile_normalization(self.expression_data)
            elif method == 'log2':
                self.normalized_data = np.log2(self.expression_data + 1)
            else:
                raise ValueError(f"Unsupported normalization method: {method}")

            logger.info(f"Data normalized using {method} method")
            return True

        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise

    def _quantile_normalization(self, data):
        """Perform quantile normalization"""
        try:
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
            
            return pd.DataFrame(normalized_data, index=data.index, columns=data.columns)
            
        except Exception as e:
            logger.error(f"Error in quantile normalization: {str(e)}")
            raise

    def get_qc_metrics(self):
        """Calculate quality control metrics"""
        try:
            if self.expression_data is None:
                raise ValueError("No expression data loaded")

            metrics = {
                'missing_values': self.expression_data.isnull().sum().sum() / (self.expression_data.shape[0] * self.expression_data.shape[1]) * 100,
                'total_features': self.expression_data.shape[0],
                'total_samples': self.expression_data.shape[1],
                'value_range': {
                    'min': self.expression_data.min().min(),
                    'max': self.expression_data.max().max()
                }
            }

            logger.info("QC metrics calculated successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating QC metrics: {str(e)}")
            raise