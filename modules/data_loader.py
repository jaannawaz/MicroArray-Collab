import pandas as pd
import numpy as np
import GEOparse
import logging
import os
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
        self.sample_groups = {
            'group1': [],
            'group2': []
        }
        self.workflow_state = {
            'qc_completed': False,
            'normalization_completed': False,
            'deg_ready': False
        }

    def load_from_geo(self, geo_id):
        """Load data from GEO database"""
        try:
            logger.info(f"Loading GEO dataset: {geo_id}")
            gse = GEOparse.get_GEO(geo=geo_id)
            
            # Extract expression data
            if len(gse.gsms) == 0:
                logger.error("No samples found in the dataset")
                return False
                
            # Get the first GPL platform
            platform = list(gse.gpls.values())[0]
            first_gsm = list(gse.gsms.values())[0]
            
            # Create expression matrix
            expression_data = pd.DataFrame()
            for gsm_name, gsm in gse.gsms.items():
                if 'VALUE' in gsm.table.columns:
                    expression_data[gsm_name] = pd.to_numeric(gsm.table['VALUE'], errors='coerce')
            
            if 'ID_REF' in first_gsm.table.columns:
                expression_data.index = first_gsm.table['ID_REF']
            
            self.expression_data = expression_data
            
            # Extract metadata
            self.metadata = pd.DataFrame({
                'sample_id': list(gse.gsms.keys()),
                'title': [gsm.metadata['title'][0] for gsm in gse.gsms.values()],
                'source': [gsm.metadata.get('source_name_ch1', ['N/A'])[0] for gsm in gse.gsms.values()]
            })
            
            self.platform_info = platform
            self.is_data_loaded = True
            
            logger.info(f"Successfully loaded dataset {geo_id}")
            logger.info(f"Expression data shape: {self.expression_data.shape}")
            logger.info(f"Number of samples: {len(self.metadata)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading GEO dataset: {str(e)}")
            self.is_data_loaded = False
            return False

    def update_sample_groups(self, groups):
        """Update sample group assignments"""
        try:
            if not isinstance(groups, dict):
                logger.error("Groups must be a dictionary")
                return False
            
            # Validate samples exist in expression data
            all_samples = []
            for group_samples in groups.values():
                all_samples.extend(group_samples)
            
            if self.expression_data is not None:
                invalid_samples = [s for s in all_samples if s not in self.expression_data.columns]
                if invalid_samples:
                    logger.error(f"Invalid samples: {invalid_samples}")
                    return False
            
            self.sample_groups = groups
            logger.info(f"Updated sample groups: {self.sample_groups}")
            
            # Update sample info for DEG analysis
            self.sample_info = {
                'group1': self.sample_groups.get('group1', []),
                'group2': self.sample_groups.get('group2', [])
            }
            
            # Reset workflow state when groups are updated
            self.workflow_state = {
                'qc_completed': False,
                'normalization_completed': False,
                'deg_ready': False
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating sample groups: {str(e)}")
            return False

    def run_qc_for_groups(self):
        """Run QC analysis for each group"""
        try:
            if not self.is_data_loaded:
                logger.error("No data loaded")
                return None

            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)

            qc_results = {}
            for group_name, group_samples in self.sample_groups.items():
                if group_samples:
                    # Get data for this group
                    group_data = self.expression_data[group_samples]
                    
                    # Calculate basic QC metrics
                    qc_metrics = {
                        'missing_values': group_data.isnull().sum().sum(),
                        'total_features': len(group_data.index),
                        'total_samples': len(group_samples),
                        'mean_expression': group_data.mean().mean(),
                        'std_expression': group_data.std().mean()
                    }
                    
                    # Save QC results
                    qc_results[group_name] = qc_metrics
                    
                    # Save group data
                    output_file = f'temp/{group_name}_raw_data.csv'
                    group_data.to_csv(output_file)
                    logger.info(f"Saved {group_name} raw data to {output_file}")

            if qc_results:
                self.workflow_state['qc_completed'] = True
                return qc_results
            else:
                logger.error("No groups available for QC analysis")
                return None

        except Exception as e:
            logger.error(f"Error in group QC analysis: {str(e)}")
            return None

    def normalize_group_data(self, method='quantile'):
        """Normalize data for each group separately"""
        try:
            if not self.workflow_state['qc_completed']:
                logger.error("QC analysis must be completed first")
                return False

            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)

            normalized_groups = {}
            for group_name, samples in self.sample_groups.items():
                if samples:
                    logger.info(f"Normalizing {group_name} data")
                    group_data = self.expression_data[samples]
                    
                    # Perform normalization
                    if method == 'quantile':
                        norm_data = self._quantile_normalization(group_data)
                    elif method == 'log2':
                        norm_data = np.log2(group_data + 1)
                    else:
                        raise ValueError(f"Unsupported normalization method: {method}")
                    
                    if norm_data is None:
                        logger.error(f"Normalization failed for {group_name}")
                        return False
                    
                    # Save normalized data
                    output_file = f'temp/{group_name}_normalized.csv'
                    norm_data.to_csv(output_file)
                    logger.info(f"Saved normalized {group_name} data to {output_file}")
                    
                    normalized_groups[group_name] = norm_data

            # Store normalized data
            if normalized_groups:
                self.normalized_data = pd.concat(normalized_groups.values(), axis=1)
                self.workflow_state['normalization_completed'] = True
                self.workflow_state['deg_ready'] = True
                return True
            else:
                logger.error("No groups to normalize")
                return False

        except Exception as e:
            logger.error(f"Error normalizing group data: {str(e)}")
            return False

    def _quantile_normalization(self, data):
        """Perform quantile normalization on the data"""
        try:
            data_array = data.values
            
            # Sort each column
            data_sorted = np.sort(data_array, axis=0)
            
            # Calculate mean of ranks
            mean_ranks = np.mean(data_sorted, axis=1)
            
            # Get ranks of original data
            temp = data_array.argsort(axis=0)
            ranks = np.empty_like(temp)
            
            # Fill in ranks
            for i in range(data_array.shape[1]):
                ranks[temp[:, i], i] = np.arange(data_array.shape[0])
            
            # Create normalized data
            normalized_data = np.zeros_like(data_array)
            for i in range(data_array.shape[1]):
                normalized_data[:, i] = mean_ranks[ranks[:, i]]
            
            # Convert back to DataFrame
            return pd.DataFrame(normalized_data, index=data.index, columns=data.columns)
            
        except Exception as e:
            logger.error(f"Error in quantile normalization: {str(e)}")
            return None

    def get_group_data(self, group_name):
        """Get data for a specific group"""
        if group_name in self.sample_groups:
            samples = self.sample_groups[group_name]
            if samples:
                return self.expression_data[samples]
        return None

    def get_normalized_group_data(self, group_name):
        """Get normalized data for a specific group"""
        if self.normalized_data is not None and group_name in self.sample_groups:
            samples = self.sample_groups[group_name]
            if samples:
                return self.normalized_data[samples]
        return None

    def get_group_samples(self):
        """Get samples for each group"""
        return self.sample_groups

    def display_data_structure(self):
        """Display current data structure information"""
        try:
            info = {
                'data_loaded': self.is_data_loaded,
                'expression_data_shape': self.expression_data.shape if self.expression_data is not None else None,
                'total_samples': len(self.expression_data.columns) if self.expression_data is not None else 0,
                'total_features': len(self.expression_data.index) if self.expression_data is not None else 0,
                'groups': self.sample_groups,
                'workflow_state': self.workflow_state
            }
            return info
        except Exception as e:
            logger.error(f"Error getting data structure: {str(e)}")
            return None