import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DEGAnalyzer:
    def __init__(self):
        self.results = None
        
    def analyze(self, group1_data, group2_data):
        """
        Perform differential expression analysis between two groups
        """
        try:
            logger.info("Starting DEG analysis")
            logger.info(f"Group 1 data shape: {group1_data.shape}")
            logger.info(f"Group 2 data shape: {group2_data.shape}")
            
            # Verify input data
            if group1_data is None or group2_data is None:
                logger.error("Missing group data")
                return None
                
            if group1_data.empty or group2_data.empty:
                logger.error("Empty group data")
                return None
                
            # Ensure we have the same genes for both groups
            common_genes = group1_data.index.intersection(group2_data.index)
            if len(common_genes) == 0:
                logger.error("No common genes between groups")
                return None
                
            group1_data = group1_data.loc[common_genes]
            group2_data = group2_data.loc[common_genes]
            
            # Calculate statistics
            results = pd.DataFrame(index=common_genes)
            
            # Mean expression
            results['mean_group1'] = group1_data.mean(axis=1)
            results['mean_group2'] = group2_data.mean(axis=1)
            
            # Log2 fold change
            epsilon = 1e-10  # Small constant to avoid division by zero
            results['log2fc'] = np.log2((results['mean_group2'] + epsilon) / (results['mean_group1'] + epsilon))
            
            # Standard deviation
            results['std_group1'] = group1_data.std(axis=1)
            results['std_group2'] = group2_data.std(axis=1)
            
            # T-test
            t_stats = []
            p_values = []
            
            for gene in common_genes:
                group1_vals = group1_data.loc[gene]
                group2_vals = group2_data.loc[gene]
                
                try:
                    t_stat, p_val = stats.ttest_ind(group1_vals, group2_vals)
                    t_stats.append(t_stat)
                    p_values.append(p_val)
                except Exception as e:
                    logger.warning(f"Error calculating t-test for gene {gene}: {str(e)}")
                    t_stats.append(np.nan)
                    p_values.append(np.nan)
            
            results['t_statistic'] = t_stats
            results['p_value'] = p_values
            
            # Adjust p-values (Benjamini-Hochberg)
            mask = ~np.isnan(results['p_value'])
            p_values_adj = np.full_like(results['p_value'], np.nan)
            p_values_adj[mask] = self._adjust_pvalues(results['p_value'][mask])
            results['adjusted_p_value'] = p_values_adj
            
            # Sort by adjusted p-value
            results = results.sort_values('adjusted_p_value')
            
            self.results = results
            logger.info(f"DEG analysis completed. Found {len(results)} genes.")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in DEG analysis: {str(e)}")
            return None
            
    def _adjust_pvalues(self, p_values):
        """
        Adjust p-values using Benjamini-Hochberg procedure
        """
        try:
            n = len(p_values)
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            # Calculate adjusted p-values
            adjusted = np.zeros_like(sorted_p)
            for i in range(n-1, -1, -1):
                if i == n-1:
                    adjusted[i] = sorted_p[i]
                else:
                    adjusted[i] = min(sorted_p[i] * n/(i+1), adjusted[i+1])
            
            # Restore original order
            final_adjusted = np.zeros_like(p_values)
            final_adjusted[sorted_idx] = adjusted
            
            return final_adjusted
            
        except Exception as e:
            logger.error(f"Error adjusting p-values: {str(e)}")
            return np.full_like(p_values, np.nan)