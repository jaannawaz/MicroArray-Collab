import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class DEGAnalyzer:
    def __init__(self):
        self.normalized_data = None
        self.sample_info = None
        self.group_annotations = {}
        self.results = None
        self.plots = {}

    def load_data(self, normalized_data, sample_info):
        """Load normalized data and sample information"""
        try:
            if normalized_data is None or sample_info is None:
                logger.error("Normalized data or sample info is None")
                return False

            self.normalized_data = normalized_data
            self.sample_info = sample_info
            logger.info(f"Loaded data with shape: {normalized_data.shape}")
            logger.info(f"Sample info contains {len(sample_info)} samples")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def update_group_annotations(self, groups_dict):
        """Update sample group annotations from the UI groups dictionary"""
        try:
            self.group_annotations = {
                group_name: group_info['samples']
                for group_name, group_info in groups_dict.items()
            }
            logger.info(f"Updated group annotations: {self.group_annotations}")
            return True
        except Exception as e:
            logger.error(f"Error updating group annotations: {str(e)}")
            return False

    def run_differential_expression(self, group1, group2):
        """Perform differential expression analysis"""
        try:
            if not all(key in self.group_annotations for key in [group1, group2]):
                logger.error(f"Missing group annotations for {group1} or {group2}")
                return False

            # Get samples for each group
            group1_samples = self.group_annotations[group1]
            group2_samples = self.group_annotations[group2]

            if not group1_samples or not group2_samples:
                logger.error("Empty sample groups")
                return False

            results = []
            for gene in self.normalized_data.index:
                # Get expression values for each group
                expr1 = self.normalized_data.loc[gene, group1_samples]
                expr2 = self.normalized_data.loc[gene, group2_samples]

                # Calculate fold change
                fc = np.mean(expr2) - np.mean(expr1)  # Log2 fold change
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(expr1, expr2)

                results.append({
                    'gene_id': gene,
                    'log2fc': fc,
                    'pvalue': p_val
                })

            # Create results DataFrame
            self.results = pd.DataFrame(results)
            self.results['padj'] = multipletests(self.results['pvalue'], method='fdr_bh')[1]

            # Generate plots
            self._generate_plots(group1, group2)

            return True

        except Exception as e:
            logger.error(f"Error in differential expression analysis: {str(e)}")
            return False

    def _generate_plots(self, group1, group2):
        """Generate visualization plots"""
        try:
            # Set style
            plt.style.use('seaborn')
            
            # Volcano plot
            plt.figure(figsize=(10, 8))
            plt.scatter(
                self.results['log2fc'],
                -np.log10(self.results['padj']),
                alpha=0.5,
                color='grey'
            )
            
            # Highlight significant DEGs
            significant = self.results['padj'] < 0.05
            up_regulated = (self.results['log2fc'] > 1) & significant
            down_regulated = (self.results['log2fc'] < -1) & significant
            
            plt.scatter(
                self.results.loc[up_regulated, 'log2fc'],
                -np.log10(self.results.loc[up_regulated, 'padj']),
                color='red',
                alpha=0.7,
                label='Up-regulated'
            )
            plt.scatter(
                self.results.loc[down_regulated, 'log2fc'],
                -np.log10(self.results.loc[down_regulated, 'padj']),
                color='blue',
                alpha=0.7,
                label='Down-regulated'
            )
            
            plt.xlabel('Log2 Fold Change')
            plt.ylabel('-log10(FDR)')
            plt.title(f'Volcano Plot\n{group2} vs {group1}')
            plt.legend()
            
            # Add significance lines
            plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.3)
            plt.axvline(x=1, color='r', linestyle='--', alpha=0.3)
            plt.axvline(x=-1, color='r', linestyle='--', alpha=0.3)

            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            self.plots['volcano'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # Heatmap
            significant_genes = self.results[self.results['padj'] < 0.05].nsmallest(50, 'padj').index
            if len(significant_genes) > 0:
                plt.figure(figsize=(12, 8))
                plot_data = self.normalized_data.loc[significant_genes, 
                                                   self.group_annotations[group1] + self.group_annotations[group2]]
                
                # Calculate z-scores for better visualization
                plot_data = (plot_data - plot_data.mean()) / plot_data.std()
                
                sns.heatmap(
                    plot_data,
                    cmap='RdBu_r',
                    center=0,
                    cbar_kws={'label': 'Z-score'},
                    xticklabels=True,
                    yticklabels=True
                )
                plt.title(f'Top 50 DEGs\n{group2} vs {group1}')
                plt.xticks(rotation=45, ha='right')
                
                # Save plot
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                self.plots['heatmap'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")

    def get_summary_stats(self):
        """Get summary statistics of DEG analysis"""
        try:
            if self.results is None:
                return None

            significant = self.results['padj'] < 0.05
            up_regulated = (self.results['log2fc'] > 1) & significant
            down_regulated = (self.results['log2fc'] < -1) & significant

            return {
                'total_genes': len(self.results),
                'significant_genes': int(sum(significant)),
                'up_regulated': int(sum(up_regulated)),
                'down_regulated': int(sum(down_regulated))
            }
        except Exception as e:
            logger.error(f"Error calculating summary stats: {str(e)}")
            return None

    def get_results_table(self):
        """Get results as a formatted table"""
        try:
            if self.results is None:
                return None

            results_table = self.results.copy()
            results_table['significant'] = results_table['padj'] < 0.05
            results_table['regulation'] = 'NS'
            results_table.loc[(results_table['log2fc'] > 1) & results_table['significant'], 'regulation'] = 'Up'
            results_table.loc[(results_table['log2fc'] < -1) & results_table['significant'], 'regulation'] = 'Down'
            
            return results_table.sort_values('padj')

        except Exception as e:
            logger.error(f"Error formatting results table: {str(e)}")
            return None