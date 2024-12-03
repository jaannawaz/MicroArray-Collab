import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class QCAnalyzer:
    def __init__(self):
        self.qc_metrics = {}
        self.qc_plots = {}
        self.qc_report = None

    def run_qc_analysis(self, expression_data):
        """Run comprehensive QC analysis"""
        try:
            if expression_data is None:
                raise ValueError("No data provided for QC analysis")

            logger.info("Running QC analysis...")
            
            # Calculate QC metrics
            self._calculate_qc_metrics(expression_data)
            
            # Generate QC plots
            self._generate_qc_plots(expression_data)
            
            # Generate QC report
            self._generate_qc_report(expression_data)
            
            logger.info("QC analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in QC analysis: {str(e)}")
            raise

    def _calculate_qc_metrics(self, expression_data):
        """Calculate quality control metrics"""
        try:
            # Sample-level metrics
            sample_metrics = {
                'total_counts': expression_data.sum(),
                'detected_genes': (expression_data > 0).sum(),
                'mean_expression': expression_data.mean(),
                'median_expression': expression_data.median(),
                'std_expression': expression_data.std()
            }

            # Gene-level metrics
            gene_metrics = {
                'mean_expression': expression_data.mean(axis=1),
                'detection_rate': (expression_data > 0).mean(axis=1),
                'variance': expression_data.var(axis=1)
            }

            # Check for mitochondrial genes
            try:
                mt_genes = expression_data.index.str.contains('MT-|mt-', regex=True)
                if mt_genes.any():
                    sample_metrics['mt_percentage'] = \
                        expression_data.loc[mt_genes].sum() / expression_data.sum() * 100
            except:
                logger.warning("Could not calculate mitochondrial percentage")

            self.qc_metrics = {
                'sample_metrics': sample_metrics,
                'gene_metrics': gene_metrics
            }

        except Exception as e:
            logger.error(f"Error calculating QC metrics: {str(e)}")
            raise

    def _generate_qc_plots(self, expression_data):
        """Generate quality control plots"""
        try:
            # Expression distribution boxplot
            plt.figure(figsize=(15, 6))
            sns.boxplot(data=expression_data)
            plt.xticks(rotation=90)
            plt.title('Expression Distribution Across Samples')
            self.qc_plots['expression_dist'] = self._get_plot_base64()
            plt.close()

            # Total counts distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.qc_metrics['sample_metrics']['total_counts'])
            plt.title('Distribution of Total Counts per Sample')
            plt.xlabel('Total Counts')
            plt.ylabel('Frequency')
            self.qc_plots['total_counts_dist'] = self._get_plot_base64()
            plt.close()

            # PCA plot
            plt.figure(figsize=(12, 8))
            # Standardize the data
            scaled_data = StandardScaler().fit_transform(expression_data.T)
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create PCA plot
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.8)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Plot of Samples')
            
            # Add sample labels
            for idx, sample in enumerate(expression_data.columns):
                plt.annotate(sample, (pca_result[idx, 0], pca_result[idx, 1]))
                
            plt.tight_layout()
            self.qc_plots['pca'] = self._get_plot_base64()
            plt.close()

            logger.info(f"Generated plots with keys: {list(self.qc_plots.keys())}")
            return True

        except Exception as e:
            logger.error(f"Error generating QC plots: {str(e)}")
            raise

    def _generate_qc_report(self, expression_data):
        """Generate QC report with quality assessment"""
        try:
            metrics = self.qc_metrics
            report = {
                'summary': {
                    'total_samples': len(expression_data.columns),
                    'total_genes': len(expression_data.index),
                    'high_quality_samples': 0,
                    'medium_quality_samples': 0,
                    'low_quality_samples': 0
                },
                'sample_quality': {},
                'recommendations': []
            }

            # Calculate thresholds
            total_counts = metrics['sample_metrics']['total_counts']
            detected_genes = metrics['sample_metrics']['detected_genes']
            
            count_thresh = {
                'low': total_counts.quantile(0.25),
                'high': total_counts.quantile(0.75)
            }
            gene_thresh = {
                'low': detected_genes.quantile(0.25),
                'high': detected_genes.quantile(0.75)
            }

            # Evaluate each sample
            for sample in expression_data.columns:
                quality_score = 0
                issues = []
                
                # Check total counts
                sample_counts = total_counts[sample]
                if sample_counts >= count_thresh['high']:
                    quality_score += 2
                elif sample_counts >= count_thresh['low']:
                    quality_score += 1
                else:
                    issues.append("Low total counts")

                # Check detected genes
                sample_genes = detected_genes[sample]
                if sample_genes >= gene_thresh['high']:
                    quality_score += 2
                elif sample_genes >= gene_thresh['low']:
                    quality_score += 1
                else:
                    issues.append("Low number of detected genes")

                # Determine quality
                if quality_score >= 3:
                    quality = "High"
                    report['summary']['high_quality_samples'] += 1
                elif quality_score >= 1:
                    quality = "Medium"
                    report['summary']['medium_quality_samples'] += 1
                else:
                    quality = "Low"
                    report['summary']['low_quality_samples'] += 1

                # Store sample quality
                report['sample_quality'][sample] = {
                    'quality': quality,
                    'score': quality_score,
                    'issues': issues,
                    'metrics': {
                        'total_counts': int(sample_counts),
                        'detected_genes': int(sample_genes)
                    }
                }

            # Generate recommendations
            if report['summary']['low_quality_samples'] > 0:
                report['recommendations'].append(
                    f"Consider removing {report['summary']['low_quality_samples']} low-quality samples "
                    "to improve analysis reliability."
                )

            self.qc_report = report

        except Exception as e:
            logger.error(f"Error generating QC report: {str(e)}")
            raise

    def _get_plot_base64(self):
        """Convert current plot to base64 string"""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def get_serializable_metrics(self):
        """Convert metrics to JSON-serializable format"""
        try:
            metrics = self.qc_metrics
            return {
                'sample_metrics': {
                    'total_counts': metrics['sample_metrics']['total_counts'].to_dict(),
                    'detected_genes': metrics['sample_metrics']['detected_genes'].to_dict(),
                    'mean_expression': metrics['sample_metrics']['mean_expression'].to_dict(),
                    'median_expression': metrics['sample_metrics']['median_expression'].to_dict(),
                    'std_expression': metrics['sample_metrics']['std_expression'].to_dict()
                },
                'gene_metrics_summary': {
                    'mean_expression': {
                        'mean': float(metrics['gene_metrics']['mean_expression'].mean()),
                        'std': float(metrics['gene_metrics']['mean_expression'].std()),
                        'min': float(metrics['gene_metrics']['mean_expression'].min()),
                        'max': float(metrics['gene_metrics']['mean_expression'].max())
                    },
                    'detection_rate': {
                        'mean': float(metrics['gene_metrics']['detection_rate'].mean()),
                        'std': float(metrics['gene_metrics']['detection_rate'].std()),
                        'min': float(metrics['gene_metrics']['detection_rate'].min()),
                        'max': float(metrics['gene_metrics']['detection_rate'].max())
                    },
                    'variance': {
                        'mean': float(metrics['gene_metrics']['variance'].mean()),
                        'std': float(metrics['gene_metrics']['variance'].std()),
                        'min': float(metrics['gene_metrics']['variance'].min()),
                        'max': float(metrics['gene_metrics']['variance'].max())
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error converting metrics to serializable format: {str(e)}")
            raise 