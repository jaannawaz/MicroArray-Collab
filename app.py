from flask import Flask, request, jsonify, render_template, redirect, session, send_from_directory
import os
import json
from modules.data_loader import GEODataLoader
from modules.utils import save_to_csv
import logging
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from io import StringIO
import warnings
from GEOparse import get_GEO
from biomart import BiomartServer
from utils.platform_config import get_platform_config

app = Flask(__name__)
# Set a secret key for session management
app.secret_key = os.urandom(24)

data_loader = GEODataLoader()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure temp directory exists
os.makedirs('temp', exist_ok=True)

def get_quality_class(score):
    """Helper function for QC quality scores"""
    if score >= 0.8:
        return 'bg-success'
    if score >= 0.6:
        return 'bg-warning'
    return 'bg-danger'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_geo_data', methods=['POST'])
def fetch_geo_data():
    try:
        data = request.get_json()
        if not data or 'geo_id' not in data:
            return jsonify({'error': 'No GEO ID provided'}), 400

        geo_id = data['geo_id'].strip()
        success = data_loader.load_from_geo(geo_id)
        
        if not success:
            return jsonify({'error': 'Failed to load data'}), 400

        # Format sample data for response
        samples = []
        for _, row in data_loader.metadata.iterrows():
            samples.append({
                'sample_id': row['sample_id'],
                'title': row['title'],
                'source': row['source']
            })

        return jsonify({
            'status': 'success',
            'data': {
                'samples': samples,
                'total_samples': len(samples),
                'total_features': len(data_loader.expression_data.index)
            }
        })

    except Exception as e:
        logger.error(f"Error in fetch_geo_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run_selected_qc', methods=['POST'])
def run_selected_qc():
    try:
        if not data_loader.is_data_loaded:
            return jsonify({
                'status': 'error',
                'error': 'No data loaded'
            }), 400

        # Get groups from request
        data = request.get_json()
        if not data or 'groups' not in data:
            return jsonify({
                'status': 'error',
                'error': 'No groups provided'
            }), 400

        groups = data['groups']
        
        # Validate groups
        if not groups['group1']['samples'] or not groups['group2']['samples']:
            return jsonify({
                'status': 'error',
                'error': 'Both groups must have samples'
            }), 400

        # Save group assignments
        success = data_loader.update_sample_groups({
            'group1': groups['group1']['samples'],
            'group2': groups['group2']['samples']
        })
        
        if not success:
            return jsonify({
                'status': 'error',
                'error': 'Failed to update groups'
            }), 400

        # Run QC analysis
        from modules.qc_analyzer import run_qc_analysis
        
        # Get data for selected samples
        selected_samples = {
            'group1': groups['group1']['samples'],
            'group2': groups['group2']['samples']
        }
        qc_results = run_qc_analysis(data_loader.expression_data, selected_samples)

        if qc_results:
            # Save QC state
            data_loader.workflow_state['qc_completed'] = True
            
            # Save group data and QC results to files
            os.makedirs('temp', exist_ok=True)
            
            # Save raw data files
            for group_name, samples in selected_samples.items():
                data_loader.expression_data[samples].to_csv(f'temp/{group_name}_raw_data.csv')
            
            # Save QC results without plots
            qc_data = {}
            for group_name, group_results in qc_results.items():
                # Save plots separately
                plots_data = group_results.pop('plots', {})
                for plot_name, plot_data in plots_data.items():
                    plot_file = f'temp/{group_name}_{plot_name}.png'
                    with open(plot_file, 'wb') as f:
                        import base64
                        f.write(base64.b64decode(plot_data))
                
                # Store metrics and plot file paths
                qc_data[group_name] = {
                    'metrics': group_results['metrics'],
                    'plot_files': {
                        plot_name: f'{group_name}_{plot_name}.png'
                        for plot_name in plots_data.keys()
                    }
                }
            
            # Save QC data to file
            with open('temp/qc_results.json', 'w') as f:
                json.dump(qc_data, f)

            return jsonify({
                'status': 'success',
                'message': 'QC analysis completed'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'QC analysis failed'
            }), 500

    except Exception as e:
        logger.error(f"Error in run_selected_qc: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/qc_results')
def qc_results():
    try:
        # Load QC results from file
        try:
            with open('temp/qc_results.json', 'r') as f:
                qc_data = json.load(f)
        except FileNotFoundError:
            return redirect('/')
            
        return render_template('qc_results.html', qc_data=qc_data, get_quality_class=get_quality_class)
    except Exception as e:
        logger.error(f"Error displaying QC results: {str(e)}")
        return redirect('/')

@app.route('/normalization')
def normalization():
    if not data_loader.workflow_state['qc_completed']:
        return redirect('/')
    return render_template('normalization.html')

@app.route('/temp/<path:filename>')
def serve_temp_file(filename):
    """Serve files from temp directory"""
    return send_from_directory('temp', filename)

@app.route('/get_group_info')
def get_group_info():
    try:
        if not data_loader.is_data_loaded:
            return jsonify({
                'status': 'error',
                'error': 'No data loaded'
            }), 400

        return jsonify({
            'status': 'success',
            'groups': data_loader.get_group_samples()
        })

    except Exception as e:
        logger.error(f"Error getting group info: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/run_normalization', methods=['POST'])
def run_normalization():
    try:
        # Clean temp directory except qc_results.json
        for file in os.listdir('temp'):
            if file != 'qc_results.json':
                os.remove(os.path.join('temp', file))

        if not data_loader.workflow_state['qc_completed']:
            return jsonify({
                'status': 'error',
                'error': 'Please complete QC analysis first'
            }), 400

        method = request.json.get('method', 'quantile')
        if method not in ['quantile', 'zscore', 'log2']:
            return jsonify({
                'status': 'error',
                'error': 'Invalid normalization method'
            }), 400

        # Get group data
        groups = data_loader.get_group_samples()
        if not groups['group1'] or not groups['group2']:
            return jsonify({
                'status': 'error',
                'error': 'Missing group data'
            }), 400

        # Run normalization
        success = data_loader.normalize_group_data(method)
        if not success:
            return jsonify({
                'status': 'error',
                'error': 'Normalization failed'
            }), 400

        # Generate plots
        import matplotlib.pyplot as plt
        import seaborn as sns
        from io import BytesIO
        import base64

        plots = {}

        # Box plots before normalization
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data_loader.expression_data)
        plt.xticks(rotation=45, ha='right')
        plt.title('Expression Distribution Before Normalization')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['before_boxplot'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Box plots after normalization
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data_loader.normalized_data)
        plt.xticks(rotation=45, ha='right')
        plt.title('Expression Distribution After Normalization')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['after_boxplot'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Density plots before normalization
        plt.figure(figsize=(12, 6))
        for col in data_loader.expression_data.columns:
            sns.kdeplot(data=data_loader.expression_data[col].dropna(), label=col)
        plt.title('Expression Density Before Normalization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['before_density'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Density plots after normalization
        plt.figure(figsize=(12, 6))
        for col in data_loader.normalized_data.columns:
            sns.kdeplot(data=data_loader.normalized_data[col].dropna(), label=col)
        plt.title('Expression Density After Normalization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['after_density'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Get statistics
        stats = {
            'before': {
                'mean': float(data_loader.expression_data.mean().mean()),
                'std': float(data_loader.expression_data.std().mean()),
                'min': float(data_loader.expression_data.min().min()),
                'max': float(data_loader.expression_data.max().max())
            },
            'after': {
                'mean': float(data_loader.normalized_data.mean().mean()),
                'std': float(data_loader.normalized_data.std().mean()),
                'min': float(data_loader.normalized_data.min().min()),
                'max': float(data_loader.normalized_data.max().max())
            }
        }

        # Save normalized data for each group
        group1_data = data_loader.normalized_data[groups['group1']]
        group2_data = data_loader.normalized_data[groups['group2']]
        
        group1_file = 'temp/group1_normalized.csv'
        group2_file = 'temp/group2_normalized.csv'
        
        group1_data.to_csv(group1_file)
        group2_data.to_csv(group2_file)

        return jsonify({
            'status': 'success',
            'message': 'Normalization completed',
            'stats': stats,
            'plots': plots,
            'files': {
                'group1': group1_file,
                'group2': group2_file
            }
        })

    except Exception as e:
        logger.error(f"Error in normalization: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/deg_analysis')
def deg_analysis_page():
    return render_template('deg_analysis.html')

def get_gene_mapping(probe_ids, platform_id):
    """Get gene symbols using biomart based on platform information."""
    try:
        # Initialize mapping dictionary
        gene_mapping = {}
        
        if not platform_id:
            logger.warning("No platform ID provided")
            return gene_mapping
            
        logger.info(f"Getting gene mapping for platform: {platform_id}")
        
        # First try GEO platform annotation
        try:
            platform_data = get_GEO(platform_id)
            if hasattr(platform_data, 'table'):
                table = platform_data.table
                if 'ID' in table.columns:
                    symbol_col = next((col for col in [
                        'Gene Symbol', 'Gene_Symbol', 'GENE_SYMBOL', 
                        'gene_symbol', 'Symbol', 'SYMBOL'
                    ] if col in table.columns), None)
                    
                    if symbol_col:
                        logger.info(f"Using GEO annotation with column: {symbol_col}")
                        gene_mapping = dict(zip(table['ID'], table[symbol_col]))
                        # Clean up mapping
                        gene_mapping = {k: str(v).split('///')[0].strip() 
                                     for k, v in gene_mapping.items() 
                                     if v and str(v).strip() != 'nan'}
                        return gene_mapping
        except Exception as e:
            logger.warning(f"Error loading GEO platform data: {str(e)}")
        
        # If GEO mapping failed, try biomart
        # Determine the correct biomart dataset and attribute based on platform
        platform_to_biomart = {
            'GPL570': ('affy_hg_u133_plus_2', 'hsapiens_gene_ensembl'),  # HG-U133_Plus_2
            'GPL96': ('affy_hg_u133a', 'hsapiens_gene_ensembl'),         # HG-U133A
            'GPL97': ('affy_hg_u133b', 'hsapiens_gene_ensembl'),         # HG-U133B
            'GPL571': ('affy_hg_u133a_2', 'hsapiens_gene_ensembl'),      # HG-U133A_2
            'GPL1261': ('illumina_mousewgdna_6', 'mmusculus_gene_ensembl'),  # Illumina MouseWG-6
            'GPL6887': ('illumina_humanht_12', 'hsapiens_gene_ensembl'),  # Illumina HumanHT-12
            'GPL6947': ('illumina_humanref_8', 'hsapiens_gene_ensembl'),  # Illumina HumanRef-8
            'GPL8321': ('illumina_ratref_12', 'rnorvegicus_gene_ensembl'),  # Illumina RatRef-12
            'GPL13534': ('illumina_humanht_12_v4', 'hsapiens_gene_ensembl'),  # Illumina HumanHT-12 V4
            'GPL6244': ('affy_hugene_1_0_st_v1', 'hsapiens_gene_ensembl'),  # HuGene-1_0-st
            'GPL11154': ('affy_hugene_2_0_st_v1', 'hsapiens_gene_ensembl'),  # HuGene-2_0-st
        }
        
        if platform_id not in platform_to_biomart:
            logger.warning(f"Platform {platform_id} not supported for biomart mapping")
            return gene_mapping
            
        array_type, dataset_name = platform_to_biomart[platform_id]
        logger.info(f"Using biomart mapping: {array_type} -> {dataset_name}")
        
        # Connect to biomart
        try:
            server = BiomartServer("http://ensembl.org/biomart")
            dataset = server.datasets[dataset_name]
            
            # Split probe_ids into chunks to avoid too long requests
            chunk_size = 100
            for i in range(0, len(probe_ids), chunk_size):
                chunk = probe_ids[i:i + chunk_size]
                
                # Prepare the query
                response = dataset.search({
                    'attributes': [array_type, 'external_gene_name'],
                    'filters': {array_type: chunk}
                })
                
                # Process results
                for line in response.iter_lines():
                    line = line.decode('utf-8').strip().split('\t')
                    if len(line) >= 2:
                        probe_id, gene_name = line
                        if gene_name:  # Only add if gene name is not empty
                            gene_mapping[probe_id] = gene_name
                
            logger.info(f"Successfully mapped {len(gene_mapping)} genes using biomart")
            return gene_mapping
            
        except Exception as e:
            logger.error(f"Error in biomart mapping: {str(e)}")
            return gene_mapping
        
    except Exception as e:
        logger.error(f"Error in gene mapping: {str(e)}")
        return {}

@app.route('/run_deg_analysis', methods=['POST'])
def run_deg_analysis():
    try:
        # Get parameters from request
        params = request.json
        control_group = params.get('control_group')
        treatment_group = params.get('treatment_group')
        pval_threshold = float(params.get('pval_threshold', 0.05))
        log2fc_threshold = float(params.get('log2fc_threshold', 1))
        top_genes = int(params.get('top_genes', 10))

        # Load QC results to get platform ID
        platform_id = None
        try:
            with open('temp/qc_results.json', 'r') as f:
                qc_data = json.load(f)
                platform_id = qc_data.get('platform_id', '').strip()
                logger.info(f"Found platform ID: {platform_id}")
        except Exception as e:
            logger.warning(f"Could not load platform info: {str(e)}")

        # Check if normalized files exist
        control_file = f'temp/{control_group}_normalized.csv'
        treatment_file = f'temp/{treatment_group}_normalized.csv'
        
        if not (os.path.exists(control_file) and os.path.exists(treatment_file)):
            return jsonify({
                'status': 'error',
                'error': 'Normalized data not found. Please complete normalization first.'
            }), 400

        # Load normalized data
        control_data = pd.read_csv(control_file, index_col=0)
        treatment_data = pd.read_csv(treatment_file, index_col=0)

        # Get gene mapping
        probe_ids = list(control_data.index)
        gene_mapping = get_gene_mapping(probe_ids, platform_id)
        
        if not gene_mapping:
            logger.warning("No gene mapping found, using probe IDs as gene symbols")
            gene_mapping = {probe_id: probe_id for probe_id in probe_ids}

        # Perform t-test for each gene
        results = []
        for gene in control_data.index:
            control_values = control_data.loc[gene].astype(float)
            treatment_values = treatment_data.loc[gene].astype(float)
            
            # Skip if all values are identical or contain NaN
            if (control_values.isna().any() or treatment_values.isna().any() or
                (control_values == control_values.iloc[0]).all() or
                (treatment_values == treatment_values.iloc[0]).all()):
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t_stat, p_val = stats.ttest_ind(control_values, treatment_values)
            except Exception as e:
                logger.warning(f"Error in t-test for gene {gene}: {str(e)}")
                continue

            mean_control = np.mean(control_values)
            mean_treatment = np.mean(treatment_values)
            
            # Calculate log2fc with safety checks
            if mean_control <= 0 or mean_treatment <= 0:
                log2fc = 0
            else:
                log2fc = np.log2(mean_treatment / mean_control)
            
            # Get gene symbol from mapping
            gene_symbol = gene_mapping.get(gene, gene)
            
            results.append({
                'probe_id': gene,
                'gene_symbol': gene_symbol,
                'log2fc': log2fc,
                'pvalue': p_val,
                'mean_control': mean_control,
                'mean_treatment': mean_treatment
            })

        if not results:
            return jsonify({
                'status': 'error',
                'error': 'No valid genes for analysis after filtering'
            }), 400

        # Create results DataFrame
        deg_results = pd.DataFrame(results)
        
        # Multiple testing correction
        deg_results['padj'] = multipletests(deg_results['pvalue'], method='fdr_bh')[1]
        
        # Sort by adjusted p-value
        deg_results = deg_results.sort_values('padj')
        
        # Generate plots
        plots = {}
        
        # Volcano plot with custom thresholds
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(deg_results['log2fc'], 
                   -np.log10(deg_results['padj']), 
                   alpha=0.5,
                   c=((abs(deg_results['log2fc']) >= log2fc_threshold) & 
                      (deg_results['padj'] <= pval_threshold)).map({True: 'red', False: 'grey'}))
        
        plt.axhline(y=-np.log10(pval_threshold), color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=-log2fc_threshold, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=log2fc_threshold, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-log10(Adjusted P-value)')
        plt.title(f'Volcano Plot\n{treatment_group} vs {control_group}')
        
        # Add gene labels for significant genes
        significant = deg_results[
            (deg_results['padj'] <= pval_threshold) & 
            (abs(deg_results['log2fc']) >= log2fc_threshold)
        ].head(top_genes)
        
        for _, gene in significant.iterrows():
            label = gene['gene_symbol'] if gene['gene_symbol'] != gene['probe_id'] else gene['probe_id']
            plt.annotate(label, 
                        (gene['log2fc'], -np.log10(gene['padj'])),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['volcano'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Heatmap of top DEGs
        significant_genes = deg_results[
            (deg_results['padj'] <= pval_threshold) & 
            (abs(deg_results['log2fc']) >= log2fc_threshold)
        ]['probe_id'].tolist()
        
        if len(significant_genes) > 50:
            significant_genes = significant_genes[:50]
        
        if significant_genes:
            combined_data = pd.concat([
                control_data.loc[significant_genes],
                treatment_data.loc[significant_genes]
            ], axis=1)
            
            # Create row labels with gene symbols
            row_labels = [f"{idx} ({gene_mapping.get(idx, idx)})" for idx in combined_data.index]
            
            plt.figure(figsize=(12, 8))
            g = sns.clustermap(combined_data,
                          cmap='RdBu_r',
                          center=0,
                          figsize=(12, 8),
                          dendrogram_ratio=(.1, .2),
                          cbar_pos=(0.02, .2, .03, .4),
                          yticklabels=row_labels)
            plt.suptitle(f'Top DEGs Heatmap: {treatment_group} vs {control_group}', y=1.02)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plots['heatmap'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
        
        # Summary statistics
        total_genes = len(deg_results)
        sig_genes = len(deg_results[
            (deg_results['padj'] <= pval_threshold) & 
            (abs(deg_results['log2fc']) >= log2fc_threshold)
        ])
        up_regulated = len(deg_results[
            (deg_results['padj'] <= pval_threshold) & 
            (deg_results['log2fc'] >= log2fc_threshold)
        ])
        down_regulated = len(deg_results[
            (deg_results['padj'] <= pval_threshold) & 
            (deg_results['log2fc'] <= -log2fc_threshold)
        ])
        
        # Prepare CSV data
        # Reorder columns for better readability
        column_order = [
            'probe_id', 'gene_symbol', 'log2fc', 'pvalue', 'padj',
            'mean_control', 'mean_treatment'
        ]
        deg_results = deg_results[column_order]
        
        csv_buffer = StringIO()
        deg_results.to_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()
        
        return jsonify({
            'status': 'success',
            'plots': plots,
            'summary': {
                'total_genes': total_genes,
                'significant_genes': sig_genes,
                'up_regulated': up_regulated,
                'down_regulated': down_regulated
            },
            'csv_data': csv_data
        })
        
    except Exception as e:
        logger.error(f"Error in DEG analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    config = get_platform_config()
    print(f"Starting server on {config['host']}:{config['port']}")
    app.run(
        host=config['host'],
        port=config['port'],
        debug=config['debug']
    )
