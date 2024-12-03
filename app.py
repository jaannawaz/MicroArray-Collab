from flask import Flask, render_template, jsonify, request, send_file, url_for, redirect, flash
import logging
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import traceback
import os
from datetime import datetime
import re

# Import our modules
from modules.data_loader import GEODataLoader
from modules.qc_analyzer import QCAnalyzer
from modules.normalizer import ExpressionNormalizer
from modules.utils import setup_logging, create_temp_directory
from modules.microarray_processor import MicroarrayProcessor
from modules.deg_analyzer import DEGAnalyzer

# Setup logging and create temp directory
setup_logging()
logger = logging.getLogger(__name__)
create_temp_directory()

app = Flask(__name__)

# Generate a random secret key if not set in environment
if not os.environ.get('FLASK_SECRET_KEY'):
    os.environ['FLASK_SECRET_KEY'] = os.urandom(24).hex()

app.secret_key = os.environ.get('FLASK_SECRET_KEY')

# Initialize our components
data_loader = GEODataLoader()
qc_analyzer = QCAnalyzer()
normalizer = ExpressionNormalizer()
microarray_processor = MicroarrayProcessor()
deg_analyzer = DEGAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_geo', methods=['POST'])
def load_geo():
    try:
        data = request.get_json()
        if not data or 'geo_id' not in data:
            return jsonify({'error': 'No GEO ID provided'}), 400
        
        geo_id = data['geo_id'].strip()
        data_loader.load_from_geo(geo_id)
        
        return jsonify({
            'message': f'Successfully loaded dataset: {geo_id}',
            'samples': len(data_loader.expression_data.columns),
            'features': len(data_loader.expression_data.index)
        })
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/qc')
def qc_page():
    try:
        if not data_loader.is_data_loaded:
            flash('Please load data first.', 'warning')
            return redirect(url_for('index'))
            
        # Run QC analysis automatically
        success = qc_analyzer.run_qc_analysis(data_loader.expression_data)
        
        if not success:
            flash('QC analysis failed.', 'error')
            return redirect(url_for('index'))
            
        # Redirect directly to results
        return render_template('qc_results.html',
                             metrics=qc_analyzer.qc_metrics,
                             plots=qc_analyzer.qc_plots,
                             report=qc_analyzer.qc_report)
                             
    except Exception as e:
        logger.error(f"Error in QC page: {str(e)}")
        flash('Error running QC analysis.', 'error')
        return redirect(url_for('index'))

@app.route('/run_qc', methods=['POST'])
def run_qc():
    try:
        if not data_loader.is_data_loaded:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400
            
        # Run QC analysis
        logger.info("Starting QC analysis...")
        success = qc_analyzer.run_qc_analysis(data_loader.expression_data)
        
        if not success:
            logger.error("QC analysis failed")
            return jsonify({'error': 'QC analysis failed'}), 500
            
        # Get serializable metrics and plots
        try:
            metrics = qc_analyzer.get_serializable_metrics()
            logger.info("Got serializable metrics")
        except Exception as e:
            logger.error(f"Error getting serializable metrics: {str(e)}")
            return jsonify({'error': 'Error processing metrics'}), 500
            
        try:
            plots = qc_analyzer.qc_plots
            logger.info(f"Available plots: {list(plots.keys())}")
        except Exception as e:
            logger.error(f"Error getting plots: {str(e)}")
            return jsonify({'error': 'Error processing plots'}), 500
            
        try:
            report = qc_analyzer.qc_report
            logger.info("Got QC report")
        except Exception as e:
            logger.error(f"Error getting report: {str(e)}")
            return jsonify({'error': 'Error processing report'}), 500

        # Verify data structure
        if not isinstance(metrics, dict):
            logger.error(f"Invalid metrics type: {type(metrics)}")
            return jsonify({'error': 'Invalid metrics format'}), 500
            
        if not isinstance(plots, dict):
            logger.error(f"Invalid plots type: {type(plots)}")
            return jsonify({'error': 'Invalid plots format'}), 500

        # Check for required plots
        required_plots = ['expression_dist', 'total_counts_dist', 'pca']
        missing_plots = [plot for plot in required_plots if plot not in plots]
        if missing_plots:
            logger.error(f"Missing required plots: {missing_plots}")
            return jsonify({'error': f'Missing required plots: {missing_plots}'}), 500

        # Prepare response
        response_data = {
            'metrics': metrics,
            'plots': plots,
            'report': report
        }

        # Verify response can be serialized
        try:
            jsonify(response_data)
            logger.info("Response data successfully serialized")
        except Exception as e:
            logger.error(f"Error serializing response: {str(e)}")
            return jsonify({'error': 'Error preparing response'}), 500

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error running QC: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/qc_results')
def qc_results():
    try:
        if not data_loader.is_data_loaded:
            return redirect(url_for('index'))
        if not qc_analyzer.qc_metrics:
            return redirect(url_for('qc_page'))
        return render_template('qc_results.html', 
                             metrics=qc_analyzer.qc_metrics, 
                             plots=qc_analyzer.qc_plots,
                             report=qc_analyzer.qc_report)
    except Exception as e:
        logger.error(f"Error loading QC results: {str(e)}")
        flash('Error loading QC results.', 'error')
        return redirect(url_for('index'))

@app.route('/normalization')
def normalization_page():
    if not data_loader.is_data_loaded:
        return redirect(url_for('index'))
    return render_template('normalization.html')

@app.route('/run_normalization', methods=['POST'])
def run_normalization():
    try:
        if not data_loader.is_data_loaded:
            logger.error("No data loaded for normalization")
            return jsonify({'error': 'No data loaded'}), 400

        logger.info(f"Starting normalization with data shape: {data_loader.expression_data.shape}")

        # Check if data is microarray
        logger.info("Detecting data type...")
        is_microarray = microarray_processor.detect_data_type(
            data_loader.expression_data,
            data_loader.platform_info
        )

        if not is_microarray:
            logger.error("Dataset is not from a microarray platform")
            return jsonify({
                'error': 'Current dataset is not from a microarray platform'
            }), 400

        # Run preprocessing
        logger.info(f"Starting preprocessing for {microarray_processor.platform_type} platform...")
        microarray_processor.preprocess_data()
        logger.info("Preprocessing completed")

        # Run normalization
        logger.info("Starting normalization...")
        microarray_processor.normalize_data()
        logger.info("Normalization completed")

        # Get results
        logger.info("Getting normalization results...")
        stats = microarray_processor.get_serializable_stats()
        plots = microarray_processor.plots

        # Store normalized data
        data_loader.normalized_data = microarray_processor.normalized_data.copy()
        logger.info(f"Normalized data shape: {data_loader.normalized_data.shape}")

        # Print debug information
        logger.info(f"Stats available: {list(stats.keys())}")
        logger.info(f"Plots available: {list(plots.keys())}")

        response_data = {
            'status': 'success',
            'stats': stats,
            'plots': plots,
            'platform_type': microarray_processor.platform_type,
            'data_shape': {
                'rows': len(microarray_processor.normalized_data),
                'columns': len(microarray_processor.normalized_data.columns)
            }
        }

        logger.info("Sending response back to client")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in normalization: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_samples')
def get_samples():
    try:
        if not data_loader.is_data_loaded:
            return jsonify({'error': 'No data loaded'}), 400
            
        samples = []
        for _, row in data_loader.metadata.iterrows():
            samples.append({
                'accession': row['sample_id'],
                'title': row['title'],
                'source': row['source']
            })
            
        return jsonify({'samples': samples})
        
    except Exception as e:
        logger.error(f"Error getting samples: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_normalized_data')
def download_normalized_data():
    try:
        if not hasattr(data_loader, 'normalized_data') or data_loader.normalized_data is None:
            logger.error("No normalized data available for download")
            return jsonify({'error': 'No normalized data available'}), 400

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'normalized_data_{timestamp}.csv'
        filepath = os.path.join(temp_dir, filename)

        # Save normalized data with gene IDs as index
        logger.info(f"Saving normalized data to {filepath}")
        data_loader.normalized_data.to_csv(filepath)
        
        logger.info("Sending normalized data file")
        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error downloading normalized data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_normalized_sample_data', methods=['POST'])
def get_normalized_sample_data():
    try:
        if data_loader.normalized_data is None:
            return jsonify({'error': 'No normalized data available'}), 400

        data = request.get_json()
        sample = data.get('sample')
        num_rows = int(data.get('rows', 10))

        if sample not in data_loader.normalized_data.columns:
            return jsonify({'error': 'Invalid sample selected'}), 400

        # Get the sample data
        sample_data = data_loader.normalized_data[sample].sort_values(ascending=False)
        
        # Get top N rows
        top_data = sample_data.head(num_rows)
        
        # Format data for response
        result = [
            {'gene_id': gene_id, 'expression': float(expr)}
            for gene_id, expr in top_data.items()
        ]

        return jsonify({
            'data': result,
            'total_genes': len(sample_data)
        })

    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_normalized_sample/<sample>')
def download_normalized_sample(sample):
    try:
        if data_loader.normalized_data is None:
            return jsonify({'error': 'No normalized data available'}), 400

        if sample not in data_loader.normalized_data.columns:
            return jsonify({'error': 'Invalid sample selected'}), 400

        # Create temp directory if it doesn't exist
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'normalized_sample_{sample}_{timestamp}.csv'
        filepath = os.path.join(temp_dir, filename)

        # Save sample data
        sample_data = data_loader.normalized_data[[sample]]
        sample_data.to_csv(filepath)

        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error downloading sample data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/deg_analysis')
def deg_analysis_page():
    try:
        # Detailed logging of data state
        logger.info("Starting DEG analysis page load")
        logger.info(f"Data loader state - Normalized data exists: {data_loader.normalized_data is not None}")
        logger.info(f"Data loader state - Sample info exists: {data_loader.sample_info is not None}")
        
        if data_loader.normalized_data is None:
            logger.error("No normalized data available for DEG analysis")
            flash('Please complete normalization first.', 'warning')
            return redirect(url_for('normalization_page'))

        if data_loader.sample_info is None:
            logger.error("No sample information available for DEG analysis")
            flash('Sample information is missing.', 'warning')
            return redirect(url_for('normalization_page'))

        # Log data details
        logger.info(f"Normalized data shape: {data_loader.normalized_data.shape}")
        logger.info(f"Sample info length: {len(data_loader.sample_info)}")
        logger.info(f"Normalized data columns: {list(data_loader.normalized_data.columns)}")
        logger.info(f"Sample info keys: {list(data_loader.sample_info.keys())}")

        try:
            # Attempt to load data into DEG analyzer with explicit error catching
            success = deg_analyzer.load_data(
                normalized_data=data_loader.normalized_data.copy(),  # Make a copy to prevent modifications
                sample_info=data_loader.sample_info.copy()
            )
            
            if success:
                logger.info("Data successfully loaded into DEG analyzer")
                return render_template('deg_analysis.html')
            else:
                logger.error("DEG analyzer reported failure in load_data")
                flash('Failed to initialize DEG analysis.', 'error')
                return redirect(url_for('normalization_page'))

        except Exception as load_error:
            logger.error(f"Error during DEG analyzer load_data: {str(load_error)}")
            flash(f'Error loading data into DEG analyzer: {str(load_error)}', 'error')
            return redirect(url_for('normalization_page'))

    except Exception as e:
        logger.error(f"Error in DEG analysis page: {str(e)}")
        flash('Error loading DEG analysis. Please ensure data is properly normalized.', 'error')
        return redirect(url_for('normalization_page'))

@app.route('/get_sample_info')
def get_sample_info():
    try:
        if data_loader.normalized_data is None:
            return jsonify({
                'status': 'error',
                'error': 'No normalized data available'
            }), 400

        return jsonify({
            'status': 'success',
            'samples': list(data_loader.normalized_data.columns)
        })

    except Exception as e:
        logger.error(f"Error getting sample info: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/run_deg_analysis', methods=['POST'])
def run_deg_analysis():
    try:
        data = request.get_json()
        groups = data.get('groups')
        group1 = data.get('group1')  # Control group
        group2 = data.get('group2')  # Treatment group

        if not all([groups, group1, group2]):
            return jsonify({
                'status': 'error',
                'error': 'Missing required parameters'
            }), 400

        # Update group annotations
        success = deg_analyzer.update_group_annotations(groups)
        if not success:
            return jsonify({
                'status': 'error',
                'error': 'Failed to update group annotations'
            }), 500

        # Run DEG analysis
        success = deg_analyzer.run_differential_expression(group1, group2)
        if not success:
            return jsonify({
                'status': 'error',
                'error': 'Failed to run differential expression analysis'
            }), 500

        # Get summary statistics
        summary_stats = deg_analyzer.get_summary_stats()
        if not summary_stats:
            return jsonify({
                'status': 'error',
                'error': 'Failed to generate summary statistics'
            }), 500

        return jsonify({
            'status': 'success',
            'summary': summary_stats,
            'plots': deg_analyzer.plots
        })

    except Exception as e:
        logger.error(f"Error in DEG analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/download_deg_results')
def download_deg_results():
    try:
        if deg_analyzer.results is None:
            return jsonify({'error': 'No DEG results available'}), 400

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'deg_results_{timestamp}.csv'
        filepath = os.path.join(temp_dir, filename)

        # Save results
        deg_analyzer.results.to_csv(filepath, index=False)

        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error downloading DEG results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_geo_data', methods=['POST'])
def fetch_geo_data():
    try:
        data = request.get_json()
        geo_id = data.get('geo_id')
        
        if not geo_id or not re.match(r'^GSE[0-9]+$', geo_id):
            return jsonify({'error': 'Invalid GEO ID format'}), 400

        logger.info(f"Fetching GEO dataset: {geo_id}")
        
        success = data_loader.load_from_geo(geo_id)
        
        if not success:
            return jsonify({'error': 'Failed to fetch GEO dataset'}), 400

        # Calculate QC metrics
        missing_values = data_loader.expression_data.isnull().sum().sum() / (data_loader.expression_data.shape[0] * data_loader.expression_data.shape[1]) * 100
        value_range = f"{data_loader.expression_data.min().min():.2f} - {data_loader.expression_data.max().max():.2f}"

        return jsonify({
            'status': 'success',
            'data': {
                'title': data_loader.platform_info.get('title', 'Not available'),
                'platform': data_loader.platform_info.get('technology', 'Not available'),
                'sample_count': len(data_loader.metadata) if data_loader.metadata is not None else 0,
                'organism': data_loader.platform_info.get('organism', 'Not available'),
                'samples': data_loader.metadata.to_dict('records'),
                'qc': {
                    'missing_values': f"{missing_values:.2f}",
                    'value_range': value_range,
                    'distribution': 'Normal'  # You might want to calculate this based on your data
                }
            }
        })

    except Exception as e:
        logger.error(f"Error fetching GEO data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/normalize_data', methods=['POST'])
def normalize_data():
    try:
        method = request.json.get('method', 'quantile')
        logger.info(f"Starting normalization with method: {method}")

        # Perform normalization
        normalized_data = normalizer.normalize(data_loader.expression_data, method=method)
        
        # Save normalized data to data_loader
        data_loader.normalized_data = normalized_data
        
        # Log success
        logger.info("Normalization completed successfully")
        logger.info(f"Normalized data shape: {normalized_data.shape}")

        return jsonify({
            'status': 'success',
            'message': 'Data normalized successfully'
        })

    except Exception as e:
        logger.error(f"Error during normalization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
