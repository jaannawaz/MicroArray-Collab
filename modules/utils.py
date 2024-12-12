import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

def create_temp_directory():
    """Create temporary directory for storing files"""
    try:
        directory = 'temp'
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created temporary directory: {directory}")
        return True
    except Exception as e:
        logging.error(f"Error creating temporary directory: {str(e)}")
        return False

def generate_timestamp():
    """Generate current timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_to_csv(data, filename, directory='temp'):
    """Save DataFrame to CSV file"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def load_from_csv(filepath):
    """Load DataFrame from CSV file"""
    try:
        data = pd.read_csv(filepath, index_col=0)
        logger.info(f"Loaded data from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise 