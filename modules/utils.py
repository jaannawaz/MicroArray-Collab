import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_temp_directory():
    """Create temporary directory for file storage"""
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

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
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def load_from_csv(filepath):
    """Load DataFrame from CSV file"""
    try:
        return pd.read_csv(filepath, index_col=0)
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise 