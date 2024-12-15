import os

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_platform_config():
    if is_colab():
        return {
            'host': '0.0.0.0',  # Allow external connections
            'port': 8888,       # Colab-friendly port
            'debug': True
        }
    elif os.environ.get('RENDER'):  # For Render
        return {
            'host': '0.0.0.0',
            'port': int(os.environ.get('PORT', 10000)),
            'debug': False
        }
    else:  # Local development
        return {
            'host': '0.0.0.0',
            'port': 5001,
            'debug': True
        } 