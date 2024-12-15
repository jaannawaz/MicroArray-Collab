# This file is specifically for running in Google Colab
import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

if __name__ == '__main__':
    install_requirements()
    from app import app
    from utils.platform_config import get_platform_config

    config = get_platform_config()
    app.run(
        host=config['host'],
        port=config['port'],
        debug=config['debug']
    ) 