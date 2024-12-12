# Microarray Analysis Tool

## Overview
The Microarray Analysis Tool is a web-based application designed for comprehensive analysis of GEO microarray data. It provides functionalities for data retrieval, quality control (QC), normalization, and differential expression analysis (DEA).

## Features
- Load GEO datasets and visualize metadata.
- Perform quality control analysis on selected samples.
- Normalize data using various methods.
- Conduct differential expression analysis and visualize results.

## Technologies Used
- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS, JavaScript (Bootstrap)

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/microarray-analysis-tool.git
   cd microarray-analysis-tool
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage
1. **Data Retrieval**: Enter the GEO accession number to load the dataset.
2. **Quality Control**: Select samples and run QC analysis to visualize metrics.
3. **Normalization**: Choose a normalization method and process the data.
4. **Differential Expression Analysis**: Select control and treatment groups to analyze DEGs.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- GEOparse for accessing GEO datasets.
- Flask for the web framework.
- Pandas, NumPy, and SciPy for data manipulation and analysis.