# NYC Taxi Data Analysis Project

This project provides a comprehensive analysis of New York City Taxi data. It processes raw data files, generates visualizations, and performs feature analysis to extract insights and build predictive models. The project includes modules for data processing, cleaning, normalization, exploratory data analysis, visualization, and feature engineering.

## Project Structure

- **data/**  
  Contains raw data files (e.g., CSVs, zipped folders) and processed data organized by year.
  
- **figures/**  
  Stores generated plots and PDFs from various analyses (e.g., correlation plots, PCA charts, distributions).
  
- **notebooks/**  
  Contains Jupyter notebooks such as [part-1.ipynb](notebooks/part-1.ipynb) for interactive exploration and analysis.
  
- **scripts/**  
  Modular Python scripts with the core functionality:
  - [`DataManipulator`, `DataNormalizer`, `DataCleaner`](scripts/data_processor.py) for processing and cleaning data.
  - [`EDA`, `DataVisualizer`](scripts/data_visualizer.py) for exploratory data analysis and visualization.
  - [`FeatureAnalysisAndGenerator`](scripts/feature_analyzer.py) for feature analysis and generation.
  
- **report/**  
  Contains generated reports, including the final analysis report in PDF format.
  
- **Miscellaneous files**  
  Include project initialization files like `__init__.py`, Git configuration, and others.

## Usage

- Run notebooks from the [notebooks](notebooks/) folder for interactive analysis.
- Use the scripts in the [scripts](scripts/) folder for batch processing or integration into other workflows.
- Example usage in a script:

    ```python
    from scripts.data_processor import DataManipulator
    dm = DataManipulator(...)
    dm.process()
    ```