# NYC Taxi Data Analysis & Machine Learning Project

This project provides a comprehensive analysis of New York City Taxi data with advanced machine learning capabilities. It processes raw taxi trip data, performs exploratory data analysis, builds predictive models for fare prediction and trip classification, and provides an interactive dashboard for visualizing results.

## 🎯 Project Overview

The NYC Taxi Data Analysis project includes:

- **Data Processing & Cleaning**: Automated data preprocessing pipeline for NYC taxi trip records
- **Exploratory Data Analysis**: Comprehensive statistical analysis and visualization of taxi patterns
- **Feature Engineering**: Advanced feature creation including temporal, spatial, and derived features
- **Machine Learning Models**: Multiple algorithms for both regression (fare prediction) and classification (trip categorization)
- **Model Evaluation**: Cross-validation, hyperparameter tuning, and comprehensive performance metrics
- **Interactive Dashboard**: Streamlit-based web application for visualizing model results and comparisons
- **Clustering Analysis**: Unsupervised learning to discover trip patterns and customer segments

## 📁 Project Structure

```
ny-taxi-project/
├── data/                                    # Raw and processed data files
│   ├── data_dictionary.pdf                 # Data schema documentation
│   ├── 2019/                              # Raw data by year
│   │   ├── 2019-01.sqlite                 # Monthly taxi trip data
│   │   └── ...
│   └── processed/                          # Processed and normalized data
│       ├── final_normalised_data_loader.pkl     # Preprocessed dataset
│       └── ...
├── output_files/                           # Generated outputs and results
│   ├── evaluations/                        # Model performance metrics
│   │   ├── *_evaluation.csv               # Final test metrics per model
│   │   ├── *_fold_evaluations.csv         # Cross-validation results
│   ├── figures/                            # Generated plots and visualizations
│   │   ├── clustering/                     # Clustering analysis plots
│   │   └── ...
│   ├── models/                             # Saved trained models
│   │   ├── *.joblib                       # Scikit-learn models
│   │   ├── *.keras                        # TensorFlow models
│   │   └── *_best_params.joblib           # Optimal hyperparameters
│   └── predictions/                        # Model predictions
│       ├── *_results.csv                  # Actual vs predicted values
├── project_codes/                          # Source code and notebooks
│   ├── jupyter_notebooks/                 # Additional analysis notebooks
│   │   ├── part-1.ipynb                   # Data exploration & EDA
│   │   └── part_2_new.ipynb              # ML pipeline notebook
│   ├── python_scripts/                    # Modular Python scripts
│   │   └── ...
│   │   └── dashboard/                         # Streamlit dashboard
│           └── dashboard.py                   # Interactive results viewer
│   ├── r_scripts/                         # R analysis scripts
│   │   └── ...
├── project_plans/                         # Project documentation
├── project_reports/                       # Generated analysis reports
│   └── ...                               
├── README.md                             # This file
├── .gitignore                            # Git ignore rules
├── .gitattributes                        # Git LFS configuration
└── __init__.py                           # Package initialization
```

## 🚀 Features

### Data Processing
- **Automated ETL Pipeline**: Processes raw NYC taxi trip data from SQLite databases
- **Data Cleaning**: Handles missing values, outliers, and data quality issues
- **Feature Engineering**: Creates temporal, spatial, and derived features
- **Data Normalization**: Standardizes features for machine learning

### Machine Learning Models
- **Regression Models** (Fare Prediction):
  - Custom k-Nearest Neighbors (from scratch)
  - Decision Tree Regressor
  - Multi-Layer Perceptron (MLP)
  - Bagging Regressor
  - Gradient Boosting Regressor
  - Deep Learning (TensorFlow/Keras)

- **Classification Models** (Trip Categorization):
  - Custom k-NN Classifier
  - Decision Tree Classifier
  - MLP Classifier
  - Bagging Classifier
  - Gradient Boosting Classifier
  - Deep Learning Classifier

### Advanced Analytics
- **Hyperparameter Optimization**: GridSearchCV for optimal model performance
- **Cross-Validation**: k-fold validation with detailed fold-by-fold metrics
- **Clustering Analysis**: K-Means and DBSCAN for pattern discovery
- **Model Comparison**: Comprehensive performance benchmarking

### Evaluation Metrics
- **Regression**: MAE, MSE, RMSE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: Fold consistency analysis and variance detection

## 🛠️ Installation & Setup

### Data Link
The data used for this project is available at: https://drive.google.com/drive/folders/1Jwqozdh2k3Q8qAcVN3akaEHiwtb8HBmJ?usp=sharing

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ny-taxi-project
```

## 📊 Usage

### 1. Data Processing & Model Training
Run the main machine learning pipeline:
```bash
# Run Jupyter notebooks
jupyter notebook part-1.ipynb
jupyter notebook part-2.ipynb

# Navigate to project codes
cd project_codes/python_scripts
python main.py
```

### 2. Start the Interactive Dashboard
```bash
# From the project root directory
cd project_codes/python_scripts/dashboard
streamlit run dashboard.py
```

### 3. Dashboard Features

The Streamlit dashboard provides three main sections:

#### 📈 Individual Model Analysis
- Detailed performance metrics for each model
- Actual vs Predicted scatter plots
- Residual analysis for regression models
- Error distribution visualizations
- Cross-validation fold details

#### 🔄 Model Comparison
- Side-by-side performance comparison
- Best model identification by metric
- Visual comparison charts
- Separate analysis for regression and classification

#### 📉 Cross-Validation Analysis
- Fold-by-fold performance tracking
- CV consistency analysis
- Variance detection for overfitting
- Summary statistics across folds

### 4. Accessing Results

After running the ML pipeline, results are automatically saved to:
- **Model Predictions**: `output_files/predictions/`
- **Evaluation Metrics**: `output_files/evaluations/`
- **Trained Models**: `output_files/models/`
- **Visualizations**: `output_files/figures/`

## 🔍 Access Dashboard

**Access Dashboard**:
   - Open browser to `http://localhost:8501`
   - Navigate through different analysis sections
   - Compare model performances
   - Analyze cross-validation results

### 📈 Model Performance Insights

The dashboard automatically displays:
- **Best Performing Models** by metric (MAE, RMSE, R², Accuracy)
- **Cross-Validation Stability** indicators
- **Overfitting Detection** through CV variance analysis
- **Prediction Quality** visualizations
- **Model Comparison** charts and tables

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:
1. Check the dashboard for model performance insights
2. Review the Jupyter notebooks for detailed analysis
3. Examine the output files for raw results
4. Open an issue for bugs or feature requests

---

**Quick Start**: After installation, run `cd project_codes/dashboard && streamlit run dashboard.py` to immediately view your model results in an interactive web interface! 🚀