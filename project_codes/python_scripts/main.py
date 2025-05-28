import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pickle
import datetime
import sqlite3
from scipy.stats import mode
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree # For kNN optimization
from sklearn.preprocessing import MinMaxScaler

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Sklearn Models
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, BaggingRegressor, BaggingClassifier

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis") # Setting a nice color palette

from data_manipulator import DataManipulator
from data_cleaner import DataCleaner
from eda import TaxiEDA
from hypothesis_tester import TaxiHypothesisTester
from feature_engineer import TaxiFeatureEngineer
from data_normalizer import DataNormalizer
from base_model import BaseModel
from custom_knn import CustomKNN
from sklearn_model import SklearnModel
from deep_learning_model import TFDeepLearningModel

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# For clustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap



# Load and combine data from multiple SQLite files
files = ['2019-01.sqlite', '2019-02.sqlite', '2019-03.sqlite', '2019-04.sqlite', '2019-05.sqlite', '2019-06.sqlite']
data_loader = DataManipulator(filename='../../data/2019/'+files[0], labels_test='fare_amount', task='regression')
for i in range(1, len(files)):
    temp_loader = DataManipulator(filename='../../data/2019/'+files[i], labels_test='fare_amount', task='regression')
    data_loader.data_train = pd.concat([data_loader.data_train, temp_loader.data_train], ignore_index=True)
    data_loader.labels_train = pd.concat([data_loader.labels_train, temp_loader.labels_train], ignore_index=True)
    data_loader.data_test = pd.concat([data_loader.data_test, temp_loader.data_test], ignore_index=True)
    data_loader.labels_test = pd.concat([data_loader.labels_test, temp_loader.labels_test], ignore_index=True)
del temp_loader


# Clean and preprocess the data
print("Before data cleaning")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)

# Data cleaning
data_cleaner = DataCleaner(data_loader)
data_cleaner.remove_duplicates()
data_cleaner.handle_missing_values()
data_cleaner.remove_outliers()

print("After data preprocessing")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)


# Initialize EDA object
eda = TaxiEDA(data_loader)

# Perform EDA visualizations
print("\n--- Performing EDA ---")
eda.plot_target_distribution()

numerical_features_for_eda = ['trip_distance', 'trip_duration_secs', 'passenger_count', 'tolls_amount']
categorical_features_for_eda = ['vendorid', 'ratecodeid', 'payment_type', 'store_and_fwd_flag']

eda.plot_numerical_distributions(numerical_features_for_eda)
eda.plot_categorical_distributions(categorical_features_for_eda) # Note: PULoc/DOLoc can be high cardinality

eda.plot_numerical_vs_target(numerical_features_for_eda)
eda.plot_categorical_vs_target(categorical_features_for_eda)

eda.plot_correlation_heatmap()

# For dimension reduction, select relevant numerical features (avoiding target)
# The selection of features should be careful. Using a subset of what's available.
features_for_dim_reduction = ['trip_distance', 'trip_distance_secs', 'tolls_amount']
# Check if these columns exist before passing
existing_features_for_dim_reduction = [col for col in features_for_dim_reduction if col in data_loader.data_train.columns]

if existing_features_for_dim_reduction:
    eda.perform_dimension_reduction_viz(existing_features_for_dim_reduction, target_for_color='ratecodeid')
else:
    print("Not enough features available for dimension reduction visualization after checks.")
    
    
# Initialize tester and run tests
tester = TaxiHypothesisTester(data_loader)
if 'trip_distance' in data_loader.data_train.columns:
    tester.test_distance_vs_fare()
if 'ratecodeid' in data_loader.data_train.columns:
    tester.test_fare_by_ratecode()
    
    
# Apply feature engineering
feature_engineer = TaxiFeatureEngineer(data_loader)
feature_engineer.create_datetime_features()
feature_engineer.create_trip_features()
feature_engineer.create_cyclical_time_features()

print("\n--- Data with Engineered Features ---")
print(feature_engineer.data_loader.data_train.head())
feature_engineer.data_loader.data_train.info()

# Check correlation after feature engineering
eda.plot_correlation_heatmap()

# Selecting relevant features for modeling
numerical_features = ['trip_distance', 'trip_duration_secs', 'tolls_amount', 'average_speed_mph']
categorical_features = ['ratecodeid', 'is_airport_trip']

data_loader.data_train = data_loader.data_train[numerical_features + categorical_features]
data_loader.data_test = data_loader.data_test[numerical_features + categorical_features]

# Normalizing the data
data_normalizer = DataNormalizer(data_loader, categorical_features, numerical_features)
data_normalizer.normalize_features()
data_normalizer.data_loader.data_train.head()

# Save the processed data loader
with open('../../data/processed/final_normalised_data_loader.pkl', 'wb') as f:
    pickle.dump(data_loader, f)
    
    
## Part 2: Model Training and Evaluation

# Define paths
RESULTS_DIR = "../../output_files/predictions"
EVALUATIONS_DIR = "../../output_files/evaluations"
MODELS_DIR = "../../output_files/models"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVALUATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global list for all final evaluations (test set)
all_evaluations = []
# Global list for all per-fold evaluations during CV
all_fold_evaluations_list = []

# Define number of CV splits (for manual CV and GridSearchCV)
N_CV_SPLITS = 5

# --- Helper function to run a model, evaluate, and save ---
def run_experiment(model_instance, X_train_data, y_train_data, X_test_data, y_test_data, target_scaler_reg=None, epochs_cv=5, epochs_final=10):
    global all_fold_evaluations_list
    print(f"\n{'='*50}")
    print(f"Running: {model_instance.model_name} ({model_instance.task_type})")
    print(f"{'='*50}")

    try:
        # Convert to numpy arrays for consistency
        X_train_np = X_train_data.values if hasattr(X_train_data, 'values') else np.array(X_train_data)
        X_test_np = X_test_data[X_train_data.columns].values if hasattr(X_test_data, 'values') else np.array(X_test_data)
        y_train_np = y_train_data.values if hasattr(y_train_data, 'values') else np.array(y_train_data)
        y_test_np = y_test_data.values if hasattr(y_test_data, 'values') else np.array(y_test_data)

        # Ensure same number of features
        if X_train_np.shape[1] != X_test_np.shape[1]:
            min_features = min(X_train_np.shape[1], X_test_np.shape[1])
            X_train_np = X_train_np[:, :min_features]
            X_test_np = X_test_np[:, :min_features]

        # Fit model
        if isinstance(model_instance, TFDeepLearningModel):
            model_instance.fit(X_train_np, y_train_np, epochs_cv=epochs_cv, epochs_final=epochs_final)
        else:
            model_instance.fit(X_train_np, y_train_np)

        # Save fold evaluations immediately after fitting
        model_instance.save_fold_evaluations()

        # Collect fold evaluations for global summary (optional)
        if model_instance.fold_evaluations:
            all_fold_evaluations_list.extend(model_instance.fold_evaluations)

        # Check if model is ready for prediction
        model_ready = False
        if isinstance(model_instance, CustomKNN):
            model_ready = (model_instance.k is not None and model_instance.k > 0)
        else:
            model_ready = (model_instance.model is not None)

        if not model_ready:
            print(f"Model {model_instance.model_name} was not successfully fitted. Skipping evaluation.")
            return model_instance

        # Make predictions
        y_pred = model_instance.predict(X_test_np)

        # For regression, inverse transform predictions if scaler provided
        if model_instance.task_type == 'regression' and target_scaler_reg is not None:
            y_pred = target_scaler_reg.inverse_transform(y_pred.reshape(-1, 1))

        # Evaluate and save
        model_instance.evaluate(y_test_np, y_pred)
        model_instance.save_results(y_test_np, y_pred)
        model_instance.save_model()

        print(f"✓ {model_instance.model_name} completed successfully")

    except Exception as e:
        print(f"✗ Error in {model_instance.model_name}: {e}")

    return model_instance

# Load Data
print("Loading data...")
try:
    with open('../../data/processed/final_normalised_data_loader.pkl', 'rb') as f:
        data_loader = pickle.load(f)

    print("✓ Data loaded successfully")
    print(f"Training data shape: {data_loader.data_train.shape}")
    print(f"Test data shape: {data_loader.data_test.shape}")
    print(f"Training labels shape: {data_loader.labels_train.shape}")
    print(f"Test labels shape: {data_loader.labels_test.shape}")

except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Use a small subset of the data for Machine Learning models
X_train = data_loader.data_train[:int(0.03 * len(data_loader.data_train))]
y_train_original = data_loader.labels_train[X_train.index]
X_test = data_loader.data_test[:int(0.03 * len(data_loader.data_test))]
y_test_original = data_loader.labels_test[X_test.index]

# # Prepare features
# X_train = data_loader.data_train
# X_test = data_loader.data_test

# y_train_original = data_loader.labels_train.copy()
# y_test_original = data_loader.labels_test.copy()

print(f"\nOriginal label stats:")
print(f"Train labels - min: {y_train_original.min():.2f}, max: {y_train_original.max():.2f}, mean: {y_train_original.mean():.2f}")
print(f"Test labels - min: {y_test_original.min():.2f}, max: {y_test_original.max():.2f}, mean: {y_test_original.mean():.2f}")

# === REGRESSION SETUP ===
# Scale target for training (will inverse transform for evaluation)
target_scaler = MinMaxScaler()
y_train_reg_scaled = target_scaler.fit_transform(y_train_original.values.reshape(-1, 1)).ravel()
y_test_reg = y_test_original.copy()  # Keep original scale for evaluation

print(f"\nRegression setup:")
print(f"Scaled train labels - min: {y_train_reg_scaled.min():.4f}, max: {y_train_reg_scaled.max():.4f}")

print("Checking for data leakage...")
train_indices = set(X_train.index)
test_indices = set(X_test.index)
overlap = train_indices.intersection(test_indices)
print(f"Overlapping indices between train and test: {len(overlap)}")
if len(overlap) > 0:
    print("WARNING: Data leakage detected!")
else:
    print("✓ No data leakage detected")
    
# === GOAL 1: REGRESSION (with CV and Fold Metrics) ===
print("\n" + "="*30 + " GOAL 1: REGRESSION (CV & Fold Metrics) " + "="*30 + "\n")

# 1. Custom kNN Regressor
knn_reg = CustomKNN(
    model_name="KNNRegressor",
    task_type="regression",
    k_options=[3, 5, 7, 9]
)
run_experiment(knn_reg, X_train, y_train_reg_scaled, X_test, y_test_reg, target_scaler_reg=target_scaler)

# 2. Decision Tree Regressor - replacing Linear Regression
dt_reg_param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [5, 10, 20],
    'criterion': ['squared_error'],
    'max_features': ['sqrt', 0.5, 0.7]
}
dt_reg_model = SklearnModel(
    model_name="DecisionTreeRegressor",
    task_type="regression",
    sklearn_model_instance=DecisionTreeRegressor(random_state=42),
    param_grid=dt_reg_param_grid
)
run_experiment(dt_reg_model, X_train, y_train_reg_scaled, X_test, y_test_reg, target_scaler_reg=target_scaler)

# 3. Bagging Regressor
bagging_reg_param_grid = {
    'n_estimators': [50, 100],
    'max_samples': [0.5, 0.7],
    'max_features': [0.5, 0.7],
    'bootstrap': [True],
    'bootstrap_features': [True]
}
bagging_reg = SklearnModel(
    model_name="BaggingRegressor",
    task_type="regression",
    sklearn_model_instance=BaggingRegressor(random_state=42, n_jobs=-1),
    param_grid=bagging_reg_param_grid
)
run_experiment(bagging_reg, X_train, y_train_reg_scaled, X_test, y_test_reg, target_scaler_reg=target_scaler)

# 4. Gradient Boosting Regressor
gb_reg_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3],
    'subsample': [0.8, 0.9],
    'max_features': ['sqrt', 0.5]
}
gb_reg = SklearnModel(
    model_name="GradientBoostingRegressor",
    task_type="regression",
    sklearn_model_instance=GradientBoostingRegressor(random_state=42),
    param_grid=gb_reg_param_grid
)
run_experiment(gb_reg, X_train, y_train_reg_scaled, X_test, y_test_reg, target_scaler_reg=target_scaler)


# Increase data for Deep Learning models
X_train = data_loader.data_train[:int(0.16 * len(data_loader.data_train))]
y_train_original = data_loader.labels_train[X_train.index]
X_test = data_loader.data_test[:int(0.16 * len(data_loader.data_test))]
y_test_original = data_loader.labels_test[X_test.index]

target_scaler = MinMaxScaler()
y_train_reg_scaled = target_scaler.fit_transform(y_train_original.values.reshape(-1, 1)).ravel()
y_test_reg = y_test_original.copy()


# 5. MLP Regressor
mlp_param_grid = {
    'hidden_layer_sizes': [(30,), (50,)],
    'learning_rate_init': [0.001, 0.01],
    'activation': ['relu'],
    'alpha': [0.01, 0.1, 1.0],
    'early_stopping': [True],
    'validation_fraction': [0.2],
    'n_iter_no_change': [10]
}
mlp_model = SklearnModel(
    model_name="MLPRegressor",
    task_type="regression",
    sklearn_model_instance=MLPRegressor(random_state=42, early_stopping=True),
    param_grid=mlp_param_grid
)
run_experiment(mlp_model, X_train, y_train_reg_scaled, X_test, y_test_reg, target_scaler_reg=target_scaler)


# 6. Deep Learning Regressor
input_dim_for_tf = X_train.shape[1] if hasattr(X_train, 'shape') else None
dl_reg_tune_params = {
    'units_layer1': [32, 64],
    'batch_size': [32, 64],
    'learning_rate': [0.0001],
    'dropout_rate': [0.3, 0.5, 0.7]
}
dl_reg = TFDeepLearningModel(
    model_name="DeepLearningRegressor",
    task_type="regression",
    input_dim=input_dim_for_tf,
    tune_params=dl_reg_tune_params
)
run_experiment(dl_reg, X_train, y_train_reg_scaled, X_test, y_test_reg, target_scaler_reg=target_scaler, epochs_cv=5, epochs_final=10)