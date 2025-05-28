import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import os
import glob

# Set page configuration
st.set_page_config(
    page_title="ML Model Predictions Dashboard",
    page_icon="🧮",
    layout="wide"
)

# Define helper functions
def load_model_predictions(file_path):
    """Load predictions from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None

def load_model_evaluation(file_path):
    """Load model evaluation from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading evaluation file {file_path}: {e}")
        return None

def calculate_metrics(actual, predicted, task_type='regression'):
    """Calculate metrics based on task type"""
    if task_type == 'regression':
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE (%)': mape
        }
    else:  # classification
        accuracy = accuracy_score(actual, predicted)
        return {
            'Accuracy': accuracy
        }

def plot_actual_vs_predicted(actual, predicted, model_name, task_type='regression'):
    """Generate scatter plot of actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if task_type == 'regression':
        sns.scatterplot(x=actual, y=predicted, alpha=0.6)
        
        # Add diagonal line (perfect predictions)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
    else:  # classification
        # Confusion matrix style plot
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual, predicted)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.grid(True, alpha=0.3)
    return fig

def plot_residuals(actual, predicted, model_name):
    """Generate residual plot (only for regression)"""
    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    return fig

def plot_error_distribution(actual, predicted, model_name, task_type='regression'):
    """Plot error distribution histogram"""
    if task_type == 'regression':
        errors = actual - predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=30)
        plt.title(f'Error Distribution - {model_name}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    else:
        # For classification, show prediction distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        unique_classes = sorted(list(set(actual.unique()) | set(predicted.unique())))
        x_pos = np.arange(len(unique_classes))
        
        actual_counts = [sum(actual == c) for c in unique_classes]
        predicted_counts = [sum(predicted == c) for c in unique_classes]
        
        width = 0.35
        plt.bar(x_pos - width/2, actual_counts, width, label='Actual', alpha=0.7)
        plt.bar(x_pos + width/2, predicted_counts, width, label='Predicted', alpha=0.7)
        
        plt.xticks(x_pos, unique_classes)
        plt.title(f'Class Distribution - {model_name}')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    return fig

def plot_model_comparison(metrics_dict, task_type='regression'):
    """Plot bar charts comparing models"""
    metrics_df = pd.DataFrame(metrics_dict).T
    
    if task_type == 'regression':
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot MSE
        axes[0].bar(metrics_df.index, metrics_df['MSE'], color='skyblue')
        axes[0].set_title('Mean Squared Error (MSE)')
        axes[0].set_ylabel('MSE')
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # Plot RMSE
        axes[1].bar(metrics_df.index, metrics_df['RMSE'], color='lightgreen')
        axes[1].set_title('Root Mean Squared Error (RMSE)')
        axes[1].set_ylabel('RMSE')
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        # Plot R²
        axes[2].bar(metrics_df.index, metrics_df['R²'], color='coral')
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('R²')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
        
        # Plot MAE
        axes[3].bar(metrics_df.index, metrics_df['MAE'], color='plum')
        axes[3].set_title('Mean Absolute Error (MAE)')
        axes[3].set_ylabel('MAE')
        plt.setp(axes[3].get_xticklabels(), rotation=45, ha='right')
    else:
        # For classification, just plot accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics_df.index, metrics_df['Accuracy'], color='lightblue')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def get_available_models():
    """Get available models from predictions and evaluations folders"""
    # Get paths to the folders
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    predictions_dir = os.path.join(base_dir, "../output_files", "predictions")
    evaluations_dir = os.path.join(base_dir, "../output_files", "evaluations")
    # print(f"Predictions Directory: {predictions_dir}")
    # print(f"Evaluations Directory: {evaluations_dir}")

    models = {}
    
    # Find prediction files
    if os.path.exists(predictions_dir):
        prediction_files = glob.glob(os.path.join(predictions_dir, "*_results.csv"))
        
        for file_path in prediction_files:
            filename = os.path.basename(file_path)
            # Extract model name and task type from filename
            # Format: ModelName_tasktype_results.csv
            parts = filename.replace('_results.csv', '').split('_')
            if len(parts) >= 2:
                model_name = '_'.join(parts[:-1])
                task_type = parts[-1]
                
                if model_name not in models:
                    models[model_name] = {}
                
                models[model_name]['predictions_file'] = file_path
                models[model_name]['task_type'] = task_type
    
    # Find evaluation files
    if os.path.exists(evaluations_dir):
        eval_files = glob.glob(os.path.join(evaluations_dir, "*_evaluation.csv"))
        fold_eval_files = glob.glob(os.path.join(evaluations_dir, "*_fold_evaluations.csv"))
        
        for file_path in eval_files:
            filename = os.path.basename(file_path)
            parts = filename.replace('_evaluation.csv', '').split('_')
            if len(parts) >= 2:
                model_name = '_'.join(parts[:-1])
                task_type = parts[-1]
                
                if model_name not in models:
                    models[model_name] = {}
                    models[model_name]['task_type'] = task_type
                
                models[model_name]['evaluation_file'] = file_path
        
        for file_path in fold_eval_files:
            filename = os.path.basename(file_path)
            parts = filename.replace('_fold_evaluations.csv', '').split('_')
            if len(parts) >= 2:
                model_name = '_'.join(parts[:-1])
                task_type = parts[-1]
                
                if model_name not in models:
                    models[model_name] = {}
                    models[model_name]['task_type'] = task_type
                
                models[model_name]['fold_evaluation_file'] = file_path
    
    return models

def display_fold_evaluations(fold_eval_df, model_name):
    """Display fold evaluation statistics"""
    st.write(f"#### Cross-Validation Results - {model_name}")
    
    if not fold_eval_df.empty:
        # Display raw fold data
        st.write("**Individual Fold Results:**")
        st.dataframe(fold_eval_df)
        
        # Calculate and display summary statistics
        st.write("**Summary Statistics Across Folds:**")
        numeric_cols = fold_eval_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['fold_id']]
        
        if len(numeric_cols) > 0:
            summary_stats = fold_eval_df[numeric_cols].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(summary_stats)
            
            # Plot fold variation
            if len(fold_eval_df) > 1:
                st.write("**Metric Variation Across Folds:**")
                fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
                if len(numeric_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(numeric_cols):
                    axes[i].plot(fold_eval_df['fold_id'], fold_eval_df[col], 'o-')
                    axes[i].set_title(f'{col} Across Folds')
                    axes[i].set_xlabel('Fold')
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

# Main app
def main():
    st.title("🧮 ML Model Predictions Dashboard")
    st.markdown("""
    This dashboard visualizes and compares predictions from different machine learning models.
    It displays both final test predictions and cross-validation results from your model training.
    """)
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("No model files found. Please ensure prediction and evaluation files are in the correct directories:")
        st.write("- Predictions: `../../../output_files/predictions/`")
        st.write("- Evaluations: `../../../output_files/evaluations/`")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", [
        "Individual Model Analysis", 
        "Model Comparison", 
        "Cross-Validation Analysis"
    ])
    
    if page == "Individual Model Analysis":
        st.header("📊 Individual Model Analysis")
        
        # Select model
        model_names = list(available_models.keys())
        selected_model = st.sidebar.selectbox("Select Model", model_names)
        
        model_info = available_models[selected_model]
        task_type = model_info.get('task_type', 'regression')
        
        # Display model information
        st.write(f"### {selected_model} ({task_type.title()})")
        
        # Load and display predictions
        if 'predictions_file' in model_info:
            st.write("#### Test Set Predictions")
            predictions_df = load_model_predictions(model_info['predictions_file'])
            
            if predictions_df is not None:
                st.write("**Sample Predictions:**")
                st.dataframe(predictions_df.head(10))
                
                if 'actual' in predictions_df.columns and 'predicted' in predictions_df.columns:
                    actual = predictions_df['actual']
                    predicted = predictions_df['predicted']
                    
                    # Calculate and display metrics
                    metrics = calculate_metrics(actual, predicted, task_type)
                    st.write("**Test Set Performance Metrics:**")
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df.style.format("{:.4f}"))
                    
                    # Display plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Actual vs Predicted**")
                        fig1 = plot_actual_vs_predicted(actual, predicted, selected_model, task_type)
                        st.pyplot(fig1)
                    
                    with col2:
                        st.write("**Error/Distribution Analysis**")
                        fig2 = plot_error_distribution(actual, predicted, selected_model, task_type)
                        st.pyplot(fig2)
                    
                    # For regression, add residual plot
                    if task_type == 'regression':
                        st.write("**Residual Analysis**")
                        fig3 = plot_residuals(actual, predicted, selected_model)
                        st.pyplot(fig3)
                else:
                    st.error("Prediction file doesn't have required 'actual' and 'predicted' columns.")
        else:
            st.warning("No prediction file found for this model.")
        
        # Load and display evaluation metrics
        if 'evaluation_file' in model_info:
            st.write("#### Final Evaluation Metrics")
            eval_df = load_model_evaluation(model_info['evaluation_file'])
            if eval_df is not None:
                st.dataframe(eval_df.style.format("{:.4f}", subset=eval_df.select_dtypes(include=[np.number]).columns))
        
        # Load and display fold evaluations
        if 'fold_evaluation_file' in model_info:
            fold_eval_df = load_model_evaluation(model_info['fold_evaluation_file'])
            if fold_eval_df is not None:
                display_fold_evaluations(fold_eval_df, selected_model)
    
    elif page == "Model Comparison":
        st.header("🔄 Model Comparison")
        
        # Separate models by task type
        regression_models = {k: v for k, v in available_models.items() if v.get('task_type') == 'regression'}
        classification_models = {k: v for k, v in available_models.items() if v.get('task_type') == 'classification'}
        
        # Regression models comparison
        if regression_models:
            st.write("### Regression Models Comparison")
            
            reg_metrics = {}
            reg_evaluations = []
            
            for model_name, model_info in regression_models.items():
                # Load evaluation metrics
                if 'evaluation_file' in model_info:
                    eval_df = load_model_evaluation(model_info['evaluation_file'])
                    if eval_df is not None and not eval_df.empty:
                        reg_evaluations.append(eval_df)
                
                # Load predictions to calculate additional metrics
                if 'predictions_file' in model_info:
                    pred_df = load_model_predictions(model_info['predictions_file'])
                    if pred_df is not None and 'actual' in pred_df.columns and 'predicted' in pred_df.columns:
                        metrics = calculate_metrics(pred_df['actual'], pred_df['predicted'], 'regression')
                        reg_metrics[model_name] = metrics
            
            if reg_evaluations:
                st.write("**Saved Evaluation Metrics:**")
                combined_eval = pd.concat(reg_evaluations, ignore_index=True)
                st.dataframe(combined_eval.style.format("{:.4f}", subset=combined_eval.select_dtypes(include=[np.number]).columns))
            
            if reg_metrics:
                st.write("**Calculated Metrics from Predictions:**")
                reg_metrics_df = pd.DataFrame.from_dict(reg_metrics, orient='index')
                st.dataframe(reg_metrics_df.style.format("{:.4f}"))
                
                # Visual comparison
                st.write("**Visual Comparison:**")
                fig = plot_model_comparison(reg_metrics, 'regression')
                st.pyplot(fig)
                
                # Best model analysis
                st.write("**Best Performing Models:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_mae = reg_metrics_df['MAE'].idxmin()
                    st.metric("Best MAE", best_mae, f"{reg_metrics_df.loc[best_mae, 'MAE']:.4f}")
                with col2:
                    best_rmse = reg_metrics_df['RMSE'].idxmin()
                    st.metric("Best RMSE", best_rmse, f"{reg_metrics_df.loc[best_rmse, 'RMSE']:.4f}")
                with col3:
                    best_r2 = reg_metrics_df['R²'].idxmax()
                    st.metric("Best R²", best_r2, f"{reg_metrics_df.loc[best_r2, 'R²']:.4f}")
        
        # Classification models comparison
        if classification_models:
            st.write("### Classification Models Comparison")
            
            clf_metrics = {}
            clf_evaluations = []
            
            for model_name, model_info in classification_models.items():
                # Load evaluation metrics
                if 'evaluation_file' in model_info:
                    eval_df = load_model_evaluation(model_info['evaluation_file'])
                    if eval_df is not None and not eval_df.empty:
                        clf_evaluations.append(eval_df)
                
                # Load predictions to calculate additional metrics
                if 'predictions_file' in model_info:
                    pred_df = load_model_predictions(model_info['predictions_file'])
                    if pred_df is not None and 'actual' in pred_df.columns and 'predicted' in pred_df.columns:
                        metrics = calculate_metrics(pred_df['actual'], pred_df['predicted'], 'classification')
                        clf_metrics[model_name] = metrics
            
            if clf_evaluations:
                st.write("**Saved Evaluation Metrics:**")
                combined_eval = pd.concat(clf_evaluations, ignore_index=True)
                st.dataframe(combined_eval.style.format("{:.4f}", subset=combined_eval.select_dtypes(include=[np.number]).columns))
            
            if clf_metrics:
                st.write("**Calculated Metrics from Predictions:**")
                clf_metrics_df = pd.DataFrame.from_dict(clf_metrics, orient='index')
                st.dataframe(clf_metrics_df.style.format("{:.4f}"))
                
                # Visual comparison
                st.write("**Visual Comparison:**")
                fig = plot_model_comparison(clf_metrics, 'classification')
                st.pyplot(fig)
                
                # Best model analysis
                best_acc = clf_metrics_df['Accuracy'].idxmax()
                st.metric("Best Accuracy", best_acc, f"{clf_metrics_df.loc[best_acc, 'Accuracy']:.4f}")
    
    else:  # Cross-Validation Analysis
        st.header("📈 Cross-Validation Analysis")
        
        # Filter models that have fold evaluations
        models_with_folds = {k: v for k, v in available_models.items() if 'fold_evaluation_file' in v}
        
        if not models_with_folds:
            st.warning("No cross-validation data found.")
            return
        
        # Model selection
        selected_models = st.sidebar.multiselect(
            "Select Models for CV Analysis", 
            list(models_with_folds.keys()),
            default=list(models_with_folds.keys())[:3]  # Default to first 3 models
        )
        
        if selected_models:
            # Load fold evaluations for selected models
            fold_data = {}
            for model_name in selected_models:
                model_info = models_with_folds[model_name]
                fold_df = load_model_evaluation(model_info['fold_evaluation_file'])
                if fold_df is not None:
                    fold_data[model_name] = fold_df
            
            if fold_data:
                st.write("### Cross-Validation Comparison")
                
                # Create comparison plots
                task_types = set([models_with_folds[m]['task_type'] for m in selected_models])
                
                for task_type in task_types:
                    task_models = [m for m in selected_models if models_with_folds[m]['task_type'] == task_type]
                    
                    if len(task_models) > 1:
                        st.write(f"#### {task_type.title()} Models CV Comparison")
                        
                        # Determine metrics to plot based on task type
                        if task_type == 'regression':
                            metrics_to_plot = ['MAE', 'RMSE', 'R2 Score']
                        else:
                            metrics_to_plot = ['Accuracy', 'F1 Score']
                        
                        # Filter metrics that actually exist in the data
                        available_metrics = set()
                        for model in task_models:
                            if model in fold_data:
                                available_metrics.update(fold_data[model].columns)
                        
                        metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
                        
                        if metrics_to_plot:
                            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 6))
                            if len(metrics_to_plot) == 1:
                                axes = [axes]
                            
                            for i, metric in enumerate(metrics_to_plot):
                                for model in task_models:
                                    if model in fold_data and metric in fold_data[model].columns:
                                        values = fold_data[model][metric]
                                        axes[i].plot(fold_data[model]['fold_id'], values, 'o-', label=model, alpha=0.7)
                                
                                axes[i].set_title(f'{metric} Across Folds')
                                axes[i].set_xlabel('Fold')
                                axes[i].set_ylabel(metric)
                                axes[i].legend()
                                axes[i].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Summary statistics table
                        st.write(f"**{task_type.title()} CV Summary Statistics:**")
                        summary_data = []
                        for model in task_models:
                            if model in fold_data:
                                numeric_cols = fold_data[model].select_dtypes(include=[np.number]).columns
                                numeric_cols = [col for col in numeric_cols if col not in ['fold_id']]
                                for col in numeric_cols:
                                    summary_data.append({
                                        'Model': model,
                                        'Metric': col,
                                        'Mean': fold_data[model][col].mean(),
                                        'Std': fold_data[model][col].std(),
                                        'Min': fold_data[model][col].min(),
                                        'Max': fold_data[model][col].max()
                                    })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df.style.format({
                                'Mean': '{:.4f}',
                                'Std': '{:.4f}',
                                'Min': '{:.4f}',
                                'Max': '{:.4f}'
                            }))
                
                # Individual model details
                st.write("### Individual Model CV Details")
                for model_name in selected_models:
                    if model_name in fold_data:
                        with st.expander(f"{model_name} Cross-Validation Details"):
                            display_fold_evaluations(fold_data[model_name], model_name)

if __name__ == "__main__":
    main()