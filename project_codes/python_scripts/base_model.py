# --- Base Model Class ---
class BaseModel:
    def __init__(self, model_name, task_type):
        self.model_name = model_name
        self.task_type = task_type
        self.model = None
        self.evaluation_metrics = {}
        self.best_params_ = None
        self.fold_evaluations = []

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        if isinstance(self, CustomKNN):
            if self._ball_tree_final is None or self.k is None:
                raise RuntimeError(f"Model {self.model_name} not fitted yet or not properly configured.")
            return self._fit_predict_single_knn(self._X_train_full_data, self._y_train_full_data, np.array(X_test), self.k)

        if self.model is None:
            raise RuntimeError(f"Model {self.model_name} not fitted yet. Call fit() first.")
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        global all_evaluations

        # Ensure arrays are 1D
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        if self.task_type == 'regression':
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            self.evaluation_metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}
        elif self.task_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            num_classes = len(np.unique(y_true))
            avg_type = 'macro' if num_classes > 2 else 'binary'

            precision = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
            recall = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
            self.evaluation_metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        print(f"\nFinal Test Set Metrics for {self.model_name}:")
        for metric, value in self.evaluation_metrics.items():
            print(f"  {metric}: {value:.4f}")

        current_eval_summary = {'model_name': self.model_name, 'task_type': self.task_type}
        if self.best_params_:
            serializable_params = {k: str(v) for k, v in self.best_params_.items()}
            current_eval_summary['best_params'] = serializable_params
        current_eval_summary.update(self.evaluation_metrics)
        all_evaluations.append(current_eval_summary)

        # Save individual model evaluation immediately
        self.save_individual_evaluation(current_eval_summary)

    def save_individual_evaluation(self, evaluation_summary):
        """Save individual model evaluation to a separate file"""
        try:
            eval_df = pd.DataFrame([evaluation_summary])
            filename = f"{self.model_name}_{self.task_type}_evaluation.csv"
            filepath = os.path.join(EVALUATIONS_DIR, filename)
            eval_df.to_csv(filepath, index=False)
            print(f"Model evaluation saved to {filepath}")
        except Exception as e:
            print(f"Error saving Model evaluation for {self.model_name}: {e}")

    def save_fold_evaluations(self):
        """Save fold evaluations for this model to a separate file"""
        try:
            if self.fold_evaluations:
                fold_df = pd.DataFrame(self.fold_evaluations)
                filename = f"{self.model_name}_{self.task_type}_fold_evaluations.csv"
                filepath = os.path.join(EVALUATIONS_DIR, filename)
                fold_df.to_csv(filepath, index=False)
                print(f"Fold evaluations saved to {filepath}")
        except Exception as e:
            print(f"Error saving fold evaluations for {self.model_name}: {e}")

    def _calculate_fold_metrics(self, y_true_fold, y_pred_fold, params_tried, fold_id):
        y_true_fold = np.array(y_true_fold).ravel()
        y_pred_fold = np.array(y_pred_fold).ravel()

        fold_metrics_dict = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'fold_id': fold_id,
            'params_tried': str(params_tried)
        }

        if self.task_type == 'regression':
            fold_metrics_dict['MSE'] = mean_squared_error(y_true_fold, y_pred_fold)
            fold_metrics_dict['RMSE'] = np.sqrt(fold_metrics_dict['MSE'])
            fold_metrics_dict['MAE'] = mean_absolute_error(y_true_fold, y_pred_fold)
            fold_metrics_dict['R2 Score'] = r2_score(y_true_fold, y_pred_fold)
        elif self.task_type == 'classification':
            num_classes = len(np.unique(y_true_fold))
            avg_type = 'macro' if num_classes > 2 else 'binary'

            fold_metrics_dict['Accuracy'] = accuracy_score(y_true_fold, y_pred_fold)
            fold_metrics_dict['Precision'] = precision_score(y_true_fold, y_pred_fold, average=avg_type, zero_division=0)
            fold_metrics_dict['Recall'] = recall_score(y_true_fold, y_pred_fold, average=avg_type, zero_division=0)
            fold_metrics_dict['F1 Score'] = f1_score(y_true_fold, y_pred_fold, average=avg_type, zero_division=0)

        self.fold_evaluations.append(fold_metrics_dict)

    def save_results(self, y_true, y_pred, filename_suffix=""):
        try:
            results_df = pd.DataFrame({
                'actual': np.array(y_true).ravel(),
                'predicted': np.array(y_pred).ravel()
            })
            filepath = os.path.join(RESULTS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}_results.csv")
            results_df.to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results for {self.model_name}: {e}")

    def save_model(self, filename_suffix=""):
        try:
            if isinstance(self, CustomKNN):
                self._save_custom_knn_model(filename_suffix)
                return
            if isinstance(self, TFDeepLearningModel):
                self._save_tf_model(filename_suffix)
                return

            filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}.joblib")
            model_to_save = self.model

            if hasattr(self.model, 'best_estimator_'):
                model_to_save = self.model.best_estimator_

            if model_to_save:
                joblib.dump(model_to_save, filepath)
                print(f"Model saved to {filepath}")
                if self.best_params_:
                    params_filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}_best_params.joblib")
                    joblib.dump(self.best_params_, params_filepath)
            else:
                print(f"No model to save for {self.model_name}")
        except Exception as e:
            print(f"Error saving model {self.model_name}: {e}")

    def load_model(self, filename_suffix=""):
        try:
            if isinstance(self, CustomKNN):
                self._load_custom_knn_model(filename_suffix)
                return
            if isinstance(self, TFDeepLearningModel):
                self._load_tf_model(filename_suffix)
                return

            filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}.joblib")
            params_filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}_best_params.joblib")

            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            if os.path.exists(params_filepath):
                self.best_params_ = joblib.load(params_filepath)
        except FileNotFoundError:
            print(f"Model files not found for {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")