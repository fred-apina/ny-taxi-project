# --- 2. Supervised Learning models (sklearn) with GridSearchCV ---
class SklearnModel(BaseModel):
    def __init__(self, model_name, task_type, sklearn_model_instance, param_grid=None):
        super().__init__(model_name, task_type)
        self.base_model_instance = sklearn_model_instance
        self.param_grid = param_grid if param_grid else {}

    def fit(self, X_train, y_train):
        self.fold_evaluations = []

        try:
            if self.param_grid:
                print(f"Running GridSearchCV for {self.model_name}...")

                # Define multiple scoring metrics
                if self.task_type == 'regression':
                    scoring = {
                        'mse': 'neg_mean_squared_error',    # MSE (sklearn returns negative, we'll convert)
                        'mae': 'neg_mean_absolute_error',   # MAE (sklearn returns negative, we'll convert)
                        'r2': 'r2'                          # R2 Score
                        # Note: RMSE = sqrt(MSE), we'll calculate it from MSE
                    }
                    refit_metric = 'r2'  # Primary metric for selecting best params
                else:  # classification
                    scoring = {
                        'accuracy': 'accuracy',
                        'precision': 'precision_macro',
                        'recall': 'recall_macro',
                        'f1': 'f1_macro'
                    }
                    refit_metric = 'accuracy'  # Primary metric for selecting best params

                grid_search = GridSearchCV(
                    estimator=self.base_model_instance,
                    param_grid=self.param_grid,
                    cv=N_CV_SPLITS,
                    scoring=scoring,
                    refit=refit_metric,  # Which metric to use for selecting best params
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=False
                )

                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                print(f"Best params: {self.best_params_}")
                print(f"Best CV score ({refit_metric}): {grid_search.best_score_:.4f}")

                # Extract fold metrics from GridSearchCV results
                self._extract_fold_metrics_from_gridsearch(grid_search)

            else:
                print(f"Fitting {self.model_name} with default parameters...")
                self.model = self.base_model_instance
                self.best_params_ = self.model.get_params()

                # Fit final model on full training data
                self.model.fit(X_train, y_train)

            print(f"{self.model_name} fitted successfully.")

        except Exception as e:
            print(f"Error fitting {self.model_name}: {e}")
            self.model = None
            self.best_params_ = None

    def _extract_fold_metrics_from_gridsearch(self, grid_search):
        """Extract fold metrics from GridSearchCV results for the best parameters"""
        try:
            # Get the index of the best parameter combination
            best_index = grid_search.best_index_

            # Get cross-validation results
            cv_results = grid_search.cv_results_

            # Extract fold scores for the best parameter combination
            for fold_id in range(N_CV_SPLITS):
                fold_metrics_dict = {
                    'model_name': self.model_name,
                    'task_type': self.task_type,
                    'fold_id': fold_id + 1,
                    'params_tried': str(self.best_params_)
                }

                if self.task_type == 'regression':
                    # Extract scores for each metric
                    # Note: sklearn returns NEGATIVE values for error metrics (neg_mean_squared_error, neg_mean_absolute_error)
                    # We need to convert them back to positive values
                    mse_score = -cv_results[f'split{fold_id}_test_mse'][best_index]  # Convert negative MSE to positive
                    mae_score = -cv_results[f'split{fold_id}_test_mae'][best_index]  # Convert negative MAE to positive
                    r2_score = cv_results[f'split{fold_id}_test_r2'][best_index]     # R2 is already positive

                    fold_metrics_dict['MSE'] = mse_score
                    fold_metrics_dict['RMSE'] = np.sqrt(mse_score)
                    fold_metrics_dict['MAE'] = mae_score
                    fold_metrics_dict['R2 Score'] = r2_score

                else:  # classification
                    fold_metrics_dict['Accuracy'] = cv_results[f'split{fold_id}_test_accuracy'][best_index]
                    fold_metrics_dict['Precision'] = cv_results[f'split{fold_id}_test_precision'][best_index]
                    fold_metrics_dict['Recall'] = cv_results[f'split{fold_id}_test_recall'][best_index]
                    fold_metrics_dict['F1 Score'] = cv_results[f'split{fold_id}_test_f1'][best_index]

                self.fold_evaluations.append(fold_metrics_dict)

        except Exception as e:
            print(f"Error extracting fold metrics from GridSearchCV: {e}")

    # predict, evaluate, save_model, load_model inherited.