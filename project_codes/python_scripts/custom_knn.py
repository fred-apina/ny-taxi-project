# --- 1. kNN Algorithm (from scratch using NumPy with BallTree for neighbor search with manual CV for k optimization) ---
class CustomKNN(BaseModel):
    def __init__(self, model_name, task_type, k_options=None):
        super().__init__(model_name, task_type)
        self.k_options = k_options if k_options else [3, 5, 7]
        self.k = None
        self._X_train_full_data = None
        self._y_train_full_data = None
        self._ball_tree_final = None

    def _fit_predict_single_knn(self, X_tr, y_tr, X_val, current_k):
        try:
            X_tr = np.array(X_tr)
            y_tr = np.array(y_tr)
            X_val = np.array(X_val)

            if current_k <= 0 or current_k > len(X_tr):
                return None

            temp_ball_tree = BallTree(X_tr)
            distances, indices = temp_ball_tree.query(X_val, k=current_k)

            predictions = []
            for i in range(X_val.shape[0]):
                neighbor_indices = indices[i]
                neighbor_labels = y_tr[neighbor_indices]
                if self.task_type == 'regression':
                    prediction = np.mean(neighbor_labels)
                else:  # classification
                    prediction = mode(neighbor_labels, keepdims=False)[0]
                predictions.append(prediction)
            return np.array(predictions)
        except Exception as e:
            print(f"Error in kNN prediction (k={current_k}): {e}")
            return None

    def fit(self, X_train_full, y_train_full):
        self._X_train_full_data = np.array(X_train_full)
        self._y_train_full_data = np.array(y_train_full)
        self.fold_evaluations = []

        if len(self._X_train_full_data) == 0:
            print(f"Error: X_train_full is empty for {self.model_name}")
            return

        kf = KFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)
        best_k_val = None
        best_avg_score = -float('inf') if self.task_type == 'regression' else -1.0

        print(f"Optimizing k for {self.model_name} using {N_CV_SPLITS}-fold CV...")

        for k_val in self.k_options:
            fold_scores_for_k = []
            print(f"  Evaluating k={k_val}...")

            for fold_id, (train_idx, val_idx) in enumerate(kf.split(self._X_train_full_data)):
                X_cv_train = self._X_train_full_data[train_idx]
                X_cv_val = self._X_train_full_data[val_idx]
                y_cv_train = self._y_train_full_data[train_idx]
                y_cv_val = self._y_train_full_data[val_idx]

                if len(X_cv_train) < k_val:
                    continue

                y_pred_cv_val = self._fit_predict_single_knn(X_cv_train, y_cv_train, X_cv_val, k_val)

                if y_pred_cv_val is None:
                    continue

                params_this_fold = {'k': k_val}
                self._calculate_fold_metrics(y_cv_val, y_pred_cv_val, params_this_fold, fold_id + 1)

                if self.task_type == 'regression':
                    score = r2_score(y_cv_val, y_pred_cv_val)
                else:
                    score = accuracy_score(y_cv_val, y_pred_cv_val)
                fold_scores_for_k.append(score)

            if fold_scores_for_k:
                avg_score_for_k = np.mean(fold_scores_for_k)
                print(f"Average CV Score ({'R²' if self.task_type == 'regression' else 'Accuracy'}): {avg_score_for_k:.4f}")

                if ((self.task_type == 'regression' and avg_score_for_k > best_avg_score) or
                    (self.task_type == 'classification' and avg_score_for_k > best_avg_score)):
                    best_avg_score = avg_score_for_k
                    best_k_val = k_val

        self.k = best_k_val
        self.best_params_ = {'k': self.k} if self.k else {}

        if self.k is not None:
            print(f"Best k found: {self.k} with score: {best_avg_score:.4f}")
            self._ball_tree_final = BallTree(self._X_train_full_data)
        else:
            print(f"Could not determine best k for {self.model_name}")

    def _save_custom_knn_model(self, filename_suffix=""):
        try:
            filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}.joblib")
            model_data = {
                'k': self.k,
                'k_options': self.k_options,
                '_X_train_full_data': self._X_train_full_data,
                '_y_train_full_data': self._y_train_full_data,
                'task_type': self.task_type,
                'best_params_': self.best_params_
            }
            joblib.dump(model_data, filepath)
            print(f"CustomKNN data saved to {filepath}")
        except Exception as e:
            print(f"Error saving CustomKNN: {e}")

    def _load_custom_knn_model(self, filename_suffix=""):
        try:
            filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}.joblib")
            model_data = joblib.load(filepath)
            self.k = model_data['k']
            self.k_options = model_data.get('k_options', [3,5,7])
            self._X_train_full_data = model_data['_X_train_full_data']
            self._y_train_full_data = model_data['_y_train_full_data']
            self.best_params_ = model_data.get('best_params_')

            if self._X_train_full_data is not None and self.k is not None:
                self._ball_tree_final = BallTree(self._X_train_full_data)
            print(f"CustomKNN loaded from {filepath}")
        except Exception as e:
            print(f"Error loading CustomKNN: {e}")