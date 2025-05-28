# --- 4. Deep Learning Model (TensorFlow) with manual CV ---
class TFDeepLearningModel(BaseModel):
    def __init__(self, model_name, task_type, input_dim, num_classes=None, tune_params=None):
        super().__init__(model_name, task_type)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.tune_params = tune_params if tune_params else {
            'units_layer1': [32, 64],
            'batch_size': [32],
            'learning_rate': [0.001]
        }

    def _build_model_dynamic(self, units_layer1=64, units_layer2=32, dropout_rate=0.5, learning_rate=0.001):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))

        # First layer
        model.add(Dense(units_layer1, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dropout(dropout_rate))

        # Second layer
        model.add(Dense(units_layer2, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dropout(dropout_rate))

        # Add batch normalization for stability
        model.add(tf.keras.layers.BatchNormalization())

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if self.task_type == 'regression':
            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:  # classification
            if self.num_classes is None or self.num_classes <= 0:
                raise ValueError("num_classes must be specified for classification")

            self.num_classes = int(self.num_classes)
            if self.num_classes <= 2:
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.add(Dense(self.num_classes, activation='softmax'))
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, X_train_full, y_train_full, epochs_cv=10, epochs_final=20):
        try:
            X_train_full = np.array(X_train_full)
            y_train_full = np.array(y_train_full)
            self.fold_evaluations = []

            if self.input_dim is None:
                self.input_dim = X_train_full.shape[1]

            kf = KFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)
            best_params_found = None
            best_avg_val_metric = -float('inf') if self.task_type == 'classification' else float('inf')

            param_combinations = []
            for lr in self.tune_params.get('learning_rate', [0.001]):
                for units in self.tune_params.get('units_layer1', [64]):
                    for batch_size in self.tune_params.get('batch_size', [32]):
                        param_combinations.append({
                            'units_layer1': units,
                            'batch_size': batch_size,
                            'learning_rate': lr
                        })

            print(f"Optimizing TensorFlow model {self.model_name}...")

            for params in param_combinations:
                print(f"  Evaluating params: {params}")
                fold_metrics = []

                for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
                    tf.keras.backend.clear_session()

                    model_fold = self._build_model_dynamic(
                        units_layer1=params['units_layer1'],
                        learning_rate=params['learning_rate']
                    )

                    X_cv_train, X_cv_val = X_train_full[train_idx], X_train_full[val_idx]
                    y_cv_train, y_cv_val = y_train_full[train_idx], y_train_full[val_idx]

                    # Prepare labels for TF training
                    y_cv_train_proc = y_cv_train.copy()
                    y_cv_val_proc = y_cv_val.copy()

                    if self.task_type == 'classification' and self.num_classes > 2:
                        # Ensure labels are 0-indexed for sparse_categorical_crossentropy
                        unique_labels = np.unique(y_cv_train)
                        if unique_labels.min() > 0:
                            y_cv_train_proc = y_cv_train - unique_labels.min()
                            y_cv_val_proc = y_cv_val - unique_labels.min()

                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=5,  # Increased patience
                        restore_best_weights=True,
                        verbose=0,
                        min_delta=0.001  # Minimum change to qualify as improvement
                    )

                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=0.00001,
                        verbose=0
                    )

                    model_fold.fit(
                        X_cv_train, y_cv_train_proc,
                        epochs=epochs_cv,
                        batch_size=params['batch_size'],
                        validation_data=(X_cv_val, y_cv_val_proc),
                        callbacks=[early_stop, reduce_lr],
                        verbose=0
                    )

                    y_pred_raw = model_fold.predict(X_cv_val, verbose=0)

                    if self.task_type == 'classification':
                        if self.num_classes <= 2:
                            y_pred_final = (y_pred_raw > 0.5).astype(int).ravel()
                        else:
                            y_pred_final = np.argmax(y_pred_raw, axis=1)
                            if unique_labels.min() > 0:
                                y_pred_final = y_pred_final + unique_labels.min()
                    else:
                        y_pred_final = y_pred_raw.ravel()

                    self._calculate_fold_metrics(y_cv_val, y_pred_final, params, fold_id + 1)

                    if self.task_type == 'classification':
                        metric = accuracy_score(y_cv_val, y_pred_final)
                    else:
                        metric = r2_score(y_cv_val, y_pred_final)
                    fold_metrics.append(metric)

                avg_metric = np.mean(fold_metrics)
                print(f"Average CV metric ({'R²' if self.task_type == 'regression' else 'Accuracy'}): {avg_metric:.4f}")

                if self.task_type == 'classification':
                    if avg_metric > best_avg_val_metric:
                        best_avg_val_metric = avg_metric
                        best_params_found = params
                else:
                    if avg_metric > best_avg_val_metric:
                        best_avg_val_metric = avg_metric
                        best_params_found = params

            self.best_params_ = best_params_found or param_combinations[0]
            print(f"Best TF params: {self.best_params_}")

            # Train final model
            print("Training final TensorFlow model...")
            tf.keras.backend.clear_session()
            self.model = self._build_model_dynamic(
                units_layer1=self.best_params_['units_layer1'],
                learning_rate=self.best_params_['learning_rate']
            )

            y_train_proc = y_train_full.copy()
            if self.task_type == 'classification' and self.num_classes > 2:
                unique_labels = np.unique(y_train_full)
                if unique_labels.min() > 0:
                    y_train_proc = y_train_full - unique_labels.min()

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            self.model.fit(
                X_train_full, y_train_proc,
                epochs=epochs_final,
                batch_size=self.best_params_['batch_size'],
                validation_split=0.15,
                callbacks=[early_stop],
                verbose=0
            )
            print(f"{self.model_name} fitted successfully.")

        except Exception as e:
            print(f"Error fitting TensorFlow model {self.model_name}: {e}")
            self.model = None

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError(f"Model {self.model_name} not fitted yet.")

        X_test = np.array(X_test)
        y_pred_raw = self.model.predict(X_test, verbose=0)

        if self.task_type == 'regression':
            return y_pred_raw.ravel()
        else:  # classification
            if self.num_classes <= 2:
                return (y_pred_raw > 0.5).astype(int).ravel()
            else:
                predictions = np.argmax(y_pred_raw, axis=1)
                # Adjust back to original label range if needed
                return predictions + 1 if hasattr(self, '_label_offset') else predictions

    def _save_tf_model(self, filename_suffix=""):
        try:
            if self.model:
                filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}.keras")
                self.model.save(filepath)
                print(f"TensorFlow model saved to {filepath}")

                if self.best_params_:
                    params_filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}_best_params.joblib")
                    joblib.dump(self.best_params_, params_filepath)
        except Exception as e:
            print(f"Error saving TensorFlow model: {e}")

    def _load_tf_model(self, filename_suffix=""):
        try:
            filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}.keras")
            params_filepath = os.path.join(MODELS_DIR, f"{self.model_name}_{self.task_type}{filename_suffix}_best_params.joblib")

            self.model = tf.keras.models.load_model(filepath)
            print(f"TensorFlow model loaded from {filepath}")

            if os.path.exists(params_filepath):
                self.best_params_ = joblib.load(params_filepath)
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
