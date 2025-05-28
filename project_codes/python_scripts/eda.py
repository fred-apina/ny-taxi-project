class TaxiEDA:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def plot_target_distribution(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data_loader.labels_train, kde=True, bins=50)
        plt.title(f'Distribution of Fare Amount')
        plt.xlabel('Fare Amount')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x=self.data_loader.labels_train)
        plt.title(f'Box Plot of Fare Amount')
        plt.xlabel('Fare Amount')
        plt.show()
        
        # Log transform for skewed data if necessary (often fare_amount is)
        if self.data_loader.labels_train.min() > 0 : # Log transform requires positive values
            plt.figure(figsize=(12, 6))
            sns.histplot(np.log1p(self.data_loader.labels_train), kde=True, bins=50)
            plt.title(f'Distribution of Log(Fare Amount)')
            plt.xlabel(f'Log(Fare Amount)')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f"Cannot log transform Fare Amount as it contains non-positive values after cleaning.")


    def plot_numerical_distributions(self, num_cols):
        for col in num_cols:
            if col in self.data_loader.data_train.columns:
                plt.figure(figsize=(10, 5))
                sns.histplot(self.data_loader.data_train[col], kde=True, bins=30)
                plt.title(f'Distribution of {col}')
                plt.show()
                plt.figure(figsize=(10, 3))
                sns.boxplot(x=self.data_loader.data_train[col])
                plt.title(f'Box Plot of {col}')
                plt.show()
            else:
                print(f"Numerical column '{col}' not found.")

    def plot_categorical_distributions(self, cat_cols):
        for col in cat_cols:
            if col in self.data_loader.data_train.columns:
                plt.figure(figsize=(10, 5))
                sns.countplot(y=self.data_loader.data_train[col], order=self.data_loader.data_train[col].value_counts().index, palette="viridis", hue=self.data_loader.data_train[col], legend=False)
                plt.title(f'Distribution of {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.show()
            else:
                print(f"Categorical column '{col}' not found.")

    def plot_numerical_vs_target(self, num_cols):
        for col in num_cols:
            if col in self.data_loader.data_train.columns:
                plt.figure(figsize=(10, 6))
                # Using a sample for scatter plots if dataframe is large, for performance
                sample_df = self.data_loader.data_train.sample(min(len(self.data_loader.data_train), 1000))
                sns.scatterplot(x=sample_df[col], y=self.data_loader.labels_train[sample_df.index], alpha=0.5)
                plt.title(f'Fare Amount vs. {col}')
                plt.show()
            else:
                print(f"Numerical column '{col}' not found for scatter plot.")


    def plot_categorical_vs_target(self, cat_cols, target_col='fare_amount'):
        for col in cat_cols:
            if col in self.data_loader.data_train.columns:
                plt.figure(figsize=(12, 7))
                # For high cardinality categoricals, consider top N or alternative plots
                if self.data_loader.data_train[col].nunique() > 20:
                    top_n = self.data_loader.data_train[col].value_counts().nlargest(15).index
                    sns.boxplot(x=self.data_loader.data_train[self.data_loader.data_train[col].isin(top_n)][col], y=self.data_loader.labels_train[self.data_loader.data_train[col].isin(top_n)], order=top_n, palette="viridis")
                    plt.title(f'Fare Amount vs. Top 15 {col}')
                else:
                    sns.boxplot(x=self.data_loader.data_train[col], y=self.data_loader.labels_train[self.data_loader.data_train[col].index], palette="viridis", hue=self.data_loader.data_train[col], legend=False)
                    plt.title(f'Fare Amount vs. {col}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            else:
                print(f"Categorical column '{col}' not found for box plot.")


    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 10))
        # Select only numerical columns for correlation heatmap
        numerical_df = pd.concat([self.data_loader.data_train, self.data_loader.labels_train], axis=1).select_dtypes(include=np.number)
        if not numerical_df.empty:
            sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
            plt.title('Correlation Heatmap of Numerical Features')
            plt.show()
        else:
            print("No numerical columns found for correlation heatmap.")
            
    def plot_feature_importance(self, n_estimators=5, n_repeats=2):
        """
        Computes and visualizes feature importance using permutation importance.
        """
        # Fit a random forest classifier to compute feature importance
        clf = RandomForestRegressor(n_estimators=n_estimators)
        clf.fit(self.data_loader.data_train.select_dtypes(include=np.number), self.data_loader.labels_train)

        # Compute permutation importance
        result = permutation_importance(clf, self.data_loader.data_train.select_dtypes(include=np.number), self.data_loader.labels_train,
                                        n_repeats=n_repeats)
        sorted_idx = result.importances_mean.argsort()

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), self.data_loader.data_train.columns[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')
        plt.show()
            
    def perform_dimension_reduction_viz(self, features_for_reduction, target_for_color=None):
        if not features_for_reduction:
            print("No features specified for dimension reduction.")
            return
        
        # Ensure features exist and are numeric, handle missing values and scale
        redux_df = self.data_loader.data_train[features_for_reduction].copy()
        redux_df.dropna(inplace=True) # Drop rows with NaN in these specific cols for DR
        
        if redux_df.empty:
            print("DataFrame is empty after selecting features and dropping NaNs for dimension reduction.")
            return

        # Impute any remaining NaNs (if any, though dropna should handle) and scale
        # This should ideally use the same imputer/scaler as the main pipeline
        num_imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        redux_df_imputed = num_imputer.fit_transform(redux_df)
        redux_scaled = scaler.fit_transform(redux_df_imputed)

        if redux_scaled.shape[0] < 2: # Need at least 2 samples
            print("Not enough samples for dimension reduction after preprocessing.")
            return

        # PCA
        print("\n--- PCA Visualization ---")
        pca = PCA(n_components=2, random_state=42)
        try:
            principal_components = pca.fit_transform(redux_scaled)
            print(f"Explained variance by PCA components: {pca.explained_variance_ratio_}")
            pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            
            plt.figure(figsize=(10, 7))
            if target_for_color and target_for_color in self.data_loader.data_train.columns and len(pc_df) == len(self.data_loader.data_train.loc[redux_df.index, target_for_color].dropna()):
                # Align target for coloring with the potentially reduced redux_df
                aligned_target = self.data_loader.data_train.loc[redux_df.index, target_for_color].copy()
                # If target is categorical, ensure it's suitable for hue
                if aligned_target.dtype.name == 'category' or aligned_target.nunique() < 15:
                    sns.scatterplot(x='PC1', y='PC2', data=pc_df, hue=aligned_target, palette='viridis', alpha=0.7, s=50)
                else: # If target is continuous or high cardinality cat, bin it or just plot points
                    sns.scatterplot(x='PC1', y='PC2', data=pc_df, color='blue', alpha=0.7, s=50)

            else:
                sns.scatterplot(x='PC1', y='PC2', data=pc_df, color='blue', alpha=0.7, s=50)
            plt.title('2D PCA Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error during PCA: {e}")