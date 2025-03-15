import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
from itertools import combinations_with_replacement
from sklearn.feature_selection import mutual_info_regression
import numpy as np


class FeatureAnalysis:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def perform_pca(self, numerical_features, explained_variance_threshold=0.8, plot_pca=True, add_pca=True):
        """
        Perform Principal Component Analysis (PCA) on the numerical features. Can only analyze
        a dataset with all categorical features at the last columns to be ignored

        Parameters:
            n_components (int): Number of principal components to retain.
            numerical_features (list): List of numerical features to perform PCA on.
            plot_pca (bool): Defines if the user pretends to plot the produced pca in the train data
            add_pca (bool): Defines if the user pretends to add the produced pca to the original train and test data


        Returns:
            pca_components_train (DataFrame): DataFrame containing the principal components of the train data.
            pca_components_test (DataFrame): DataFrame containing the principal components of the test data.
        """
        try:
            # Check if data is not None
            if self.data_loader.data_train is None and self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Perform PCA
            pca = PCA()
            pca.fit(self.data_loader.data_train[numerical_features])

            # Determine the number of components to retain
            explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(explained_variance_ratio_cumulative >= explained_variance_threshold) + 1

            # Perform PCA with the determined number of components
            pca = PCA(n_components=n_components)
            pca_components_train = pca.fit_transform(self.data_loader.data_train[numerical_features])
            pca_components_train = pd.DataFrame(data=pca_components_train,
                                                columns=[f"PC{i}" for i in range(1, n_components + 1)])
            pca_components_test = pca.transform(self.data_loader.data_test[numerical_features])
            pca_components_test = pd.DataFrame(data=pca_components_test,
                                               columns=[f"PC{i}" for i in range(1, n_components + 1)])

            if plot_pca:
                self._plot_pca_components(pca_components_train)
            if add_pca:
                self._add_pca_components(pca_components_train, pca_components_test)

            print("PCA performed successfully.")
            return pca_components_train, pca_components_test

        except ValueError as ve:
            print("Error:", ve)

    def _plot_pca_components(self, pca_components):
        """
        Plot all possible combinations of PCA components.

        Parameters:
            pca_components (DataFrame): DataFrame containing the principal components.
        """
        num_components = pca_components.shape[1]
        combinations = list(itertools.combinations(range(num_components), 2))

        fig, axes = plt.subplots(len(combinations), 1, figsize=(8, 5 * len(combinations)))

        for i, (component1, component2) in enumerate(combinations):
            ax = axes[i] if len(combinations) > 1 else axes

            ax.scatter(pca_components.iloc[:, component1], pca_components.iloc[:, component2])
            ax.set_title(f"PCA Component {component1 + 1} and PCA Component {component2 + 1}")
            ax.set_xlabel(f"PCA Component {component1 + 1}")
            ax.set_ylabel(f"PCA Component {component2 + 1}")

        plt.tight_layout()
        plt.show()

    def _add_pca_components(self, pca_components_train, pca_components_test):
        """
        Add PCA components to the original datasets.

        Parameters:
            pca_components_train (DataFrame): DataFrame containing the principal components for the training dataset.
            pca_components_test (DataFrame): DataFrame containing the principal components for the test dataset.
        """

        # Generate column names for PCA components
        pca_column_names = [f"PCA_{i}" for i in range(1, pca_components_train.shape[1] + 1)]

        # Assign column names to PCA components
        pca_components_train.columns = pca_column_names
        pca_components_test.columns = pca_column_names

        # Set indices to match before concatenating
        pca_components_train.index = self.data_loader.data_train.index
        pca_components_test.index = self.data_loader.data_test.index

        # Concatenate the PCA components with the original datasets
        self.data_loader.data_train = pd.concat([self.data_loader.data_train, pca_components_train], axis=1)
        self.data_loader.data_test = pd.concat([self.data_loader.data_test, pca_components_test], axis=1)

    def relevant_feature_identification(self, num_features=10):
        """
        Perform feature relevant feature identification using mutual information between each feature and the target
        variable. Mutual information measures the amount of information obtained about one random variable through
        another random variable. It quantifies the amount of uncertainty reduced for one variable given the knowledge
        of another variable. In feature selection, mutual information helps identify the relevance of features with
        respect to the target variable.

        Parameters:
            num_features (int): Number of features to select.

        Returns:
            selected_features (list): List of selected feature names.
        """
        # try:
        # Check if data_train is not None
        if self.data_loader.data_train is None or self.data_loader.labels_train is None:
            raise ValueError("Training data or labels have not been loaded yet.")

        # Perform feature selection using mutual information
        mutual_info = mutual_info_regression(self.data_loader.data_train, self.data_loader.labels_train)

        selected_features_indices = np.argsort(mutual_info)[::-1][:num_features]
        selected_features = self.data_loader.data_train.columns[selected_features_indices]

        print(f"{num_features} relevant features identified.")
        print(selected_features.tolist())
        return selected_features.tolist()

        # except ValueError as ve:
        #     print("Error:", ve)


class FeatureAnalysisAndGenerator(FeatureAnalysis):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def generate_features(self):
        """
        Generate and add new features to the dataset.

        This method orchestrates the creation of new features by calling
        individual methods responsible for different types of feature engineering.

        Returns:
            None
        """
        self.add_statistical_features()
        self.add_interaction_features()
        self.add_nonlinear_interaction_features()

    def add_statistical_features(self):
        """
        Add statistical features to the dataset.

        Statistical features provide insights into how much each sample's value is above or below the mean of the feature.

        Returns:
            None
        """
        # Statistical features
        for feature in self.data_loader.data_train.columns:
            if feature not in ['trip_distance', 'trip_duration']:
                mean_value = self.data_loader.data_train[feature].mean()
                self.data_loader.data_train[f'{feature}_deviation_from_mean'] = self.data_loader.data_train[
                                                                                    feature] - mean_value
                self.data_loader.data_test[f'{feature}_deviation_from_mean'] = self.data_loader.data_test[
                                                                                   feature] - mean_value

    def add_interaction_features(self):
        """
        Add interaction features to the dataset.

        Interaction features capture interactions between existing attributes, potentially revealing
        complex relationships not captured by individual features alone.

        Returns:
            None
        """
        interaction_features = list(combinations_with_replacement(['tolls_amount', 'extra', 'ratecodeid'], 2))

        for feat1, feat2 in interaction_features:
            if feat1 != feat2:
                self.data_loader.data_train[f"{feat1}_x_{feat2}"] = self.data_loader.data_train[feat1] * \
                                                                    self.data_loader.data_train[feat2]
                self.data_loader.data_test[f"{feat1}_x_{feat2}"] = self.data_loader.data_test[feat1] * \
                                                                   self.data_loader.data_test[feat2]

        print("Interaction features added to the dataset.")

    def add_nonlinear_interaction_features(self):
        """
        Add nonlinear interaction features to the dataset.

        Nonlinear interaction features capture nonlinear relationships between existing attributes,
        potentially revealing complex patterns not captured by linear interactions.

        Returns:
            None
        """
        interaction_features = list(combinations_with_replacement(['tolls_amount', 'extra', 'ratecodeid'], 2))

        for feat1, feat2 in interaction_features:
            if feat1 != feat2:
                self.data_loader.data_train[f"{feat1}_+_{feat2}_sigmoid"] = self._sigmoid(
                    self.data_loader.data_train[feat1] + self.data_loader.data_train[feat2])
                self.data_loader.data_test[f"{feat1}_+_{feat2}_sigmoid"] = self._sigmoid(
                    self.data_loader.data_test[feat1] + self.data_loader.data_test[feat2])

        print("Nonlinear interaction features added to the dataset.")

    def _sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x (array-like): Input data.

        Returns:
            array-like: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))