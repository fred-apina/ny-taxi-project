import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


class EDA:
    """
    A class responsible for exploratory data analysis (EDA).

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        perform_eda(): Performs exploratory data analysis.
        plot_distributions(): Plots distributions of the features.
        plot_correlation_heatmap(): Plots a correlation heatmap between features and labels.
        plot_feature_importance(): Computes and visualizes feature importance using permutation importance.
    """

    def __init__(self, data_loader):
        """
        Initializes the EDA class with a DataLoader object.
        """
        self.data_loader = data_loader

    def perform_eda(self):
        """
        Performs exploratory data analysis.
        """
        print("Exploratory Data Analysis (EDA) Report:")
        print("--------------------------------------")

        # Summary statistics
        print("\nSummary Statistics for train data:")
        print(self.data_loader.data_train.describe())
        print("\nSummary Statistics for test data:")
        print(self.data_loader.data_test.describe())

        # Distribution analysis
        self.plot_distributions()

        # Correlation analysis
        self.plot_correlation_heatmap()

        # Feature Importance analysis
        self.plot_feature_importance()

    def plot_distributions(self):
        """
        Plots distributions of the features.
        """
        num_cols = len(self.data_loader.data_train.columns)
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
        for i, feature in enumerate(self.data_loader.data_train.columns):
            sns.histplot(data=self.data_loader.data_train, x=feature, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plots a correlation heatmap between features and labels.
        """
        # Concatenate features and labels horizontally for correlation calculation
        data_with_labels = pd.concat([self.data_loader.data_train, self.data_loader.labels_train], axis=1)

        # Compute the correlation matrix
        corr_matrix = data_with_labels.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap between Features and Labels")
        plt.show()

    def plot_feature_importance(self, n_estimators=5, n_repeats=2):
        """
        Computes and visualizes feature importance using permutation importance.
        """
        # Fit a random forest classifier to compute feature importance
        clf = RandomForestRegressor(n_estimators=n_estimators)
        clf.fit(self.data_loader.data_train, self.data_loader.labels_train)

        # Compute permutation importance
        result = permutation_importance(clf, self.data_loader.data_train, self.data_loader.labels_train,
                                        n_repeats=n_repeats)
        sorted_idx = result.importances_mean.argsort()

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), self.data_loader.data_train.columns[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')
        plt.show()


class DataVisualizer:
    """
    A class responsible for data visualization.

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        perform_visualization(): Performs data visualization.
    """

    def __init__(self, data_loader):
        """
        Initializes the DataVisualization class with a DataLoader object.
        """
        self.data_loader = data_loader

    def perform_visualization(self):
        """
        Performs data visualization.
        """
        # Pairplot
        self.plot_pairplot()

        # Boxplot
        self.plot_boxplot()

        # Ridgeplot
        self.plot_ridgeplot()

    def plot_pairplot(self):
        """
        Plots pairplot for all features.
        """
        sns.pairplot(self.data_loader.data_train, diag_kind='kde')
        plt.title("Pairplot of Features")
        plt.show()

    def plot_boxplot(self):
        """
        Plots boxplot for all features.
        """
        # Create a single figure and axis for all boxplots
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot boxplots for each feature
        sns.boxplot(data=self.data_loader.data_train, ax=ax)
        ax.set_title("Boxplot of all Features")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.set_xticks(range(len(self.data_loader.data_train.columns)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_ridgeplot(self):
        """
        Plots overlapping densities (ridge plot) for all features.
        """

        # Create a single figure and axis for all boxplots
        fig, axes = plt.subplots(len(self.data_loader.data_train.columns), 1, figsize=(10, 8), sharex=True)

        # Generate a gradient of darker colors for the plots
        num_plots = len(self.data_loader.data_train.columns)
        cmap = plt.get_cmap('Blues')
        colors = [cmap(1 - i / (num_plots + 1)) for i in range(1, num_plots + 1)]

        # Plot overlapping densities for each numerical feature
        for i, (feature, color) in enumerate(zip(self.data_loader.data_train.columns, colors)):
            sns.kdeplot(data=self.data_loader.data_train[feature], ax=axes[i], color=color, fill=True, linewidth=2)
            axes[i].set_ylabel(feature, rotation=0, labelpad=40)  # Rotate y-axis label
            axes[i].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position
            axes[i].spines['top'].set_visible(False)

            # Remove box structure around the plots
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)

        # Adjust plot aesthetics
        axes[-1].set_xlabel("Value")

        plt.tight_layout()
        plt.show()