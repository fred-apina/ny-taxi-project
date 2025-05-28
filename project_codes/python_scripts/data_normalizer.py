class DataNormalizer:
    """
    Class responsible for normalizing the data.

    Methods:
        _normalize_features(): Normalizes all features using standard scaling for numerical and onehot encoding for categorical.
    """

    def __init__(self, data_loader, categorical_features, numerical_features):
        """
        Initializes the DataNormalizer class.
        """
        self.data_loader = data_loader
        self.categorical_features = categorical_features
        self.real_numerical_features = numerical_features

    def normalize_features(self):
        """
        Normalizes features using using standard scaling for numerical and onehot encoding for categorical.
        """
        try:
            # Check if data_train and data_test are not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")
            # Check if labels_train and labels_test are not None
            if self.data_loader.labels_train is None or self.data_loader.labels_test is None:
                raise ValueError("Labels have not been loaded yet.")

            # Normalize real numerical features using StandardScaler
            scaler = StandardScaler()
            self.data_loader.data_train[self.real_numerical_features] = scaler.fit_transform(
                self.data_loader.data_train[self.real_numerical_features])
            self.data_loader.data_test[self.real_numerical_features] = scaler.transform(
                self.data_loader.data_test[self.real_numerical_features])
            
            # convert to one hot encoding for categorical features
            self.data_loader.data_train = pd.get_dummies(self.data_loader.data_train, columns=self.categorical_features, dtype=int)
            self.data_loader.data_test = pd.get_dummies(self.data_loader.data_test, columns=self.categorical_features, dtype=int)

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)