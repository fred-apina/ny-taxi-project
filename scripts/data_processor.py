import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import sqlite3

import warnings
warnings.filterwarnings("ignore")


class DataLoader:
    """
    Generic Class responsible for loading the dataset and splitting it into training and testing sets.

    Attributes:
        filename (str): The filename of the dataset to load.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int or None): The seed used by the random number generator for reproducibility.

    Attributes (after loading the data):
        data_train (DataFrame): The training data features.
        labels_train (Series): The training data labels.
        data_test (DataFrame): The testing data features.
        labels_test (Series): The testing data labels.

    Methods:
        _load_data(): Loads the dataset, splits it into training and testing sets,
                      and assigns the data and labels to the appropriate attributes.
    """

    def __init__(self, filename, labels_test, test_size=0.2, random_state=None):
        """
        Initializes the DataLoader with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        self.filename = filename
        self.test_size = test_size
        self.random_state = random_state
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = labels_test

        # Load data
        self._load_data()

    def _load_data(self):
        """
        Loads the dataset from the specified filename,
        splits it into training and testing sets using train_test_split(),
        and assigns the data and labels to the appropriate attributes.
        """
        try:
            # Load Dataset from sqlite database
            conn = sqlite3.connect(self.filename)
            query = "SELECT * FROM tripdata"
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Split the data into features and labels
            X = df.drop(columns=self.labels_test)
            y = df[self.labels_test]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                random_state=self.random_state)

            # Assign the data and labels to attributes
            self.data_train = X_train
            self.labels_train = y_train
            self.data_test = X_test
            self.labels_test = y_test

            print("Data loaded successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")


class DataManipulator(DataLoader):
    """
    A subclass of DataLoader that specializes to manipulate the forest cover types dataset.

    Methods:
        _combine_soil_types(): Combines the Soil_Type features into a single encoded column.
        _combine_wilderness_area(): Combines the Wilderness_Area features into a single encoded column.
    """

    def __init__(self, filename, labels_test, test_size=0.2, random_state=None):
        """
        Initializes the DataManipulator with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        super().__init__(filename, labels_test, test_size, random_state)

        # Manipulate data
        self._convert_features()

    def _convert_features(self):
        """
        Convert columns to appropriate data types and create trip_duration column.
        """
        try:
            # Convert datetime columns to datetime type
            self.data_train['tpep_pickup_datetime'] = pd.to_datetime(self.data_train['tpep_pickup_datetime'], errors='coerce')
            self.data_test['tpep_pickup_datetime'] = pd.to_datetime(self.data_test['tpep_pickup_datetime'], errors='coerce')
            self.data_train['tpep_dropoff_datetime'] = pd.to_datetime(self.data_train['tpep_dropoff_datetime'], errors='coerce')
            self.data_test['tpep_dropoff_datetime'] = pd.to_datetime(self.data_test['tpep_dropoff_datetime'], errors='coerce')
            
            # Get trip duration in seconds
            self.data_train['trip_duration'] = (self.data_train['tpep_dropoff_datetime'] - self.data_train['tpep_pickup_datetime']).dt.total_seconds()
            self.data_test['trip_duration'] = (self.data_test['tpep_dropoff_datetime'] - self.data_test['tpep_pickup_datetime']).dt.total_seconds()
            
            # Drop datetime columns
            self.data_train.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)
            self.data_test.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

            # Convert categorical columns to int type
            self.data_train['payment_type'] = self.data_train['payment_type'].astype(int)
            self.data_test['payment_type'] = self.data_test['payment_type'].astype(int)
            self.data_train['passenger_count'] = self.data_train['passenger_count'].astype(int)
            self.data_test['passenger_count'] = self.data_test['passenger_count'].astype(int)
            self.data_train['store_and_fwd_flag'] = self.data_train['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
            self.data_test['store_and_fwd_flag'] = self.data_test['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
            
            # Convert vendorid, ratecodeid, passenger_count to int type
            self.data_train['vendorid'] = self.data_train['vendorid'].astype(int)
            self.data_test['vendorid'] = self.data_test['vendorid'].astype(int)
            self.data_train['ratecodeid'] = self.data_train['ratecodeid'].astype(int)
            self.data_test['ratecodeid'] = self.data_test['ratecodeid'].astype(int)
            self.data_train['pulocationid'] = self.data_train['pulocationid'].astype(int)
            self.data_test['pulocationid'] = self.data_test['pulocationid'].astype(int)
            self.data_train['dolocationid'] = self.data_train['dolocationid'].astype(int)
            self.data_test['dolocationid'] = self.data_test['dolocationid'].astype(int)
            
            # Rearrange columns
            rearragned_columns = ['vendorid', 'trip_duration', 'passenger_count', 'trip_distance', 'ratecodeid', 'store_and_fwd_flag', 'pulocationid', 'dolocationid', 'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge']
            self.data_train = self.data_train[rearragned_columns]
            self.data_test = self.data_test[rearragned_columns]

            print("Datetime and categorical features converted to respective type successfully.")

        except ValueError as ve:
            print("Error:", ve)


class DataNormalizer:
    """
    Class responsible for preprocessing the loaded dataset. Need to pass the data with first columns as numerical and the last columns as categorical, then indicate how many categorical features are present in the dataset

    Methods:
        _normalize_features(): Normalizes all features using standard scaling for numerical and min-max scalling for categorical.
    """

    def __init__(self, data_loader, categorical_features, numerical_features):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.categorical_features = categorical_features
        self.real_numerical_features = numerical_features

    def normalize_features(self):
        """
        Normalizes features using using standard scaling for numerical and min-max scalling for categorical.
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
            # self.data_loader.data_train = pd.get_dummies(self.data_loader.data_train, columns=self.categorical_features.columns.difference(['pulocationid', 'dolocationid']).tolist())
            # self.data_loader.data_test = pd.get_dummies(self.data_loader.data_test, columns=self.categorical_features.columns.difference(['pulocationid', 'dolocationid']).tolist())
            
            # Normalize encoded features using MinMaxScaler
            scaler = MinMaxScaler()
            self.data_loader.data_train[self.categorical_features] = scaler.fit_transform(
                self.data_loader.data_train[self.categorical_features])
            self.data_loader.data_test[self.categorical_features] = scaler.transform(
                self.data_loader.data_test[self.categorical_features])

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)


class DataCleaner:
    """
    Class for cleaning operations.

    Methods:
        remove_duplicates(): Remove duplicate rows from the dataset.
        handle_missing_values(strategy='mean'): Handle missing values using the specified strategy.
        remove_outliers(threshold=3): Remove outliers from the dataset
    """

    def __init__(self, data_loader):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader

    def remove_duplicates(self):
        """
        Remove duplicate rows from the train dataset.
        """
        try:
            # Check if data and labels are not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")
            if self.data_loader.labels_train is None:
                raise ValueError("Labels have not been loaded yet.")
            
            row_count_before = self.data_loader.data_train.shape[0]
            
            # Remove duplicate rows from training data (do not apply to test data)
            self.data_loader.data_train.drop_duplicates(inplace=True)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            if row_count_before != self.data_loader.data_train.shape[0]:
                print("Duplicate rows removed from training data.")
            else:
                print("No duplicate rows found in the training data.")

        except ValueError as ve:
            print("Error:", ve)

    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values using the specified strategy.

        Parameters:
            strategy (str): The strategy to handle missing values ('mean', 'median', 'most_frequent', or a constant value).
        """
        try:
            # Check if data is not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Check if there are missing values
            if self.data_loader.data_train.isnull().sum().sum() == 0 and self.data_loader.data_test.isnull().sum().sum() == 0:
                print("No missing values found in the data.")
                return
            
            # Check missing values in a column and remove the column if it has more than 50% missing values
            missing_values = self.data_loader.data_train.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            missing_values = missing_values[missing_values > 0.5 * self.data_loader.data_train.shape[0]]
            self.data_loader.data_train.drop(columns=missing_values.index, inplace=True)
            self.data_loader.data_test.drop(columns=missing_values.index, inplace=True)
            if len(missing_values) > 0:
                removed_columns = ', '.join(missing_values.index)
                removed_columns = removed_columns.split(',')
                print(f"{removed_columns} Columns which have more than 50% missing values have been removed from the dataset.")
                
            # Handle missing values based on the specified strategy
            if strategy == 'mean':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mean(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mean(), inplace=True)
            elif strategy == 'median':
                self.data_loader.data_train.fillna(self.data_loader.data_train.median(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.median(), inplace=True)
            elif strategy == 'most_frequent':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mode().iloc[0], inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mode().iloc[0], inplace=True)
            elif strategy == 'fill_nan':
                self.data_loader.data_train.fillna(strategy, inplace=True)
                self.data_loader.data_test.fillna(strategy, inplace=True)
            elif strategy == 'drop':
                self.data_loader.data_train = self.data_loader.data_train.dropna(axis=0)
                self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
                self.data_loader.data_test = self.data_loader.data_test.dropna(axis=0)
                self.data_loader.labels_test = self.data_loader.labels_test[self.data_loader.data_test.index]

            else:
                raise ValueError("Invalid strategy.")
            print("Missing values handled using strategy:", strategy)

        except ValueError as ve:
            print("Error:", ve)

    def _detect_outliers(self, threshold=4):
        """
        Detect outliers in numerical features using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.

        Returns:
            outliers (DataFrame): DataFrame containing the outliers.
        """
        try:
            # Check if test data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Identify numerical features
            numerical_features = self.data_loader.data_train.select_dtypes(include=['number'])

            # Calculate z-scores for numerical features
            z_scores = (numerical_features - numerical_features.mean()) / numerical_features.std()

            # Find outliers based on threshold
            outliers = self.data_loader.data_train[(z_scores.abs() > threshold).any(axis=1)]

            return outliers

        except ValueError as ve:
            print("Error:", ve)

    def remove_outliers(self, threshold=3.5):
        """
        Remove outliers from the dataset using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.
        """
        try:
            # Check if data_loader.data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Detect outliers
            outliers = self._detect_outliers(threshold)

            # Remove outliers from the dataset
            self.data_loader.data_train = self.data_loader.data_train.drop(outliers.index)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Outliers removed from the dataset.")

        except ValueError as ve:
            print("Error:", ve)