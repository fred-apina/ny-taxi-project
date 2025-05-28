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

        except ValueError as ve:
            print("Error:", ve)

    def remove_outliers(self):
        """
        Remove outliers from the dataset.
        """
        try:
            # Check if data_loader.data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Remove outliers whose fare_amount <= 0 and > 300
            self.data_loader.data_train = self.data_loader.data_train[(self.data_loader.labels_train > 0) & (self.data_loader.labels_train <= 300)]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            # Remove outliers whose passenger_count < 1
            self.data_loader.data_train = self.data_loader.data_train[self.data_loader.data_train['passenger_count'] > 0]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
            
            # Remove outliers whose ratecodeid > 6
            # Handle ratecodeid as categorical with valid values 1-6
            valid_ratecodes = [1, 2, 3, 4, 5, 6]
            self.data_loader.data_train = self.data_loader.data_train[self.data_loader.data_train['ratecodeid'].astype(int).isin(valid_ratecodes)]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
            
            valid_vendorid = [1, 2]
            self.data_loader.data_train = self.data_loader.data_train[self.data_loader.data_train['vendorid'].astype(int).isin(valid_vendorid)]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            # Remove outliers whose trip_distance <= 0 and > 60
            self.data_loader.data_train = self.data_loader.data_train[(self.data_loader.data_train['trip_distance'] > 0) & (self.data_loader.data_train['trip_distance'] <= 60)]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
            
            # Remove outliers whose trip_duration_secs is outside the range of < 0  and >= 10000.0
            self.data_loader.data_train = self.data_loader.data_train[(self.data_loader.data_train['trip_duration_secs']> 0) & (self.data_loader.data_train['trip_duration_secs'] <= 10000.0)]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            # Remove outliers whose tolls_amount is outside the range of <= 0  and > 7.42
            self.data_loader.data_train = self.data_loader.data_train[(self.data_loader.data_train['tolls_amount'] >= 0) & (self.data_loader.data_train['tolls_amount'] <= 7.42)]
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Outliers removed from the dataset.")

        except ValueError as ve:
            print("Error:", ve)