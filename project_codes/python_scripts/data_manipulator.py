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

    def __init__(self, filename, labels_test, test_size=0.1, random_state=None, task='regression'):
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
        self.task = task

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
            
            if self.task == 'classification':
                # Create fare classes based on total_amount
                conditions = [
                    (y < 10),
                    (y >= 10) & (y < 30),
                    (y >= 30) & (y < 60),
                    (y >= 60)
                ]
                choices = [1, 2, 3, 4]
                y = pd.Series(np.select(conditions, choices, default=0), index=y.index)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                random_state=self.random_state)

            # Assign the data and labels to attributes
            self.data_train = X_train
            self.labels_train = y_train
            self.data_test = X_test
            self.labels_test = y_test

            print("Data loaded successfully.")
            
            print("--- Initial Data Info ---")
            df.info()
            print("\n--- Initial Descriptive Stats ---")
            print(df.describe(include='all'))
            print("\n--- Missing Values ---")
            print(df.isnull().sum())

        except FileNotFoundError:
            print("File not found. Please check the file path.")


class DataManipulator(DataLoader):
    """
    A subclass of DataLoader that specializes to manipulate the forest cover types dataset.

    Methods:
        _convert_features(): Converts columns to appropriate data types.
        _engineer_base_features(): Engineers base features such as trip duration.
        _handle_data_leakages(): Handles data leakage by removing features that can provide information leakage.
    """

    def __init__(self, filename, labels_test, test_size=0.1, random_state=None, task='regression'):
        """
        Initializes the DataManipulator with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        super().__init__(filename, labels_test, test_size, random_state, task=task)

        # Manipulate data
        self._convert_features()
        self._engineer_base_features()
        self._handle_data_leakages()

    def _convert_features(self):
        """
        Convert columns to appropriate data types.
        """
        print("\n--- Converting Data Types ---")
        try:
            # Convert datetime columns to datetime type
            datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
            for col in datetime_cols:
                self.data_train[col] = pd.to_datetime(self.data_train[col], errors='coerce')
                self.data_test[col] = pd.to_datetime(self.data_test[col], errors='coerce')

            # Categorical features
            self.data_train['store_and_fwd_flag'] = self.data_train['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
            self.data_test['store_and_fwd_flag'] = self.data_test['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
            categorical_cols = ['vendorid', 'ratecodeid', 'pulocationid', 'dolocationid', 'payment_type', 'store_and_fwd_flag']
            for col in categorical_cols:
                self.data_train[col] = self.data_train[col].astype('int')
                self.data_test[col] = self.data_test[col].astype('int')
            print("Data types converted.")

        except ValueError as ve:
            print("Error:", ve)

    def _engineer_base_features(self):
        print("\n--- Engineering Base Features ---")
        # Trip duration
        self.data_train['trip_duration_secs'] = (self.data_train['tpep_dropoff_datetime'] - self.data_train['tpep_pickup_datetime']).dt.total_seconds()
        self.data_test['trip_duration_secs'] = (self.data_test['tpep_dropoff_datetime'] - self.data_test['tpep_pickup_datetime']).dt.total_seconds()
        # self.data_train['trip_duration_mins'] = self.data_train['trip_duration_secs'] / 60.0
        # self.data_test['trip_duration_mins'] = self.data_test['trip_duration_secs'] / 60.0

    def _handle_data_leakages(self):
        """
        Handling Data Leakage from the dataset.
        """
        print("\n--- Handling Data Leakage ---")
        # These features are direct components or results of the fare calculation including the fare itself.
        # tip_amount is added after fare is known. total_amount includes fare_amount.
        # MTA_tax is triggered based on the metered rate in use.
        # Extra is Miscellaneous extras and surcharge.
        leakage_cols = ['total_amount', 'tip_amount', 'extra', 'mta_tax', 'improvement_surcharge']
    
        # Drop features that can provide information leakage
        self.data_train.drop(columns=leakage_cols, inplace=True)
        self.data_test.drop(columns=leakage_cols, inplace=True)