class TaxiFeatureEngineer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def create_datetime_features(self):
        print("\n--- Engineering Datetime Features ---")
        
        for df in [self.data_loader.data_train, self.data_loader.data_test]:
            # 1. Pickup Hour
            df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
            # 2. Pickup Day of Week (Monday=0, Sunday=6)
            df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
            # 3. Pickup Month
            df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
            # 4. Is Weekend?
            df['pickup_is_weekend'] = df['pickup_day'].isin([5, 6]).astype(int)
            # 5. Time of Day Segment (simple version)
            def assign_time_segment(hour):
                if 5 <= hour < 12: return 'Morning'
                elif 12 <= hour < 17: return 'Afternoon'
                elif 17 <= hour < 21: return 'Evening'
                else: return 'Night'
            df['time_of_day_segment'] = df['pickup_hour'].apply(assign_time_segment).astype('category')
            # 6. Rush Hour Pickup
            # Weekdays: 7-9 AM (morning rush) or 5-7 PM (15-19 evening rush)
            df['is_rush_hour'] = (((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) | \
                                    ((df['pickup_hour'] >= 15) & (df['pickup_hour'] <= 19))) & \
                                    (df['pickup_day'] < 5) # Monday to Friday
            df['is_rush_hour'] = df['is_rush_hour'].astype(int)

        print("Datetime features created: pickup_hour, pickup_day, pickup_month, pickup_is_weekend, time_of_day_segment, is_rush_hour.")

    def create_trip_features(self):
        print("\n--- Engineering Trip Features ---")
        for df in [self.data_loader.data_train, self.data_loader.data_test]:
            # 7. Is Airport Trip (based on RateCodeID from data dictionary
            # RateCodeID: 2=JFK, 3=Newark
            df['is_airport_trip'] = df['ratecodeid'].isin([2, 3]).astype(int)
            print("Feature 'is_airport_trip' created.")

            # 8. Pickup and Dropoff in Same Location
            df['is_same_location_trip'] = (df['pulocationid'] == df['dolocationid']).astype(int)
            print("Feature 'is_same_location_trip' created.")

            # 9. Average Speed (mph)
            # trip_duration_hours, ensuring no division by zero
            trip_duration_hours = df['trip_duration_secs'] / 3600.0
            df['average_speed_mph'] = df['trip_distance'] / trip_duration_hours.replace(0, np.nan)
            df['average_speed_mph'].fillna(0, inplace=True) # If duration was 0, speed is 0
            # Cap speed at a reasonable max, e.g., 100 mph
            df['average_speed_mph'] = np.clip(df['average_speed_mph'], 0, 100)
            print("Feature 'average_speed_mph' created.")
        
    def create_cyclical_time_features(self):
        print("\n--- Engineering Cyclical Time Features ---")
        for df in [self.data_loader.data_train, self.data_loader.data_test]:
            # These can help models understand the cyclical nature of time.
            # 10. Hour Sin/Cos
            if 'pickup_hour' in df.columns:
                df['pickup_hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24.0)
                df['pickup_hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24.0)
                print("Cyclical features 'pickup_hour_sin', 'pickup_hour_cos' created.")

            # 11. Day of Week Sin/Cos
            if 'pickup_day' in df.columns:
                df['pickup_day_sin'] = np.sin(2 * np.pi * df['pickup_day'] / 7.0)
                df['pickup_day_cos'] = np.cos(2 * np.pi * df['pickup_day'] / 7.0)
                print("Cyclical features 'pickup_day_sin', 'pickup_day_cos' created.")
