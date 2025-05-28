class TaxiHypothesisTester:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def test_distance_vs_fare(self, distance_col='trip_distance', target_col='fare_amount'):
        print(f"\n--- Hypothesis Test 1: {distance_col} vs {target_col} ---")
        if distance_col not in self.data_loader.data_train.columns:
            print(f"Required columns {distance_col} not found.")
            return

        subset_df = pd.concat([self.data_loader.data_train[distance_col], self.data_loader.labels_train], axis=1).dropna()
        if len(subset_df) < 2:
            print("Not enough data points for correlation test after dropping NaNs.")
            return

        print(f"H0: There is no significant linear correlation between {distance_col} and {target_col}.")
        print(f"HA: There is a significant linear correlation between {distance_col} and {target_col}.")
        
        correlation, p_value = stats.pearsonr(subset_df[distance_col], subset_df[target_col])
        print(f"Pearson Correlation: {correlation:.4f}, P-value: {p_value:.4g}")
        alpha = 0.05
        if p_value < alpha:
            print(f"Conclusion: Reject H0. There is a statistically significant linear correlation between {distance_col} and {target_col}.")
        else:
            print(f"Conclusion: Fail to reject H0. There is no statistically significant linear correlation.")

    def test_fare_by_ratecode(self, rate_col='ratecodeid', target_col='fare_amount'):
        print(f"\n--- Hypothesis Test 2: {target_col} by {rate_col} ---")
        if rate_col not in self.data_loader.data_train.columns:
            print(f"Required columns '{rate_col}' not found.")
            return
        
        # ANOVA requires at least two groups
        grouped_data = [group[target_col].dropna().values for name, group in pd.concat([self.data_loader.data_train, self.data_loader.labels_train], axis=1).groupby(rate_col) if len(group[target_col].dropna()) >= 2]

        if len(grouped_data) < 2:
            print(f"Not enough groups (found {len(grouped_data)}) in '{rate_col}' with sufficient data for ANOVA test.")
            return

        print(f"H0: The mean {target_col} is the same across all {rate_col} groups.")
        print(f"HA: The mean {target_col} differs for at least one {rate_col} group.")

        # Check for normality and homogeneity of variances if being extremely rigorous,
        # but ANOVA is somewhat robust. For simplicity, directly apply.
        f_statistic, p_value = stats.f_oneway(*grouped_data)
        print(f"ANOVA F-statistic: {f_statistic:.4f}, P-value: {p_value:.4g}")
        alpha = 0.05
        if p_value < alpha:
            print(f"Conclusion: Reject H0. There is a statistically significant difference in mean {target_col} across {rate_col} groups.")
        else:
            print(f"Conclusion: Fail to reject H0. No statistically significant difference found.")