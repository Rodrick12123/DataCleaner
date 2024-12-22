import pandas as pd
import re
from sklearn.ensemble import IsolationForest


import numpy as np

class CSVFeatureAnalyzer:

    def feature_has_too_many_outliers(data, threshold=1.5, outlier_percentage_threshold=0.1):
        """
        Check for outliers using the IQR method and return True if the number of outliers exceeds the specified percentage of the data.
        
        Args:
            data: The data to check for outliers.
            threshold: The IQR multiplier to determine outlier bounds.
            outlier_percentage_threshold: The maximum percentage of outliers allowed (default is 0.1 for 10%).
        
        Returns:
            bool: True if too many outliers are found, False otherwise.
        """
        if pd.api.types.is_numeric_dtype(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            
            # Count the number of outliers
            outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Calculate the total number of data points
            total_count = len(data)
            
            # Calculate the maximum allowed outliers based on the percentage threshold
            max_allowed_outliers = total_count * outlier_percentage_threshold
            
            # Return True if the number of outliers exceeds the allowed threshold
            return outlier_count > max_allowed_outliers
        
        return False
    
    @staticmethod
    def feature_is_duration_related(column_name, data):
        """
        Check if a column is likely duration-related based on its values.
        Duration-related columns are often differences between dates or times (e.g., '5 days', '3 hours').
        This function handles both string durations (e.g., '3 hours') and numeric durations (e.g., '142' as hours or minutes).
        """
        non_na_data = data.dropna()

        # Sample a few values (ensure we sample only non-NaN values)
        sample_values = non_na_data.sample(min(10, len(non_na_data)))  # Sample a few values, but no more than the available data

        # Check if any of the values are datetime, which should be excluded for duration check
        if sample_values.apply(lambda x: isinstance(x, pd.Timestamp)).any():
            return False  # If any value is a datetime, it's likely date-related

        # Check if the column contains recognizable duration formats (e.g., '5 days', '3 hours', etc.)
        duration_pattern = r'(\d+)\s*(days?|hours?|minutes?)'  # Pattern for matching durations like "5 days" or "3 hours"
        if sample_values.apply(lambda x: isinstance(x, str) and bool(re.match(duration_pattern, x.strip()))).any():
            return True  # Likely duration-related
        sample_values = pd.to_numeric(sample_values, errors='coerce').dropna()
        def is_potential_duration(value):
            if isinstance(value, (int, float)):  # Numeric values
                return 0 < value < 10000  # Reasonable range for durations (e.g., seconds, minutes, hours)
            return False
        if sample_values.empty:
            return False
        if sample_values.apply(is_potential_duration).any() :
            return True  # Likely duration-related if we have numbers in a reasonable range

        # Try to convert values to Timedelta (duration-related)
        try:
            pd.to_timedelta(sample_values, errors='raise')
            return True  # Likely duration-related
        except (ValueError, TypeError):
            return False  # Not duration-related
    
    def feature_is_date_related(column_name, data):
        """
        Check if a column is likely date-related based on its values.
        Date-related columns represent absolute points in time (e.g., '2024-12-20').
        """
        # sample_values = data.sample(min(10, len(data))) 
        sample_values = data 
        # Check if any of the values are timedelta, which should be excluded for date check
        if sample_values.apply(lambda x: isinstance(x, pd.Timedelta)).any():
            return False  # If any value is a Timedelta, it's likely duration-related
        
        # Check for date-like patterns (e.g., '2024-12-20', '1995-02-10', etc.)
        date_pattern = r'\d{4}-\d{2}-\d{2}'  # Common date format (YYYY-MM-DD)
        if sample_values.apply(lambda x: isinstance(x, str) and bool(re.match(date_pattern, x.strip()))).any():
            return True  # Likely date-related
        
        try:
            pd.to_datetime(sample_values, errors='raise')
            return True  # Likely date-related
        except (ValueError, TypeError):
            return False  # Not date-related
        
    def feature_is_datetime_related(column_name, data):
        """
        Check if a column is likely to be datetime-related, duration-related, or contain time-related data.

        Args:
            column_name (str): The name of the column.
            data (pd.Series): The data in the column.

        Returns:
            bool: True if the column is datetime-related or duration-related, False otherwise.
        """
        # Constants
        datetime_keywords = ['date', 'time', 'timestamp', 'duration', 'elapsed', 'interval', 'created', 'updated']
        duration_keywords = ['day', 'hour', 'minute', 'second', 'week', 'month', 'year', 'duration']
        duration_pattern = r'(\d+\s*(day|hour|minute|second|week|month|year)s?)|(\d+[hms])'

        # Sampling size (10% of data or 10 rows)
        sample_size = max(1, min(10, len(data)))  # Ensure sample is between 1 and 10 rows

        # Check if the column name contains any common date/time-related keywords
        for keyword in datetime_keywords:
            if keyword in column_name.lower():
                print(f"Date/Time keyword '{keyword}' detected in column name '{column_name}'.")
                return True

        # Sample the data to inspect the values
        sample_data = data.sample(min(sample_size, len(data)))

        # Check if the data is numeric, if so, it can't be date/time-related
        if pd.api.types.is_numeric_dtype(sample_data):
            return False

        # Convert sample data to string type to safely apply string-based operations
        sample_data_str = sample_data.astype(str)

        # Try parsing the sample data to datetime using pandas' to_datetime function
        try:
            parsed_dates = pd.to_datetime(sample_data_str, errors='coerce')
            # Check if a significant portion (80%) of the data can be parsed as datetime
            if parsed_dates.notna().sum() >= 0.8 * len(sample_data):
                print(f"Datetime detected in column: {column_name}")
                return True
        except (ValueError, TypeError):
            pass  # If conversion fails, continue to the next checks

        # Check for duration-like patterns using regex
        matching_count = sample_data_str.str.contains(duration_pattern, case=False, regex=True).sum()
        if matching_count >= 0.8 * len(sample_data):  # Require 80% of samples to match
            print(f"Duration detected in column: {column_name}")
            return True

        return False


    def feature_has_poor_distribution( data, skew_threshold=1):
        """Check for features with poor distribution based on skewness."""
        if pd.api.types.is_numeric_dtype(data):
            skewness = data.skew()
            return abs(skewness) > skew_threshold
        return False

    def feature_has_high_noise(data, noise_threshold=0.1):
        """Check for features with high noise levels based on standard deviation."""
        if pd.api.types.is_numeric_dtype(data):
            std_dev = data.std()
            mean = data.mean()
            return (std_dev / mean) > noise_threshold if mean != 0 else False
        return False
    


    def feature_has_constant_values( data):
        """Check for constant features with only one unique value."""
        return len(data.unique()) <= 1

    def feature_has_high_cardinality(data, threshold=0.9):
        """Check for high cardinality features (e.g., many unique values)."""
        unique_ratio = len(data.unique()) / len(data)
        return unique_ratio > threshold


    # def feature_has_sparse_categories( data, min_frequency=2):
    #     """Check for sparse categories with very few occurrences."""
    #     value_counts = data.value_counts()
    #     return (value_counts < min_frequency).sum() > 0

    def feature_includes_placeholders(data):
        """Check for placeholder or dummy variables."""
        common_placeholders = ["unknown", "n/a", "none", "placeholder"]
        return data.astype(str).str.lower().isin(common_placeholders).mean() > 0.5

    def feature_is_temporal(data):
        """Check for features containing date or timestamp-like data."""
        try:
            pd.to_datetime(data)
            return True
        except (ValueError, TypeError):
            return False

    def feature_is_redundant(original_data, data, other_columns):
        """Check for redundancy by comparing with other columns."""
        return any(data.equals(original_data[col]) for col in other_columns if col != data.name)

    def feature_has_sparse_values( data):
        """Check for sparse features with mostly missing or empty values."""
        return data.isna().mean() > 0.5

    def feature_is_phone_number( data):
        """Identify columns that may contain phone numbers."""

        # Check if column contains mostly numeric data or has phone-like patterns
        if data.dtype == 'object' or pd.api.types.is_numeric_dtype(data):
            phone_pattern = re.compile(r'^\+?[\d\s\-()]{7,15}$')  # Phone number regex
            match_count = data.astype(str).str.match(phone_pattern).sum()
            return match_count / len(data) > 0.8  # Threshold: 80% matches indicate phone numbers
        return False

    def feature_is_zip_codes( data):
        """
        Enhanced ZIP code check for large datasets.
        Parameters:
            - data: Pandas Series
            - sample_fraction: Fraction of data to sample for large datasets
            - max_unique_ratio: Maximum ratio of unique values to total values
        """
        if pd.api.types.is_numeric_dtype(data):
            # Check numeric range for typical ZIP codes
            within_range = (data >= 1000) & (data <= 99999)

            # Check cardinality using nunique
            unique_ratio = data.nunique() / len(data)

            # Combine range check and cardinality
            if within_range.mean() > 0.9 and unique_ratio < .1:
                return True
        elif data.dtype == 'object':
            # Regex to match ZIP code patterns (5 digits or 5+4 format)
            zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
            
            # Use vectorized string matching with a sample if necessary
            match_count = data.astype(str).str.match(zip_pattern).sum()

            # Check if most values match the ZIP code format
            if match_count / len(data) > 0.8:
                return True

        return False


    def feature_is_unique_Id(data, uniqueness_threshold=0.9, duplicate_threshold=0.1):
        """
        Identify columns that are likely to be IDs or unique identifiers.

        Args:
            data (pd.Series): A single column of the dataset to analyze.
            uniqueness_threshold (float): The minimum proportion of unique values required (default: 0.9).
            duplicate_threshold (float): The maximum proportion of duplicate values allowed (default: 0.1).

        Returns:
            bool: True if the column is identified as a potential ID, False otherwise.
        """
        # Ensure input is a pandas Series
        if not isinstance(data, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        # Handle empty Series
        if data.empty:
            return False  # An empty Series cannot be a unique ID

        # Calculate the proportion of unique values
        unique_ratio = data.nunique() / len(data)

        # Calculate the proportion of duplicate values
        duplicate_ratio = 1 - unique_ratio

        # Check if the column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(data)

        # Additional numeric checks for meaningful range and patterns
        if is_numeric:
            # Safely filter out non-finite values
            numeric_data = pd.to_numeric(data, errors='coerce').dropna()
            numeric_data = numeric_data[np.isfinite(numeric_data)]  # Remove inf/-inf values
            meaningful_range = numeric_data.max() - numeric_data.min() > 1e-5
            is_integer_like = np.array_equal(numeric_data, numeric_data.astype(int))  # Check if values are integer-like
        else:
            meaningful_range = False
            is_integer_like = False

        # Additional checks for potential patterns in string data
        if pd.api.types.is_string_dtype(data):
            # Check if all values follow a pattern like UUID, alphanumeric, etc.
            string_pattern_consistent = all(data.str.match(r"^[A-Za-z0-9\-_]+$").fillna(False))
        else:
            string_pattern_consistent = False

        # Conditions for identifying an ID
        is_potential_id = (
            unique_ratio >= uniqueness_threshold  # Mostly unique values
            and duplicate_ratio <= duplicate_threshold  # Low duplicates
            and (
                (not is_numeric)  # Non-numeric data
                or (is_numeric and not meaningful_range and is_integer_like)  # Integer-like numeric data with no meaningful range
                or (pd.api.types.is_string_dtype(data) and string_pattern_consistent)  # String data must match the pattern
            )
            and (not is_numeric or not meaningful_range)  # Exclude numeric with meaningful range
        )

        return is_potential_id



    def feature_has_outliers_ml(data, contamination=0.1):
        """Detect outliers using Isolation Forest."""
        if pd.api.types.is_numeric_dtype(data):
            model = IsolationForest(contamination=contamination)
            outliers = model.fit_predict(data.values.reshape(-1, 1))
            return (outliers == -1).sum() > (len(data) * contamination)
        return False


    