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

    def feature_is_datetime_related(column_name, data):
        """
        Check if a column is likely to be datetime-related based on its name and sample data.

        Args:
            column_name (str): The name of the column.
            sample_data (pd.Series): A sample of the data in the column.

        Returns:
            bool: True if the column is likely datetime-related, False otherwise.
        """
        # Check if the data is numeric
        if pd.api.types.is_numeric_dtype(data):
            return False  # Numeric values should not be considered datetime-related
        
        # Check for common keywords in the column name
        datetime_keywords = ['date', 'time', 'timestamp']
        if any(keyword in column_name.lower() for keyword in datetime_keywords):
            return True

        # Check if the sample data can be converted to datetime
        try:
            pd.to_datetime(data, errors='raise')
            return True
        except (ValueError, TypeError):
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
            meaningful_range = data.max() - data.min() > 1e-5
            is_integer_like = np.array_equal(data, data.astype(int))  # Check if values are integer-like
            
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


    