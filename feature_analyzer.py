import pandas as pd
import re

class CSVFeatureAnalyzer:

    def feature_has_outliers(self, data, threshold=1.5, outlier_percentage_threshold=0.1):
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

    def feature_has_poor_distribution(self, data, skew_threshold=1):
        """Check for features with poor distribution based on skewness."""
        if pd.api.types.is_numeric_dtype(data):
            skewness = data.skew()
            return abs(skewness) > skew_threshold
        return False

    def feature_has_high_noise(self, data, noise_threshold=0.1):
        """Check for features with high noise levels based on standard deviation."""
        if pd.api.types.is_numeric_dtype(data):
            std_dev = data.std()
            mean = data.mean()
            return (std_dev / mean) > noise_threshold if mean != 0 else False
        return False
    


    def feature_has_constant_values(self, data):
        """Check for constant features with only one unique value."""
        return len(data.unique()) <= 1

    def feature_has_high_cardinality(self, data, threshold=0.9):
        """Check for high cardinality features (e.g., many unique values)."""
        unique_ratio = len(data.unique()) / len(data)
        return unique_ratio > threshold

    def feature_has_sparse_categories(self, data, min_frequency=2):
        """Check for sparse categories with very few occurrences."""
        value_counts = data.value_counts()
        return (value_counts < min_frequency).sum() > 0

    def feature_includes_placeholders(self, data):
        """Check for placeholder or dummy variables."""
        common_placeholders = ["unknown", "n/a", "none", "placeholder"]
        return data.astype(str).str.lower().isin(common_placeholders).mean() > 0.5

    def feature_is_temporal(self, data):
        """Check for features containing date or timestamp-like data."""
        try:
            pd.to_datetime(data)
            return True
        except (ValueError, TypeError):
            return False

    def feature_is_redundant(self, data, other_columns):
        """Check for redundancy by comparing with other columns."""
        return any(data.equals(self.data[col]) for col in other_columns if col != data.name)

    def feature_has_sparse_values(self, data):
        """Check for sparse features with mostly missing or empty values."""
        return data.isna().mean() > 0.5

    def feature_is_phone_number(self, data):
        """Identify columns that may contain phone numbers."""

        # Check if column contains mostly numeric data or has phone-like patterns
        if data.dtype == 'object' or pd.api.types.is_numeric_dtype(data):
            phone_pattern = re.compile(r'^\+?[\d\s\-()]{7,15}$')  # Phone number regex
            match_count = data.astype(str).str.match(phone_pattern).sum()
            return match_count / len(data) > 0.8  # Threshold: 80% matches indicate phone numbers
        return False

    def feature_is_zip_codes(self, data):
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


    def feature_is_unique_Id(self, data, uniqueness_threshold=0.9, duplicate_threshold=0.1):
        """
        Identify columns that are likely to be IDs or unique identifiers.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            uniqueness_threshold (float): The minimum proportion of unique values required (default: 0.9).
            duplicate_threshold (float): The maximum proportion of duplicate values allowed (default: 0.1).

        Returns:
            list: Columns identified as potential IDs.
        """
            
        # Calculate the proportion of unique values
        unique_ratio = data.nunique() / len(data)
        
        # Calculate the proportion of duplicate values
        duplicate_ratio = 1 - unique_ratio

        # Check if the column is numeric and spans a meaningful range
        if pd.api.types.is_numeric_dtype(data):
            meaningful_range = data.max() - data.min() > 1e-5
        else:
            meaningful_range = False

        # Conditions for identifying an ID
        if (
            unique_ratio >= uniqueness_threshold  # Mostly unique values
            and duplicate_ratio <= duplicate_threshold  # Low duplicates
            and not meaningful_range  # Exclude numeric columns with a meaningful range
        ):
            return True
        else:
            return False
        