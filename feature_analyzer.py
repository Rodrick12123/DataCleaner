import pandas as pd
import re

class FeatureAnalyzer:

    def check_constant_features(self, data):
        """Check for constant features with only one unique value."""
        return len(data.unique()) <= 1

    def check_high_cardinality(self, data, threshold=0.9):
        """Check for high cardinality features (e.g., many unique values)."""
        unique_ratio = len(data.unique()) / len(data)
        return unique_ratio > threshold

    def check_sparse_categories(self, data, min_frequency=2):
        """Check for sparse categories with very few occurrences."""
        value_counts = data.value_counts()
        return (value_counts < min_frequency).sum() > 0

    def check_placeholder_features(self, data):
        """Check for placeholder or dummy variables."""
        common_placeholders = ["unknown", "n/a", "none", "placeholder"]
        return data.astype(str).str.lower().isin(common_placeholders).mean() > 0.5

    def check_identifiers(self, data):
        """Check for unique identifiers or IDs."""
        return len(data.unique()) == len(data)

    def check_date_or_timestamp(self, data):
        """Check for features containing date or timestamp-like data."""
        try:
            pd.to_datetime(data)
            return True
        except (ValueError, TypeError):
            return False

    def check_redundant_features(self, data, other_columns):
        """Check for redundancy by comparing with other columns."""
        return any(data.equals(self.data[col]) for col in other_columns if col != data.name)

    def check_sparse_values(self, data):
        """Check for sparse features with mostly missing or empty values."""
        return data.isna().mean() > 0.5




    def csv_feature_is_phone_number(self, data,sample_fraction=0.2):
        """Identify columns that may contain phone numbers."""

        if len(data) > 1e5:  # Arbitrary threshold for large datasets
            data = data.sample(frac=sample_fraction, random_state=42)

        # Check if column contains mostly numeric data or has phone-like patterns
        if data.dtype == 'object' or pd.api.types.is_numeric_dtype(data):
            phone_pattern = re.compile(r'^\+?[\d\s\-()]{7,15}$')  # Phone number regex
            match_count = data.astype(str).str.match(phone_pattern).sum()
            return match_count / len(data) > 0.8  # Threshold: 80% matches indicate phone numbers
        return False

    def csv_feature_is_zip_codes(self, data, sample_fraction=0.2):
        """
        Enhanced ZIP code check for large datasets.
        Parameters:
            - data: Pandas Series
            - sample_fraction: Fraction of data to sample for large datasets
            - max_unique_ratio: Maximum ratio of unique values to total values
        """
        # For very large datasets, sample data
        if len(data) > 1e5:  # Arbitrary threshold for large datasets
            data = data.sample(frac=sample_fraction, random_state=42)

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


    def csv_feature_is_unique_Id(self, data, uniqueness_threshold=0.9, duplicate_threshold=0.1):
        """
        Identify columns that are likely to be IDs or unique identifiers.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            uniqueness_threshold (float): The minimum proportion of unique values required (default: 0.9).
            duplicate_threshold (float): The maximum proportion of duplicate values allowed (default: 0.1).

        Returns:
            list: Columns identified as potential IDs.
        """
        
        # for very large datasets, sample data
        if len(data) > 1e5:  # Arbitrary threshold for large datasets
            data = data.sample(frac=.2, random_state=42)
            
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
        