import os
import re
import string
import chardet
import pandas as pd
from feature_analyzer import CSVFeatureAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np



class CSVCleaner:
    def __init__(self, file_path):
        """
        Initialize the CSVCleaner with the file path and load the data.
        
        Args:
            file_path (str): The path to the CSV file to be cleaned.
        """
        self.file_path = file_path
        self.data = self.load_data(file_path)
        self.init_number_rows = len(self.data)
        self.init_number_cols = self.data.shape[1]
        self.cols_removed = []
        self.rows_removed = []
        self.percentage_rows_removed = 0
        self.percentage_cols_removed = 0
        self.comment_columns = []
        self.protected_columns = []

    def detect_delimiter(self, file_path):
        """
        Detect the delimiter used in the CSV file based on the first line.
        
        Args:
            file_path (str): The path to the CSV file.
        
        Returns:
            str: The detected delimiter.
        """
        with open(file_path, 'r') as f:
            first_line = f.readline()
            # Check for common delimiters
            if ',' in first_line:
                return ','
            elif ';' in first_line:
                return ';'
            elif '\t' in first_line:
                return '\t'
            else:
                return ','  # Default to comma if no common delimiter is found

    def load_data(self, file_path):
        """
        Load the CSV data into a DataFrame, handling encoding and delimiter detection.
        
        Args:
            file_path (str): The path to the CSV file.
        
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Detect the file encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))  # Read the first 10,000 bytes
            encoding = result['encoding']
            print(f"Detected encoding: {encoding}")

        # Detect the delimiter
        delimiter = self.detect_delimiter(file_path)
        print(f"Detected delimiter: {delimiter}")
        
        # Try reading the CSV with detected encoding and dynamic delimiter
        try:
            data = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, on_bad_lines='warn')
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing the file: {e}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {e}")

        # Check if the DataFrame is empty
        if data.empty:
            raise ValueError("The DataFrame is empty after loading the data.")
        

        return data

    def convert_to_datetime(self, column):
        """
        Convert a column to datetime format with error handling.
        
        Args:
            column (pd.Series): The column to convert.
        
        Returns:
            pd.Series: The column converted to datetime format.
        """
        # Input validation
        if not isinstance(column, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        if column.empty:
            print("Warning: The input column is empty. Returning an empty Series.")
            return column

        # Attempt to convert to datetime
        try:
            print(f"Converting column '{column}' to datetime format.")
            converted_column = pd.to_datetime(column, errors='raise')  
            return converted_column
        except ValueError as ve:
            print(f"ValueError: {ve}. Unable to convert the column to datetime.")
            return column  # Return the original column if conversion fails
        except TypeError as te:
            print(f"TypeError: {te}. The column contains non-convertible data.")
            return column  # Return the original column if conversion fails
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            return column  # Return the original column if conversion fails
        
    def process_datetime_column(self, column_name):
        negligible_duration = 0

        # Check if the column contains more duration-related or date-related values
        if CSVFeatureAnalyzer.feature_is_duration_related(column_name, self.data[column_name]):
            print(f"Majority values in column '{column_name}' are durations (numeric values).")
            self.data[column_name] = self.data[column_name].apply(
                lambda x: x.total_seconds() if isinstance(x, pd.Timedelta)  # Convert Timedelta to seconds
                else pd.to_numeric(x, errors='coerce') if isinstance(x, object)  # Convert object to numeric if possible
                else x  # Keep numeric values as is (int or float)
            )

            self.data[column_name] = pd.to_numeric(self.data[column_name], errors='coerce')
            self.data[column_name].fillna(negligible_duration, inplace=True)
            

        elif CSVFeatureAnalyzer.feature_is_date_related(column_name, self.data[column_name]):
            print(f"Majority values in column '{column_name}' are date-time values.")

            negligible_duration = 0  # Value to replace invalid durations or missing values

            first_valid_date = self.data[column_name].dropna().iloc[0]
            
            # inferred_format = pd.to_datetime(first_valid_date, errors='coerce')

            self.data[column_name] = pd.to_datetime(self.data[column_name], errors='coerce')

            if self.data[column_name].isnull().any():
                print(f"Some values in column '{column_name}' could not be converted to datetime.")

                self.data[column_name].fillna(pd.Timestamp('1970-01-01'), inplace=True)
            
            #possible encoding step
            # min_timestamp = self.data[column_name].min()
            # self.data[column_name] = self.data[column_name].apply(
            #     lambda x: (x - min_timestamp).total_seconds() if isinstance(x, pd.Timestamp) else negligible_duration
            # )
            

        else:
            print(f"Unable to classify column '{column_name}' as either duration or date-time related.")

    def analyze_features(self, force_remove_column=[]):
        """
        Analyze features in the DataFrame to identify irrelevant features and convert types.
        """
        irrelevant_features = []
        # Create a new list for renamed columns
        self.rename_duplicate_columns()

        for column in self.data.columns:
            # Check for object dtype and convert to numeric if necessary
            self.feature_object_to_numeric_fillna(column)

            if column in self.data.columns:

                # Make optional
                if CSVFeatureAnalyzer.feature_is_datetime_related(column, self.data[column]):
                    
                    self.process_datetime_column(column)
                    self.protected_columns.append(column)
                    



                if self.is_irrelevant_feature(column, force_remove_column=force_remove_column):
                    irrelevant_features.append(column)

        # Call remove features on irrelevant_features
        if len(irrelevant_features) > 0:
            if self.init_number_cols > 0:
                self.percentage_cols_removed = (len(irrelevant_features) / self.init_number_cols) * 100
            else:
                self.percentage_cols_removed = 0  # Handle the case where there are no initial columns

            if self.percentage_cols_removed < 50:
                self.remove_irrelevant_features(irrelevant_features)  # Call the method to remove the columns

    def rename_duplicate_columns(self):
        """
        Automatically rename duplicate columns by appending a suffix.
        """
        new_columns = []
        for col in self.data.columns:
            if col in new_columns:
                # If the column name is already in the new list, append a suffix
                count = new_columns.count(col) + 1
                new_columns.append(f"{col}_{count}")  # Append a suffix
            else:
                new_columns.append(col)  # Keep the original name

        # Assign the new column names to the DataFrame
        self.data.columns = new_columns

    def feature_object_to_numeric_fillna(self, column, fill="mean", threshold=0.5):
        """
        Convert strings to numeric and fill NaN values with mean, mode, etc.
        If the percentage of missing values exceeds the threshold, remove the column.
        
        Args:
            column (str): The name of the column to process.
            fill (str): The method to fill NaN values ('mean', 'mode', etc.).
            threshold (float): The threshold for the percentage of missing values (default is 0.5 for 50%).
        
        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        
        # Skip numeric conversion for columns not predominantly numeric
        if self.data[column].dtype == 'object':
            # Skip processing for ID-like columns
            if "id" in column.lower():
                print(f"Column '{column}' identified as an ID. Skipping processing.")
                return self.data
            
            if CSVFeatureAnalyzer.feature_is_datetime_related(column, self.data[column]):
                print(f"Column '{column}' identified as date-time related. Skipping processing.")

                return self.data

            # Drop NaN values and sample the data
            non_na_data = self.data[column].dropna()
            sample_size = min(len(non_na_data), 100)  # Limit sample size to 100
            if sample_size == 0:
                print(f"Column '{column}' contains only NaN values. Skipping processing.")
                return self.data

            # Preprocess and check numeric dominance
            def preprocess_numeric_like(value):
                """
                Preprocess values to correct common numerical formatting issues.

                Args:
                    value (str): The raw input value.

                Returns:
                    str: The preprocessed string for numeric checks.
                """
                if isinstance(value, str):
                    # Remove currency symbols (e.g., $, €, £) and any whitespace
                    value = value.replace('$', '').replace('€', '').replace('£', '').strip()

                    # Remove commas (thousands separators)
                    value = value.replace(',', '')

                    # If there are multiple periods ('.'), treat them as thousands separators
                    if value.count('.') > 1:
                        # Split the value by the period
                        parts = value.split('.')
                        # Rejoin all parts except the last one (treat as thousands separator), keeping the last part as the decimal point
                        value = ''.join(parts[:-1]) + '.' + parts[-1]
                    
                    # Return the cleaned value
                    return value
                
                return value

            sample = non_na_data.sample(sample_size, random_state=42)
            preprocessed_sample = sample.apply(preprocess_numeric_like)
            numeric_proportion = preprocessed_sample.apply(
                lambda x: bool(re.match(r'^-?\d+(\.\d+)?$', str(x).strip()))
            ).mean()
            if numeric_proportion < 0.8:  # Threshold for numeric dominance
                print(f"Column '{column}' is not predominantly numeric after preprocessing. Skipping processing.")
                return self.data

            # Replace empty strings and strings with only whitespace with NaN
            self.data[column].replace(r'^\s*$', float('nan'), regex=True, inplace=True)

            # Function to extract numeric values from mixed strings
            def extract_numeric(value):
                if isinstance(value, str):
                    value = value.replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip()
                    if value.count('.') > 1:
                        parts = value.split('.')
                        value = ''.join(parts[:-1]) + '.' + parts[-1]
                    try:
                        return float(value)
                    except ValueError:
                        return float('nan')  # Return NaN for non-convertible strings
                return value

            # Apply the extraction function to the column
            self.data[column] = self.data[column].apply(extract_numeric)

            # Attempt to convert the entire column to numeric
            try:
                self.data[column] = pd.to_numeric(self.data[column], errors='raise')
            except ValueError:
                print(f"Column '{column}' is not numeric after processing.")
                return self.data

        # Calculate the percentage of missing values
        missing_percentage = self.data[column].isna().mean()

        # Handle missing values or remove the column if necessary
        if missing_percentage > threshold:
            print(f"Column '{column}' has {missing_percentage * 100:.2f}% missing values. Removing the column.")
            self.data.drop(columns=[column], inplace=True)
            self.cols_removed.extend([column])
            return self.data

        # Fill NaN values based on the specified method
        if fill == "mean":
            mean_value = self.data[column].mean()
            self.data[column].fillna(mean_value, inplace=True)
        elif fill == "mode":
            mode_value = self.data[column].mode()[0]  # Get the first mode
            self.data[column].fillna(mode_value, inplace=True)

        return self.data

    def non_informative_categorical_feature(self, data):
        """
        Check if the data includes non-informative features (e.g., unique identifiers, IDs).
        
        Args:
            data (pd.Series): The data to check.
        
        Returns:
            bool: True if the data is non-informative.
        """
        if CSVFeatureAnalyzer.feature_is_phone_number(data):
            return True
        
        if CSVFeatureAnalyzer.feature_is_zip_codes(data):
            return True
        
        if CSVFeatureAnalyzer.feature_is_unique_Id(data):
            return True
        
        return False

    def is_irrelevant_feature(self, column, sample_fraction=.2, constants=True,
                                non_informative_categories=True, outliers=False, 
                                redundant_features=True, placeholders=True,
                                temporal_features=False,
                                sparse_values=True, high_cardinality=False,
                                poor_distribution=False, high_noise=False,
                                force_remove_column=[]):
        """
        Identify irrelevant features based on various criteria.
        
        Args:
            column (str): The name of the column to check.
            sample_fraction (float): Fraction of data to sample for large datasets.
            constants (bool): Check for constant features.
            non_informative_categories (bool): Check for non-informative categorical features.
            outliers (bool): Check for outliers.
            redundant_features (bool): Check for redundant features.
            placeholders (bool): Check for placeholder values.
            temporal_features (bool): Check for temporal features.
            sparse_values (bool): Check for sparse values.
            high_cardinality (bool): Check for high cardinality features.
            poor_distribution (bool): Check for poor distribution.
            high_noise (bool): Check for high noise levels.
        
        Returns:
            bool: True if the feature is irrelevant.
        """
        # Automatically remove columns in force_remove_column
        if column in force_remove_column:
            print(f"Column '{column}' is marked for removal.")
            return True
        
        # Protect columns in protected_columns
        if column in self.protected_columns:
            if not self.data[column].empty and self.data[column].nunique(dropna=True) > 1:
                        
                print(f"Column '{column}' is protected and will not be removed.")
                return False
            else:
                if not self.data[column].empty:
                    print(f"Column '{column}' is overrided, because {column} is empty.")
                elif self.data[column].nunique(dropna=True) > 1:
                    print(f"Column '{column}' is overrided, because {column} has constant values.")

        if len(self.data) > 1e5:  # Arbitrary threshold for large datasets
            data = self.data.sample(frac=sample_fraction, random_state=42)
            feature = data[column]
        else:
            feature = self.data[column]

        # Check for non-informative categorical features (IDs/unique identifiers)
        if non_informative_categories and self.non_informative_categorical_feature(feature):
            return True
        
        # Check for very noisy data (e.g., large Na)
        if sparse_values and CSVFeatureAnalyzer.feature_has_sparse_values(feature):
            return True
        
        if temporal_features and CSVFeatureAnalyzer.feature_is_temporal(feature):
            return True
        
        if placeholders and CSVFeatureAnalyzer.feature_includes_placeholders(feature):
            return True

        if high_cardinality and CSVFeatureAnalyzer.feature_has_high_cardinality(feature):
            return True

        # Check for outliers
        if outliers and CSVFeatureAnalyzer.feature_has_too_many_outliers(feature):
            return True
        
        # Check for poor distribution
        if poor_distribution and CSVFeatureAnalyzer.feature_has_poor_distribution(feature):
            return True
        
        # Check for high noise
        if high_noise and CSVFeatureAnalyzer.feature_has_high_noise(feature):
            return True

        # Check for redundant features (duplicates)
        if redundant_features:
            column_index = self.data.columns.get_loc(column)
            other_columns = self.data.iloc[:, column_index + 1:]  
            if CSVFeatureAnalyzer.feature_is_redundant(self.data, feature, other_columns):
                return True

        # Check for constant features
        if constants and CSVFeatureAnalyzer.feature_has_constant_values(feature):
            return True

        return False


    def remove_irrelevant_features(self, irrelevant_features):
        """
        Remove irrelevant features from the DataFrame.
        
        Args:
            irrelevant_features (list): List of columns to remove.
        """
        self.data.drop(columns=irrelevant_features, inplace=True)
        self.cols_removed.extend(irrelevant_features)
        print(f"Removed columns: {self.cols_removed}, Percentage removed: {self.percentage_cols_removed:.2f}%")
        return self.data

    def remove_duplicates(self):
        """Remove duplicate rows from the DataFrame."""
        initial_count = len(self.data)
        self.data.drop_duplicates(inplace=True)
        final_count = len(self.data)
        print(f"Removed {initial_count - final_count} duplicate rows.")

    def fill_missing_values(self, strategy='mean'):
        """Fill missing values in the DataFrame based on the specified strategy."""
        for column in self.data.columns:
            if self.data[column].isnull().any():
                if strategy == 'mean':
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.data[column].fillna(self.data[column].median(), inplace=True)
                elif strategy == 'mode':
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                print(f"Filled missing values in column: {column} using {strategy}.")

    def normalize_numerical_features(self, fill_method='mean'):
        """Normalize numerical features to a range of [0, 1]."""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            if not CSVFeatureAnalyzer.feature_is_datetime_related(col, self.data[col]):
                # Handle NaN values based on the specified fill method
                if self.data[col].isnull().any():
                    if fill_method == 'mean':
                        fill_value = self.data[col].mean()
                    elif fill_method == 'median':
                        fill_value = self.data[col].median()
                    elif fill_method == 'drop':
                        self.data.dropna(subset=[col], inplace=True)
                        print(f"Column '{col}' had NaN values. Dropped rows with NaN.")
                        continue  # Skip normalization if rows are dropped
                    else:
                        raise ValueError("Invalid fill method. Choose 'mean', 'median', or 'drop'.")
                    
                    self.data[col].fillna(fill_value, inplace=True)
                    print(f"Column '{col}' had NaN values. Filled with {fill_method}: {fill_value}.")

                # Calculate min and max values
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                
                # Check if the column has more than one unique value
                if self.data[col].nunique() <= 1:
                    print(f"Column '{col}' has constant values ({min_val}). Normalization not applied.")
                    continue  # Skip normalization for this column

                # Normalize the column
                self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
                print(f"Normalized column '{col}': min={min_val}, max={max_val}.")

        print("Normalized numerical features.")
        return self.data 

    def remove_rows_with_high_nan_percentage(self, threshold=0.5):
        """
        Remove rows with a percentage of NaN values above the specified threshold.
        
        Args:
            threshold (float): The threshold for the percentage of NaN values (default is 0.5 for 50%).
        """
        # Calculate the percentage of NaN values for each row
        nan_percentage = self.data.isna().mean(axis=1)

        # Identify rows where the percentage of NaN values exceeds the threshold
        rows_to_drop = nan_percentage > threshold

        # Drop the identified rows
        if rows_to_drop.any():
            initial_count = len(self.data)
            self.data = self.data[~rows_to_drop]  # Keep rows that do not exceed the threshold
            final_count = len(self.data)
            print(f"Removed {initial_count - final_count} rows with more than {threshold * 100:.0f}% missing values.")

    def encode_categorical_features(self):
        """Encode categorical features dynamically based on their characteristics."""
        total_samples = self.data.shape[0]  # Total number of samples (rows)

        # Loop through all categorical columns
        for column in self.data.select_dtypes(include=['object']).columns:
            if column not in self.comment_columns:
                # Call the clustering function to combine similar values(In need of fix)
                #self.cluster_similar_values(column)  
                
                # After clustering, update unique values
                unique_values = self.data[column].unique()
                print(f"Processing column: {column} with unique values: {unique_values}")

                # Convert all values to string to avoid mixed type issues
                self.data[column] = self.data[column].astype(str)

                # Determine thresholds dynamically
                low_cardinality_threshold = max(5, int(0.05 * total_samples))  # Minimum of 5 unique values
                high_cardinality_threshold = min(70, total_samples)  # Maximum of 70 unique values

                if len(unique_values) <= low_cardinality_threshold:
                    # One-Hot Encoding for low cardinality features
                    self.data = pd.get_dummies(self.data, columns=[column], drop_first=True)
                    print(f"One-hot encoded column: {column}")

                elif len(unique_values) <= high_cardinality_threshold:
                    # Label Encoding for medium cardinality features
                    le = LabelEncoder()
                    self.data[column] = le.fit_transform(self.data[column])
                    print(f"Label encoded column: {column}")

                else:
                    # Dynamically determine top_n for high cardinality features
                    top_n = max(5, min(int(0.2 * len(unique_values)), 50))  # Keep top 20% of unique categories
                    top_categories = self.data[column].value_counts().nlargest(top_n).index
                    self.data[column] = self.data[column].where(self.data[column].isin(top_categories), other='Other')
                    self.data = pd.get_dummies(self.data, columns=[column], drop_first=True)
                    print(f"Limited one-hot encoded column: {column} with top {top_n} categories.")

        # Ensure all features are numeric
        self.data = self.data.apply(pd.to_numeric, errors='ignore')

        print("Encoded categorical features.")

    def preprocess_and_encode_datetime(self):
        """
        Preprocess and encode datetime-like features.
        
        Returns:
            pd.DataFrame: The modified DataFrame with extracted datetime features.
        """
        for column in self.data.columns:
            # Skip columns that are already datetime-like (datetime64[ns])
            if self.data[column].dtype == 'datetime64[ns]':
                print(f"Column '{column}' is already in datetime format.")
                # Directly extract datetime components if it's already datetime-like
                self._encode_datetime_components(column)
                continue  # Skip to next column

            # # Check if the column is likely to be datetime-related
            # if CSVFeatureAnalyzer.feature_is_datetime_related(column, self.data[column]):
            #     negligible_duration = pd.Timedelta(seconds=0)

            #     # Replace problematic values with the negligible duration
            #     self.data[column] = self.data[column].replace(
            #         to_replace=[None, 'Nan', 'Inf', 'Not Applicable', '-', ''],  # List of values to replace
            #         value=negligible_duration
            #     )

            #     # Convert non-numeric entries to NaT, and then numeric entries to the correct datetime
            #     self.data[column] = pd.to_datetime(self.data[column], errors='coerce')

            #     # Handle remaining NaN values (e.g., replace with negligible duration)
            #     self.data[column].fillna(negligible_duration, inplace=True)

            #     # Check for conversion issues
            #     if self.data[column].isnull().all():
            #         print(f"Warning: All values in column '{column}' could not be converted to datetime. Keeping original values.")
            #         continue  # Skip to the next column if conversion fails

            #     # Log how many values were converted successfully
            #     successful_conversions = self.data[column].notnull().sum()
            #     print(f"Converted {successful_conversions} values in column '{column}' to datetime.")

            #     # Now encode the datetime components like year, month, day, etc.
            #     self._encode_datetime_components(column)

            # else:
            #     print(f"Column '{column}' is not recognized as datetime-related.")

        return self.data  # Return the modified DataFrame after processing all columns


    def _encode_datetime_components(self, column):
        """
        Encodes datetime features by extracting individual components such as year, month, day, etc.
        Adds new columns for each component.
        """
        # Extract datetime components
        year_values = self.data[column].dt.year
        month_values = self.data[column].dt.month
        day_values = self.data[column].dt.day
        hour_values = self.data[column].dt.hour
        minute_values = self.data[column].dt.minute
        second_values = self.data[column].dt.second
        dayofweek_values = self.data[column].dt.dayofweek  # Monday=0, Sunday=6

        num_added_columns = 0

        # Create new columns only if there is more than 1 unique value (excluding NaN)
        if year_values.nunique(dropna=True) > 1:
            self.data[f'{column}_year'] = year_values.fillna(0).astype(int)
            num_added_columns += 1
        if month_values.nunique(dropna=True) > 1:
            self.data[f'{column}_month'] = month_values.fillna(0).astype(int)
            num_added_columns += 1
        if day_values.nunique(dropna=True) > 1:
            self.data[f'{column}_day'] = day_values.fillna(0).astype(int)
            num_added_columns += 1
        if hour_values.nunique(dropna=True) > 1:
            self.data[f'{column}_hour'] = hour_values.fillna(0).astype(int)
            num_added_columns += 1
        if minute_values.nunique(dropna=True) > 1:
            self.data[f'{column}_minute'] = minute_values.fillna(0).astype(int)
            num_added_columns += 1
        if second_values.nunique(dropna=True) > 1:
            self.data[f'{column}_second'] = second_values.fillna(0).astype(int)
            num_added_columns += 1
        if dayofweek_values.nunique(dropna=True) > 1:
            self.data[f'{column}_dayofweek'] = dayofweek_values.fillna(0).astype(int)
            num_added_columns += 1

        if num_added_columns > 0:
            # Check for NaN values in the new columns
            for new_col in [f'{column}_year', f'{column}_month', f'{column}_day', f'{column}_hour', f'{column}_minute', f'{column}_second', f'{column}_dayofweek']:
                if new_col in self.data.columns:
                    nan_count = self.data[new_col].isnull().sum()
                    total_count = len(self.data[new_col])
                    
                    if nan_count > 0:
                        nan_percentage = (nan_count / total_count) * 100
                        print(f"Warning: Column '{new_col}' has {nan_count} NaN values ({nan_percentage:.2f}%). This indicates conversion issues.")

            # Optionally drop the original column if desired
            self.data.drop(columns=[column], inplace=True)

            print(f"Processed and encoded datetime features from column '{column}'.")

    def detect_and_treat_outliers(self, method='remove'):
        """
        Identify and handle outliers in numerical columns.
        
        Args:
            method (str): The method to handle outliers ('remove' to remove them, 'mean' to replace with mean).
        """
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            # Calculate the first and third quartiles
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1  # Interquartile range

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]

            if not outliers.empty:
                if method == 'remove':
                    # Remove outliers
                    self.data = self.data[~self.data.index.isin(outliers.index)]
                    print(f"Removed {len(outliers)} outliers from column: {col}.")
                elif method == 'mean':
                    # Replace outliers with the mean
                    mean_value = self.data[col].mean()
                    self.data.loc[(self.data[col] < lower_bound) | (self.data[col] > upper_bound), col] = mean_value
                    print(f"Replaced {len(outliers)} outliers in column: {col} with mean value: {mean_value}.")
                else:
                    raise ValueError("Invalid method. Choose 'remove' or 'mean'.")

    def clean_text_data(self):
        """
        Clean text data by removing special characters, converting to lowercase, etc.
        """
        text_columns = self.data.select_dtypes(include=['object']).columns  # Identify text columns
        
        for column in text_columns:
            
            
            
            # Check if the column contains string-like data and is not numeric
            if not pd.api.types.is_numeric_dtype(self.data[column]):
                # Remove special characters and convert to lowercase
                self.data[column] = self.data[column].str.replace(f"[{string.punctuation}]", "", regex=True)  # Remove punctuation
                self.data[column] = self.data[column].str.lower()  # Convert to lowercase
                print(f"Cleaned text data in column: {column}")
            else:
                print(f"Skipped cleaning for column: {column} as it does not contain string data or is numeric.")

    def identify_text_columns(self):
        """
        Identify columns that are likely to contain comments or user feedback.
        """
        text_columns = self.data.select_dtypes(include=['object']).columns
        self.comment_columns = []

        for column in text_columns:
            # Check for long strings or specific keywords in the column name
            if self.data[column].str.len().mean() > 50 or 'comment' in column.lower() or 'feedback' in column.lower():
                self.comment_columns.append(column)

        print(f"Identified comment/feedback columns: {self.comment_columns}")
        return self.comment_columns

    def encode_text_data(self):
        """
        Encode text data using TF-IDF.
        """
        for column in self.comment_columns:
            vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features
            tfidf_matrix = vectorizer.fit_transform(self.data[column].fillna(''))  # Fill NaN with empty string
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
            self.data = pd.concat([self.data, tfidf_df], axis=1)  # Concatenate the TF-IDF features to the original DataFrame
            print(f"Encoded text data in column: {column} using TF-IDF.")

    #Needs fixing consider using NLP instead
    def cluster_similar_values(self, column, n_clusters=2):
        """
        Cluster similar values in a specified column using K-Means clustering.
        
        Args:
            column (str): The name of the column to process.
            n_clusters (int): The number of clusters to form.
        """
        # Extract unique values from the column
        unique_values = self.data[column].unique()
        
        # Vectorize the unique values using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(unique_values)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # Create a mapping from original values to cluster labels
        value_mapping = {}
        for idx, label in enumerate(kmeans.labels_):
            value_mapping[unique_values[idx]] = f'cluster_{unique_values[idx]}'

        # Replace values in the specified column based on the mapping
        self.data[column] = self.data[column].replace(value_mapping)
        print(f"Clustered similar values in column: {column} with mapping: {value_mapping}")

    


    def get_data(self):
        """Return the cleaned DataFrame."""
        return self.data
    
    # Getter for init_number_rows
    def get_init_number_rows(self):
        """Return the initial number of rows."""
        return self.init_number_rows

    # Getter for init_number_cols
    def get_init_number_cols(self):
        """Return the initial number of columns."""
        return self.init_number_cols

    # Getter for cols_removed
    def get_cols_removed(self):
        """Return the list of removed columns."""
        return self.cols_removed

    # Getter for percentage_rows_removed
    def get_percentage_rows_removed(self):
        """Return the percentage of rows removed."""
        return self.percentage_rows_removed

    # Getter for percentage_cols_removed
    def get_percentage_cols_removed(self):
        """Return the percentage of columns removed."""
        return self.percentage_cols_removed

    def preprocess(self):
        """
        Preprocess the data by identifying comment columns, encoding datetime, and cleaning text data.
        """
        # Identify comment columns
        self.comment_columns = self.identify_text_columns()

        # Preprocessing steps
        self.preprocess_and_encode_datetime()
        
        self.encode_text_data()

        self.encode_categorical_features()
        self.normalize_numerical_features()

    def run(self, preprocess=False):
        """
        Run the data cleaning process.
        
        Args:
            preprocess (bool): Whether to preprocess the data after cleaning.
        """
        # Analyze dataset
        self.remove_rows_with_high_nan_percentage()
        self.analyze_features()
        self.remove_duplicates()
        self.clean_text_data()
        #self.detect_and_treat_outliers(method='remove')  

        #Analyze dataset
        
        # Check if user wishes to preprocess
        if preprocess:
            self.preprocess()

        #Analyze dataset