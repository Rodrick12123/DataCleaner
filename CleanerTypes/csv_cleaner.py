import pandas as pd
import os
from feature_analyzer import CSVFeatureAnalyzer 

class CSVCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data(file_path)
        self.init_number_rows = len(self.data)
        self.init_number_cols = self.data.shape[1]
        self.cols_removed = []
        self.rows_removed = []
        self.percentage_rows_removed = 0
        self.percentage_cols_removed = 0


    def load_data(self, file_path):

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Try to read the CSV file
        try:
            data = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the file. Please check the file format.")
        
        # Check if the DataFrame is empty
        if data.empty:
            raise ValueError("The DataFrame is empty after loading the data.")
        
        return data
    
    def analyze_features(self):
        irrelevant_features = []
        # Create a new list for renamed columns
        self.rename_duplicate_columns()

        for column in self.data.columns:

            #check for correct dtypes
            self.feature_object_to_numeric_fillna(column)

            if(self.is_irrelevant_features()):
                irrelevant_features.append(column)
                continue
            
        
         # Call remove features on irrelevant_features
        if len(irrelevant_features) > 0:
            if self.init_number_cols > 0:
                self.percentage_cols_removed = (len(irrelevant_features) / self.init_number_cols) * 100
            else:
                self.percentage_cols_removed = 0  # Handle the case where there are no initial columns

            if self.percentage_cols_removed < 50:
                self.cols_removed = irrelevant_features  # Store the removed columns
                self.remove_irrelevant_features()  # Call the method to remove the columns
                print(f"Removed columns: {self.cols_removed}, Percentage removed: {self.percentage_cols_removed:.2f}%")


    # Function to automatically rename duplicate columns
    def rename_duplicate_columns(self):
        # Create a new list for renamed columns
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

    def analyze_samples():
        pass


    
    # Convert strings to numeric and fill NaN values with mean,mode,etc
    def feature_object_to_numeric_fillna(self,column,fill="mean"):

        # Check if the feature is meant to be numeric but is stored as an object
        if self.data[column].dtype == 'object':
            # Replace empty strings with NaN
            self.data[column].replace('', float('nan'), inplace=True)
            try:
                # Attempt to convert to numeric
                pd.to_numeric(self.data[column], errors='raise')
            except ValueError:
                # If conversion fails, it indicates the feature is not numeric
                return self.data
        # Fill NaN values with the mean of the column
        mean_value = self.data[column].mean()
        self.data[column].fillna(mean_value, inplace=True)
        
        return self.data
    

    
    #Check to see if the data includes information that is not helpful (e.g unique identifiers/ Ids)
    def non_informative_categorical_feature(data):
        """
        non_informative includes:
        phone numbers
        zipcodes
        Ids
        Unique Identifiers

        Returns True if the data is non_informative
        """
        
        return True
    
    def is_irrelevant_features(self, column, sample_fraction=.2, constants=True,
                                non_informative_categories=True, outliers=False, 
                                redundant_features=True):
        """
        Identify irrelevant features based on...

        Causes for removal:
        Highly correlated features
        Irrelavant Features(nothing to do with target value)
        *Duplicate Features
        Features with high missing values (e.g more than 50%)
        Outliers
        Non-informative categorical features
        Features with high noise levels
        Redundant features
        Features with irrelevant units
        Temporal Features with no predictive power
        Features with high cardinality
        Features with poor distribution
        Domain Knowledge

        
        Parameters:
        - threshold: The variance threshold below which features are considered irrelevant.
        
        Returns:
        - True if the feature is useless
        """
        
        if len(self.data) > 1e5:  # Arbitrary threshold for large datasets
            data = self.data.sample(frac=sample_fraction, random_state=42)
            feature = data[column]
        else:
            feature = self.data[column]

        #check for non-informative categorical features (IDs/unique identifiers)
        if non_informative_categories and self.non_informative_categorical_feature(feature):
            return True

        if outliers:
            pass

        #Check for redundant features (duplicates)
        if redundant_features:
                # Get the index of the specified column
            column_index = self.data.columns.get_loc(feature)
            other_columns = self.data.iloc[:, column_index + 1:]  
            if CSVFeatureAnalyzer.feature_is_redundant(feature,other_columns):
                return True

        # Check for constant features
        if CSVFeatureAnalyzer.feature_has_constant_values(feature) and constants:
            return True
        

        return False
    

    #Check to see if percentage of removed columns are to high and act
    def remove_irrelevant_features(self):
        """
        main call

        Returns: self.data with features removed
        """

        #Get all irrelevant_features in a list


        #Drop all features (columns) in the list from the data

        return self.data

    
    def get_data(self):
        return self.data
    
     # Getter for init_number_rows
    def get_init_number_rows(self):
        return self.init_number_rows

    # Getter for init_number_cols
    def get_init_number_cols(self):
        return self.init_number_cols

    # Getter for cols_removed
    def get_cols_removed(self):
        return self.cols_removed
    
    # Getter for cols_removed
    def get_cols_removed(self):
        return self.cols_removed

    # Getter for percentage_rows_removed
    def get_percentage_rows_removed(self):
        return self.percentage_rows_removed

    # Getter for percentage_rows_removed
    def get_percentage_cols_removed(self):
        return self.percentage_cols_removed