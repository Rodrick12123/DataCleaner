import pandas as pd
import os

class CSVCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data(file_path)
        self.init_number_rows = len(self.data)
        self.init_number_cols = self.data.shape[1]
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
    
    #Check to see if the data includes information that is not helpful such as IDs
    #Consider all dtypes
    #Returns True if the data is non_informative
    def non_informative_categorical_feature(data):
        
        return True
    
    def get_irrelevant_features(self, threshold=0.01, constants=True, irrelevant_categories=True, outliers=False):
        """
        Identify irrelevant features based on variance and constant values.

        More causes for removal:
        Highly correlated features
        Irrelavant Features(nothing to do with target value)
        Duplicate Features
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
        - A list of irrelevant feature names.
        """
        
        irrelevant_features = []

        for column in self.data.columns:
            data = self.data[column]

            #check for non-informative categorical features (IDs/unique identifiers)
            if irrelevant_categories:
                pass

            if outliers:
                pass

            # Check for constant features
            if data.nunique() <= 1 and constants:
                irrelevant_features.append(column)
            
            

        # Check for low variance features
        # low_variance_features = self.data.columns[self.data.var() < threshold]
        # irrelevant_features.extend(low_variance_features)

        return irrelevant_features
    
    #Calls get_irrelevant_features to create a list of irrelevant features -> removes all features(columns) in list from data
    def remove_irrelevant_features(self):
        pass

    
    def get_cleaned_data(self):
        return self.data