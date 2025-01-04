import os
import re

import pandas as pd
import openpyxl  
from .base_cleaner import BaseCleaner


class ExcelCleaner(BaseCleaner):
    def __init__(self, file_path, sheet_name=None, header_row=0, rename_unnamed=True):
        super().__init__()
        """
        Initialize the ExcelCleaner with the file path and load the data.
        
        Args:
            file_path (str): The path to the Excel file to be cleaned.
            sheet_name (str, optional): Name of the sheet to process. If None, reads first sheet.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        
        self.header_row = header_row
        self.rename_unnamed = rename_unnamed
        self.data = self.load_data(file_path)
        self.init_number_rows = len(self.data)
        self.init_number_cols = self.data.shape[1]
        

    

    # def load_data(self, file_path):
    #     """
    #     Load the Excel data into a DataFrame with proper column handling.
    #     """
    #     if not os.path.isfile(file_path):
    #         raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    #     try:
    #         # Read Excel file
    #         data = pd.read_excel(
    #             file_path,
    #             sheet_name=0 if self.sheet_name is None else self.sheet_name,
    #             header=self.header_row,
    #             na_values=['NA', 'N/A', '', ' '],
    #             keep_default_na=True
    #         )
            
    #         # Handle unnamed columns
    #         if self.rename_unnamed:
    #             # Check for unnamed columns
    #             unnamed_pattern = re.compile(r'^Unnamed:|^$')
    #             unnamed_cols = [bool(unnamed_pattern.match(str(col))) for col in data.columns]
                
    #             if any(unnamed_cols):
    #                 print("Found unnamed columns. Renaming them...")
    #                 new_columns = []
    #                 for i, (col, is_unnamed) in enumerate(zip(data.columns, unnamed_cols)):
    #                     if is_unnamed:
    #                         new_columns.append(f'Column_{i+1}')
    #                     else:
    #                         new_columns.append(col)
    #                 data.columns = new_columns
            
    #         # Remove empty rows and columns
    #         data = data.dropna(how='all', axis=1)
    #         data = data.dropna(how='all', axis=0)
            
    #         # Verify we have valid column names
    #         if data.columns.duplicated().any():
    #             print("Warning: Duplicate column names found. Adding suffixes...")
    #             data.columns = pd.Series(data.columns).apply(
    #                 lambda x: f"{x}_{pd.Series(data.columns).value_counts()[x]}" 
    #                 if pd.Series(data.columns).value_counts()[x] > 1 
    #                 else x
    #             )
            
    #         return data

        # except Exception as e:
        #     raise ValueError(f"Error reading Excel file: {e}")

    def load_data(self, file_path):
        """
        Load the Excel data into a DataFrame with automatic header detection.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        try:
            # First, read a sample of rows to detect header
            sample_data = pd.read_excel(
                file_path,
                sheet_name=0 if self.sheet_name is None else self.sheet_name,
                nrows=20,  # Read first 20 rows for analysis
                header=None
            )
            
            # Detect header row
            header_row = self._detect_header_row(sample_data)
            print(f"Detected header row at index: {header_row}")
            
            # Read the full file with detected header
            data = pd.read_excel(
                file_path,
                sheet_name=0 if self.sheet_name is None else self.sheet_name,
                header=header_row,
                na_values=['NA', 'N/A', '', ' '],
                keep_default_na=True
            )
            
            # Handle unnamed columns
            if self.rename_unnamed:
                data.columns = self._handle_unnamed_columns(data.columns)
            
            # Remove empty rows and columns
            data = data.dropna(how='all', axis=1).dropna(how='all', axis=0)
            
            # Handle duplicate column names
            if data.columns.duplicated().any():
                data.columns = self._handle_duplicate_columns(data.columns)
            
            return data

        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

    def detect_header_row(self, sample_data):
        """
        Detect the most likely header row based on data patterns.
        """
        max_score = -1
        best_header_row = 0
        
        for i in range(min(len(sample_data), 10)):  # Check first 10 rows
            row = sample_data.iloc[i]
            score = 0
            
            # Check for common header characteristics
            score += sum([
                # Non-numeric values are more likely to be headers
                sum(isinstance(val, str) for val in row) * 2,
                
                # Unique values are more likely to be headers
                (len(set(row)) / len(row)) * 3,
                
                # find keywords
                sum(any(keyword in str(val).lower() for keyword in 
                    ['id', 'name', 'date', 'total', 'number', 'code', 'type']) 
                    for val in row) * 2,
                
                # check amount of Na
                (1 - row.isna().mean()) * 3,
                
                #shorter strings
                -sum(len(str(val)) > 50 for val in row)
            ])
            
            # Check if values below this row are consistent in type
            if i < len(sample_data) - 1:
                below_rows = sample_data.iloc[i+1:]
                type_consistency = sum(
                    below_rows[col].apply(type).nunique() == 1
                    for col in below_rows.columns
                ) / len(below_rows.columns)
                score += type_consistency * 4
            
            if score > max_score:
                max_score = score
                best_header_row = i
        
        return best_header_row

    def handle_unnamed_columns(self, columns):
        """
        Handle unnamed columns with improved naming.
        """
        unnamed_pattern = re.compile(r'^Unnamed:|^$')
        new_columns = []
        
        for i, col in enumerate(columns, 1):
            if unnamed_pattern.match(str(col)):
                new_columns.append(f'Column_{i}')
            else:
                new_columns.append(col)
        
        return new_columns

    def handle_duplicate_columns(self, columns):
        """
        Handle duplicate column names with improved suffixes.
        """
        seen = {}
        new_columns = []
        
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 1
                new_columns.append(col)
        
        return new_columns

    def get_sheet_names(self):
        """
        Get list of sheet names in the Excel file.
        
        Returns:
            list: List of sheet names.
        """
        try:
            xl = pd.ExcelFile(self.file_path)
            return xl.sheet_names
        except Exception as e:
            print(f"Error reading sheet names: {e}")
            return []

    def change_sheet(self, sheet_name):
        """
        Change to a different sheet in the Excel file.
        
        Args:
            sheet_name (str): Name of the sheet to switch to.
        """
        if sheet_name in self.get_sheet_names():
            self.sheet_name = sheet_name
            self.data = self.load_data(self.file_path)
            self.init_number_rows = len(self.data)
            self.init_number_cols = self.data.shape[1]
            print(f"Switched to sheet: {sheet_name}")
        else:
            raise ValueError(f"Sheet '{sheet_name}' not found in the Excel file.")




    def run(self, preprocess=False, sheet_name=None):
        """
        Run the data cleaning process.
        
        Args:
            preprocess (bool): Whether to preprocess the data after cleaning.
            sheet_name (str, optional): Name of the sheet to process.
        """
        # Change sheet if specified
        if sheet_name:
            self.change_sheet(sheet_name)

        # Analyze dataset
        self.remove_rows_with_high_nan_percentage()
        self.analyze_features()
        self.remove_duplicates()
        self.clean_text_data()

        # Check if user wishes to preprocess
        if preprocess:
            self.preprocess()