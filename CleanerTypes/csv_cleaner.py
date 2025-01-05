import os

import chardet
import numpy as np
import pandas as pd


from .base_cleaner import BaseCleaner




class CSVCleaner(BaseCleaner):
    def __init__(self, file_path, load_sample=None):
        """
        Initialize the CSVCleaner with the file path and load the data.
        
        Args:
            file_path (str): The path to the CSV file to be cleaned.
            load_sample (bool): If True, load sample data instead of full file.
        """
        super().__init__()
        self.file_path = file_path
        
        if load_sample:
            self.data = self.load_sample_data()  # Remove file_path parameter
        else:
            self.data = self.load_data(file_path)
        
        self.init_number_rows = len(self.data)
        self.init_number_cols = self.data.shape[1]

        

    def load_data(self, file_path):
        """
        Load the CSV data into a DataFrame with robust encoding and delimiter detection.
        
        Args:
            file_path (str): The path to the CSV file.
        
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # List of encodings to try
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'ascii']
        
        # Try to detect encoding first
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                if result['confidence'] > 0.7:  # Only use if confidence is high
                    encodings.insert(0, result['encoding'])
        except Exception as e:
            print(f"Warning: Encoding detection failed: {e}")

        # Detect delimiter
        delimiter = self.detect_delimiter(file_path)
        print(f"Detected delimiter: {delimiter}")
        
        # Try different encodings
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                data = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    on_bad_lines='skip',  # Skip problematic lines
                    low_memory=False,     # Avoid mixed type inference warnings
                    encoding_errors='replace'  # Replace invalid characters
                )
                
                if not data.empty:
                    print(f"Successfully loaded with encoding: {encoding}")
                    return data
                    
            except Exception as e:
                print(f"Failed with encoding {encoding}: {e}")
                continue
        
        raise ValueError("Failed to load the CSV file with any of the attempted encodings.")

    def load_sample_data(self, sample_size=0.1):
        """
        Load a sample of the CSV data into a DataFrame.
        
        Args:
            file_path (str): The path to the CSV file.
            sample_size (float): Fraction of rows to load (default: 0.1 or 10%).
        
        Returns:
            pd.DataFrame: The sampled DataFrame.
        """
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        # First, count total rows in file
        total_rows = sum(1 for _ in open(self.file_path, 'r'))
        rows_to_read = int(total_rows * sample_size)
        
        # Ensure we read at least 10000 rows for reliable analysis
        rows_to_read_max = max(10000, rows_to_read)
        if rows_to_read_max > total_rows:
            rows_to_read = total_rows
        else:
            rows_to_read = rows_to_read_max
        print(f"Loading {rows_to_read:,} rows out of {total_rows:,} total rows")
        
        # List of encodings to try
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'ascii']
        
        # Try to detect encoding first
        try:
            with open(self.file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                if result['confidence'] > 0.7:
                    encodings.insert(0, result['encoding'])
        except Exception as e:
            print(f"Warning: Encoding detection failed: {e}")

        # Detect delimiter
        delimiter = self.detect_delimiter(self.file_path)
        print(f"Detected delimiter: {delimiter}")
        
        # Try different encodings
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                
                # Read the header first
                header = pd.read_csv(
                    self.file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    nrows=0  # Only read header
                ).columns
                
                # Generate random row indices
                skip_rows = sorted(np.random.choice(
                    range(1, total_rows), 
                    size=total_rows - rows_to_read, 
                    replace=False
                ))
                
                # Read sampled data
                data = pd.read_csv(
                    self.file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    skiprows=skip_rows,
                    names=header,
                    on_bad_lines='skip',
                    low_memory=False,
                    encoding_errors='replace'
                )
                
                if not data.empty:
                    print(f"Successfully loaded {len(data):,} rows with encoding: {encoding}")
                    return data
                    
            except Exception as e:
                print(f"Failed with encoding {encoding}: {e}")
                continue
        
        raise ValueError("Failed to load the CSV file with any of the attempted encodings.")


    def detect_delimiter(self, file_path):
        """
        Enhanced delimiter detection for CSV files.
        
        Args:
            file_path (str): The path to the CSV file.
        
        Returns:
            str: The detected delimiter.
        """
        common_delimiters = [',', ';', '\t', '|']
        
        try:
            # Try reading first few lines with different encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        first_lines = ''.join([f.readline() for _ in range(5)])
                        
                        # Count occurrences of each delimiter
                        counts = {delimiter: first_lines.count(delimiter) 
                                for delimiter in common_delimiters}
                        
                        # Return the most common delimiter
                        if counts:
                            max_count = max(counts.values())
                            if max_count > 0:
                                return max(counts, key=counts.get)
                        
                        break  # If successful, no need to try other encodings
                        
                except UnicodeDecodeError:
                    continue
                    
        except Exception as e:
            print(f"Warning: Delimiter detection failed: {e}")
        
        return ','  # Default to comma if detection fails

    

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