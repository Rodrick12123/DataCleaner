from CleanerTypes.csv_cleaner import CSVCleaner
from CleanerTypes.excel_cleaner import ExcelCleaner
from IPython.display import display
from warning_config import configure_warnings
import os
import numpy as np

import chardet
import pandas as pd


# Configure warnings at startup
configure_warnings()

def determine_file_type():
    pass

def load_csv_data(file_path, sample_size=0.1):
    """
    Load a sample of the CSV data into a DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        sample_size (float): Fraction of rows to load (default: 0.1 or 10%).
    
    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # First, count total rows in file
    total_rows = sum(1 for _ in open(file_path, 'r'))
    rows_to_read = int(total_rows * sample_size)
    
    # Ensure we read at least 10000 rows for reliable analysis
    rows_to_read = max(10000, rows_to_read)
    print(f"Loading {rows_to_read:,} rows out of {total_rows:,} total rows")
    
    # List of encodings to try
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'ascii']
    
    # Try to detect encoding first
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            if result['confidence'] > 0.7:
                encodings.insert(0, result['encoding'])
    except Exception as e:
        print(f"Warning: Encoding detection failed: {e}")

    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    print(f"Detected delimiter: {delimiter}")
    
    # Try different encodings
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            
            # Read the header first
            header = pd.read_csv(
                file_path,
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
                file_path,
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

def detect_delimiter( file_path):
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

def main():

    # Define the datasets directory path
    datasets_dir = "datasets"

    # Update file paths to use the datasets directory
    file_path = f"{datasets_dir}/example_data.csv"
    file_path2 = f"{datasets_dir}/messy_IMDB_dataset.csv"
    file_path3 = f"{datasets_dir}/GOOG.csv"
    file_path4 = f"{datasets_dir}/hotel_bookings.csv"

    file_path5 = f"{datasets_dir}/Inventory-Records-Sample-Data.xlsx"
    file_path6 = f"{datasets_dir}/Cola.xlsx"
    file_path7 = f"{datasets_dir}/AB_NYC_2019.csv"

    big_file1 = f"{datasets_dir}/train.csv"
    
    user_likes = True

    
    if user_likes:
        print("Loading whole dataset for cleaning")
        cleaner = CSVCleaner(big_file1)
        
    else:
        #sample large data for testing
        print("Loading sample dataset for cleaning")
        csvdata = load_csv_data(big_file1)
        cleaner = CSVCleaner(big_file1, csvdata)
        

    
    #Cleaner setup
    #choose cleaner type
    # Create an instance of CSVCleaner

    #cleaner = ExcelCleaner(file_path=file_path5)
    
    #configure clean logistics
    #cleaner.add_protected_columns("ORIGIN_CALL")
    # cleaner.add_force_remove_column("IMBD title ID")
    
    #Clean data
    #Could check with user if they want to remove features using cleaner.get_irrelevant_features()
    cleaner.run(preprocess=True)
    
    # Get the cleaned data
    cleaned_data = cleaner.get_data()
    
    
    # Display 10 random samples from the cleaned data
    random_samples = cleaned_data.sample(n=10, random_state=10)  # Set random_state for reproducibility
    print("Random Samples from Cleaned Data:")
    display(random_samples)
    


if __name__ == "__main__":
    main()