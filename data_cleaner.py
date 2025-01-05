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



def display_data_summary(data, title="Dataset Summary"):
    """Display a summary of the dataset with key information."""
    print(f"\n=== {title} ===")
    print(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    
    # Display sample data
    print("\n--- Random Sample (10 rows) ---")
    random_samples = data.sample(n=10, random_state=10)
    display(random_samples)
    
    # Display basic statistics
    print("\n--- Column Information ---")
    col_info = pd.DataFrame({
        'Type': data.dtypes,
        'Non-Null': data.count(),
        'Null %': (data.isnull().sum() / len(data) * 100).round(2),
        'Unique': data.nunique()
    })
    display(col_info)
    
    # Display numeric column statistics
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\n--- Numeric Column Statistics ---")
        display(data[numeric_cols].describe())




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
    
    user_likes = False

    exlfile = file_path6
    csvfile= file_path
    if user_likes:
        print("Loading whole dataset for cleaning")
        cleaner = CSVCleaner(csvfile)

        #cleaner = ExcelCleaner(file_path=file_path)
    else:
        #sample large data for testing
        print("Loading sample dataset for cleaning")
        cleaner = CSVCleaner(csvfile, load_sample=True)

        #cleaner = ExcelCleaner(file_path=exlfile,load_sample=True)
        

    
    #Cleaner setup
    #choose cleaner type
    # Create an instance of CSVCleaner

    #cleaner = ExcelCleaner(file_path=file_path5)
    
    #configure clean logistics
    #cleaner.add_protected_columns("ORIGIN_CALL")
    # cleaner.add_force_remove_column("IMBD title ID")
    
    #Clean data
    #Could check with user if they want to remove features using cleaner.get_irrelevant_features()
    #remember to fix preprocess later
    cleaner.run(preprocess=False)
    
    # Get the cleaned data
    cleaned_data = cleaner.get_data()
    display_data_summary(cleaned_data, "Cleaned Data")

    
    
    # Display 10 random samples from the cleaned data
    # random_samples = cleaned_data.sample(n=10, random_state=10)  # Set random_state for reproducibility
    # print("Random Samples from Cleaned Data:")
    # display(random_samples)
    


if __name__ == "__main__":
    main()