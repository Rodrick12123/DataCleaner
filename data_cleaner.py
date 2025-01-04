from CleanerTypes.csv_cleaner import CSVCleaner
from CleanerTypes.excel_cleaner import ExcelCleaner
from IPython.display import display
from warning_config import configure_warnings

# Configure warnings at startup
configure_warnings()

def determine_file_type():
    pass

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

    #Cleaner setup
    #choose cleaner type
    # Create an instance of CSVCleaner
    cleaner = CSVCleaner(file_path2)
    #cleaner = ExcelCleaner(file_path=file_path5)
    
    #configure clean logistics
    # cleaner.add_protected_columns("IMBD title ID")
    # cleaner.add_force_remove_column("IMBD title ID")
    
    #Clean data
    #Could check with user if they want to remove features using cleaner.get_irrelevant_features()
    cleaner.run()
    
    # Get the cleaned data
    cleaned_data = cleaner.get_data()
    
    
    # Display 10 random samples from the cleaned data
    random_samples = cleaned_data.sample(n=10, random_state=10)  # Set random_state for reproducibility
    print("Random Samples from Cleaned Data:")
    display(random_samples)
    


if __name__ == "__main__":
    main()