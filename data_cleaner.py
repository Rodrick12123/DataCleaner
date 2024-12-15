from CleanerTypes.csv_cleaner import CSVCleaner

def main():
    # Specify the path to your file
    file_path = 'path/to/your/data.csv'

    
    # Create an instance of CSVCleaner
    cleaner = CSVCleaner(file_path)
    
    #Clean data
    
    #Could check with user if they want to remove features using cleaner.get_irrelevant_features()
    cleaner.remove_irrelevant_features()
    
    # Get the cleaned data
    cleaned_data = cleaner.get_cleaned_data()
    
    # Display or save the cleaned data
    print(cleaned_data)

if __name__ == "__main__":
    main()