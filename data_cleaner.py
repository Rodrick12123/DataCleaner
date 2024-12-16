from CleanerTypes.csv_cleaner import CSVCleaner

def main():
    # Specify the path to your file
    file_path = '"example_data.csv"'

    
    # Create an instance of CSVCleaner
    cleaner = CSVCleaner(file_path)
    
    #Clean data
    
    #Could check with user if they want to remove features using cleaner.get_irrelevant_features()
    cleaner.remove_irrelevant_features()
    
    # Get the cleaned data
    cleaned_data = cleaner.get_cleaned_data()
    
    # Display 10 random samples from the cleaned data
    random_samples = cleaned_data.sample(n=10, random_state=42)  # Set random_state for reproducibility
    print("Random Samples from Cleaned Data:")
    print(random_samples)

if __name__ == "__main__":
    main()