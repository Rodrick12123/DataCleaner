from CleanerTypes.csv_cleaner import CSVCleaner
from IPython.display import display

def main():
    # Specify the path to your file
    file_path = "example_data.csv"

    
    # Create an instance of CSVCleaner
    cleaner = CSVCleaner(file_path)
    
    #Clean data
    
    #Could check with user if they want to remove features using cleaner.get_irrelevant_features()
    cleaner.run()
    
    # Get the cleaned data
    cleaned_data = cleaner.get_data()
    
    
    # Display 10 random samples from the cleaned data
    random_samples = cleaned_data.sample(n=10, random_state=10)  # Set random_state for reproducibility
    print("Random Samples from Cleaned Data:")
    display(random_samples)
    print(cleaner.cols_removed) 



if __name__ == "__main__":
    main()