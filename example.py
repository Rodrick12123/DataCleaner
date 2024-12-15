import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and display the data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to clean the data
def clean_data(df):
    # Handle missing values
    if st.checkbox("Remove rows with missing values"):
        df = df.dropna()
    
    # Remove duplicates
    if st.checkbox("Remove duplicate rows"):
        df = df.drop_duplicates()
    
    # Fill missing values
    fill_option = st.selectbox("Fill missing values with:", ["None", "Mean", "Median", "Mode"])
    if fill_option == "Mean":
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(df[col].mean(), inplace=True)
    elif fill_option == "Median":
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(df[col].median(), inplace=True)
    elif fill_option == "Mode":
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# Function to visualize data
def visualize_data(df):
    st.subheader("Data Visualization")
    if st.checkbox("Show pairplot"):
        sns.pairplot(df)
        st.pyplot()

# Function to detect and remove outliers
def remove_outliers(df):
    if st.checkbox("Remove outliers using IQR"):
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    return df

# Streamlit app layout
st.title("Advanced Dynamic Data Cleaner App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = load_data(uploaded_file)
    
    # Display the original data
    st.subheader("Original Data")
    st.write(df)
    
    # Clean the data
    st.subheader("Data Cleaning Options")
    cleaned_df = clean_data(df)
    
    # Remove outliers
    cleaned_df = remove_outliers(cleaned_df)
    
    # Visualize data
    visualize_data(cleaned_df)
    
    # Display the cleaned data
    st.subheader("Cleaned Data")
    st.write(cleaned_df)

    # Download cleaned data
    st.download_button("Download Cleaned Data", cleaned_df.to_csv(index=False), "cleaned_data.csv", "text/csv")