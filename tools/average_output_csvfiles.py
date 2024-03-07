import pandas as pd

# Replace these file paths with the actual paths to your CSV files
filenames = ['../dataset/cycles_dataset/ML_model_collected_dataset_cycles1.csv',
             '../dataset/cycles_dataset/ML_model_collected_dataset_cycles2.csv',
             '../dataset/cycles_dataset/ML_model_collected_dataset_cycles3.csv',
             '../dataset/cycles_dataset/ML_model_collected_dataset_cycles4.csv',
             '../dataset/cycles_dataset/ML_model_collected_dataset_cycles5.csv']

# Read all CSV files and store them in a list
dataframes = [pd.read_csv(filename) for filename in filenames]

# Concatenate all the dataframes vertically
concatenated_df = pd.concat(dataframes)

# Calculate the mean for each column, ignoring the 'Matrix Size' if it's an identifier
mean_df = concatenated_df.groupby('index', as_index=False).mean()

# Save the resulting DataFrame to a new CSV file
mean_df.to_csv('../dataset/cycles_dataset/ML_model_collected_dataset_cycles_avg.csv', index=False)
