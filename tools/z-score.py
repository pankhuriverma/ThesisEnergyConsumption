import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
data = pd.read_csv('../dataset/cycles_dataset/old_dataset/ML_model_collected_dataset_cycles_avg.csv')
print(data)

# Remove rows with 'cpu energy' equal to 0.0
data_cleaned_0removed = data[data['cpu energy'] != 0.0]
print(data_cleaned_0removed)

mean = data_cleaned_0removed['cpu energy'].mean()
std_dev = data_cleaned_0removed['cpu energy'].std()

# Set the threshold (e.g., 3 standard deviations)
threshold = 3

# Calculate Z-score for each data point
data_cleaned_0removed['Z_score'] = (data_cleaned_0removed['cpu energy'] - mean) / std_dev

# Identify outliers based on the threshold
outliers = data_cleaned_0removed[np.abs(data_cleaned_0removed['Z_score']) > threshold]

# Remove rows with outliers
cleaned_data = data_cleaned_0removed[np.abs(data_cleaned_0removed['Z_score']) <= threshold]

# Optionally, remove the 'Z_score' column from the cleaned data
cleaned_data = cleaned_data.drop(columns=['Z_score'])
print(cleaned_data)

averages = cleaned_data.mean()
print(averages)
print(averages['cpu energy'] / averages['cycles'])

plt.figure(figsize=(10, 6))
plt.scatter(cleaned_data['cycles'], cleaned_data['cpu energy'], label='cpu energy')

# Adding title and labels
plt.title('ipc vs cpu energy total')
plt.xlabel('ipc')
plt.ylabel('cpu energy')

# Adding a grid for better readability
plt.grid(True)

# Showing the plot
plt.legend()
plt.show()

