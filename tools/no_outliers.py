import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/old_datasets/ML_model_collected_dataset.csv')
print(data)
Q1 = data[['dram energy']].quantile(0.25)
Q3 = data[['dram energy']].quantile(0.75)
IQR = Q3 - Q1
threshold_multiplier = 0
outlier_rows = ((data[['dram energy']] < (Q1 - threshold_multiplier * IQR)) | (data[['dram energy']] > (Q3 + threshold_multiplier * IQR))).any(axis=1)
print(outlier_rows)
data_cleaned = data[~outlier_rows]
print(data_cleaned)




plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['Total Instructions'], data_cleaned['dram energy'], label='dram energy')

# Adding title and labels
plt.title('ins vs dram energy total')
plt.xlabel('ins')
plt.ylabel('dram energy')

# Adding a grid for better readability
plt.grid(True)

# Showing the plot
plt.legend()
plt.show()

