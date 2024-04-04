import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

# Path to your CSV file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import dataset



# Creating a DataFrame from the CSV data
data = pd.read_csv("../dataset/ipc_cycles_dataset/ML_model_svm_ipc_cycles_dataset_10_iterations_avg.csv")


#For CPU Model
# Independent variables
X = data['ipc']

# Dependent variables
Y = data['cpu energy']


"""#For DRAM Model

independent_vars = ['L1 Cache Misses', 'Conditional branch ins', 'Total Cycles']

# Dependent variables
dependent_var = 'dram energy'"""


"""
# Create three subplots
fig, axs = plt.subplots(1, 1, figsize=(15, 5), sharey=True)

# Iterate through each independent variable and create scatter plots
for i, independent_var in enumerate(independent_vars):
    axs[i].plot(data[independent_var], data[dependent_var])
    axs[i].set_xlabel(independent_var)

# Set common y-axis label
axs[0].set_ylabel(dependent_var)

# Set the overall title for the subplots
fig.suptitle(f"Scatter Plots of Independent Variables vs {dependent_var}")

# Show the plots
plt.show()
"""

""" Create a 3x3 subplot
fig, axs = plt.subplots(1, 3, figsize=(10, 10))

# Loop through each independent variable
for i, ivar in enumerate(independent_vars):
    # Loop through each dependent variable
    for j, dvar in enumerate(dependent_vars):
        # Create scatter plot for each combination of independent and dependent variables
        axs[j, i].scatter(data[ivar], data[dvar])
        axs[j, i].set_xlabel(ivar)
        axs[j, i].set_ylabel(dvar)
        axs[j, i].set_title(f'{ivar} vs {dvar}')

# Adjust layout for better readability
plt.tight_layout()
plt.show()"""

plt.figure(figsize=(10, 10))
plt.scatter(X, Y, label='graph')

# Adding title and labels
plt.title('cycles vs cpu energy')
plt.xlabel('X')
plt.ylabel('Y')

# Adding a grid for better readability
plt.grid(True)

# Showing the plot
plt.legend()
plt.show()




