import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression



def clean_data(data):

    mean = data['dram energy'].mean()
    std_dev = data['dram energy'].std()

    # Set the threshold (e.g., 3 standard deviations)
    threshold = 3

    # Calculate Z-score for each data point
    data['Z_score'] = (data['dram energy'] - mean) / std_dev

    # Identify outliers based on the threshold
    outliers = data[np.abs(data['Z_score']) > threshold]

    # Remove rows with outliers
    cleaned_data = data[np.abs(data['Z_score']) <= threshold]

    # Optionally, remove the 'Z_score' column from the cleaned data
    cleaned_data = cleaned_data.drop(columns=['Z_score'])
    print(cleaned_data)

    X = cleaned_data[['cycles']]


    # Dependent variable
    y = cleaned_data[["dram energy"]]

    return X, y

def plot_graph(X,y1, y2):
    """    plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, label='graph')

    # Adding title and labels
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Adding a grid for better readability
    plt.grid(True)

    # Showing the plot
    plt.legend()
    plt.show()"""

    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel('X axis label')
    ax1.set_ylabel('Y1 axis label', color=color)
    ax1.scatter(X, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel('Y2 axis label', color=color)
    ax2.plot(X, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph with two Y values and one X value')
    ax1.grid(True)

    # Show the plot
    plt.show()





csv_file_path = '../dataset/cycles_dataset/ML_model_cycles_dataset_joules_avg.csv'
dataset_train = pd.read_csv(csv_file_path)

X, y = clean_data(dataset_train)



# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

"""# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)"""

"""csv_file_path = '../dataset/cycles_dataset/ML_model_collected_dataset_cycles.csv'
dataset_test = pd.read_csv(csv_file_path)
X, y = clean_data(dataset_test)
plot_graph(X,y)

# Split the data into training/testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.90, random_state=42)"""

"""# Make predictions using the testing set
y_pred = regr.predict(X_test)"""
"""lasso_model = Lasso(alpha=1.0)

# Train the model
lasso_model.fit(X_train, y_train)

# Predict using the trained model

y_pred = lasso_model.predict(X_test)"""
# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
plot_graph(X_test, y_test, y_pred)


# Evaluate the model
lasso_mse = mean_absolute_error(y_test, y_pred)
lasso_r2 = r2_score(y_test, y_pred)

# Output the coefficients and the intercept
lasso_coefficients = regr.coef_
lasso_intercept = regr.intercept_


print("Lasso Regression R2 Score:", lasso_r2)
print("Lasso Regression Coefficients:", lasso_coefficients)
print("Lasso Regression Intercept:", lasso_intercept)

print("Predicted result")

y_pred_list = []
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

to_csvfile = {}

to_csvfile["pred"] = y_pred_list
to_csvfile["true"] = y_test["dram energy"]



df = pd.DataFrame(to_csvfile)
csv_file = '../../dataset/cycles_dataset/old_dataset/ML_Lasso_model_cycles_result_joules_compare.csv'  # Specify your CSV file name
df.to_csv(csv_file, index=False, mode = 'w')
# The coefficients
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Mean abs error: %.2f' % mean_absolute_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, y_pred))
