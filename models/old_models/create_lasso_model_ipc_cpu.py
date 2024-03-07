import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RANSACRegressor



def clean_data(data):

    mean = data['cpu energy'].mean()
    std_dev = data['cpu energy'].std()

    # Set the threshold (e.g., 3 standard deviations)
    threshold = 3

    # Calculate Z-score for each data point
    data['Z_score'] = (data['cpu energy'] - mean) / std_dev

    # Identify outliers based on the threshold
    outliers = data[np.abs(data['Z_score']) > threshold]

    # Remove rows with outliers
    cleaned_data = data[np.abs(data['Z_score']) <= threshold]

    # Optionally, remove the 'Z_score' column from the cleaned data
    cleaned_data = cleaned_data.drop(columns=['Z_score'])
    #print(cleaned_data)
    cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)

    X = cleaned_data[['ipc']]


    # Dependent variable



    cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)

    y = cleaned_data[["cpu energy"]]
    print(X)
    print(y)

    return X, y

def plot_graph(X, y1, y2):
    """plt.figure(figsize=(10, 10))
    plt.plot(X, Y, label='graph')

    # Adding title and labels
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Adding a grid for better readability
    plt.grid(True)


    # Showing the plot
    plt.legend()
    plt.show()"""

    # Create a figure and a set of subplots
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




csv_file_path = '../../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'
dataset_train = pd.read_csv(csv_file_path)

X, y = clean_data(dataset_train)




# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

"""# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)"""
plot_graph(X_test,y_train)
reg = RANSACRegressor(random_state=0).fit(X_train, y_train)
output = reg.score(X, y)
print(output)

csv_file_path = "../../dataset/ipc_dataset/old_dataset/ML_model_linear_test_ipc_10iterations_avg.csv"
dataset_test = pd.read_csv(csv_file_path)
X, y = clean_data(dataset_test)


# Split the data into training/testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.90, random_state=42)

# Make predictions using the testing set
#y_pred = regr.predict(X_test1)
"""lasso_model = Lasso(alpha=0.2)

# Train the model
lasso_model.fit(X_train, y_train)

# Predict using the trained model

y_pred = lasso_model.predict(X_test)
plot_graph(y_test,y_pred)"""


"""plot_graph(X_test1, y_test1, y_pred)



# Evaluate the model
lasso_mae = mean_absolute_error(y_test1, y_pred)
lasso_r2 = r2_score(y_test1, y_pred)
lasso_mse = mean_squared_error(y_test1,y_pred)

# Output the coefficients and the intercept
lasso_coefficients = regr.coef_
lasso_intercept = regr.intercept_


print("Lasso Regression R2 Score:", lasso_r2)
print("Lasso Regression mse:", lasso_mse)
print("Lasso Regression mae:", lasso_mae)

print("Lasso Regression Coefficients:", lasso_coefficients)
print("Lasso Regression Intercept:", lasso_intercept)

print("Predicted result")

y_pred_list = []
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])
print(y_pred_list)
to_csvfile = {}

to_csvfile["pred"] = y_pred_list
to_csvfile["true"] = y_test1["cpu energy"]
"""


"""df = pd.DataFrame(to_csvfile)
csv_file = '../dataset/ipc_dataset/ML_model_linear8000_test_compare_avg.csv' # Specify your CSV file name
df.to_csv(csv_file, index=False, mode = 'w')"""