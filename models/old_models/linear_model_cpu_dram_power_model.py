import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file_path = '../../results/matrix_results/output_run_average.csv'

# Reading the dataset from a CSV file
data = pd.read_csv(csv_file_path)

# Independent variables
X = data[["Total Instructions", "Total Cycles", "L1 Cache Misses"]]


# Dependent variable
y = data["cpu energy"]

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting the cpu energy for the test set
y_pred = model.predict(X_test)


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_


print("R2 Score:", r2)
print("Coefficients:", coefficients)
print("Intercept:", intercept)




# Plotting the actual vs predicted values for training set
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7, color='blue')  # Plotting the actual vs. predicted values
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Plotting a diagonal line for reference
plt.title('Training set: Actual vs. Predicted')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

# Plotting the actual vs predicted values for test set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7, color='red')  # Plotting the actual vs. predicted values
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Plotting a diagonal line for reference
plt.title('Test set: Actual vs. Predicted')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

plt.tight_layout()
plt.show()
