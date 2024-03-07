import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

csv_file_path = '../../dataset/old_datasets/dataset_all_54_events.csv'
data = pd.read_csv(csv_file_path)

X = data[['L1 Cache Misses', 'Conditional branch ins', 'Total Cycles']]
print(X)

# Dependent variable
y = data[["dram energy"]]
print(y)

X_testdata = data[['L1 Cache Misses', 'Conditional branch ins', 'Total Cycles']]
print(X)



# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the Lasso regression model
# alpha is the regularization strength; larger values specify stronger regularization.
lasso_model = Lasso(alpha=1.0)

# Train the model
lasso_model.fit(X_train, y_train)

# Predict using the trained model
predict = lasso_model.predict(X_train)
y_train_pred = predict
y_test_pred = lasso_model.predict(X_test)

# Evaluate the model
lasso_mse = mean_absolute_error(y_test, y_test_pred)
lasso_r2 = r2_score(y_test, y_test_pred)

# Output the coefficients and the intercept
lasso_coefficients = lasso_model.coef_
lasso_intercept = lasso_model.intercept_


print("Lasso Regression R2 Score:", lasso_r2)
print("Lasso Regression Coefficients:", lasso_coefficients)
print("Lasso Regression Intercept:", lasso_intercept)








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

