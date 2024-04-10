from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calculate_co2(y_pred_list,y_test_list):
    germany_co2_intensity = 365.477
    pred_co2_list = []
    test_co2_list = []
    for pred,test in zip(y_pred_list, y_test_list):
        pred_co2 = (pred/3600000) * germany_co2_intensity
        pred_co2_list.append(pred_co2)
        test_co2 = (test/3600000) * germany_co2_intensity
        test_co2_list.append(test_co2)

    return pred_co2_list, test_co2_list


def plot_2d_graph(X, y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_true, color='blue', label='Actual')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


def train_val_test_split(X, y, test_size=0.2, val_size=0.25, random_state=None):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted,
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


csv_file_path = '../../dataset/ipc_cycles_dataset/test.csv'
data = pd.read_csv(csv_file_path)


X = data[['cycles','ins']]
y = data[['cpu energy']]

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, test_size=0.2, val_size=0.25, random_state=42)
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape )
# Normalizing the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_val_scaled = scaler.fit_transform(y_val)
y_test_scaled = scaler.transform(y_test)

# Define the neural network model
np.random.seed(42)
regr = LinearRegression()
regr.fit(X_train_scaled, y_train_scaled)

y_pred = regr.predict(X_test_scaled)

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print('Mean squared error: %.2f')
print(mean_absolute_error(y_test_scaled, y_pred))
print('Coefficient of determination: %.2f')
print(r2_score(y_test_scaled, y_pred))
X_test_unscaled = scaler.inverse_transform(X_test_scaled)
y_test_unscaled = scaler.inverse_transform(y_test_scaled)
y_pred_unscaled = scaler.inverse_transform(y_pred)

print("X_test_unscaled")
print(X_test_unscaled[1:10])
print(X_test_unscaled.shape)
print("y test unscaled")
print(y_test_unscaled[1:10])
print(y_test_unscaled.shape)
print("y pred unscaled")
print(y_pred_unscaled[1:10])
print(y_pred_unscaled.shape)

plot_2d_graph(X_test_unscaled, y_test_unscaled, y_pred_unscaled)

y_pred_list = []
for i in range(len(y_pred_unscaled)):
    y_pred_list.append(y_pred_unscaled[i][0])

y_test_list = []
for i in range(len(y_test_unscaled)):
    y_test_list.append(y_test_unscaled[i][0])

to_csvfile = {}

to_csvfile["pred energy"] = y_pred_list
to_csvfile["true energy"] = y_test_list

y_pred_co2emm, y_test_co2emm = calculate_co2(y_pred_list, y_test_list)
to_csvfile["pred co2"] = y_pred_co2emm
to_csvfile["true co2"] = y_test_co2emm
print(y_pred_co2emm)
print(y_test_co2emm)

df = pd.DataFrame(to_csvfile)
csv_file = '../../dataset/ipc_cycles_dataset/NN_model_ins_cycles_huber_loss_compare.csv' # Specify your CSV file name
df.to_csv(csv_file, index=False, mode = 'w')

