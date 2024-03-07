import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../dataset/ipc_dataset/old_dataset/ML_model_ipc_results_compare.csv")
X = data['pred']
Y = data['true']

print(r2_score(X, Y))