import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('piPAPI_NN_output.csv')
column1 = df1['No of Epochs']
column2 = df1['Cycles']

"""df2 = pd.read_csv('piRAPL_output.csv')
column2 = df2['Total Energy']"""

plt.figure(figsize=(10, 6))
plt.plot(column1, column2)
plt.xlabel('Instruction Count')
plt.ylabel('Total Energy')
plt.title('Instruction Count vs Total Energy')
plt.grid(True)
plt.show()
