import pandas as pd


# Read the data into a pandas DataFrame
df = pd.read_csv('../dataset/old_datasets/ML_model_collected_dataset_ins_cyc.csv')

# Calculate the average of all rows for each column
averages = df.mean()


total_energy_consumed = df['cpu energy'].sum()
total_instructions_executed = df['Total Instructions'].sum()

# Calculate energy per instruction (EPI)
print(total_energy_consumed)
print(total_instructions_executed)
energy_per_instruction = total_energy_consumed / total_instructions_executed

print("Energy per Instruction (EPI): {:.5f} energy units per instruction".format(energy_per_instruction))


"""print(averages)
print(averages['cpu energy']/averages['Total Instructions'])"""
