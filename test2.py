"""# Define the actual and expected values
actual_value = 3000000000.0
expected_value = 3321983132.0

# Calculate the absolute error
absolute_error = abs(actual_value - expected_value)

# Calculate the error percentage
error_percentage = (absolute_error / expected_value) * 100

# Print the error percentage
print(f"Error Percentage: {error_percentage}%")
"""

import pandas as pd

# Example lists
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c', 'd', 'e']
list3 = [True, False, True, False, True]
list4 = [2.2, 3.3, 4.4, 5.5, 6.6]
list5 = ['one', 'two', 'three', 'four', 'five']

# Combine into a dictionary
data = {
    'Numbers': list1,
    'Letters': list2,
    'Booleans': list3,
    'Floats': list4,
    'Words': list5
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
csv_file = 'output.csv'  # Specify your CSV file name
df.to_csv(csv_file, index=False)
