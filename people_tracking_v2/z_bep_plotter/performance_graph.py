import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
decision_data_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/Case 1 Data/decision_data.csv'
ground_truth_data_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/Case 1 Data/Ground truth.ods'

# Read the CSV and ODS files
decision_df = pd.read_csv(decision_data_path)
ground_truth_df = pd.read_excel(ground_truth_data_path, engine='odf')

# Merge the dataframes on Frame Number
merged_df = pd.merge(decision_df, ground_truth_df, on='Frame Number')

# Define the conditions for true positives, true negatives, false positives, and false negatives
conditions = [
    (merged_df['Operator ID'] > 0) & (merged_df['Operator ID'] == merged_df['Real operator']),
    (merged_df['Operator ID'] == -1) & (merged_df['Real operator'] == -1),
    (merged_df['Operator ID'] > 0) & (merged_df['Operator ID'] != merged_df['Real operator']),
    (merged_df['Operator ID'] == -1) & (merged_df['Real operator'] != -1),
]

# Define the corresponding categories
choices = ['True Positive', 'True Negative', 'False Positive', 'False Negative']

# Apply the conditions to create a new column for the results
merged_df['Result'] = np.select(conditions, choices, default='Unknown')

# Print the summary of results
summary = merged_df['Result'].value_counts()
print(summary)

# Plot the results over time
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Frame Number'] / 15, merged_df['Result'].eq('True Positive').cumsum(), label='True Positives', color='g')
plt.plot(merged_df['Frame Number'] / 15, merged_df['Result'].eq('True Negative').cumsum(), label='True Negatives', color='b')
plt.plot(merged_df['Frame Number'] / 15, merged_df['Result'].eq('False Positive').cumsum(), label='False Positives', color='r')
plt.plot(merged_df['Frame Number'] / 15, merged_df['Result'].eq('False Negative').cumsum(), label='False Negatives', color='m')

plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Count')
plt.title('Detection Results Over Time')
plt.legend()
plt.grid(True)
plt.show()
