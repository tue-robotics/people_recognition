import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
decision_data_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/all together data/decision_data_all.csv'
ground_truth_data_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/all together data/Ground truth_all.ods'

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

# Count the number of each result type
result_counts = merged_df['Result'].value_counts()
true_positives = result_counts.get('True Positive', 0)
true_negatives = result_counts.get('True Negative', 0)
false_positives = result_counts.get('False Positive', 0)
false_negatives = result_counts.get('False Negative', 0)
total_frames = len(merged_df)

# Print the results
print(f"True Positives: {true_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Total Frames: {total_frames}")

# Map result types to y-values for the scatter plot
result_mapping = {
    'True Positive': 'Tp',
    'True Negative': 'Tn',
    'False Positive': 'Fp',
    'False Negative': 'Fn'
}
merged_df['Result Type'] = merged_df['Result'].map(result_mapping)

# Plot the results over time
plt.figure(figsize=(16, 3))  # Reduce the height

# Scatter plot for each result type
colors = {'Tp': 'g', 'Tn': 'b', 'Fp': 'r', 'Fn': 'orange'}
y_ticks = ['Tp', 'Tn', 'Fp', 'Fn']
y_tick_labels = ['Tp', 'Tn', 'Fp', 'Fn']
y_positions = np.arange(len(y_ticks))

for result_type, color in colors.items():
    subset = merged_df[merged_df['Result Type'] == result_type]
    plt.scatter(subset['Frame Number'] / 15, [result_type] * len(subset), c=color, s=40)  # Increased dot size

plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Type', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(y_positions, y_tick_labels, fontsize=18)
plt.title('Association Type over Time', fontsize=24)
plt.grid(True)
plt.tight_layout()  # Adjust the layout to ensure everything fits
plt.show()
