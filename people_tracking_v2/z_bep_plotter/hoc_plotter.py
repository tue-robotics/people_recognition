import csv
import matplotlib.pyplot as plt

def plot_hoc_values(csv_file_path):
    timestamps = []
    operator_ids = []
    hoc_values = []

    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamps.append(float(row['Timestamp']))
            operator_ids.append(int(row['Operator ID']))
            hoc_values.append(float(row['Operator HoC Value']))

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, hoc_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('HoC Value')
    plt.title('HoC Value of the Operator Over Time')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    csv_file_path = '/home/miguel/Documents/BEP-Testing/Test Case 2/Excel Sat Jun 29 Test case 2 full tracker/decision_data.csv'  # Update this path
    plot_hoc_values(csv_file_path)
