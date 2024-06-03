import numpy as np
import matplotlib.pyplot as plt
import os

# Expand the user directory
npz_file_path = os.path.expanduser('~/hoc_data/hoc_data.npz')

# Load the HoC arrays from the .npz file
with np.load(npz_file_path) as data:
    all_hue_histograms = data['hue']
    all_sat_histograms = data['sat']

# Select the first histogram for plotting (or change index as needed)
hoc_hue = all_hue_histograms[0]
hoc_sat = all_sat_histograms[0]

# Plot the Hue and Saturation HoC arrays
plt.figure(figsize=(10, 6))
plt.plot(hoc_hue, label='Hue')
plt.plot(hoc_sat, label='Saturation')
plt.title('Histogram of Colors (HoC)')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.legend()
plt.show()
