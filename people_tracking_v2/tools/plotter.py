import numpy as np
import matplotlib.pyplot as plt
import os

# Expand the user directory
hue_file_path = os.path.expanduser('~/hoc_data/hoc_hue_detection_1.npy')
sat_file_path = os.path.expanduser('~/hoc_data/hoc_sat_detection_1.npy')

# Load the HoC arrays from the saved files
hoc_hue = np.load(hue_file_path)
hoc_sat = np.load(sat_file_path)

# Plot the Hue and Saturation HoC arrays
plt.figure(figsize=(10, 6))
plt.plot(hoc_hue, label='Hue')
plt.plot(hoc_sat, label='Saturation')
plt.title('Histogram of Colors (HoC)')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.legend()
plt.show()
