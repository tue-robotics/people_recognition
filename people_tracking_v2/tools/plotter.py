import numpy as np
import matplotlib.pyplot as plt
import os

# Expand the user directory
npz_file_path = os.path.expanduser('~/hoc_data/hoc_data.npz')

# Load the HoC arrays from the .npz file
with np.load(npz_file_path) as data:
    all_hue_histograms = data['hue']
    all_sat_histograms = data['sat']

# Ensure we are dealing with the correct batch size
print("Number of Hue Histograms:", len(all_hue_histograms))
print("Number of Saturation Histograms:", len(all_sat_histograms))

# Select the first histogram for plotting (or change index as needed)
hoc_hue = all_hue_histograms
hoc_sat = all_sat_histograms

# Print some statistics about the histograms
print("Hue Histogram - Min:", np.min(hoc_hue), "Max:", np.max(hoc_hue), "Mean:", np.mean(hoc_hue))
print("Saturation Histogram - Min:", np.min(hoc_sat), "Max:", np.max(hoc_sat), "Mean:", np.mean(hoc_sat))

# Print first 10 values of the histograms
print("First 10 values of Hue Histogram:", hoc_hue[:10])
print("First 10 values of Saturation Histogram:", hoc_sat[:10])

# Plot the Hue and Saturation HoC arrays
plt.figure(figsize=(10, 6))
plt.plot(hoc_hue, label='Hue')
plt.plot(hoc_sat, label='Saturation')
plt.title('Histogram of Colors (HoC)')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.legend()
plt.show()
