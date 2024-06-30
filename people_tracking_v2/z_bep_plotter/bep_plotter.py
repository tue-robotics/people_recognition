import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Path to the folder containing frames
folder_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/Frames Sat Jun 29 Test Case 1/rgb'

# Frame rate of the video
frame_rate = 15

# Get list of all frames
frames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])

# Parameters for selecting frames
total_frames = len(frames)
selected_frame_indices = np.linspace(0, total_frames-1, 12, dtype=int)  # Select 12 evenly spaced frames

# List to store selected frames
selected_frames = []

for i in selected_frame_indices:
    # Read the frame
    frame = cv2.imread(frames[i])
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Calculate timestamp
    timestamp = i / frame_rate
    
    # Put timestamp on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{timestamp:.1f}'
    font_scale = 2  # Increase the font scale for larger text
    thickness = 3  # Increase the thickness for better visibility
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = frame_rgb.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Append frame to the list
    selected_frames.append(frame_rgb)

# Determine grid size for subplot
rows = int(np.ceil(len(selected_frames) / 4))
cols = 4

# Plot frames in a grid
fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
axes = axes.flatten()

for ax, frame in zip(axes, selected_frames):
    ax.imshow(frame)
    ax.axis('off')

# Hide any remaining empty subplots
for i in range(len(selected_frames), rows * cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('composite_image.png')
plt.show()
