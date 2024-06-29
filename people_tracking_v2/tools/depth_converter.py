import cv2
import os

def save_image(image, prefix, index, subfolder='depth_png'):
    directory = os.path.join(os.getcwd(), subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{prefix}_{index}.png"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved {filename} to {subfolder}/")

def process_depth_image(image_path, output_index):
    # Load the depth image
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Make sure to load it in the original bit depth
    
    # Normalize the depth image
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to 8-bit
    depth_image_8bit = cv2.convertScaleAbs(depth_image_normalized)
    
    # Save the 8-bit depth image
    save_image(depth_image_8bit, 'depth', output_index)

# Example usage
process_depth_image('/home/miguel/Documents/BEP-Testing/Test Case 1/Frames Sat Jun 29 Test Case 1/depth/depth_000422.png' , 1)
