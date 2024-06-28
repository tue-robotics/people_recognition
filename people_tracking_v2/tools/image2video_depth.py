import cv2
import os
import glob

def create_video_from_images(image_folder, output_video_file, fps=30, target_size=(1280, 720)):
    # Print the absolute path to verify correctness
    abs_image_folder = os.path.abspath(image_folder)
    print(f"Absolute path to images: {abs_image_folder}")

    # Verify the directory exists
    if not os.path.exists(abs_image_folder):
        raise Exception(f"The specified image folder does not exist: {abs_image_folder}")

    # Check if directory is readable (permission check)
    if not os.access(abs_image_folder, os.R_OK):
        raise Exception(f"The script does not have permission to read the directory: {abs_image_folder}")

    # Get all the image paths with correct extension and sorted
    images = sorted(glob.glob(os.path.join(abs_image_folder, '*.png')))
    print(f"Found images: {len(images)}")  # Print count of found images for debugging

    # Proceed if images are found
    if not images:
        raise Exception("No images found. Check your folder path and image extensions.")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, target_size)

    # Write each resized image as a frame in the video
    for image in images:
        frame = cv2.imread(image)
        if frame is None:
            print(f"Warning: Could not read image {image}. Skipping.")
            continue

        # Resize the frame
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        # Write the resized frame to the video
        out.write(resized_frame)

    out.release()  # Release everything when job is finished
    print("Video processing complete.")

# Usage
image_folder = '/home/miguel/Documents/BEP-Testing/TestCase2/Frames Tue Jun 25 Test case 2/depth'  # Corrected path
output_video_file = '/home/miguel/Documents/BEP-Testing/TestCase2/TestCase2_depth.mp4'  # Desired output file
fps = 32.095  # Frames per second of the output video

create_video_from_images(image_folder, output_video_file, fps)
