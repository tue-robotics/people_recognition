import cv2
import os
import glob

def create_video_from_images(image_folder, output_video_file, fps=30, target_size=None):
    print(f"Processing images from: {image_folder}")
    
    # Verify the directory exists and is readable
    abs_image_folder = os.path.abspath(image_folder)
    if not os.path.exists(abs_image_folder):
        raise Exception(f"The specified image folder does not exist: {abs_image_folder}")
    if not os.access(abs_image_folder, os.R_OK):
        raise Exception(f"The script does not have permission to read the directory: {abs_image_folder}")

    # Get all the image paths with correct extension and sorted
    images = sorted(glob.glob(os.path.join(abs_image_folder, '*.png')))
    if not images:
        raise Exception("No images found. Check your folder path and image extensions.")
    
    print(f"Found {len(images)} images in {image_folder}")

    # Load the first image to get video properties if target_size is not specified
    frame = cv2.imread(images[0])
    if frame is None:
        raise Exception("Could not read the first image. Ensure the image is accessible and not corrupted.")
    if target_size is None:
        target_size = (frame.shape[1], frame.shape[0])

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, target_size)

    # Write each image (resized if necessary) as a frame in the video
    for image in images:
        frame = cv2.imread(image)
        if frame is None:
            print(f"Warning: Could not read image {image}. Skipping.")
            continue
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        out.write(frame)

    out.release()  # Release everything when job is finished
    print(f"Video created at {output_video_file}")

def main():
    base_folder = '/home/miguel/Documents/BEP-Testing/TestCase4/Frames Tue Jun 25 Test case 4'
    
    # RGB Video Processing
    rgb_folder = os.path.join(base_folder, 'rgb')
    rgb_output = '/home/miguel/Documents/BEP-Testing/TestCase4/TestCase4_rgb.mp4'
    create_video_from_images(rgb_folder, rgb_output, fps=20)

    # Depth Video Processing
    depth_folder = os.path.join(base_folder, 'depth')
    depth_output = '/home/miguel/Documents/BEP-Testing/TestCase4/TestCase4_depth.mp4'
    depth_fps = 20 * len(glob.glob(os.path.join(depth_folder, '*.png'))) / len(glob.glob(os.path.join(rgb_folder, '*.png')))
    create_video_from_images(depth_folder, depth_output, fps=depth_fps, target_size=(1280, 720))

if __name__ == '__main__':
    main()
