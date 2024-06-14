#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class VideoRecorder:
    def __init__(self, save_path):
        rospy.init_node('video_recorder', anonymous=True)
        
        # Subscribe to the RGB and depth image topics
        self.rgb_image_sub = rospy.Subscriber('/hero/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_image_callback)
        self.depth_image_sub = rospy.Subscriber('/hero/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_image_callback)
        
        # Create a CvBridge object
        self.bridge = CvBridge()
        
        # Initialize the video writers
        self.rgb_out = None
        self.depth_out = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = 30  # You can adjust this based on your requirements

        # Ensure the save path exists
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def rgb_image_callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV image
            rgb_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            
            # Initialize the RGB video writer once we get the first image
            if self.rgb_out is None:
                self.rgb_frame_size = (rgb_image.shape[1], rgb_image.shape[0])
                rgb_filename = os.path.join(self.save_path, 'rgb_output.avi')
                self.rgb_out = cv2.VideoWriter(rgb_filename, self.fourcc, self.fps, self.rgb_frame_size)
            
            # Write the RGB image to the video file
            self.rgb_out.write(rgb_image)

            # Optionally, display the RGB image
            cv2.imshow('RGB Camera Feed', rgb_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(f'CvBridge Error: {e}')

    def depth_image_callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(data, '16UC1')
            
            # Normalize depth image to 8-bit for visualization
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            depth_image_colored = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

            # Initialize the depth video writer once we get the first image
            if self.depth_out is None:
                self.depth_frame_size = (depth_image_colored.shape[1], depth_image_colored.shape[0])
                depth_filename = os.path.join(self.save_path, 'depth_output.avi')
                self.depth_out = cv2.VideoWriter(depth_filename, self.fourcc, self.fps, self.depth_frame_size)
            
            # Write the depth image to the video file
            self.depth_out.write(depth_image_colored)

            # Optionally, display the depth image
            cv2.imshow('Depth Camera Feed', depth_image_colored)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(f'CvBridge Error: {e}')

    def cleanup(self):
        # Release the video writers and close OpenCV windows
        if self.rgb_out is not None:
            self.rgb_out.release()
        if self.depth_out is not None:
            self.depth_out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    save_path = '~/hero_videos'  # Replace with your desired save path
    recorder = VideoRecorder(save_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        recorder.cleanup()
        print('Shutting down')
