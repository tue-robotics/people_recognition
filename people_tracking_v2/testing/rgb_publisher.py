#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

def publish_images_from_folder(folder_path):
    pub = rospy.Publisher("/hero/head_rgbd_sensor/rgb/image_raw", Image, queue_size=10)
    bridge = CvBridge()

    # List all image files in the folder and sort them
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        rospy.logerr(f"No image files found in {folder_path}")
        return

    rate = rospy.Rate(18.7)  # Set the desired rate

    for image_file in image_files:
        if rospy.is_shutdown():
            break

        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            rospy.logerr(f"Error reading image file {image_path}")
            continue

        try:
            image_message = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(image_message)
            rate.sleep()
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('rgb_image_folder_publisher_node', anonymous=True)
    folder_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/Frames Sat Jun 29 Test Case 1/rgb' #/home/miguel/Documents/BEP-Testing/Test Case 1/Frames Sat Jun 29 Test Case 1/rgb
    try:
        publish_images_from_folder(folder_path)
    except rospy.ROSInterruptException:
        pass
