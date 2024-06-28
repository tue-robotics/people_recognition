#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def publish_video(video_path):
    pub = rospy.Publisher("/hero/head_rgbd_sensor/rgb/image_raw", Image, queue_size=10)
    bridge = CvBridge()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        rospy.logerr(f"Error opening video file {video_path}")
        return

    rate = rospy.Rate(20)  # Modify based on your video's fps

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                image_message = bridge.cv2_to_imgmsg(frame, "bgr8")
                pub.publish(image_message)
                rate.sleep()
            except CvBridgeError as e:
                rospy.logerr(e)
        else:
            break

    cap.release()

if __name__ == '__main__':
    rospy.init_node('rgb_video_publisher_node', anonymous=True)
    video_path = '/home/miguel/Documents/BEP-Testing/TestCase1/TestCase1_rgb.mp4'
    try:
        publish_video(video_path)
    except rospy.ROSInterruptException:
        pass
