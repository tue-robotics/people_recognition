#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$


import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


laptop = True
name_subscriber_RGB = '/hero/head_rgbd_sensor/rgb/image_raw' if not laptop else 'video_frames'


class PeopleTracker:
    def __init__(self) -> None:

        # Initialize YOLO
        model_path = "~/MEGA/developers/Donal/yolov8n-seg.pt"
        device = "cuda"
        self.model = YOLO(model_path).to(device)
        self.person_class = 0  # person class = 0

        # ROS Initialize
        rospy.init_node('listener', anonymous=True)
        self.publisher = rospy.Publisher('/hero/segmented_image', Image)
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.callback, queue_size = 1)

    @staticmethod
    def detect(model, frame):
        """
            Return segemented image per class type.
        """
        results = model(frame)
        result = results[0]
        segmentation_contours_idx = [np.array(seg, dtype=np.int32) for seg in result.masks.xy]
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        return class_ids, segmentation_contours_idx

    def callback(self, data):
        rospy.loginfo("got message")
        seconds = rospy.get_time()
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        rospy.loginfo("converted message")

        classes, segmentations = self.detect(self.model, cv_image)

        mask = np.zeros_like(cv_image)

        for class_id, seg in zip(classes, segmentations):
            if class_id == self.person_class:
                # Fill the region enclosed by the polyline with white color (255)
                cv2.fillPoly(mask, [seg], (255, 255, 255))
        # Use the mask to cut out the regions from the original image
        cv_image[mask == 0] = 0  # Set the regions outside the mask to black (or any desired color)

        # # cv2.imshow("Segmented Image", cv_image)
        # # cv2.waitKey(1)

        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

        self.publisher.publish(image_message)   # Send image with boundaries human


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
