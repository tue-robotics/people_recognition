#!/usr/bin/env python

# ROS modules
import rospy
from visualization_msgs.msg import MarkerArray
import random

# People recognition 3D modules
from people_recognition_msgs.msg import Person3D, People3D
from people_recognition_msgs.srv import RecognizePeople3D, RecognizePeople3DResponse
from geometry_msgs.msg import Point, Pose


class DummyPeopleRecognition3DNode:
    def __init__(self):
        self._recognize_people_3d_srv = rospy.Service('detect_people_3d',
                RecognizePeople3D,
                self._recognize_people_3d_srv)

        self._counter = 0

        rospy.loginfo("PeopleRecognition3DNode initialized:")

    def _generate_dummy_person3d(self, rgb, depth, cam_info, name=None):
        person = Person3D()
        person.header = rgb.header
        person.name = name if name else "Person{}".format(self._counter)
        person.age = 20 + self._counter
        person.gender = random.choice([0, 1])
        person.gender_confidence = random.normalvariate(mu=0.75, sigma=0.2)
        person.posture = random.choice(['standing', 'sitting'])
        person.emotion = 'happy'
        colors = ['black', 'orange', 'yellow']
        random.shuffle(colors)
        person.shirt_colors = colors
        # person.body_parts_pose
        person.position = Point()
        person.position.z = 3.0

        xs = [-2, -1, -0.5, 0.5, 1, 2]
        random.shuffle(xs)
        person.position.x = xs.pop()

        return person


    def _recognize_people_3d_srv(self, req):
        """
        Callback when the RecognizePeople3D service is called

        :param req: RecognizePeople3DRequest (with .image_rgb, .image_depth and
            .camera_info_depth attributes)
        :return: RecognizePeople3DResponse (with a .people attribute)
        """
        # Convert to opencv images
        rospy.loginfo("Detecting people in 3D from incoming RGB-D image")
        names_str = raw_input("Please enter the names of the people the robot should see, comma-separated: ")

        names = [name.strip() for name in names_str.split(",")]
        people3d = [self._generate_dummy_person3d(req.image_rgb,
                                                  req.image_depth,
                                                  req.camera_info_depth,
                                                  name) for name in names]

        return RecognizePeople3DResponse(people=people3d)


if __name__ == '__main__':
    rospy.init_node('people_recognition_3d')
    node = DummyPeopleRecognition3DNode()
    rospy.spin()
