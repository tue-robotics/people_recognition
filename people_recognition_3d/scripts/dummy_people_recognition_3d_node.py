#!/usr/bin/env python

# ROS modules
import math

import rospy
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import MarkerArray
import random

# People recognition 3D modules
from people_recognition_msgs.msg import Person3D, People3D
from people_recognition_msgs.srv import RecognizePeople3D, RecognizePeople3DResponse
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import RegionOfInterest


class DummyPeopleRecognition3DNode:
    def __init__(self, prompt):
        self._recognize_people_3d_srv = rospy.Service('detect_people_3d',
                RecognizePeople3D,
                self._recognize_people_3d_srv)

        self._prompt = prompt
        self._counter = 0

        rospy.loginfo("PeopleRecognition3DNode initialized:")

    def _generate_dummy_person3d(self, rgb, depth, cam_info, name=None):
        person = Person3D()
        person.header = rgb.header
        person.name = name if name else ""  # Empty name makes this person unknwown
        person.age = 20 + self._counter
        person.gender = random.choice([0, 1])
        person.gender_confidence = random.normalvariate(mu=0.75, sigma=0.2)
        person.posture = random.choice(['standing', 'sitting'])
        person.emotion = 'happy'
        colors = ['black', 'orange', 'yellow']
        random.shuffle(colors)
        person.shirt_colors = colors
        person.tags = ['LWave', 'RWave'] + random.choice([[], ["is_pointing"]])
        # person.body_parts_pose
        person.position = Point()
        person.face.roi = RegionOfInterest(width=200, height=200)

        xs = [-2, -1, -0.5, 0.5, 1, 2]
        random.shuffle(xs)
        person.position.x = xs.pop()

        zs = [0.5, 1, 2, 3]
        random.shuffle(zs)
        person.position.z = zs.pop()

        person.pointing_pose = Pose(
            position=person.position,
            orientation=Quaternion(*quaternion_from_euler(0, 0, math.pi / 2))
        )

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
        if self._prompt:
            names_str = raw_input("Please enter the names of the people the robot should see, comma-separated: ")
            names = [name.strip() for name in names_str.split(",")]
        else:
            names = [""] * random.randint(0, 2)  # Randomly insert 0, 1, 2 people without a name

        people3d = [self._generate_dummy_person3d(req.image_rgb,
                                                  req.image_depth,
                                                  req.camera_info_depth,
                                                  name) for name in names]

        return RecognizePeople3DResponse(people=people3d)


if __name__ == '__main__':
    rospy.init_node('people_recognition_3d')
    prompt = rospy.get_param('~prompt', False)
    if prompt:
        rospy.logwarn("{} will ask for persons to be 'found'".format(rospy.get_name()))
    node = DummyPeopleRecognition3DNode(prompt)
    rospy.spin()
