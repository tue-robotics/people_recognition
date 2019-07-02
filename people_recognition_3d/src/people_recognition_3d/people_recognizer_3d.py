#!/usr/bin/env python
from __future__ import print_function, division

import PyKDL as kdl
from collections import namedtuple
import numpy as np

import image_geometry
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from people_recognition_msgs.srv import RecognizePeople2D, RecognizePeople2DResponse
from people_recognition_msgs.msg import Person3D


def _get_and_wait_for_service(srv_name, srv_class):
    """
    Function to start and wait for dependent service
    :param: srv_name: Service name
    :param: srv_class: Service class
    :return: started ServiceProxy object
    """
    service = rospy.ServiceProxy('{}'.format(srv_name), srv_class)
    rospy.loginfo("Waiting for service {} ...".format(service.resolved_name))
    service.wait_for_service()
    return service


def _get_service_response(srv, args):
    """
    Method to get service response with checks
    :param: srv: service
    :param: args: Input arguments of the service request
    :return: response
    """
    try:
        response = srv(args)
    except Exception as e:
        rospy.logwarn("{} service call failed: {}".format(
            srv.resolved_name, e))
        raise
    else:
        return response


Joint = namedtuple('Joint', ['group_id', 'name', 'p', 'point'])


def geometry_msg_point_to_kdl_vector(msg):
    return kdl.Vector(msg.x, msg.y, msg.z)


def get_frame_from_vector(x_vector,
                          translation,
                          z_direction=kdl.Vector(0, 0, 1)):
    """
    Function to generate an affine transformation frame given the x_vector, z_direction and
    translation of the frame.

    How this works:
        Any two given vectors form a plane so, x_vector and z_direction can be
        considered as such vectors. Taking vector cross-product of these two
        vectors will give a vector perpendicular to the plane.

        1. First normalize the x_vector to get a unit_x vector.
        2. Take cross product of z_direction and unit_x, the will give the
            y_direction. Normalize y_direction to get the unit_y vector.
        3. Take the cross product between unit_x and unit_y to get unit_z

    :param: x_vector: The x_vector in some coordinate frame.
    :param: origin: The origin of the frame to be created
    :param: z_direction (default kdl.Vector(0, 0, 1)): The direction of z
    :return: frame: KDL frame
    """
    unit_x = x_vector / x_vector.Norm()
    unit_y = (z_direction * unit_x) / (z_direction * unit_x).Norm()
    unit_z = unit_x * unit_y

    rotation = kdl.Rotation(unit_x, unit_y, unit_z)

    return kdl.Frame(rotation, translation)


class Skeleton(object):
    """
    Dictionary of all joints, the following joins could be available:

    Nose
    Neck
    {L, R}{Shoulder, Elbow, Wrist, Hip, Knee, Ankle, Eye, Ear}
    """

    def __init__(self, body_parts):
        """
        Constructor

        :param body_parts: {name: Joint}
        """
        self.links = [
            # Head left half
            ('LEar', 'LEye'),
            ('LEye', 'Nose'),
            # Head right half
            ('REar', 'REye'),
            ('REye', 'Nose'),
            # Head center
            ('Nose', 'Neck'),

            # Upper body left half
            ('Neck', 'LShoulder'),
            ('LShoulder', 'LElbow'),
            ('LElbow', 'LWrist'),
            # Upper body right half
            ('Neck', 'RShoulder'),
            ('RShoulder', 'RElbow'),
            ('RElbow', 'RWrist'),

            # Lower body left half
            ('Neck', 'LHip'),
            ('LHip', 'LKnee'),
            ('LKnee', 'LAnkle'),
            # Lower body right half
            ('Neck', 'RHip'),
            ('RKnee', 'RAnkle'),
            ('RHip', 'RKnee'),
        ]

        self.body_parts = body_parts

    def filter_body_parts(self, threshold):
        """
        Method to remove body parts from a Skeleton object based on the
        maximum length of a link

        :param: threshold: Maximum length of a link
        :return: Skeleton object containing body parts within the threshold
        """
        return_list = set()
        for (a, b) in self.links:
            if a in self.body_parts and b in self.body_parts:
                p1 = self.body_parts[a].point
                p2 = self.body_parts[b].point

                l = (geometry_msg_point_to_kdl_vector(p1) -
                     geometry_msg_point_to_kdl_vector(p2)).Norm()
                if l <= threshold:
                    return_list.add(a)
                    return_list.add(b)

        return Skeleton({
            name: joint
            for name, joint in self.body_parts.items() if name in return_list
        })

    # def __iter__(self):
    #     return self.body_parts.__iter__()
    #
    # def __index__(self, value):
    #     return self.body_parts.__index__(value)
    #
    def __getitem__(self, key):
        return self.body_parts.__getitem__(key)

    def __contains__(self, item):
        return self.body_parts.__contains__(item)

    #
    # def items(self):
    #     return self.body_parts.items()

    def get_links(self):
        """
        :returns [Point], with point pairs for all the links
        """
        for (a, b) in self.links:
            if a in self.body_parts and b in self.body_parts:
                yield self.body_parts[a].point
                yield self.body_parts[b].point
            else:
                pass

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.body_parts)


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i + 1  # skip the first color (black)
        for j in range(8):
            r |= bitget(c, 0) << 7 - j
            g |= bitget(c, 1) << 7 - j
            b |= bitget(c, 2) << 7 - j
            c >>= 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class PeopleRecognizer3D(object):
    def __init__(self, recognize_people_srv_name, probability_threshold,
                 link_threshold, heuristic, arm_norm_threshold,
                 neck_norm_threshold, wave_threshold, vert_threshold,
                 hor_threshold, padding):

        self._recognize_people_srv = _get_and_wait_for_service(
            recognize_people_srv_name, RecognizePeople2D)

        self._bridge = CvBridge()

        # parameters
        self._threshold = probability_threshold
        self._link_threshold = link_threshold
        self._heuristic = heuristic
        self._arm_norm_threshold = arm_norm_threshold
        self._neck_norm_threshold = neck_norm_threshold
        self._wave_threshold = wave_threshold
        self._vert_threshold = vert_threshold
        self._hor_threshold = hor_threshold
        self._padding = padding

        rospy.loginfo('People recognizer 3D initialized')

    def recognize(self, rgb, depth, camera_info):
        """
        Service call function
        :param: rgb: RGB Image msg
        :param: depth: Depth Image_msg
        :param: depth_info: Depth CameraInfo msg
        """
        assert isinstance(rgb, Image)
        assert isinstance(depth, Image)
        assert isinstance(camera_info, CameraInfo)

        rospy.loginfo('Got recognize service call')
        cam_model = image_geometry.PinholeCameraModel()
        cam_model.fromCameraInfo(camera_info)

        t = rospy.Time.now()

        recognize_people_response = _get_service_response(
            self._recognize_people_srv, rgb)

        people2d = recognize_people_response.people
        rospy.loginfo('PeopleRecognizer2D took %f seconds',
                      (rospy.Time.now() - t).to_sec())
        rospy.loginfo('Found {} people'.format(len(people2d)))

        max_group = max(person2d.body_parts[0].group_id for person2d in people2d) if people2d else 0
        rospy.loginfo("max_group = {}".format(max_group))
        cmap = color_map(N=max_group+1, normalized=True)

        markers = MarkerArray()
        delete_all = Marker(action=Marker.DELETEALL)
        delete_all.header.frame_id = rgb.header.frame_id
        markers.markers.append(delete_all)

        cv_depth = self._bridge.imgmsg_to_cv2(depth)
        people3d = []
        depth_image_scaling = 1.0

        if rgb.width != depth.width or rgb.height != depth.height:
            depth_image_scaling = depth.width / rgb.width
            rospy.logdebug(
                "RGB and D don't have same dimensions, using scaling factor '%f' on ROIs",
                depth_image_scaling)

        regions_viz = np.zeros_like(cv_depth)

        for person2d in people2d:
            i = person2d.body_parts[0].group_id
            color_i = cmap[i, 0], cmap[i, 1], cmap[i, 2]
            joints = self.recognitions_to_joints(person2d.body_parts, cv_depth,
                                                 cam_model, regions_viz,
                                                 depth_image_scaling)

            # visualize joints
            rospy.logdebug('found %s objects for group %s', len(joints), i)

            # Skip the person for who has no 3D joints
            if not joints:
                continue

            points = [j.point for j in joints]
            markers.markers.append(
                Marker(header=rgb.header,
                       ns='joints',
                       id=i,
                       type=Marker.SPHERE_LIST,
                       action=Marker.ADD,
                       points=points,
                       scale=Vector3(0.07, 0.07, 0.07),
                       color=ColorRGBA(color_i[0], color_i[1], color_i[2], 1.0)))

            unfiltered_skeleton = Skeleton({j.name: j for j in joints})
            skeleton = unfiltered_skeleton.filter_body_parts(
                self._link_threshold)

            # visualize links
            markers.markers.append(
                Marker(header=rgb.header,
                       ns='links',
                       id=i,
                       type=Marker.LINE_LIST,
                       action=Marker.ADD,
                       points=list(skeleton.get_links()),
                       scale=Vector3(0.03, 0, 0),
                       color=ColorRGBA(color_i[0] * 0.9, color_i[1] * 0.9, color_i[2] * 0.9, 1.0)))

            # If the skeleton has no body parts do not add the recognition in
            # the list of 3D people
            if any(skeleton.body_parts):
                try:
                    point3d = skeleton['Neck'].point
                except KeyError:
                    x = []
                    y = []
                    z = []
                    for _, joint in skeleton.body_parts.iteritems():
                        x.append(joint.point.x)
                        y.append(joint.point.y)
                        z.append(joint.point.z)

                    x = np.average(x)
                    y = np.average(y)
                    z = np.average(z)
                    point3d = Vector3(x, y, z)
            else:
                rospy.logwarn(
                    "3D recognition of {} failed as no body parts found".
                    format(person2d.name))
                continue

            # rospy.loginfo("Skeleton: {}".format(skeleton))

            person3d = Person3D(
                header=rgb.header,
                name=person2d.name,
                age=person2d.age,
                gender=person2d.gender,
                gender_confidence=person2d.gender_confidence,
                posture=person2d.posture,
                emotion=person2d.emotion,
                shirt_colors=person2d.shirt_colors,
                body_parts_pose=person2d.body_parts,
                position=point3d,
                tags=self.get_person_tags(skeleton),
            )

            pointing_pose = self.get_pointing_pose(skeleton)

            if pointing_pose:
                person3d.tags.append("is_pointing")
                person3d.pointing_pose = pointing_pose

                markers.markers.append(
                    Marker(header=rgb.header,
                           ns='pointing_pose',
                           id=i,
                           type=Marker.ARROW,
                           action=Marker.ADD,
                           pose=person3d.pointing_pose,
                           scale=Vector3(0.5, 0.05, 0.05),
                           color=ColorRGBA(color_i[0], color_i[1], color_i[2], 1.0)))

            people3d.append(person3d)

        # After completion of people recognition, the regions_viz matrix is
        # populated with the depth data of all recognized people
        regions_viz = self._bridge.cv2_to_imgmsg(regions_viz)
        regions_viz.header = rgb.header

        rospy.loginfo("Done. Found {} people, {} markers".format(
            len(people3d), len(markers.markers)))
        return people3d, markers, regions_viz

    def recognitions_to_joints(self, recognitions, cv_depth, cam_model,
                               regions_viz, scale):
        """
        Method to convert 2D recognitions of body parts to Joint named tuple
        :param: recognitions: List of body part recognitions
        :param: cv_depth: cv2 Depth image
        :param: cam_model: Depth camera model
        :param: regions_viz: numpy array the size of cv_depth to store depth
                values of the ROIs
        :param: scale: Scaling factor of ROIs based on difference in size of RGB
                and D images
        :return: joints: List of joints of type Joint
        """
        joints = []
        joints_with_invalid_3D_roi = []

        for r in recognitions:
            assert len(r.categorical_distribution.probabilities) == 1
            pl = r.categorical_distribution.probabilities[0]
            label = pl.label
            p = pl.probability

            if p < self._threshold:
                continue

            roi = r.roi
            x_min = int((roi.x_offset - self._padding) * scale)
            x_max = int((roi.x_offset + roi.width + self._padding) * scale)
            y_min = int((roi.y_offset - self._padding) * scale)
            y_max = int((roi.y_offset + roi.height + self._padding) * scale)

            if x_min < 0 or y_min < 0 or x_max > cv_depth.shape[
                    1] or y_max > cv_depth.shape[0]:
                continue  # outside of the image
            # rospy.loginfo('roi=[%d %d %d %d] in %dx%d', x_min, x_max, y_min, y_max, depth.width, depth.height)

            region = cv_depth[y_min:y_max, x_min:x_max]

            # debugging viz
            regions_viz[y_min:y_max, x_min:x_max] = region

            u = (x_min + x_max) // 2
            v = (y_min + y_max) // 2

            ray = np.array(cam_model.projectPixelTo3dRay((u, v)))

            # Create a dummy joint for full nan and correct the position if non
            # nan regions based joints exist
            if np.all(np.isnan(region)):
                joints_with_invalid_3D_roi.append(
                    Joint(r.group_id, label, p, Point(*ray)))
                continue

            d = np.nanmedian(region)

            # project to 3d
            point3d = ray * d

            point = Point(*point3d)
            joints.append(Joint(r.group_id, label, p, point))

        new_joints = []
        if joints:
            if joints_with_invalid_3D_roi:
                mean_z = np.mean([j.point.z for j in joints])

                for j in joints_with_invalid_3D_roi:
                    j.point.x *= mean_z
                    j.point.y *= mean_z
                    j.point.z *= mean_z

                new_joints = joints + joints_with_invalid_3D_roi

            else:
                new_joints = joints

        return new_joints

    def get_person_tags(self, skeleton):
        """
        Method to get tags for a skeleton. The possible elements of the tag
        list are:
            1. LWave/LPointing | RWave/RPointing
            2. LLaying/LSitting | RLaying/RSitting

        :param: skeleton: The filtered skeleton of a person
        :return: tags: List of tags for the person
        """
        tags = []

        for side in ('L', 'R'):
            if self._heuristic == 'shoulder':
                try:
                    other = skeleton[side + 'Shoulder'].point
                except KeyError:
                    return tags
            else:
                raise ValueError('wrong heuristic')

            try:
                wrist = skeleton[side + 'Wrist'].point
            except KeyError:
                pass
            else:
                if wrist.y < (other.y - self._wave_threshold) and wrist.x < (
                        other.x + self._hor_threshold):
                    tags.append(side + 'Wave')

                elif wrist.x > (other.x + self._hor_threshold):
                    tags.append(side + 'Pointing')

            try:
                knee = skeleton[side + 'Knee'].point
            except KeyError:
                pass
            else:
                if knee.y < (other.y + self._vert_threshold) and knee.x > (
                        other.x + self._hor_threshold):
                    tags.append(side + 'Laying')

                elif knee.y < (other.y + self._vert_threshold) and knee.x < (
                        other.x + self._hor_threshold):
                    tags.append(side + 'Sitting')

        rospy.logdebug(tags)
        return tags

    def get_pointing_pose(self, skeleton):
        # We do required the shoulders for pointing calculation
        # if "Neck" not in skeleton or "Nose" not in skeleton:
        #     return None
        #
        # neck = geometry_msg_point_to_kdl_vector(skeleton['Neck'].point)
        # nose = geometry_msg_point_to_kdl_vector(skeleton['Nose'].point)
        # neck_vector = (nose - neck) / (nose - neck).Norm()
        neck_vector = kdl.Vector(0, -1, 0)

        # Check if arms are pointing
        # left_arm_valid = "LWrist" in skeleton and "LElbow" in skeleton and "LShoulder" in skeleton
        # right_arm_valid = "RWrist" in skeleton and "RElbow" in skeleton and "RShoulder" in skeleton
        left_arm_valid = "LElbow" in skeleton and "LShoulder" in skeleton
        right_arm_valid = "RElbow" in skeleton and "RShoulder" in skeleton

        if left_arm_valid:

            left_elbow = geometry_msg_point_to_kdl_vector(
                skeleton['LElbow'].point)
            left_shoulder = geometry_msg_point_to_kdl_vector(
                skeleton['LShoulder'].point)

            left_upper_arm_vector = (left_elbow - left_shoulder) / (
                left_elbow - left_shoulder).Norm()
            left_frame = get_frame_from_vector(left_upper_arm_vector,
                                               left_elbow)

            left_arm_neck_norm = (neck_vector * left_upper_arm_vector).Norm()

            if "LWrist" in skeleton:
                left_wrist = geometry_msg_point_to_kdl_vector(
                    skeleton['LWrist'].point)
                left_lower_arm_vector = (left_wrist - left_elbow) / (
                    left_wrist - left_elbow).Norm()

                left_arm_norm = (left_lower_arm_vector *
                                 left_upper_arm_vector).Norm()

                if left_arm_norm > self._arm_norm_threshold:
                    left_arm_valid = False
                else:
                    left_arm_vector = (left_wrist - left_shoulder) / (
                        left_wrist - left_shoulder).Norm()
                    left_frame = get_frame_from_vector(left_arm_vector,
                                                       left_wrist)

                rospy.logdebug("Left arm norm: %.2f", left_arm_norm)
        else:
            rospy.logdebug(
                "Left arm not valid because it does not contain all required body parts"
            )

        if right_arm_valid:

            right_elbow = geometry_msg_point_to_kdl_vector(
                skeleton['RElbow'].point)
            right_shoulder = geometry_msg_point_to_kdl_vector(
                skeleton['RShoulder'].point)

            right_upper_arm_vector = (right_elbow - right_shoulder) / (
                right_elbow - right_shoulder).Norm()
            right_frame = get_frame_from_vector(right_upper_arm_vector,
                                                right_elbow)

            right_arm_neck_norm = (neck_vector * right_upper_arm_vector).Norm()

            if "RWrist" in skeleton:
                right_wrist = geometry_msg_point_to_kdl_vector(
                    skeleton['RWrist'].point)
                right_lower_arm_vector = (right_wrist - right_elbow) / (
                    right_wrist - right_elbow).Norm()

                right_arm_norm = (right_lower_arm_vector *
                                  right_upper_arm_vector).Norm()

                if right_arm_norm > self._arm_norm_threshold:
                    right_arm_valid = False
                else:
                    right_arm_vector = (right_wrist - right_shoulder) / (
                        right_wrist - right_shoulder).Norm()
                    right_frame = get_frame_from_vector(
                        right_arm_vector, right_wrist)

                rospy.logdebug("Right arm norm: %.2f", right_arm_norm)
        else:
            rospy.logdebug(
                "Right arm not valid because it does not contain all required body parts"
            )

        rospy.logdebug("Arm norm threshold: %.2f", self._arm_norm_threshold)

        # Constraint based on parralelliness arm and neck
        if left_arm_valid and left_arm_neck_norm < self._neck_norm_threshold:
            rospy.logdebug(
                "Rejecting left arm because of neck norm threshold ...")
            left_arm_valid = False
        if right_arm_valid and right_arm_neck_norm < self._neck_norm_threshold:
            rospy.logdebug(
                "Rejecting right arm because of neck norm threshold ...")
            right_arm_valid = False

        # Optimize
        if left_arm_valid and right_arm_valid:
            if left_arm_neck_norm > right_arm_neck_norm:
                rospy.loginfo("Right arm is pointing the most, using this one")
                frame = right_frame
            else:
                rospy.loginfo("Left arm is pointing the most, using this one")
                frame = left_frame
        # if left_arm_valid and right_arm_valid:
        #     if left_arm_norm > right_arm_norm:
        #         rospy.loginfo("Right arm is pointing the most, using this one")
        #         frame = right_frame
        #     else:
        #         rospy.loginfo("Left arm is pointing the most, using this one")
        #         frame = left_frame
        elif left_arm_valid:
            frame = left_frame
        elif right_arm_valid:
            frame = right_frame
        else:
            rospy.logdebug("No valid arms found ...")
            return None

        return Pose(position=Point(*frame.p),
                    orientation=Quaternion(*frame.M.GetQuaternion()))
