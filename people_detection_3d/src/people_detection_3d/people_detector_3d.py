#!/usr/bin/env python
from __future__ import print_function, division

import math
import PyKDL as kdl
from collections import namedtuple
from itertools import groupby
import numpy as np

import image_geometry
# import message_filters
import rospy
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from image_recognition_msgs.srv import Recognize, DetectPeople
from people_detection_3d_msgs.msg import Person3D

def _get_and_wait_for_services(service_names, service_class):
    """
    Function to start and wait for dependent services
    """
    services = {s: rospy.ServiceProxy('{}'.format(s), service_class) for s in service_names}
    for service in services.values():
        rospy.loginfo("Waiting for service {} ...".format(service.resolved_name))
        service.wait_for_service()
    return services


Joint = namedtuple('Joint', ['group_id', 'name', 'p', 'point'])

def geometry_msg_point_to_kdl_vector(msg):
    return kdl.Vector(msg.x, msg.y, msg.z)


def get_frame_from_vector(x_vector, origin):
    unit_z = kdl.Vector(0, 0, 1)
    unit_z_cross_diff = (unit_z * x_vector) / (unit_z * x_vector).Norm()
    y_vector = x_vector * unit_z_cross_diff
    z_vector = x_vector * y_vector

    rotation = kdl.Rotation(x_vector, y_vector, z_vector)
    translation = origin

    return kdl.Frame(rotation, translation)


class Skeleton(object):
    """
    Dictionary of all joints, the following joins could be available:

    Nose
    Neck
    nose
    {L,R}{Shoulder,Elbow,Wrist,Hip,Knee,Ankle,Eye,Ear}
    """

    def __init__(self, bodyparts):
        """Constructor

        :param bodyparts: {name: Joint}
        """
        self.links = [
            # head
            ('Nose', 'Neck'),
            ('LEar', 'LEye'),
            ('LEye', 'Nose'),
            ('REar', 'REye'),
            ('REye', 'Nose'),

            # body
            ('LShoulder', 'Neck'),
            ('LShoulder', 'LElbow'),
            ('LElbow', 'LWrist'),
            ('RElbow', 'RWrist'),
            ('RShoulder', 'Neck'),
            ('RShoulder', 'RElbow'),

            # legs
            ('LHip', 'Neck'),
            ('LAnkle', 'LKnee'),
            ('LKnee', 'LHip'),
            ('RHip', 'Neck'),
            ('RAnkle', 'RKnee'),
            ('RKnee', 'RHip'),
        ]

        self.bodyparts = bodyparts

    def filter_bodyparts(self, threshold):
        filter_list = set()
        for (a, b) in self.links:
            if a in self.bodyparts and b in self.bodyparts:
                p1 = self.bodyparts[a].point
                p2 = self.bodyparts[b].point

                l = (geometry_msg_point_to_kdl_vector(p1) - geometry_msg_point_to_kdl_vector(p2)).Norm()
                if l > threshold:
                    filter_list.add(a)
                    filter_list.add(b)

        return Skeleton({name: joint for name, joint in self.bodyparts.items() if name not in filter_list})

    # def __iter__(self):
    #     return self.bodyparts.__iter__()
    #
    # def __index__(self, value):
    #     return self.bodyparts.__index__(value)
    #
    def __getitem__(self, key):
        return self.bodyparts.__getitem__(key)

    def __contains__(self, item):
        return self.bodyparts.__contains__(item)

    #
    # def items(self):
    #     return self.bodyparts.items()

    def get_links(self):
        """
        :returns [Point], with point pairs for all the links
        """
        for (a, b) in self.links:
            if a in self.bodyparts and b in self.bodyparts:
                # rospy.loginfo("Add link {}".format((a, b)))
                yield self.bodyparts[a].point
                yield self.bodyparts[b].point
            else:
                # rospy.logwarn("Not all bodyparts of link {} found".format((a, b)))
                pass

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.bodyparts)


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


class PeopleDetector3D(object):

    def __init__(self, detect_people_srv_name, probability_threshold, link_threshold, heuristic,
            arm_norm_threshold, wave_threshold, vert_threshold, hor_threshold,
            padding):

        self._detect_people_srv_name = detect_people_srv_name

        self._detect_people_services = _get_and_wait_for_services([
            self._detect_people_srv_name
        ], DetectPeople)

        self._bridge = CvBridge()

        # parameters
        self.threshold = probability_threshold
        self.link_threshold = link_threshold
        self.heuristic = heuristic
        self.arm_norm_threshold = arm_norm_threshold
        self.wave_threshold = wave_threshold
        self.vert_threshold = vert_threshold
        self.hor_threshold = hor_threshold
        self.padding = padding

        rospy.loginfo('People detector 3D initialized')

    def _get_detect_people(self, rgb_imgmsg):
        """
        Get recognitions from openpose and openface
        :param: rgb_imgmsg: RGB Image msg people detector service
        """
        return {srv_name: srv(image=rgb_imgmsg) for srv_name, srv in self._detect_people_services.iteritems()}


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
        try:
            people2d = self._get_detect_people(rgb)['detect_people'].people
        except rospy.ServiceException as e:
            rospy.logwarn('PeopleDetector2D call failed: %s', e)
            return
        rospy.loginfo('PeopleDetector2D took %f seconds', (rospy.Time.now() - t).to_sec())
        rospy.loginfo('Found {} people'.format(len(people2d)))

        cmap = color_map(N=len(people2d), normalized=True)

        markers = MarkerArray()
        delete_all = Marker(action=Marker.DELETEALL)
        delete_all.header.frame_id = rgb.header.frame_id
        markers.markers.append(delete_all)

        people3d = []
        for i, person2d in enumerate(people2d):
            joints = self.recognitions_to_joints(person2d.body_parts, rgb, depth, cam_model)

            # visualize joints
            rospy.logdebug('found %s objects for group %s', len(joints), i)

            points = [j.point for j in joints]
            markers.markers.append(Marker(header=rgb.header,
                                          ns='joints',
                                          id=i,
                                          type=Marker.SPHERE_LIST,
                                          action=Marker.ADD,
                                          points=points,
                                          scale=Vector3(0.07, 0.07, 0.07),
                                          color=ColorRGBA(cmap[i, 0], cmap[i, 1], cmap[i, 2], 1.0)))

            unfiltered_skeleton = Skeleton({j.name: j for j in joints})
            skeleton = unfiltered_skeleton.filter_bodyparts(self.link_threshold)

            # visualize links
            markers.markers.append(Marker(header=rgb.header,
                                          ns='links',
                                          id=i,
                                          type=Marker.LINE_LIST,
                                          action=Marker.ADD,
                                          points=list(skeleton.get_links()),
                                          scale=Vector3(0.03, 0, 0),
                                          color=ColorRGBA(cmap[i, 0] * 0.9, cmap[i, 1] * 0.9, cmap[i, 2] * 0.9, 1.0)))

            point3d = Vector3(i, i, i)
            try:
                point3d = skeleton['Neck'].point
            except KeyError:
                try:
                    point3d = skeleton['Head'].point
                except KeyError:
                    if any(skeleton.bodyparts):
                        x = np.average([joint.point.x for _, joint in skeleton.bodyparts.iteritems()])
                        y = np.average([joint.point.y for _, joint in skeleton.bodyparts.iteritems()])
                        z = np.average([joint.point.z for _, joint in skeleton.bodyparts.iteritems()])
                        point3d = Vector3(x, y, z)
                    else:
                        rospy.logwarn("There are no bodyparts to average")
            rospy.loginfo('Position: {}'.format(point3d))

            person3d = Person3D(
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

            pointing_pose = self.get_pointing_pose(skeleton, self.arm_norm_threshold)
            if pointing_pose:
                person3d.tags.append("is_pointing")
                person3d.pointing_pose = pointing_pose

            people3d.append(person3d)

            # visualize persons
            # height = 1.8
            # head = 0.2

            # q = tf.transformations.quaternion_from_euler(-math.pi / 2, 0, 0)
            if "is_pointing" in person3d.tags:
                markers.markers.append(Marker(header=rgb.header,
                                              ns='pointing_pose',
                                              id=i,
                                              type=Marker.ARROW,
                                              action=Marker.ADD,
                                              pose=person3d.pointing_pose,
                                              scale=Vector3(0.5, 0.05, 0.05),
                                              color=ColorRGBA(cmap[i, 0], cmap[i, 1], cmap[i, 2], 1.0)))

        # self.person_pub.publish(People(header=rgb.header, people=people3d))
        # publish all markers in one go
        self.markers_pub.publish(markers)
        rospy.loginfo("Done. Found {} people, {} markers".format(len(people3d), len(markers.markers)))
        return people3d

    def recognitions_to_joints(self, recognitions, rgb, depth, cam_model):
        cv_depth = self._bridge.imgmsg_to_cv2(depth)
        regions_viz = np.zeros_like(cv_depth)

        joints = list()
        for r in recognitions:
            assert len(r.categorical_distribution.probabilities) == 1
            pl = r.categorical_distribution.probabilities[0]
            label = pl.label
            p = pl.probability

            if p < self.threshold:
                continue

            roi = r.roi
            x_min = roi.x_offset - self.padding
            x_max = roi.x_offset + roi.width + self.padding
            y_min = roi.y_offset - self.padding
            y_max = roi.y_offset + roi.height + self.padding

            if rgb.width != depth.width or rgb.height != depth.height:
                factor = depth.width / rgb.width
                rospy.logdebug("using hack for Sjoerd's rgbd stuff, scaling factor %f", factor)

                x_min = int(x_min * factor)
                x_max = int(x_max * factor)
                y_min = int(y_min * factor)
                y_max = int(y_max * factor)

            if x_min < 0 or y_min < 0 or x_max > depth.width or y_max > depth.height:
                continue  # outside of the image
            # rospy.loginfo('roi=[%d %d %d %d] in %dx%d', x_min, x_max, y_min, y_max, depth.width, depth.height)

            region = cv_depth[y_min:y_max, x_min:x_max]

            # debugging viz
            regions_viz[y_min:y_max, x_min:x_max] = cv_depth[y_min:y_max, x_min:x_max]
            self.regions_viz_pub.publish(self._bridge.cv2_to_imgmsg(regions_viz))

            u = (x_min + x_max) // 2
            v = (y_min + y_max) // 2

            # skip fully nan
            if np.all(np.isnan(region)):
                continue

            # Sjoerd's rgbd implementation returns 0 on invalid
            if not np.all(region):
                joints.append(Joint(r.group_id, label, p, Point(x=u, y=v, z=None)))
                continue

            median = np.nanmedian(region)
            rospy.logdebug('region p=%f min=%f, max=%f, median=%f', p, np.nanmin(region), np.nanmax(region), median)

            # project to 3d
            d = median
            ray = np.array(cam_model.projectPixelTo3dRay((u, v)))
            point3d = ray * d

            rospy.logdebug('3d point of %s is %d,%d: %s', label, u, v, point3d)
            point = Point(*point3d)
            joints.append(Joint(r.group_id, label, p, point))

        new_joints = list()
        for joint in joints:
            if joint.point.z:
                new_joints.append(joint)
            else:
                zs = list()
                for j in joints:
                    if j.group_id == joint.group_id and j.name != joint.name and j.point.z:
                        zs.append(j.point.z)

                if zs:
                    mean_z = np.mean(zs)
                    ray = np.array(cam_model.projectPixelTo3dRay((joint.point.x, joint.point.y)))
                    point3d = ray * mean_z

                    new_joint = Joint(joint.group_id, joint.name, joint.p, Point(*point3d))
                    new_joints.append(new_joint)
                else:
                    new_joints.append(joint)

        return new_joints

    def get_person_tags(self, skeleton):
        tags = list()
        for side in ('L', 'R'):
            try:
                if self.heuristic == 'shoulder':
                    other = skeleton[side + 'Shoulder'].point
                elif self.heuristic == 'head':
                    other = skeleton['Head'].point
                else:
                    raise ValueError('wrong heuristic')

                wrist = skeleton[side + 'Wrist'].point
            except KeyError:
                continue

            if wrist.y < (other.y - self.wave_threshold) and wrist.x < (other.x + self.hor_threshold):
                tags.append(side + 'Wave')

        for side in ('L', 'R'):
            try:
                if self.heuristic == 'shoulder':
                    other = skeleton[side + 'Shoulder'].point
                elif self.heuristic == 'head':
                    other = skeleton['Head'].point
                else:
                    raise ValueError('wrong heuristic')

                wrist = skeleton[side + 'Wrist'].point
            except KeyError:
                continue

            if wrist.x > (other.x + self.hor_threshold):
                tags.append(side + 'Pointing')

        for side in ('L', 'R'):
            try:
                if self.heuristic == 'shoulder':
                    other = skeleton[side + 'Shoulder'].point
                elif self.heuristic == 'head':
                    other = skeleton['Head'].point
                else:
                    raise ValueError('wrong heuristic')

                knee = skeleton[side + 'Knee'].point
            except KeyError:
                continue

            if knee.y < (other.y + self.vert_threshold) and knee.x > (other.x + self.hor_threshold):
                tags.append(side + 'Laying')

        for side in ('L', 'R'):
            try:
                if self.heuristic == 'shoulder':
                    other = skeleton[side + 'Shoulder'].point
                elif self.heuristic == 'head':
                    other = skeleton['Head'].point
                else:
                    raise ValueError('wrong heuristic')

                knee = skeleton[side + 'Knee'].point
            except KeyError:
                continue

            if knee.y < (other.y + self.vert_threshold) and knee.x < (other.x + self.hor_threshold):
                tags.append(side + 'Sitting')

        rospy.logdebug(tags)
        return tags

    @staticmethod
    def get_pointing_pose(skeleton, arm_norm_threshold=0.3, neck_norm_threshold=0.7):
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

            left_elbow = geometry_msg_point_to_kdl_vector(skeleton['LElbow'].point)
            left_shoulder = geometry_msg_point_to_kdl_vector(skeleton['LShoulder'].point)

            left_upper_arm_vector = (left_elbow - left_shoulder) / (left_elbow - left_shoulder).Norm()
            left_frame = get_frame_from_vector(left_upper_arm_vector, left_elbow)

            left_arm_neck_norm = (neck_vector * left_upper_arm_vector).Norm()

            if "LWrist" in skeleton:
                left_wrist = geometry_msg_point_to_kdl_vector(skeleton['LWrist'].point)
                left_lower_arm_vector = (left_wrist - left_elbow) / (left_wrist - left_elbow).Norm()

                left_arm_norm = (left_lower_arm_vector * left_upper_arm_vector).Norm()

                if left_arm_norm > arm_norm_threshold:
                    left_arm_valid = False
                else:
                    left_arm_vector = (left_wrist - left_shoulder) / (left_wrist - left_shoulder).Norm()
                    left_frame = get_frame_from_vector(left_arm_vector, left_wrist)

                rospy.logdebug("Left arm norm: %.2f", left_arm_norm)
        else:
            rospy.logdebug("Left arm not valid because it does not contain all required bodyparts")

        if right_arm_valid:

            right_elbow = geometry_msg_point_to_kdl_vector(skeleton['RElbow'].point)
            right_shoulder = geometry_msg_point_to_kdl_vector(skeleton['RShoulder'].point)

            right_upper_arm_vector = (right_elbow - right_shoulder) / (right_elbow - right_shoulder).Norm()
            right_frame = get_frame_from_vector(right_upper_arm_vector, right_elbow)

            right_arm_neck_norm = (neck_vector * right_upper_arm_vector).Norm()

            if "RWrist" in skeleton:
                right_wrist = geometry_msg_point_to_kdl_vector(skeleton['RWrist'].point)
                right_lower_arm_vector = (right_wrist - right_elbow) / (right_wrist - right_elbow).Norm()

                right_arm_norm = (right_lower_arm_vector * right_upper_arm_vector).Norm()

                if right_arm_norm > arm_norm_threshold:
                    right_arm_valid = False
                else:
                    right_arm_vector = (right_wrist - right_shoulder) / (right_wrist - right_shoulder).Norm()
                    right_frame = get_frame_from_vector(right_arm_vector, right_wrist)

                rospy.logdebug("Right arm norm: %.2f", right_arm_norm)
        else:
            rospy.logdebug("Left arm not valid because it does not contain all required bodyparts")

        rospy.logdebug("Arm norm threshold: %.2f", arm_norm_threshold)

        # Constraint based on parralelliness arm and neck
        if left_arm_valid and left_arm_neck_norm < neck_norm_threshold:
            rospy.logdebug("Rejecting left arm because of neck norm threshold ...")
            left_arm_valid = False
        if right_arm_valid and right_arm_neck_norm < neck_norm_threshold:
            rospy.logdebug("Rejecting right arm because of neck norm threshold ...")
            right_arm_valid = False

        # Optimize
        frame = None
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

        if left_arm_valid:
            frame = left_frame
        if right_arm_valid:
            frame = right_frame

        if not frame:
            rospy.logdebug("No valid arms found ...")
            return None

        return Pose(position=Point(*frame.p), orientation=Quaternion(*frame.M.GetQuaternion()))
