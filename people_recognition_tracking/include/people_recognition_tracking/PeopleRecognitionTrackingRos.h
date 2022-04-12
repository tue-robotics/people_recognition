#ifndef __PEOPLE_RECOGNITION_TRACKING_ROS_H__
#define __PEOPLE_RECOGNITION_TRACKING_ROS_H__

#include <actionlib/server/simple_action_server.h>
#include <people_recognition_msgs/TrackOperatorAction.h>
#include <people_recognition_tracking/Tracking.h>
#include <ros/ros.h>

namespace people_recognition
{
class PeopleRecognitionTrackingRos
{
public:
  PeopleRecognitionTrackingRos();
  ~PeopleRecognitionTrackingRos();

private:
  ros::NodeHandle nh_, pnh_;
  action
};  // class PeopleRecognitionTrackingRos
}  // namespace people_recognition

#endif  // __PEOPLE_RECOGNITION_TRACKING_ROS_H__
