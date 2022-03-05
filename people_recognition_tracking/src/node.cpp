#include <ros/ros.h>
#include <people_recognition_tracking/PeopleRecognitionTrackingRos.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "people_recognition_tracking_node");

  people_recognition::PeopleRecognitionTrackingRos prt;
  ros::spin();

  return 0;
}
