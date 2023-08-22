#include <ros/ros.h>

#include <iostream>
#include <string.h>

#include <rgbd/ros/conversions.h>
#include <rgbd/image.h>
#include <rgbd/image_buffer/image_buffer.h>
#include <rgbd/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <people_recognition_msgs/RecognizePeople2D.h>

sensor_msgs::Image rgbd2ros(rgbd::ImageConstPtr image)
{
    rgbd::View view(*image, image->getRGBImage().cols);

    // Convert to image messages
    sensor_msgs::Image msg;
    sensor_msgs::CameraInfo info_msg;

    rgbd::convert(image->getRGBImage(), view.getRasterizer(), msg, info_msg);

    msg.header.stamp.fromSec(image->getTimestamp());
    msg.header.frame_id = image->getFrameId();
    return msg;
}


// TODO docstring and usage
int main(int argc, char** argv)
{
    ros::init(argc, argv, "people_tracker");
    ros::NodeHandle nh("~");

    std::string rgbd_topic;
    if (!nh.getParam("rgbd_topic", rgbd_topic))
    {
        ROS_FATAL("[People Tracker] could not read rgbd_topic from parameter server");
        return 1;
    }
    
    std::string frame_id = "base_link";
    if (!nh.getParam("frame_id", frame_id))
    {
        ROS_FATAL("[People Tracker] could not read frame_id from parameter server");
        return 1;
    }

    // Optional
    double rate = 100;
    if (!nh.getParam("rate", rate))
    {
        ROS_DEBUG_STREAM("[People Tracker] could not read rate from parameter server, defaulting to " << rate);
    }

    // get depth sensor integrator parameters
    std::string people_recognition_service;
    if (!nh.getParam("people_recognition_service", people_recognition_service))
    {
        ROS_DEBUG_STREAM("[People Tracker] could not read people_recognition2d from parameter server");
    }

    rgbd::ImageBuffer image_buffer;
    image_buffer.initialize(rgbd_topic, frame_id);
    ROS_INFO_STREAM("set up image buffer on topic: " << rgbd_topic.c_str());

    ros::ServiceClient srv_people = nh.serviceClient<people_recognition_msgs::RecognizePeople2D>(people_recognition_service);
    ROS_INFO_STREAM("set up client to " << people_recognition_service.c_str());

    ros::Rate r(rate);
    while (ros::ok())
    {
        rgbd::ImageConstPtr image;
        geo::Pose3D sensor_pose;

        // get new image
        if (!image_buffer.waitForRecentImage(image, sensor_pose, r.expectedCycleTime().toSec()))
        {
            r.sleep(); // So we do sleep after getting an image again after failing to get an image
            continue;
        }

        std::vector <geo::Vector3> measurements;

        // Get people detections
        people_recognition_msgs::RecognizePeople2D srv;
        srv.request.image = rgbd2ros(image);
        if(srv_people.call(srv))
        {
            ROS_INFO_STREAM("got " << srv.response.people.size() << " detections");
        }
        else
        {
            ROS_ERROR_STREAM("Failed to call service " << people_recognition_service.c_str() );
        }

        // perform magic tracking

        // fill output message
        if (r.cycleTime() > r.expectedCycleTime())
        {
            ROS_WARN_STREAM("Could not make the desired cycle time of " << std::to_string(r.expectedCycleTime().toSec()) << ", instead took " << std::to_string(r.cycleTime().toSec()) );
        }
        r.sleep();
    }
}