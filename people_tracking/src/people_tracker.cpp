#include <ros/ros.h>

#include <iostream>
#include <string.h>

#include <rgbd/ros/conversions.h>
#include <rgbd/image.h>
#include <rgbd/image_buffer/image_buffer.h>
#include <rgbd/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <visualization_msgs/MarkerArray.h>

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

visualization_msgs::MarkerArray fillMessage(people_recognition_msgs::RecognizePeople2D& srv)
{
    visualization_msgs::MarkerArray array_msg;

    double r = 0.5; // visualization distance
    double fx = 1.0/500;
    int image_width = 640;
    int image_height = 480;

    int marker_id = 4321;
    for (int i=0; i<srv.response.people.size(); i++)
    {
        for (int j=0; j<srv.response.people[i].body_parts.size(); j++)
        {
            sensor_msgs::RegionOfInterest output_roi = srv.response.people[i].body_parts[0].roi;
            double x = r * fx * (output_roi.x_offset + 0.5*output_roi.width - image_width);
            double y = r * fx * (output_roi.y_offset + 0.5*output_roi.height - image_height);

            // fill message
            visualization_msgs::Marker marker_msg;
            marker_msg.header.frame_id = "head_rgbd_sensor_link"; //TODO hardcoded frame
            marker_msg.header.stamp = ros::Time::now();
            marker_msg.id = marker_id;
            marker_msg.type = visualization_msgs::Marker::SPHERE;
            marker_msg.action = 0;
            marker_msg.pose.position.x = x;
            marker_msg.pose.position.y = y;
            marker_msg.pose.position.z = r;
            marker_msg.pose.orientation.x = 0.0;
            marker_msg.pose.orientation.y = 0.0;
            marker_msg.pose.orientation.z = 0.0;
            marker_msg.pose.orientation.w = 1.0;
            marker_msg.scale.x = 0.05;
            marker_msg.scale.y = 0.05;
            marker_msg.scale.z = 0.05;
            marker_msg.color.r = 1.0;
            marker_msg.color.g = 0.0;
            marker_msg.color.b = 1.0;
            marker_msg.color.a = 1.0;
            array_msg.markers.push_back(marker_msg);
            marker_id++;
        }
    }
    return array_msg;
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

    // set up camera image feed
    rgbd::ImageBuffer image_buffer;
    const std::string namespaced_rgbd_topic = ros::names::resolve(rgbd_topic);
    image_buffer.initialize(namespaced_rgbd_topic, frame_id);
    ROS_INFO_STREAM("set up image buffer on topic: " << namespaced_rgbd_topic);

    // set up image processing services
    ros::ServiceClient srv_people = nh.serviceClient<people_recognition_msgs::RecognizePeople2D>(people_recognition_service);
    ROS_INFO_STREAM("set up client to " << people_recognition_service.c_str());

    // set up output stream
    ros::Publisher pub_marker = nh.advertise<visualization_msgs::MarkerArray>("people_tracking_markers", 1);

    ros::Rate r(rate);
    while (ros::ok())
    {
        rgbd::ImageConstPtr image;
        geo::Pose3D sensor_pose;

        // get new image
        if (!image_buffer.waitForRecentImage(image, sensor_pose, r.expectedCycleTime().toSec()))
        {
            r.sleep(); // So we do sleep after getting an image again after failing to get an image
            ROS_WARN_STREAM("Could not get image");
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
        visualization_msgs::MarkerArray marker_msg = fillMessage(srv);
        pub_marker.publish(marker_msg);

        if (r.cycleTime() > r.expectedCycleTime())
        {
            ROS_WARN_STREAM("Could not make the desired cycle time of " << std::to_string(r.expectedCycleTime().toSec()) << ", instead took " << std::to_string(r.cycleTime().toSec()) );
        }
        r.sleep();
    }
}