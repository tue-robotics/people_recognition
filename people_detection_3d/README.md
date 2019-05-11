# people_detection_3d

Uses the image_recognition package to annotate 2D RGB images with people.
This package then does the extention to 3D and adds more annotations on the added information.

## Usage:
To provide some test data:
```bash
cd test/assets
rosbag play 2019-05-03-* --loop
```

```bash
rosrun people_detection_3d people_detection_3d_node _enable_topic_mode:=true rgb:=/hero/head_rgbd_sensor/rgb/image_raw depth:=/hero/head_rgbd_sensor/depth_registered/image camera_info:=/hero/head_rgbd_sensor/rgb/camera_info
```
