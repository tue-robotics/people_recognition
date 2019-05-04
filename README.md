# People Detector
Node for people detection in 2D. Defined as a service and not a publisher/subscriber.

## Custom dependencies
- [image_recognition_openpose](https://github.com/tue-robotics/image_recognition/image_recognition_openpose)
- [image_recognition_openface](https://github.com/tue-robotics/image_recognition/image_recognition_openpose)
- [image_recognition_keras](https://github.com/tue-robotics/image_recognition/image_recognition_keras)
- [image_recognition_util](https://github.com/tue-robotics/image_recognition/image_recognition_util)
- [image_recognition_msgs](https://github.com/tue-robotics/image_recognition/image_recognition_msgs)

# How To
### Start the dependent nodes
The namespaces of the dependent nodes must be the same as the values of the [parameters](#parameters) of people detection node 
```
rosrun image_recognition_openpose openpose_node __ns:=openpose

rosrun image_recognition_openface face_recognition_node __ns:=openface

rosrun image_recognition_keras face_properties_node __ns:=keras

rosrun image_recognition_util colour_extractor_node __ns:=colour_extractor
```

### Run the people detection node and service
```
rosrun image_recognition_people_detector people_detection_node
```
This will create a service `detect_people` of type `DetectPeople` and which requires a colour image as a message
(`sensor_msgs/Image`) as input in the service request and returns `people` in the response which is an array of 
custom message type `image_recognition_msgs/Person`

### Parameters
| Name                           | Default Value      |
|--------------------------------|--------------------|
| `~openpose_srv_prefix`         | `openpose`         |
| `~openface_srv_prefix`         | `openface`         |
| `~keras_srv_prefix`            | `keras`            |
| `~colour_extractor_srv_prefix` | `colour_extractor` |

### Message definition of Person
```
string name
uint8 age
uint8 gender
float64 gender_confidence
string posture
string emotion
string[] shirt_colors
image_recognition_msgs/Recognition[] body_parts
  image_recognition_msgs/CategoricalDistribution categorical_distribution
    image_recognition_msgs/CategoryProbability[] probabilities
      string label
      float32 probability
    float32 unknown_probability
  sensor_msgs/RegionOfInterest roi
    uint32 x_offset
    uint32 y_offset
    uint32 height
    uint32 width
    bool do_rectify
  uint32 group_id
```
### Test service
```
roscd image_recognition_people_detector/scripts
./get_people_detections_srv_test image ../test/assets/example.jpg
```

