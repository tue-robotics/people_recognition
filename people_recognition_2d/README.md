# People Recognition 2D
Node for people recognition in 2D. Defined as a service and not a publisher/subscriber.

## Custom dependencies
- [image_recognition_openpose](https://github.com/tue-robotics/image_recognition/image_recognition_openpose)
- [image_recognition_openface](https://github.com/tue-robotics/image_recognition/image_recognition_openpose)
- [image_recognition_keras](https://github.com/tue-robotics/image_recognition/image_recognition_keras)
- [image_recognition_util](https://github.com/tue-robotics/image_recognition/image_recognition_util)
- [image_recognition_msgs](https://github.com/tue-robotics/image_recognition/image_recognition_msgs)

# How To
### Start the dependent nodes
The relative namespaces of the dependent nodes must be as follows so that their advertised services are the same as the values of the [parameters](#parameters) of people recognition node:
```
rosrun image_recognition_openpose openpose_node __ns:=openpose

rosrun image_recognition_openface face_recognition_node __ns:=face_recognition

rosrun image_recognition_keras face_properties_node __ns:=face_recognition

rosrun image_recognition_util color_extractor_node
```

### Run the people recognition node and service
```
rosrun people_recognition_2d people_recognition_2d_node
```
This will create a service `detect_people` of type `RecognizePeople2D` and which requires a color image
(`sensor_msgs/Image`) as input in the service request and returns `people` in the response which is an array of
custom message type `people_recognition_msgs/Person2D`

### Parameters
| Name                         | Default Value                         |
|------------------------------|---------------------------------------|
| `~openpose_srv_name`         | `openpose/recognize`                  |
| `~openface_srv_name`         | `face_recognition/recognize`          |
| `~keras_srv_name`            | `face_recognition/get_face_properties`|
| `~color_extractor_srv_name`  | `extract_color`                       |

### Message definition of Person2D
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
roscd people_recognition_2d/scripts
./people_recognition_2d_srv_test image ../test/assets/example.jpg
```

# Work Flow
The node first calls the recognize services of the openpose and openface nodes. ROIs of faces are extracted from
the recognitions returned by OpenPose and are associated with the recognitions returned by OpenFace through the
face ROIs to create a `Person2D` object. The ROIs of body parts returned by OpenPose are associated with each
`Person2D` object. Face images are sent to the Keras node and properties (age and gender) are extracted and
assoicated with each 'Person2D' object. The ROIs of the faces are shifted vertically to approximate the ROIs of
the shirts. These are sent to the color extractor to get the dominant colors.
