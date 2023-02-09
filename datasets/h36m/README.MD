# Human 3.6 million

[Human 3.6m million](http://vision.imar.ro/human3.6m/description.php) is an image-based dataset recorded at 50Hz. Event-based dataset can be created using [v2e](https://github.com/SensorsINI/v2e).

### Usage

- `Run_v2e_on_h36m.py` is used to create high temporal resolution events from frame inputs. Cropping can be used in which the video is converted to `640x480` centered on the person in the image.
- `h36_h5_to_yarp.py` is used to convert the output of `v2e` to the YARP format.
- `export_to_yarp.py` is used to convert the ground-truth and images to YARP format. If cropping is used in step 1, the same cropping can be used to modify the images and ground-truth joint positions.
- `export_xxx.py` is used to convert YARP format events to image-based formats used for training and testing detectors.

