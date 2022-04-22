# Datasets

This folder contains scripts for reading, preprocessing and exporting HPE datasets to a common format that can be used 
for training and evaluating models for human pose prediction and joints velocity tracking.

Figure 1 shows a diagram of the export pipeline.

![Figure 1](readme_imgs/events_conversion_pipeline.png)

[//]: # (datasets to yarp)
[//]: # (yarp to eros)
[//]: # (yarp to json)

Datasets based on RGB frames are first converted to events (if possible) using the [v2e](https://github.com/SensorsINI/v2e) tool.
The resulting events are then exported to the IIT [YARP](https://www.yarp.it) framework format and event frames are generated
for training deep neural networks based on convolutional layers.

## Data Format

### YARP Format - Events
DVS events (timestamped address-event data) are exported to IIT's EDPR YARP format (see the 
[Bimvee's github page](https://github.com/event-driven-robotics/bimvee)). The output is a set of log and xml files containing encoded
events data (timestamp, x and y coordinates and polarity) and metadata for external tools.

### YARP Format - Skeletons
Ground truth skeletons are saved using a similar format to the events' one, where each line of the data.log file has the
following structure

`<line_id> <ts> SKLT (<list of joints coordinates>) <head_size> <torso_size>`

where
- `<line_id>` is an incremental integer
- `<ts>` is the timestamp of the labeled pose
- `<list of joints coordinates>` is a sequence of flattened joints coordinates
- `<head_size>` is the size of the person's head if provided by the dataset, -1 otherwise (the size is usually computed 
  as the diagonal of the head's bounding box)
- `<torso_size>` is the size of the person's torso if provided by the dataset, -1 otherwise (the size is usually computed 
  as the distance between the joints of a shoulder and the hip on the opposite side) 

### YARP Format - Frames?

### Event Frames
In order to be able to train "standard" convolutional neural networks, events are converted to image-like frames.
Two representations have been implemented, [EROS]() (the default one) and [TOS]().

Figure 2 shows EROS examples obtained from dataset DHP19.

![Figure 2](readme_imgs/eros_frames_example.png)

### Event Frames for Image-Based Datasets
?


## How to Convert Datasets

See each dataset's README file for steps on how to convert them to YARP format and detailed info on the available scripts.
- [DHP19](dhp19/README.md)
- [H3.6M](h36m/README.md)
- [MPII](mpii/README.md).


## Create Event Frames
(MPII exception?)