# Dataset Conversion

This folder contains scripts for reading, preprocessing and exporting HPE datasets to common formats to enable valid comparison of human pose estimation algorithms. As each dataset has a unique format, tailored scripts are required per-dataset. As a final step we use [BIMVEE](https://github.com/event-driven-robotics/bimvee) to export events to any common format. 

We currently support the folowing datasets
* [DHP19](dhp19/README.md)
* [H3.6M](h36m/README.md)
* [MPII](mpii/README.md)

Human Pose across all hpe-core modules is defined with 13 joints, with the following map for joints (unless explicitly mentioned):
```Python
KEYPOINTS_MAP = {'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'elbow_right': 3, 'elbow_left': 4,
                     'hip_left': 5, 'hip_right': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9, 'knee_left': 10,
                     'ankle_right': 11, 'ankle_left': 12}
```
