
import numpy as np

from datasets.utils.constants import HPECoreSkeleton


# DVS camera
DHP19_SENSOR_HEIGHT = 260
DHP19_SENSOR_WIDTH = 346
DHP19_CAM_FRAME_EVENTS_NUM = 7500  # fixed number of events used in DHP19 for generating event frames
DHP19_CAM_NUM = 4  # number of synchronized cameras used for recording events

DHP19_NUM_OF_SESSIONS = 5
DHP19_NUM_OF_SUBJECTS = 17

# map from body parts to indices for dhp19
DHP19_BODY_PARTS_INDICES = {
    'head': 0,
    'shoulderR': 1,
    'shoulderL': 2,
    'elbowR': 3,
    'elbowL': 4,
    'hipL': 5,
    'hipR': 6,
    'handR': 7,
    'handL': 8,
    'kneeR': 9,
    'kneeL': 10,
    'footR': 11,
    'footL': 12
}

DHP19_TO_HPECORE_SKELETON_IND = [
    DHP19_BODY_PARTS_INDICES['head'],
    DHP19_BODY_PARTS_INDICES['shoulderL'],
    DHP19_BODY_PARTS_INDICES['shoulderR'],
    DHP19_BODY_PARTS_INDICES['elbowL'],
    DHP19_BODY_PARTS_INDICES['elbowR'],
    DHP19_BODY_PARTS_INDICES['handL'],
    DHP19_BODY_PARTS_INDICES['handR'],
    DHP19_BODY_PARTS_INDICES['hipL'],
    DHP19_BODY_PARTS_INDICES['hipR'],
    DHP19_BODY_PARTS_INDICES['kneeL'],
    DHP19_BODY_PARTS_INDICES['kneeR'],
    DHP19_BODY_PARTS_INDICES['footL'],
    DHP19_BODY_PARTS_INDICES['footR']
]
