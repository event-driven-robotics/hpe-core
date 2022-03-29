
import numpy

class Movenet:

    KEYPOINTS_COCO = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                     'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                     'right_ankle']

    KEYPOINTS_NUM = 13


class HPECoreSkeleton:

    KEYPOINTS_MAP = {'head': 0, 'left_shoulder': 1, 'right_shoulder': 2, 'left_elbow': 3, 'right_elbow': 4,
                     'left_wrist': 5, 'right_wrist': 6, 'left_hip': 7, 'right_hip': 8, 'left_knee': 9, 'right_knee': 10,
                     'left_ankle': 11, 'right_ankle': 12}

    @staticmethod
    def compute_torso_sizes(skeletons: numpy.array) -> float:

        if len(skeletons.shape) == 2:
            l_hip = skeletons[HPECoreSkeleton.KEYPOINTS_MAP['left_hip'], :]
            r_shoulder = skeletons[HPECoreSkeleton.KEYPOINTS_MAP['right_shoulder'], :]
            torso_sizes = numpy.linalg.norm(l_hip - r_shoulder)

        elif len(skeletons.shape) == 3:
            l_hips = skeletons[:, HPECoreSkeleton.KEYPOINTS_MAP['left_hip'], :]
            r_shoulders = skeletons[:, HPECoreSkeleton.KEYPOINTS_MAP['right_shoulder'], :]
            torso_sizes = numpy.linalg.norm(l_hips - r_shoulders, axis=1)

        return torso_sizes
