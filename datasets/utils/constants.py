
import numpy


class Movenet:

    KEYPOINTS_COCO = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                     'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                     'right_ankle']

    KEYPOINTS_NUM = 13


class HPECoreSkeleton:

    # the order of the joints is the same as the DHP19 one
    # differences:
    # - more descriptive labels
    # - we use the more common 'wrist' and 'ankle' labels instead of DHP19's 'hand' and 'foot' ones
    # - all joints are listed as first 'right' and then 'left' except for the 'hip' ones: it is not an error,
    #   it is DHP19's order
    # KEYPOINTS_MAP = {'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'elbow_right': 3, 'elbow_left': 4,
    #                  'hip_left': 5, 'hip_right': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9, 'knee_left': 10,
    #                  'ankle_right': 11, 'ankle_left': 12}
    KEYPOINTS_MAP = {'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'hip_left': 3, 'hip_right': 4,
                     'elbow_right': 5, 'elbow_left': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9, 'knee_left': 10,
                     'ankle_right': 11, 'ankle_left': 12}
    @staticmethod
    def compute_torso_sizes(skeletons: numpy.array) -> float:

        if len(skeletons.shape) == 2:
            l_hip = skeletons[HPECoreSkeleton.KEYPOINTS_MAP['hip_left'], :]
            r_shoulder = skeletons[HPECoreSkeleton.KEYPOINTS_MAP['shoulder_right'], :]
            torso_sizes = numpy.linalg.norm(l_hip - r_shoulder)

        elif len(skeletons.shape) == 3:
            l_hips = skeletons[:, HPECoreSkeleton.KEYPOINTS_MAP['hip_left'], :]
            r_shoulders = skeletons[:, HPECoreSkeleton.KEYPOINTS_MAP['shoulder_right'], :]
            torso_sizes = numpy.linalg.norm(l_hips - r_shoulders, axis=1)

        return torso_sizes
