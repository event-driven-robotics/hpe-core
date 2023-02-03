
import numpy as np


class PCK:
    """
    Class for computing HPE metric PCK (Percentage of Correct Keypoints)
    """

    def __init__(self, threshold: float):
        """
        Class initializer.
        Parameters:
            threshold (float): threshold value used for classifying joints as correct or not correct; must be > 0.0 and <= 1.0
        """

        assert .0 < threshold <= 1.0, 'PCK threshold value must be > 0.0 and <= 1.0'

        self.threshold = threshold
        self.predicted_joints = None
        self.gt_joints = None
        self.reference_sizes = None

    def update_samples(self, predicted_joints: np.array, gt_joints: np.array, reference_sizes: np.array) -> None:
        """
        Accumulates data that will be used for computing PCK (with function get_value()).
        It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
        not annotated joints as [-1, -1] or [-1, -1, -1]
        Parameters:
            predicted_joints (numpy.array): predicted joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            gt_joints (numpy.array): ground truth joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            reference_sizes (numpy.array): sizes of a body part (either head or torso, used for classifying joints as correct or not correct) as a numpy array with shape [batch_size, joints_num]
        """
        if self.predicted_joints is None:
            self.predicted_joints = predicted_joints
            self.gt_joints = gt_joints
            self.reference_sizes = reference_sizes

        else:
            self.predicted_joints = np.append(self.predicted_joints, predicted_joints, axis=0)
            self.gt_joints = np.append(self.gt_joints, gt_joints, axis=0)
            self.reference_sizes = np.append(self.reference_sizes, reference_sizes, axis=0)

    def get_value(self):
        """
        Computes metric using the accumulated data.
        Return:
            numpy.array: PCK values for each single joint
            float: average of PCK values computed over all joints
        """

        if self.reference_sizes is None:
            return np.zeros(1, dtype=float), .0

        # compute PCK's threshold as percentage of head size in pixels for each pose
        thresholds_head = self.reference_sizes * self.threshold
        # thresholds_head = thresholds_head.reshape([-1, 1]).tile((1, self.gt_joints.shape[1]))
        thresholds_head = np.tile(thresholds_head.reshape([-1, 1]), (1, self.gt_joints.shape[1]))

        # compute euclidean distances between joints
        distances = np.linalg.norm(self.predicted_joints - self.gt_joints, axis=2)

        # compute correct keypoints
        correct_keypoints = (distances <= np.array(thresholds_head)).astype(int)

        # remove not annotated keypoints from pck computation
        correct_keypoints = correct_keypoints * (self.gt_joints[:, :, 0] != -1).astype(int)
        annotated_keypoints_num = np.sum((self.gt_joints[:, :, 0] != -1).astype(int), axis=0)

        # compute pck
        pck_joints = np.sum(correct_keypoints, axis=0) / annotated_keypoints_num
        pck_avg = np.mean(pck_joints)

        return pck_joints, pck_avg


class RMSE:

    def __init__(self):
        self.predicted_joints = None
        self.gt_joints = None

    def update_samples(self, predicted_joints: np.array, gt_joints: np.array) -> None:
        """
        Accumulates data that will be used for computing RSME (with function get_value()).
        It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
        not annotated joints as [-1, -1] or [-1, -1, -1]
        Parameters:
            predicted_joints (numpy.array): predicted joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            gt_joints (numpy.array): ground truth joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
        """
        if self.predicted_joints is None:
            self.predicted_joints = predicted_joints
            self.gt_joints = gt_joints

        else:
            self.predicted_joints = np.append(self.predicted_joints, predicted_joints, axis=0)
            self.gt_joints = np.append(self.gt_joints, gt_joints, axis=0)

    def get_value(self):
        """
        Computes metric using the accumulated data.
        Return:
            numpy.array: normalized RMSE values computed on x and y coordinates of each single joint
            numpy.array: average of normalized RMSE values computed on x and y coordinates of all joints
            numpy.array: maximum RMSE values between the x and y coordinates of all joints
        """

        rmse = np.sqrt(np.sum(np.square(self.predicted_joints - self.gt_joints), axis=0) / len(self.gt_joints))
        # normalized_rmse = (rmse / abs(np.mean(self.predicted_joints, axis=0))) * 100
        normalized_rmse_avg = np.mean(normalized_rmse, axis=0)
        max_normalized_rmse = np.max(normalized_rmse, axis=1)

        return normalized_rmse.flatten(), normalized_rmse_avg.flatten(), max_normalized_rmse


def compute_mpjpe(predicted_joints, gt_joints):
    """
    Computes MPJPE (Mean Per Joint Position Error)
    """

    # # compute euclidean distances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)
    
    # TODO: what if a gt joint is not present in a frame? get the number of gt joints by
    # gt_num = np.sum(gt_joints == 0)
    mpje = np.sum(distances, axis=0) / len(distances)
    
    
    # compute errors wihtout considering non detected joints    
    dist = np.zeros((len(gt_joints),13))
    count = np.zeros((13))
    for i in range(13):
        for j in range(len(gt_joints)):
            if(predicted_joints[j,i,0]!=0 and predicted_joints[j,i,1]!=0):
                dist[j,i] = np.linalg.norm(predicted_joints[j,i,:] - gt_joints[j,i,:], axis=0)
                count[i] = count[i] + 1
    res = np.sum(dist, axis=0) / count

    # print('distances = ')
    # print(distances[1:4,:])
    # print('dist = ')
    # print(dist[1:4,:])

    # return mpje
    return res

def print_mpjpe(mpjpe, keypoints_str):
    for ei, error in enumerate(mpjpe):
        print(f'{keypoints_str[ei]}: {error}')
