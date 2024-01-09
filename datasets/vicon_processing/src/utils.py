import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.signal import butter, lfilter, freqz, filtfilt

def extract_frame(events, ts_start, ts_end, img_shape):
        """Extract a single frame from the events"""

        id_start = np.searchsorted(events['ts'], ts_start)
        id_end = np.searchsorted(events['ts'], ts_end)

        img = np.zeros(img_shape[:2])

        # img[
        #     events['y'][id_start:id_end],
        #     events['x'][id_start:id_end],
        #     :
        # ] += 1 

        for i, (y, x) in enumerate(zip(events['y'][id_start:id_end], events['x'][id_start:id_end])):
              img[y, x] = events['ts'][id_start + i] - events['ts'][id_start]

        # img = 1 / np.exp(img)

        vmin = np.min(img)
        vmax = np.max(img)

        img = ((img / vmax) * 255)

        final_img = np.zeros(img_shape)
        final_img[:, :, 0] = img
        final_img[:, :, 1] = img
        final_img[:, :, 2] = img

        return final_img

def align_vectors_to_axis(vectors):
    # target = np.eye(3)

    # rot, _ = Rotation.align_vectors(vectors, target)

    return np.linalg.inv(vectors)


def _geometric_error_with_K(m, world_points, image_points, K):
        projected =  (K @ m.reshape(3, 4) @ world_points.transpose()).transpose()
        for r in projected:
            r /= r[-1]

        return (np.linalg.norm(image_points - projected, axis=1)).sum()

def _geometric_error_with_K_2(m, world_points, image_points, K, D):
        
        r_eul = Rotation.from_euler('XYZ', m[:3])
        t = m[3:]

        M = np.zeros((4, 4))
        M[:3, :3] = r_eul.as_matrix()
        M[:-1, -1] = t
        M[-1, -1] = 1.0

        projected, _ = cv2.projectPoints(world_points[:, :-1].transpose(), 
                          r_eul.as_rotvec(), t, K, D)
        
        projected = projected[:, 0, :]

        # projected = project_to_frame(
        #     transform_points(world_points, M),
        #     K
        # )

        return ((np.linalg.norm(image_points[:, :2] - projected, axis=1))).mean()

def project_to_frame(points, K):
    """
    Given a set of points, projects them on to the 2D image unsing the 
    intrinsic matrik self.K"""
    P_id = np.zeros((3, 4))
    P_id[:, :3] = np.eye(3)

    if points.shape[-1] < 4:
        # points are not in homogeneous coordinates
        points = np.hstack((points, np.ones((points.shape[0], 1))))

    image_points = (K @ P_id @ points.transpose()).transpose()

    for p in image_points:
        p /= p[-1]

    return image_points[:, :-1]


def transform_points(points, T):
    
    transformed_points = (T @ points.transpose()).transpose()

    for p in transformed_points:
        p /= p[-1]


    return transformed_points

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y