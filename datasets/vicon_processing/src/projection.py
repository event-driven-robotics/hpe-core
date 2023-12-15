import os
import os.path
import warnings

import numpy as np
import c3d
import scipy.stats
import cv2
import yaml
import matplotlib.pyplot as plt

from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.spatial.transform import Rotation

from bimvee.importIitYarp import importIitYarp

from . import utils

class ProjectionHelper():

    # def __init__(self, world_points: np.ndarray, image_points: np.ndarray):
    #     """
    #     If the points come from different frames (aka different points in time)
    #     they are assumed to be transformed in the reference frame defined by the camera markers.
    #     If they come from a single point in time they can also be in world coordinates
    #     """

    #     assert world_points.shape[0] == image_points.shape[0]

    #     self.world_points = world_points
    #     self.image_points = image_points

    #     for wp in world_points:
    #         wp /= wp[-1]

    #     self.with_calibration = False

    def __init__(self, vicon_points_dict: dict=None, dvs_points_dict: dict=None):
        """
        Takes the world points and image points as dict with timestampts and creates
        the numpy array """
        if vicon_points_dict is None and dvs_points_dict is None:
            return
        
        n_points = len(vicon_points_dict['points'][0])
        world_points = np.ones((n_points * len(vicon_points_dict['points']), 4))
        image_points = np.ones((n_points * len(vicon_points_dict['points']), 3))
        labels = list(vicon_points_dict['points'][0].keys())
        
        for k, vicon_frame in enumerate(vicon_points_dict['points']):
            for i, l in enumerate(labels):

                vicon_time = vicon_points_dict['times'][k]
                dvs_id = (np.abs(dvs_points_dict['times'] - vicon_time)).argmin()

                try:
                    # vicon_points = vicon_points_dict['points'][vicon_frame][l][:3]
                    world_points[k*n_points + i, :3] = vicon_frame[l][:3]
                    p_dict = dvs_points_dict['points'][dvs_id][l]
                    image_points[k*n_points + i, :2] = [p_dict['x'], p_dict['y']]
                except Exception as e:
                    print("The stored image labels probably don't match with the vicon labels used.")
                    print(e)

        self.world_points = world_points
        self.image_points = image_points

        print(f"Number of 3d points: {len(self.world_points)}")
        print(f"Number of image points: {len(self.image_points)}")

        assert self.world_points.shape[0] == self.image_points.shape[0], "Not equal numbers of image and 3D points"

    def undistort_image_points(self):
        """
        Apply undistortion to image points. The function returns the undistorted points 
        and changes the saved image points to the undistoreted ones."""
        try:
            print(f"Undistort points using coeffs: {self.D}")
        except:
            print("Camera calibration not imported, trying to import ../data/temp_calib.txt")
            self.import_camera_calbration('..data/temp_calib.txt')

        undistorted = cv2.undistortPoints(self.image_points[:, :2].astype(np.float64), self.K, self.D, P=self.K)
        undistorted = undistorted[:, 0, :]
        undistorted = np.hstack((undistorted, np.ones((undistorted.shape[0], 1))))  

        # self.image_points = undistorted
        return undistorted

    def import_camera_calbration(self, calib_file):
        """Import the camera calibration from file"""

        calib = np.genfromtxt(calib_file, delimiter=" ", skip_header=1, dtype=object)
        calib_dict = {}
        keys = calib[:, 0].astype(str)
        vals = calib[:, 1].astype(float)
        calib_dict = {
            key: value for key, value in zip(keys, vals)
        }
        self.img_shape = (calib_dict['h'], calib_dict['w'])
        # intrinsic
        self.K = np.array([
            [calib_dict['fx'], 0.0, calib_dict['cx']],
            [0.0, calib_dict['fy'], calib_dict['cy']],
            [0.0, 0.0, 1.0]
        ])
        # camera distortion
        self.D = np.array([calib_dict['k1'], calib_dict['k2'], calib_dict['p1'], calib_dict['p2']])

        self.with_calibration = True

        return self.img_shape, self.K, self.D
    
    def initial_estimate(self):
        s = self.world_points.shape[0]
        A = np.zeros((2*s, 12))
        for i in range(s):
            A[2*i, :4] = self.world_points[i]
            A[2*i + 1, 4:8] = self.world_points[i]

            A[2*i, 8:] = self.world_points[i] * (-self.image_points[i][0])
            A[2*i+1, 8:] = self.world_points[i] * (-self.image_points[i][1])

        # compute At x A
        A_ = np.matmul(A.T, A)
        # compute its eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(A_)
        # find the eigenvector with the minimum eigenvalue
        # (numpy already returns sorted eigenvectors wrt their eigenvalues)
        m = eigenvectors[:, 11]
        return m
    
    def find_projection(self):
        """This does not use the calibration information. It finds the projection
        matrix that includes calibration and transformation"""
        return None
    
    def find_transform(self):
        """
        It find the transformation using the calibration provided"""

        assert self.with_calibration is True

        m = self.initial_estimate()
        m = m.reshape(3, 4)
        m = np.linalg.inv(self.K) @ m
        m = m.flatten()
        # m = np.ones((12, 1))
        result = minimize(utils._geometric_error_with_K, m, args=(self.world_points, 
                                                                 self.image_points, 
                                                                 self.K))
        
        if result.success is False:
            warnings.warn(result.message)

        print(result)
        
        T = result.x.reshape(3, 4)

        return T
    
    def find_R_t_opencv_refine(self):
        T = self.find_R_t_opencv()

        # measure error
        r = np.array(Rotation.from_matrix(T[:3, :3]).as_rotvec())
        t = np.array(T[:3, -1])
        m = np.concatenate((r, t))
        print(r)
        print(t)

        proj_points = np.copy(self.world_points[:, :-1])
        img_points = np.copy(self.image_points[:, :-1])
        r, t = cv2.solvePnPRefineLM(proj_points, img_points, self.K, self.D, r, t)

        T = np.zeros((4, 4))
        T[:3, :3] = Rotation.from_rotvec(r.reshape(3, )).as_matrix()
        T[:3, -1] = t.reshape(3, )
        T[-1, -1] = 1.0

        # measure error
        m = np.concatenate((r, t))
        # m = np.squeeze(m, -1)
        err = utils._geometric_error_with_K_2(m, self.world_points, self.image_points, self.K, self.D)
        print(f"Error at the end: {err}")

        return T
        
    
    def find_R_t_opencv(self):
        T, s = self._find_R_t_opencv()
        return T
    
    def _find_R_t_opencv(self):
        proj_points = np.copy(self.world_points[:, :-1])
        img_points = np.copy(self.image_points[:, :-1])

        s, r, t, e = cv2.solvePnPGeneric(proj_points, 
                    img_points, 
                    self.K, self.D, flags=cv2.SOLVEPNP_SQPNP)
        
        r = r[0]
        t = t[0]
        e = e[0][0]
        
        
        T = np.zeros((4, 4))
        T[:3, :3] = Rotation.from_rotvec(r.reshape(3, )).as_matrix()
        T[:3, -1] = t.reshape(3, )
        T[-1, -1] = 1.0

        return T, e
    
    def find_R_t_opencv_ransac(self):
        T, s = self._find_R_t_opencv_ransac()
        return T
    
    def _find_R_t_opencv_ransac(self):

        proj_points = np.copy(self.world_points[:, :-1])
        img_points = np.copy(self.image_points[:, :-1])

        s, r, t, inl = cv2.solvePnPRansac(proj_points, 
                    img_points, 
                    self.K, self.D,
                    reprojectionError=7)
        
        T = np.zeros((4, 4))
        T[:3, :3] = Rotation.from_rotvec(r.reshape(3, )).as_matrix()
        T[:3, -1] = t.reshape(3, )
        T[-1, -1] = 1.0

        # measure error
        if s is False:
            err = np.inf
        else:
            m = np.concatenate((r, t))
            m = np.squeeze(m, -1)
            err = utils._geometric_error_with_K_2(m, self.world_points[inl[:, 0]], self.image_points[inl[:, 0]], self.K, self.D)
        print(f"Error at the end: {err}")
        

        return T, err

    def find_R_t(self, constrain_translation=False):
        T, result = self._find_R_t(constrain_translation=constrain_translation)
        return T
    
    def _find_R_t(self, constrain_translation=False):
        """The functiom finds the transformation (rotation and translation) that best 
        project the world_points to image_points using the intrinsic parameters K.
        im = K @ (R | t) @ P_w"""

        assert self.with_calibration is True

        # in the initial configuration we assume that the frame is almost correct already
        # we start with zero rotation and traslation
        start_m = np.zeros((6, ))
        # the parameters are encoded in a 6d vector. The first 3 represent the rotation and
        # the last 3 the translation
        # the rotation in encoded using euler angles in xyz order
        
        # this is not used
        def _front_camera_constraint(m):
            """The function is used to impose a constraint on projected points
            It should find a solution that keeps the points in front of the camera. 
            TODO it does not seem to work all the times.
            See Cheirality, p.515 in Multiple View Geometry in Computer Vision, Second Edition 
            """
            nonlocal self 

            world_points = self.world_points
            K = self.K

            r_eul = Rotation.from_euler('XYZ', m[:3])
            t = m[3:]

            M = np.zeros((3, 4))
            M[:, :3] = r_eul.as_matrix()
            M[:, -1] = t

            P = K @ M

            XC = np.copy(world_points[:, :-1])
            for row in XC:
                row -= t

            m3 = P[2, :-1]
            w = m3.reshape(1, 3) @ XC.transpose()

            depth_sign = np.multiply(w, world_points[:, -1]) * np.linalg.det(P[:, :3])

            return depth_sign.flatten()

        # the linear constraint only affects the translation (A is all zeros in the upper half)
        # it should limit the movement of the camera to only 25 (mm ?) = 2.5 cm combined = 5 cm per axis
        A = np.zeros((6, 6))
        A[3:, 3:] = np.eye(3)
        translation_constraint = LinearConstraint(A, lb=-30, ub=30, keep_feasible=False)

        # again, not used
        # positive_depth = NonlinearConstraint(_front_camera_constraint, lb=0.0, ub=np.inf, keep_feasible=False)
        if not constrain_translation:
            result = minimize(utils._geometric_error_with_K_2, start_m, 
                                                                args=(self.world_points, 
                                                                self.image_points, 
                                                                self.K, self.D))
        else:
            result = minimize(utils._geometric_error_with_K_2, start_m, 
                                                                    constraints = [
                                                                        translation_constraint, 
                                                                        # positive_depth
                                                                        ],
                                                                    args=(self.world_points, 
                                                                    self.image_points, 
                                                                    self.K, self.D))
        
        if result.success is False:
            warnings.warn(result.message)

        print(result)

        m = result.x
        r_eul = Rotation.from_euler('XYZ', m[:3])
        t = m[3:]

        T = np.zeros((4, 4))
        T[:3, :3] = r_eul.as_matrix()
        T[:-1, -1] = t
        T[-1, -1] = 1.0

        return T, result
    
    def T_to_transform(self, T):
        Tr = np.zeros((4, 4))

        Tr[:3, :3] = T[:, :3]
        Tr[:3, -1] = T[:, -1]
        Tr[-1, -1] = 1.0

        return Tr

    def transform_points(self, T):

        transformed_points = (T @ self.world_points.transpose()).transpose()

        for p in transformed_points:
            p /= p[-1]

        return transformed_points
    
    def transform_points(self, points, T):
        
        return utils.transform_points(points, T)
    
    def project_to_frame(self, points):
        """
        Given a set of points, projects them on to the 2D image unsing the 
        intrinsic matrik self.K"""
        ps = np.copy(points)

        ps = ps[ps[:, 2] > 0]
        P_id = np.zeros((3, 4))
        P_id[:, :3] = np.eye(3)
        image_points = (self.K @ P_id @ ps.transpose()).transpose()

        for p in image_points:
            p /= p[-1]

        return image_points

    def project_to_frame_opencv(self, points):
        rep_points, _ = cv2.projectPoints(points[:, :-1], np.zeros((3,)), np.zeros((3,)), self.K, self.D)
        return np.hstack((rep_points[:, 0, :], np.ones((rep_points.shape[0], 1))))