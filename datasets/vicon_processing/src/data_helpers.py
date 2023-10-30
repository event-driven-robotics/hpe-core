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

class C3dHelper:

    def __init__(self, file_path, wand_zero_time=False, delay=0.0):
        
        self.reader = c3d.Reader(open(file_path, 'rb'))
        self.wand_zero = wand_zero_time
        self.delay = delay
        self.calculate_frame_times()

        # the obtained transforms are saved in a dict for later use if needed
        # the keys are the frame ids
        self.markers_T = {}

    def find_start_time(self):
        """
        To synchronize the camera recording and the vicon data
        we take as start time the moment the lights on the calibration 
        stick turn on. 
        For the vicon recordings we can look a the number of points in
        each frame. When there is a jump in the number of points (5)
        it should be the moment the lights are turned on.
        """

        if not self.wand_zero:
            self.start_time = 0.0
            return self.start_time

        n_points= []
        for i, points, analog in self.reader.read_frames():
            valid_points = points[points[:, 0] > 0.0]
            n_points.append(valid_points.shape[0])

        mode, count = scipy.stats.mode(n_points)

        # the calibration stick has 5 lights on it
        # when it is turned on we should see 5 more points
        candidates_id = np.argwhere(n_points >= mode[0] + 5)
        if len(candidates_id) == 0:
            start_frame_id = 0
        else:
            start_frame_id = candidates_id[0]

        rate = self.reader.point_rate
        time_step = 1.0 / rate

        self.start_time = start_frame_id * time_step
        return self.start_time
    
    def calculate_frame_times(self):

        self.find_start_time()
        self.start_time += self.delay

        rate = self.reader.point_rate # vicon rate
        time_step = 1.0 / rate # t between frames

        n_points= []
        for i, points, analog in self.reader.read_frames():
            valid_points = points[points[:, 0] > 0.0]
            n_points.append(valid_points.shape[0])

        times = np.linspace(0, self.reader.frame_count * time_step,
                    self.reader.frame_count)
        # the zero time is when the calibration stick lights turn on
        times -= self.start_time
        self.frame_times = times

        return self.frame_times
    
    def get_frame_time(self, times):
        frame_ids = []
        for t in times:
            frame_ids.append(np.searchsorted(self.frame_times, t))

        return frame_ids
    
    def get_points_frame(self, frame_id):
        frame_points_all = None
        for i, points, analog in self.reader.read_frames():
            if i == frame_id:
                frame_points_all = points
                break
        if frame_points_all is not None:
            return frame_points_all
        print(f"The recording does not have the frame id {frame_id}")
        return None
    
    def get_points_dict(self, frame_id):
        points = self.get_points_frame(frame_id)
        dict_out = {}
        for i, l in enumerate(self.reader.point_labels):
            dict_out[l.strip()] = points[i][:3]

        return dict_out
    
    def filter_dict_labels(self, old_dict, labels):
        new_dict = {
            key: old_dict[key] for key in labels
        }
        return new_dict

    def marker_T_at_frame_vector(self, frame_id):
        """
        This method returns the tranformation T that described the frame of reference defined by
        the 3 marker placed on the camera.
        Returns the 4x4 transformation matrix T that transform points from world frame to the marker frame
        The function actually first computes the inverse transformation from marker frame to world and then inverts the matrix"""
        
        # different sequences sometime use different labels for the camera markers
        # this should not be necessary in the final version. The vicon processing should keep the labels consistent.
        try:
            labels = [
                'event_camera:side',
                'event_camera:front',
                'event_camera:top'
                ]
            camera_points = self.filter_dict_labels(self.get_points_dict(frame_id), labels)
            camera_front = camera_points['event_camera:front'][:3]
            camera_side = camera_points['event_camera:side'][:3]
            camera_top = camera_points['event_camera:top'][:3]
        except:
            pass

        try:
            labels = [
                'camera:side',
                'camera:front',
                'camera:top'
                ]
            camera_points = self.filter_dict_labels(self.get_points_dict(frame_id), labels)
            camera_front = camera_points['camera:front'][:3]
            camera_side = camera_points['camera:side'][:3]
            camera_top = camera_points['camera:top'][:3]
        except Exception as e:
            # print(e)
            pass
        
        # TODO remove, just for test
        try:
            labels = [
                'camera:cam_back',
                'camera:cam_right',
                'camera:cam_left'
                ]
            camera_points = self.filter_dict_labels(self.get_points_dict(frame_id), labels)
            camera_front = camera_points['camera:cam_right'][:3]
            camera_side = camera_points['camera:cam_left'][:3]
            camera_top = camera_points['camera:cam_back'][:3]
        except Exception as e:
            print(e)
            
        
        # mid point between top and side marker
        # even if not precise it is considered the origin of the new frame of reference
        side_top_mid = (camera_side + camera_top) / 2

        z = camera_front - side_top_mid
        z = z / np.linalg.norm(z)

        t = camera_top - side_top_mid
        t = t / np.linalg.norm(t)

        x = np.cross(t, z)
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        # x, y, z are all normalized and orthogonal
        # the 3 vectors stacked define a rotation matrix
        rot_mat = np.vstack((x, y, z))

        T = np.zeros((4, 4))
        T[:3, :3] = rot_mat.transpose()
        T[:-1, -1] = np.array(side_top_mid)
        T[-1, -1] = 1

        T = np.linalg.inv(T)

        self.markers_T[frame_id] = np.copy(T)

        return np.copy(T)
    
    def get_vicon_points(self, frames_id, labels):
        vicon_points_frames = [self.get_points_dict(idx) for idx in frames_id]
        vicon_points_frames = [self.filter_dict_labels(old_dict, labels) 
                            for old_dict in vicon_points_frames]
        
        out = {
            'points': vicon_points_frames,
            'times': np.array([self.frame_times[idx] for idx in frames_id]),
            'frame_ids': (frames_id)
        }
        
        return out
    
    def transform_points_to_marker_frame(self, vicon_points):
        """The vicon_points is a dict containing the points and the the times"""
        # self.find_markers_p0()

        transformed_points = vicon_points.copy()

        for f, points in zip(transformed_points['frame_ids'], transformed_points['points']):
            try:
                T  = self.marker_T_at_frame_vector(f)
            except Exception as e:
                T = np.eye(4)

            for pl in points:
                points[pl] = T @ np.append(points[pl], 1.0)

        return transformed_points
    
    def points_dict_to_array(self, points_dict):
        points_all = []
        try:
            for points in points_dict['points']:
                for l in sorted(points.keys()):
                    points_all.append(points[l])
        except:
            # print('dict with only the points')

            try:
                for l in sorted(points_dict.keys()):
                    points_all.append(points_dict[l])
            except:
                print('error converting the dict to array')
        
        points_all = np.array(points_all)
        # add 1s to as the last column
        points_all = np.hstack((points_all, np.ones((points_all.shape[0], 1))))

        return np.array(points_all)


class DvsHelper():
    def __init__(self, file_path):

        self.loaded_events = False
        self.file_path = file_path
        try:
            self.flash_time = self.read_annotation(file_path)
        except Exception as e:
            print("no manual zero time found, using zero instead")
            self.flash_time = 0.0

    def read_annotation(self, file_path):
        """
        The annotation is only the flash time in the dvs recording
        the time needs to be labelled manually. Using mustard the data is
        saved in a file ground_truth.csv
        """
        flash_time = None
        for dirpath, dirnames, filenames in os.walk(file_path):
            for filename in filenames:
                if filename == "ground_truth.csv":
                    annotation_data = np.loadtxt(os.path.join(dirpath, filename), delimiter=" ")
                    flash_time = annotation_data[0, 0]
                    break
        if flash_time is None:
            raise Exception("No ground_truth.csv found, you need to label the start time")
        return flash_time
    
    def read_events(self):
        """
        Read the recorded events from the dvs
        """
        dvs_data = importIitYarp(filePathOrName=self.file_path)
        self.events = dvs_data['data']['left']['dvs']

        # adjust the time given the synchronized beginning
        self.events['ts'] -= self.flash_time

        self.loaded_events = True

        return self.events
    
    def read_points_labels(self, file_path):
        """
        The points annotations are saved in a yaml file. The data consists of a list of frames.
        For each frame there is an associated time and a dictionary of lables. The dictionary
        contains the labels and the x, y position in the image
        """
        with open(file_path, 'r') as stream:
            data_loaded = yaml.load(stream, Loader=yaml.Loader)

        self.labeled_points = data_loaded
        return self.labeled_points

class DvsLabeler():

    def __init__(self, events, img_shape):

        self.events = events
        self.img_shape = img_shape

        self.labels_done = False

        return None
    
    def on_click(event, x, y, p1 , p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{x}, {y}")

    def label_data(self, times, labels, duration=0.02):
        """
        Create the labels for the image points. The parameter 'times' controlls at which times
        the labels are recorded. 
        The labelling is then done manually and saved in a yaml file.
        """
        dict_out = {
            'points': [],
            'times': []
        }
        self.dvs_frames = []

        for t in times:
            dvs_frame = utils.extract_frame(self.events, t, t+duration, self.img_shape)
            self.dvs_frames.append(dvs_frame)

            # extract the points
            points_dict, frame = self.label_frame(dvs_frame, labels)

            cv2.imwrite(f"/home/schiavazza/code/hpe/vicon_recordings/data/frame_{t}.png", frame)

            dict_out['points'].append(points_dict)
            dict_out['times'].append(t)

        cv2.destroyWindow('image')
        cv2.waitKey(1)

        self.labeled_dict = dict_out
        self.labels_done = True
        return dict_out
    
    def save_labeled_points(self, out_file):
        assert self.labels_done == True

        with open(out_file, 'w') as yaml_file:
            yaml.dump(self.labeled_dict, yaml_file, default_flow_style=False)

    def label_frame(self, frame, labels):
        points = []
        finished = False
        current_label_id = 0

        def on_click(event, x, y, p1, p2):
            nonlocal current_label_id
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                current_label_id += 1
                points.append([x, y])
                
                
        while not finished:
            img = np.copy(frame)
            cv2.putText(img, f"Click on: {labels[current_label_id]}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            for p in points:
                cv2.circle(img, np.asarray(p, dtype=int), 6, (255, 0, 0), -1)
            cv2.imshow("image", img)

            cv2.setMouseCallback('image', on_click)
            cv2.waitKey(100)

            if current_label_id >= len(labels):
                finished = True

        points_dict = {}
        for p, l in zip(points, labels):
            points_dict[l] = {
                'x': int(p[0]),
                'y': int(p[1])
            }

        return points_dict, img
