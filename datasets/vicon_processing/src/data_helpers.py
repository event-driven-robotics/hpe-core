"""
This file contains the class helpers to read and process the data from the 
c3d data (from the vicon) and the event data from the cameras
"""

# TODO the different helpers could me moved to different files to improve readability

import os
import os.path

import numpy as np
import c3d
import scipy.stats
import cv2
import yaml

from bimvee.importIitYarp import importIitYarp

from . import utils

class PointsInfo:
    """Class used to represent a set of points
    Stores the point positions, the times and and the associated
    vicon frame id
    
    This class is also used to represent labeled points from the dvs
    In this case the frames_ids field is not used
    
    The different fields are accessed as a dictionary, e.g. example_points['times']
    
    Trying to retrieve or set any other field returns an error"""

    def __init__(self):
        self.times = None
        self.points = None
        self.frame_ids = None
        
    def __getitem__(self, key):
        match key:
            case "points":
                return self.points
            case "times":
                return self.times
            case "frame_ids":
                return self.frame_ids
            
        raise KeyError("Invalid Key")
    

    def __setitem__(self, key, value):
        match key:
            case "points":
                self.points = value
            case "times":
                self.times = value
            case "frame_ids":
                self.frame_ids = value
            case _: 
                raise KeyError("Invalid Key")

class C3dHelper:
    """Class for processing the c3d files"""

    def __init__(self, file_path, wand_zero_time=False, delay=0.0, camera_markers=True, filter_camera_markers=True):
        
        self.reader = c3d.Reader(open(file_path, 'rb'))
        self.wand_zero = wand_zero_time
        self.delay = delay
        self.camera_markers = camera_markers
        self.filter_camera_markers = filter_camera_markers

        self.calculate_frame_times()
        # the obtained transforms are saved in a dict for later use if needed
        # the keys are the frame ids
        self.markers_T = {}
        self.process_all_frames()

        if not self.camera_markers:
            print("Selected the option to not use the markers on the camera, the identity transformation will be used instead")
        else:
            self.process_camera_markers()
    
    def calculate_frame_times(self) -> np.ndarray:
        """Each frame does not have a timestamp, however the realative times 
        can be calculated from the known constant frame rate """

        self.start_time = 0.0
        # self.start_time -= self.delay

        rate = self.reader.point_rate # vicon rate
        time_step = 1.0 / rate # t between frames

        times = np.linspace(self.start_time, self.reader.frame_count * time_step,
                    self.reader.frame_count)
        
        # delay is used to synchnoize the vicon data and the dvs
        # the value is subtracted to all the times
        times -= self.delay
        
        self.frame_times = times

        return self.frame_times
    
    def get_frame_time(self, times : list[float]) -> list[int]:
        """Get a list of ids corresponding to the passed list of times"""
        frame_ids = []
        for t in times:
            idx = np.searchsorted(self.frame_times, t)
            if idx >= self.reader.frame_count:
                break
            frame_ids.append(idx)

        return frame_ids
    
    def get_points_frame(self, frame_id: int) -> list[float]:
        assert frame_id >= 0 and frame_id < self.reader.frame_count

        return self.frame_points[frame_id]

    def find_start_time_stick(self):
        """
        NOT USED ANYMORE
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
    
    def set_delay(self, delay: float):
        self.delay = delay
        self.calculate_frame_times()

    def get_vicon_points(self, frames_id: list[int], labels:list[str]) -> PointsInfo:
        vicon_points_frames = [self.get_points_dict(idx) for idx in frames_id]
        vicon_points_frames = [self.filter_dict_labels(old_dict, labels) 
                            for old_dict in vicon_points_frames]

        out = PointsInfo()
        out['points'] = vicon_points_frames
        out['times'] = np.array([self.frame_times[idx] for idx in frames_id])
        out['frame_ids'] = (frames_id)
        
        return out

    def process_camera_markers(self):
        """For each vicon frame, read the position of the markers on the camera, 
        calculate the corresponding frame of reference for the camera
        filter the poses and store the trajectory"""

        # TODO: read the labels froma config file
        labels = [
            'camera:cam_back',
            'camera:cam_right',
            'camera:cam_left'
            ]
        vicon_points = self.get_vicon_points(range(1, self.reader.frame_count), labels)

        camera_front = []
        camera_side = []
        camera_top = []
        for f in vicon_points['points']:
            camera_front.append(f['camera:cam_right'][:3])
            camera_side.append(f['camera:cam_left'][:3])
            camera_top.append(f['camera:cam_back'][:3])

        camera_front = np.array(camera_front)
        camera_side = np.array(camera_side)
        camera_top = np.array(camera_top)

        if self.filter_camera_markers:
            self.camera_front_filt = self.filter_pose(camera_front)
            self.camera_side_filt = self.filter_pose(camera_side)
            self.camera_top_filt = self.filter_pose(camera_top)
        else:
            self.camera_front_filt = camera_front
            self.camera_side_filt = camera_side
            self.camera_top_filt = camera_top
    
    def process_all_frames(self):
        self.frame_points = {}
        for i, points, analog in self.reader.read_frames():
            self.frame_points[i] = points
    
    def get_points_dict(self, frame_id: int) -> dict:
        points = self.get_points_frame(frame_id)
        dict_out = {}
        for i, l in enumerate(self.reader.point_labels):
            dict_out[l.strip()] = points[i][:3]

        return dict_out
    
    def filter_dict_labels(self, old_dict: dict, labels: list[str]) -> dict:
        new_dict = {
            key: old_dict[key] for key in labels
        }
        return new_dict

    def marker_T_at_frame_vector(self, frame_id: int, time: int=-1) -> np.ndarray:
        """
        This method returns the tranformation T that described the frame of reference defined by
        the 3 marker placed on the camera.
        Returns the 4x4 transformation matrix T that transform points from world frame to the marker frame
        The function actually first computes the inverse transformation from marker frame to world and then inverts the matrix"""

        # if self.camera_markers is False, use a zero T
        if not self.camera_markers:
            return np.eye(4) 
        
        if frame_id >= self.camera_front_filt.shape[0]:
            return self.marker_T_at_frame_vector(frame_id-1)
        
        t1 = self.frame_times[frame_id -1]
        t2 = self.frame_times[frame_id]
        if time < 0:
            time = t2
        f = (time - t1) / (t2 - t1)
        
        camera_front = self.interpolate_point_array(
            self.camera_front_filt[frame_id-1],
            self.camera_front_filt[frame_id],
            f)
        camera_side = self.interpolate_point_array(
            self.camera_side_filt[frame_id-1],
            self.camera_side_filt[frame_id],
            f)
        camera_top = self.interpolate_point_array(
            self.camera_top_filt[frame_id-1],
            self.camera_top_filt[frame_id],
            f)
        
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
    
    def interpolate_point_array(self, arr1:np.ndarray, arr2:np.ndarray, f:float) -> np.ndarray:
        p_n = arr1 + (arr2 - arr1) * f
        return p_n
    
    def interpolate_point_dict(self, dict_t1:dict, dict_t2:dict, f:float) -> dict:
        # desired time should be between 0.0 and 1.0

        out_dict = {}

        for l in dict_t1.keys():
            p1 = dict_t1[l]
            p2 = dict_t2[l]

            p_n = p1 + (p2 - p1) * f
            out_dict[l] = p_n

        return out_dict

    def get_vicon_points_interpolated(self, dvs_points:PointsInfo) -> PointsInfo:
        """the frames is represent the ids of the frames corresponding to the label
        times floored to match with a vicon frame. The next id gives the upper value
        We can use the two values to interpolate to the right time
        
        The input is a dict: {
            points: [...]
            times: [...]
        }"""

        vicon_points_frames = []
        frames_id = self.get_frame_time(dvs_points['times'])
        desired_times = dvs_points['times']

        for i, (idx, d_t) in enumerate(zip(frames_id, desired_times)):
            labels = dvs_points['points'][i].keys()
            if idx == 0:
                vicon_points_frames = self.get_points_dict(idx)

                vicon_points_frames = self.filter_dict_labels(vicon_points_frames, labels)
                vicon_points_frames.append(vicon_points_frame)
                continue
            
            vicon_points_frame_t1 = self.get_points_dict(idx - 1)
            vicon_points_frame_t2 = self.get_points_dict(idx)

            vicon_points_frame_t1 = self.filter_dict_labels(vicon_points_frame_t1, labels)
            vicon_points_frame_t2 = self.filter_dict_labels(vicon_points_frame_t2, labels)

            t1 = self.frame_times[idx - 1]
            t2 = self.frame_times[idx]
            f = (d_t - t1) / (t2 - t1)
            vicon_points_frame = self.interpolate_point_dict(vicon_points_frame_t1, vicon_points_frame_t2, f)

            vicon_points_frames.append(vicon_points_frame)
        
        out = PointsInfo()
        out['points'] = vicon_points_frames
        out['times'] = desired_times
        out['frame_ids'] = (frames_id)
        
        return out
    
    def filter_pose(self, x:np.ndarray, order:int = 3, fs:int = 100.0, cutoff:int = 4) -> np.ndarray:
        out = np.empty_like(x)
        
        for i in range(x.shape[1]):
            out[:, i] = utils.butter_lowpass_filter(x[:, i], cutoff, fs, order)

        return out
    
    def transform_points_to_marker_frame(self, vicon_points:PointsInfo) -> PointsInfo:
        """The vicon_points is a dict containing the points and the the times"""
        # self.find_markers_p0()

        transformed_points = vicon_points.copy()

        for f, t,points in zip(transformed_points['frame_ids'], 
                             transformed_points['times'],
                             transformed_points['points']):
            try:
                T  = self.marker_T_at_frame_vector(f, time=t)
            except Exception as e:
                # TODO explicit error
                T = np.eye(4)

            for pl in points:
                points[pl] = T @ np.append(points[pl], 1.0)

        return transformed_points
    
    def points_dict_to_array(self, points_dict: PointsInfo) -> np.ndarray:
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

    def read_annotation(self, file_path):
        """
        NOT USED ANYMORE
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
    
    def find_start_moving_time(self):
        """The camera start moving time is determined by the event rate
        The time can then be used to synchronize the dvs with the vicon data"""
        if not self.loaded_events:
            self.read_events()

        # search only the first 2 seconds
        id_range_end = np.searchsorted(self.events['ts'], 2.0)
        times = self.events['ts'][:id_range_end]
        duration = times[-1]- times[0]
        time_bins = np.linspace(times[0], times[-1], int(1000 * duration)) # 1000 bins per second

        hist, bin_edges = np.histogram(times, time_bins)
        idx = np.argmax(np.diff(hist) > 1000)
        return bin_edges[idx]

class DvsLabeler():

    def __init__(self, img_shape, events=None):

        self.events = events
        self.img_shape = img_shape

        self.labels_done = False

        return None
    
    def on_click(event, x, y, p1 , p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{x}, {y}")

    def generate_frames(self, times, save_folder, duration=0.02):
        self.dvs_frames = []

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_data = []

        for t in times:
            min_t = max(0.0, t-duration)
            dvs_frame = utils.extract_frame(self.events, min_t, t, self.img_shape)
            self.dvs_frames.append(dvs_frame)

            img_path = os.path.join(save_folder, f"frame_{str(t)}.png")
            cv2.imwrite(img_path, dvs_frame)

            save_data.append([t, f"frame_{str(t)}.png"])

        # save frames times
        # np.savetxt(os.path.join(save_folder, "times.txt"), times, fmt="%.9f")

        with open(os.path.join(save_folder, "times.yml"), 'w') as f:
            yaml.dump(save_data, f, default_flow_style=False)

        return save_folder

    def label_data(self, frames_folder:str, labels:list[str], subject:str="P11", manual:bool=False) -> PointsInfo:
        """
        Create the labels for the image points. The parameter 'times' controlls at which times
        the labels are recorded. 
        The labelling is then done manually and saved in a yaml file.
        """
        dict_out = PointsInfo()
        self.dvs_frames = []

        with open(os.path.join(frames_folder, "times.yml")) as f:
            times = yaml.load(f, Loader=yaml.Loader)

        print(times)

        for t, filenameext in times:
            file_name, ext = os.path.splitext(filenameext)
            if ext != ".png":
                continue
            print(f"Labeling {filenameext}")

            dvs_frame = cv2.imread(os.path.join(frames_folder, filenameext))
            self.dvs_frames.append(dvs_frame)

            if not manual: 
                # extract the points
                success, points_dict, frame = self.label_frame(dvs_frame, labels, subject=subject)
                if not success:
                    continue
            else:
                success, points_dict, frame = self.label_frame_manual(dvs_frame, subject=subject)
                if not success:
                    continue

            labeled_folder_path = os.path.join(frames_folder, "labeled")
            if not os.path.exists(labeled_folder_path):
                os.makedirs(labeled_folder_path)
            file_save_path = os.path.join(labeled_folder_path, filenameext)

            try:
                cv2.imwrite(file_save_path, frame)
            except Exception as e:
                print(e)

            dict_out['points'].append(points_dict)
            dict_out['times'].append(t)

        cv2.destroyWindow('image')
        cv2.waitKey(1)

        self.labeled_dict = dict_out
        self.labels_done = True
        return dict_out
    
    def save_labeled_points(self, out_file:str):
        assert self.labels_done == True

        with open(out_file, 'w') as yaml_file:
            yaml.dump(self.labeled_dict, yaml_file, default_flow_style=False)

    def label_frame(self, frame:np.ndarray, labels:list[str], subject:str="P11") -> (bool, PointsInfo, np.ndarray):
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
                cv2.circle(img, np.asarray(p, dtype=int), 4, (255, 0, 0), -1)
            cv2.imshow("image", img)

            cv2.setMouseCallback('image', on_click)
            c = cv2.waitKey(100)
            if ' ' == chr(c & 255):
                print("space pressed, skipping")
                return False, None, None

            if current_label_id >= len(labels):
                finished = True

        img = np.copy(frame)
        for p in points:
            cv2.circle(img, np.asarray(p, dtype=int), 6, (255, 0, 0), -1)

        points_dict = PointsInfo()
        for p, l in zip(points, labels):
            complete_label = f"{subject}:{l}"
            points_dict[complete_label] = {
                'x': int(p[0]),
                'y': int(p[1])
            }

        return True, points_dict, img
    
    def label_frame_manual(self, frame:np.ndarray, subject:str="P11") -> (bool, PointsInfo, np.ndarray):
        """This is different from the previous method. It is still for labeling the frames
        Instead of using a fixed list of labels, the user select the label for each point."""
        points = []
        finished = False
        current_label_id = 0
        points_dict = PointsInfo()

        # load all the marker labels
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../scripts/config/labels_tags.yml")
        with open(filename) as f:
            marker_labels = yaml.load(f, Loader=yaml.Loader)

        def on_click(event, x, y, p1, p2):
            nonlocal current_label_id
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                current_label_id += 1
                points.append([x, y])

                # query user for label id
                print("Select a label: ", "\t".join([f"{idx}):{n}" for idx, n in enumerate(marker_labels)]))
                label_id = input()
                label_val = marker_labels[int(label_id)]

                label = f"{subject}:{label_val}"
                points_dict[label] = {
                    "x": int(x),
                    "y": int(y)
                }

                print(f"current labels: {points_dict}")

                
        while not finished:
            img = np.copy(frame)
            for p in points:
                cv2.circle(img, np.asarray(p, dtype=int), 4, (255, 0, 0), -1)
            cv2.imshow("image", img)

            cv2.setMouseCallback('image', on_click)
            c = cv2.waitKey(100)
            if ' ' == chr(c & 255):
                print("space pressed, skipping")
                return False, None, None

            if c==ord('k'):
                finished = True

        img = np.copy(frame)
        for p in points:
            cv2.circle(img, np.asarray(p, dtype=int), 6, (255, 0, 0), -1)

        return True, points_dict, img