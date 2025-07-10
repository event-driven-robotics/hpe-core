#functions for use with vicon processing

import numpy as np
import math
import cv2
import os
import yaml

from scipy.spatial.transform import Rotation

# dropdown menu for labeling points
import tkinter as tk
from tkinter import simpledialog

def makeT(Rot, Trans):

    Rot = np.array(Rot) * (math.pi / 180.0)

    yawMatrix = np.matrix([
    [math.cos(Rot[2]), -math.sin(Rot[2]), 0],
    [math.sin(Rot[2]), math.cos(Rot[2]), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(Rot[1]), 0, math.sin(Rot[1])],
    [0, 1, 0],
    [-math.sin(Rot[1]), 0, math.cos(Rot[1])]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(Rot[0]), -math.sin(Rot[0])],
    [0, math.sin(Rot[0]), math.cos(Rot[0])]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    T = np.zeros((4,4), dtype = np.float64)
    T[0:3, 0:3] = R
    T[3,3] = 1.0
    Trans = (R @ Trans.transpose()).transpose()
    T[0,3] = -Trans[0]
    T[1,3] = -Trans[1]
    T[2,3] = -Trans[2]
    return T

def marker_p(c3d_labels, c3d_points, mark_name):
        
    i = 0
    for label in c3d_labels:
        k = label.find(':') + 1
        if(label[k:k+4] == mark_name):
            break
        i = i + 1
    #marker_p = lambda index: np.array([np.append(val[index][0:3], 1) for val in points_3d.values()])

    return np.array([np.append(val[i][0:3], 1) for val in c3d_points])

def calc_indices(e_ts, period):
        
    time_tags = np.arange(e_ts[0], e_ts[-1], period)
    index_tags = np.empty(len(time_tags), dtype=int)
    i = 0
    for j, t in enumerate(time_tags):
        while e_ts[i] < t:
            i = i + 1               
        #print(e_ts[i])
        index_tags[j] = i
    return time_tags, index_tags  

def read_points_labels(file_path):
    # read labeled points from YAML file.
    
    with open(file_path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.Loader)

    return data_loaded  

class DvsLabeler:
    # functions relative to the labeling of the sequences
    
    def __init__(self, img_shape, subject, events=None):
        self.events = events
        self.img_shape = img_shape
        self.labels_done = False
        self.labeled_dict = None
        self.subject = subject
        
    def label_data(self, e_ts, e_us, e_vs, event_indices, time_tags, period):
        # Go though every event frame and call function to do the labelling.
        
        dict_out = {'points': [], 'times': []}
        self.dvs_frames = []
        
        ft = e_ts[0]
        img = np.ones(self.img_shape, dtype=np.uint8) * 255
        i = 0
        
        while i < len(e_ts):
            if e_ts[i] >= ft:
                # Show the current event frame
                img[e_vs[i], e_us[i]] = 0
                
                success, points_dict, frame = self.label_frame(img, ft)
                if not success:
                    img = np.ones(self.img_shape, dtype = np.uint8)*255
                else:
                    self.dvs_frames.append(frame)
                    dict_out['points'].append(points_dict)
                    dict_out['times'].append(float(ft))
                    
                    img = np.ones(self.img_shape, dtype = np.uint8)*255
                
                ft = ft + period    # maybe 2*period, just to skip some frames as they are a lot
                    
            img[e_vs[i],e_us[i]] = 0
            i += 1
            
        cv2.destroyAllWindows()
        self.labeled_dict = dict_out
        self.labels_done = True
        return dict_out

    def save_labeled_points(self, out_file: str):
        # Save labeled points to a YAML file.
        
        assert self.labels_done is True
        with open(out_file, 'w') as yaml_file:
            yaml.dump(self.labeled_dict, yaml_file, default_flow_style=False)
        
    # TODO: add method to match markers from first estimated projection 
    # and manually match it to object in the scene

    def select_label_tkinter(self, marker_labels):
        # dropdown menu to select labels.
        # TODO: make it better
        
        root = tk.Tk()
        root.withdraw()

        selected = simpledialog.askstring(
            "Select Label",
            "Choose a label:\n" + "\n".join(f"{i}: {l}" for i, l in enumerate(marker_labels)),
            parent=root
        )
        root.destroy()
        return selected

    def label_frame(self, frame: np.ndarray, timestamp: float = None):
        # allow user to label points in the current frame.
        
        points = []
        finished = False
        points_dict = {}

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../scripts/config/labels_tags.yml")   # check later for modification of yaml file
        with open(filename) as f:
            marker_labels = yaml.load(f, Loader=yaml.Loader)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                label_val = self.select_label_tkinter(marker_labels)
                if label_val is None:
                    print("Labeling aborted by user.")
                    self.abort_labeling = True
                    return
                try:
                    marker_name = marker_labels[int(label_val)]
                except (ValueError, IndexError):
                    marker_name = label_val
                marker_name = f"{self.subject}:{marker_name}"
                points.append([x, y])
                points_dict[marker_name] = {"x": int(x), "y": int(y)}
                print(f"current labels: {points_dict}")

        self.abort_labeling = False
        cv2.imshow("image", frame)
        cv2.setMouseCallback('image', on_click)

        while not finished:
            img = np.copy(frame)
            # draw labeled points on the image
            for p in points:
                cv2.circle(img, np.asarray(p, dtype=int), 4, (255, 0, 0), -1)
            # Draw timestamp on the top right corner
            if timestamp is not None:
                text = f"t = {timestamp:.6f}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                color = (0, 0, 0)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = img.shape[1] - text_width - 10
                y = text_height + 10
                cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.imshow("image", img)
            c = cv2.waitKey(100)
            cv2.setMouseCallback('image', on_click)
            if ' ' == chr(c & 255): # space bar
                finished = True
                # skip current frame by pressing space bar
                print("space pressed, skipping")
                return False, None, None
            elif c == ord('s'):     # s key
                # save labeled points for the current frame
                finished = True
                return True, points_dict, img
            # elif c == ord('q') or c == 27 or getattr(self, 'abort_labeling', False):    # q key or esc
            #     # abort labeling by pressing q or esc
            #     print("Labeling aborted by user.")
            #     cv2.destroyWindow("image")
            #     return False, None, None
            elif c == 8:            # backspace
                # remove latest added point for the current frame
                if points:
                    removed_point = points.pop() # it may be useful to keep the removed point
                    if points_dict:
                        last_marker = list(points_dict.keys())[-1]
                        points_dict.pop(last_marker)
                    print("Removed last labeled marker.")

        img = np.copy(frame)
        for p in points:
            cv2.circle(img, np.asarray(p, dtype=int), 6, (255, 0, 0), -1)

        return True, points_dict, img  
    
   
    
class ViconHelper:
    # functions relative to the extraction of the vicon data from c3d files.
    
    def __init__(self, frame_times, points_3d, delay, frame_count, point_rate, point_labels, marker_T_at_frame_vector=None):
        self.frame_times = frame_times
        self.points_3d = points_3d
        self.delay = delay
        self.frame_count = frame_count
        self.point_rate = point_rate
        self.point_labels = [l.strip() for l in point_labels]
        self.marker_T_at_frame_vector = marker_T_at_frame_vector
        
        self.calculate_frame_times()
    
    def calculate_frame_times(self) -> np.ndarray:
        # calculate the frame times based on the frame rate and delay

        self.start_time = 0.0

        rate = self.point_rate # vicon rate
        time_step = 1.0 / rate # t between frames

        times = np.linspace(self.start_time, self.frame_count * time_step,
                    self.frame_count)
        
        # delay is used to synchnoize the vicon data and the dvs
        times += self.delay
        
        self.frame_times = times


    def get_frame_time(self, times):
        # attribute each label an id based on the timestamp.
        
        frame_ids = []
        for t in times:
            idx = np.searchsorted(self.frame_times, t)
            if idx >= len(self.frame_times):
                break
            frame_ids.append(idx)
        return frame_ids

    def get_points_dict(self, frame_id):
        # get 3D points for a specific frame id.
        
        points = self.points_3d[frame_id]
        return {l: points[i][:3] for i, l in enumerate(self.point_labels)}

    def interpolate_point_dict(self, dict_t1, dict_t2, f):
        out_dict = {}
        for l in dict_t1.keys():
            p1 = dict_t1[l]
            p2 = dict_t2[l]
            out_dict[l] = p1 + (p2 - p1) * f
        return out_dict

    def get_vicon_points_interpolated(self, dvs_points):
        # For each labeled DVS point (with a timestamp), interpolate the Vicon 3D marker positions to match the DVS times
        
        vicon_points_frames = []
        frames_id = self.get_frame_time(dvs_points['times'])    # get closest vicon frame for each dvs timestamp
        desired_times = dvs_points['times']

        for i, (idx, d_t) in enumerate(zip(frames_id, desired_times)):
            frame_labels = dvs_points['points'][i]  # extract labels points
            if not frame_labels:
                continue
            labels = frame_labels.keys()

            if idx == 0:
                vicon_points_frame = self.get_points_dict(idx)
                #vicon_points_frame = self.filter_dict_labels(vicon_points_frame, labels)
                
                # Only keep the labels present in the DVS frame
                vicon_points_frame = {k: v for k, v in vicon_points_frame.items() if k in labels}
                vicon_points_frames.append(vicon_points_frame)
                continue

            vicon_points_frame_t1 = self.get_points_dict(idx - 1)
            vicon_points_frame_t2 = self.get_points_dict(idx)
            #vicon_points_frame_t1 = self.filter_dict_labels(vicon_points_frame_t1, labels)
            #vicon_points_frame_t2 = self.filter_dict_labels(vicon_points_frame_t2, labels)
            
            # Only keep the labels present in the DVS frame
            vicon_points_frame_t1 = {k: v for k, v in vicon_points_frame_t1.items() if k in labels}
            vicon_points_frame_t2 = {k: v for k, v in vicon_points_frame_t2.items() if k in labels}

            # interpolation
            t1 = self.frame_times[idx - 1]
            t2 = self.frame_times[idx]
            f = (d_t - t1) / (t2 - t1) if (t2 - t1) != 0 else 0.0

            vicon_points_frame = self.interpolate_point_dict(vicon_points_frame_t1, vicon_points_frame_t2, f)
            vicon_points_frames.append(vicon_points_frame)

        out = {}
        out['points'] = vicon_points_frames
        out['times'] = desired_times
        out['frame_ids'] = frames_id
        
        return out

    def transform_points_to_marker_frame(self, vicon_points):
        # if 3 markers are placed on the camera, this function should transform the points
        # from the world frame to the marker frame.
        # may be useful to actually implement it.
        
        if self.marker_T_at_frame_vector is None:
            return vicon_points  # No transformation
        transformed_points = vicon_points.copy()
        for f, t, points in zip(transformed_points['frame_ids'], transformed_points['times'], transformed_points['points']):
            try:
                T = self.marker_T_at_frame_vector(f, time=t)
            except Exception:
                T = np.eye(4)
            for pl in points:
                points[pl] = T @ np.append(points[pl], 1.0)
        return transformed_points

# Not being used, but i wanted to implement everything in a nicer way and also allow for initial estimation
# class ProjectionHelper:
#     # class to help with the projection of points and calculation of camera position and orientation
    
#     def __init__(self, vicon_points, labeled_points, K, D):
#         # get the points from the vicon and the labeled points from the dvs and store them.
        
#         if vicon_points is None or labeled_points is None:
#             return
        
#         world_points = []
#         image_points = []

#         for dvs_frame, vicon_frame in zip(labeled_points['points'], vicon_points['points']):
#             labels = dvs_frame.keys()

#             for l in labels:
#                 try:
#                     w_p = vicon_frame[l]
#                     i_p = [
#                         dvs_frame[l]['x'],
#                         dvs_frame[l]['y'],
#                         1.0
#                     ]

#                     world_points.append(w_p)
#                     image_points.append(i_p)
#                 except Exception as e:
#                     print("The stored image labels probably don't match with the vicon labels used.")
#                     print(e)

#         self.world_points = np.array(world_points)
#         self.image_points = np.array(image_points)

#         assert self.world_points.shape[0] == self.image_points.shape[0], "Not equal numbers of image and 3D points"    
        
#         self.K = K
#         self.D = D

#     # if want to use known translation
#     # def find_R_t_opencv(self, known_translation):
#     #     proj_points = np.copy(self.world_points)
#     #     img_points = np.copy(self.image_points[:, :2])

#     #     if known_translation is not None:
#     #         # Center 3D points by subtracting known translation
#     #         proj_points = proj_points - known_translation.reshape(1, 3)
#     #         # Initial guess for rotation (zero) and translation (known)
#     #         rvec_init = np.zeros((3, 1), dtype=np.float64)
#     #         tvec_init = known_translation.reshape(3, 1)
#     #         # Solve PnP with fixed translation
#     #         success, rvec, tvec = cv2.solvePnP(
#     #             proj_points, img_points, self.K, self.D,
#     #             rvec=rvec_init, tvec=tvec_init,
#     #             useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
#     #         )
#     #         R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
#     #         T = np.eye(4)
#     #         T[:3, :3] = R
#     #         T[:3, 3] = known_translation
#     #         return T, None
#     #     else:
#     #         # Default: estimate both rotation and translation
#     #         s, r, t, e = cv2.solvePnPGeneric(proj_points, img_points, self.K, self.D, flags=cv2.SOLVEPNP_SQPNP)
#     #         r = r[0]
#     #         t = t[0]
#     #         e = e[0][0]
#     #         T = np.zeros((4, 4))
#     #         T[:3, :3] = Rotation.from_rotvec(r.reshape(3,)).as_matrix()
#     #         T[:3, -1] = t.reshape(3,)
#     #         T[-1, -1] = 1.0
#     #         return T, e
        
#     # no need for known translation
#     def find_R_t_opencv(self):        
#         proj_points = np.copy(self.world_points)
#         img_points = np.copy(self.image_points[:, :-1])
        
#         r, t, e = cv2.solvePnP(proj_points, img_points, self.K, self.D, flags=cv2.SOLVEPNP_SQPNP)
#         r = r[0]
#         t = t[0]
#         e = e[0][0]
#         T = np.zeros((4, 4))
#         T[:3, :3] = Rotation.from_rotvec(r.reshape(3,)).as_matrix() # check for angles
#         T[:3, -1] = t.reshape(3,)
#         T[-1, -1] = 1.0
#         return T, e
    
#     # first initial estimation to project points on plane and then match them by clicking the correct points          
#     # TODO: implement an initial estimate to match the projections without needing labeling and knowing the markers
#     def initial_estimate(self):
#         #Find an inital estimate for transformation to match the projections
#         s = self.world_points.shape[0]
#         A = np.zeros((2*s, 12))
#         for i in range(s):
#             A[2*i, :4] = self.world_points[i]
#             A[2*i + 1, 4:8] = self.world_points[i]

#             A[2*i, 8:] = self.world_points[i] * (-self.image_points[i][0])
#             A[2*i+1, 8:] = self.world_points[i] * (-self.image_points[i][1])

#         # compute At x A
#         A_ = np.matmul(A.T, A)
#         # compute its eigenvectors and eigenvalues
#         eigenvalues, eigenvectors = np.linalg.eig(A_)
#         # find the eigenvector with the minimum eigenvalue
#         # (numpy already returns sorted eigenvectors wrt their eigenvalues)
#         m = eigenvectors[:, 11]
#         return m