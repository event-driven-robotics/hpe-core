#functions for use with vicon processing

import numpy as np
import math
import cv2
import os
import yaml
import c3d

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

    mark_name_clean = mark_name.strip()
    i = None
    for idx, label in enumerate(c3d_labels):
        label_clean = label.strip()
        if label_clean == mark_name_clean:
            i = idx
            break
    if i is None:
        raise ValueError(f"Marker name '{mark_name}' not found in c3d_labels. Searched for '{mark_name_clean}'. Available labels: {[l.strip() for l in c3d_labels]}")
    out = []
    for val in c3d_points:
        if i < val.shape[0]:
            out.append(val[i][0:3])     # check shape
        else:
            continue
    return np.array(out)

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

def project_vicon_to_event_plane(
    marker_names, 
    c3d_data, 
    points_3d, 
    marker_t, 
    T, 
    K,
    cam_res, 
    delay, 
    e_ts, 
    e_us, 
    e_vs, 
    period,
    visualize
):
    # Project points from Vicon to event plane using the transformation matrix T

    projected_points = {}

    # For each marker, transform and project
    for mark_name in marker_names:
        ps = marker_p(c3d_data.point_labels, points_3d.values(), mark_name)
        
        if ps.shape[1] == 3:
            ps = np.hstack([ps, np.ones((ps.shape[0], 1))])
            
        # Apply transformation
        ps_trans = (T @ ps.T).T
        ps_trans = ps_trans / ps_trans[:, [3]]

        ps_trans = ps_trans[:, :3]

        # Project to image plane
        img_pts, _ = cv2.projectPoints(ps_trans, np.zeros(3), np.zeros(3), K, None)
        img_pts = img_pts.reshape(-1, 2)
        projected_points[mark_name] = img_pts
        
        # plt.plot(marker_t, img_pts[:, 0:2])
        # plt.legend(['U', 'V'])
        # plt.plot([marker_t[0], marker_t[-1]], [cam_res[0], cam_res[0]], 'tab:orange', linestyle=':')
        # plt.plot([marker_t[0], marker_t[-1]], [cam_res[1], cam_res[1]], 'b:')
        # plt.plot([marker_t[0], marker_t[-1]], [0, 0], 'k:')
        # plt.title(mark_name)
        # plt.show()

    if visualize:
        # Visualization
        img = np.ones(cam_res, dtype=np.uint8) * 255
        tic_markers = marker_t[0] + period
        tic_events = e_ts[0] + delay + period
        i_markers = 0
        i_events = 0

        while tic_markers < marker_t[-1] and tic_events < e_ts[-1]:
            while marker_t[i_markers] < tic_markers:
                for mark_name in marker_names:
                    u = int(projected_points[mark_name][i_markers][0])
                    v = int(projected_points[mark_name][i_markers][1])
                    if 0 <= u < cam_res[1] and 0 <= v < cam_res[0]:
                        cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                        cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                i_markers += 1

            while e_ts[i_events] < tic_events:
                img[e_vs[i_events], e_us[i_events]] = 0
                i_events += 1

            cv2.imshow('Projected Points', img)
            c = cv2.waitKey(int(period * 1000))
            if c == ord('q'):
                cv2.destroyAllWindows()
                return projected_points
            img = np.ones(cam_res, dtype=np.uint8) * 255
            tic_markers += period
            tic_events += period

    cv2.destroyAllWindows()
    return projected_points
 

class DvsLabeler:
    # functions relative to the labeling of the sequences
    
    def __init__(self, img_shape, subject, events):
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

    def label_frame(self, frame, timestamp):
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
            # elif c == ord('q') or c == 27:    # q key or esc
            #     # abort labeling by pressing q or esc
            #     # TODO: quit and save            
            #     print("Labeling aborted by user.")
            #     cv2.destroyWindow("image")
            #     return False, None, None
            elif c == 8:            # backspace
                # remove latest added point for the current frame
                if points:
                    removed_point = points.pop() # it may be useful to keep the removed point?
                    if points_dict:
                        last_marker = list(points_dict.keys())[-1]
                        points_dict.pop(last_marker)
                    print("Removed last labeled marker.")

        # display labeled points on the image
        img = np.copy(frame)
        for p in points:
            cv2.circle(img, np.asarray(p, dtype=int), 6, (255, 0, 0), -1)

        return True, points_dict, img  
    
    def correct_data(
        self, e_ts, e_us, e_vs, event_indices, time_tags, period,
        T, marker_names, c3d_data, points_3d, marker_t, K, cam_res
    ):
        # showcase markers from projection and allow user to correct the labels one frame at a time.
        # TODO: do i save them in a new file? do i overwrite the old one?
        
        dict_out = {'points': [], 'times': []}
        self.corrected_frames = []

        ft = e_ts[0]
        img = np.ones(self.img_shape, dtype=np.uint8) * 255
        i = 0
        frame_idx = 0

        while i < len(e_ts):
            if e_ts[i] >= ft:
                # Show the current event frame
                img[e_vs[i], e_us[i]] = 0

                success, points_dict, frame = self.correct_labels(
                    img, ft, T, marker_names, c3d_data, points_3d, marker_t, frame_idx, K, cam_res
                )
                if not success:
                    img = np.ones(self.img_shape, dtype=np.uint8) * 255
                else:
                    self.corrected_frames.append(frame)
                    dict_out['points'].append(points_dict)
                    dict_out['times'].append(float(ft))
                    img = np.ones(self.img_shape, dtype=np.uint8) * 255

                ft = ft + period
                frame_idx += 1

            img[e_vs[i], e_us[i]] = 0
            i += 1

        cv2.destroyAllWindows()
        self.corrected_dict = dict_out
        self.corrections_done = True
        return dict_out    
    
    def correct_labels(
        self, frame, timestamp, T, marker_names, c3d_data, points_3d, marker_t, frame_idx, K, cam_res
    ):
        # Given first projection of points, allow user to correct the labels by selecting the corresponding ones
        
        # Project markers for this frame only
        # TODO: need to iterate through all of the frames correctly -> function and call inside loop
        projected_points = {}
        for mark_name in marker_names:
            ps = marker_p(c3d_data.point_labels, points_3d.values(), mark_name)
            if ps.shape[1] == 3:
                ps = np.hstack([ps, np.ones((ps.shape[0], 1))])
            ps_trans = (T @ ps.T).T
            ps_trans = ps_trans / ps_trans[:, [3]]
            ps_trans = ps_trans[:, :3]
            img_pts, _ = cv2.projectPoints(ps_trans, np.zeros(3), np.zeros(3), K, None)
            img_pts = img_pts.reshape(-1, 2)
            projected_points[mark_name] = img_pts[frame_idx]
        
        corrected_points = []
        finished = False
        corrected_points_dict = {}
        
        # load markers name file, don't even need it tbf
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../scripts/config/labels_tags.yml")   # check later for modification of yaml file
        with open(filename) as f:
            marker_labels = yaml.load(f, Loader=yaml.Loader)
            
        # call to function to project on single frame ?
            
        def on_click(event, x, y, flags, param): # ???
            if event == cv2.EVENT_LBUTTONDOWN:
                # projected_points: dict {label: [u, v]}
                min_dist = float('inf')
                closest_label = None
                for label, pt in projected_points.items():
                    u, v = pt  # or pt = projected_points[label][frame_idx]
                    dist = np.linalg.norm(np.array([x, y]) - np.array([u, v]))
                    if dist < min_dist:
                        min_dist = dist
                        closest_label = label
                print(f"Closest marker: {closest_label} at distance {min_dist}")
                # after selecting the closest marker, add it to the corrected points by selecting the second click
                # TODO: modify from old projection and attribute correction
        
        # TODO: check delay
        
        while not finished:
            img = np.copy(frame)
            
            # TODO: reproject every iteration???
            for mark_name, pt in projected_points.items():
                u, v = int(pt[0]), int(pt[1])
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (u, v), 6, (0, 0, 255), 2)
                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
            
            # draw labeled points on the image
            for p in corrected_points:
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
                
            cv2.imshow("Correct Markers", img)
            c = cv2.waitKey(100)
            cv2.setMouseCallback("Correct Markers", on_click)
            
            if ' ' == chr(c & 255):  # space bar
                finished = True
                # skip current frame by pressing space bar
                print("space pressed, skipping")
                return False, None, None
            
            elif c == 8:            # backspace
                # remove latest added point for the current frame
                if corrected_points:
                    removed_corrected_point = corrected_points.pop() # it may be useful to keep the removed point
                    if corrected_points_dict:
                        last_marker = list(corrected_points_dict.keys())[-1]
                        corrected_points_dict.pop(last_marker)
                    print("Removed last labeled marker.")
                    
            elif c == ord('s'):     # s key
                # save labeled points for the current frame
                
                # TODO: actually properly introduce method to save
                
                finished = True
                return True, corrected_points_dict, img

            elif c == ord('q') or c == 27:
                # abort labeling by pressing q or esc
                
                # TODO: quit and save
                
                print("Labeling aborted by user.")
                cv2.destroyWindow("image")
                return False, None, None
                
            # TODO: fix projection not following event data?

        return True, corrected_points_dict, img   
       
    
class ViconHelper:
    # functions relative to the extraction of the vicon data from c3d files.
    
    def __init__(self, frame_times, points_3d, delay, frame_count, point_rate, point_labels, camera_markers):
        self.frame_times = frame_times
        self.points_3d = points_3d
        self.delay = delay
        self.frame_count = frame_count
        self.point_rate = point_rate
        self.point_labels = [l.strip() for l in point_labels]
        self.camera_markers = camera_markers
        
        self.calculate_frame_times()
        
        self.marker_T_vector = {}
        
        if not self.camera_markers:
            print("Selected the option to not use the markers on the camera, the identity transformation will be used instead")
        else:
            self.process_camera_markers()
    
    def calculate_frame_times(self):
        # calculate the frame times based on the frame rate and delay

        self.start_time = 0.0

        rate = self.point_rate # vicon rate
        time_step = 1.0 / rate # t between frames

        times = np.linspace(self.start_time, self.frame_count * time_step,
                    self.frame_count)
        
        # delay is used to synchronize the vicon data and the dvs
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
    
    def interpolate_point_array(self, arr1, arr2, f):
        # ?
        p_n = arr1 + (arr2 - arr1) * f
        return p_n

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
            
            print(f"Interpolating frame {idx} at time {d_t:.6f}s between {t1:.6f}s and {t2:.6f}s")
            
            f = (d_t - t1) / (t2 - t1) if (t2 - t1) != 0 else 0.0

            vicon_points_frame = self.interpolate_point_dict(vicon_points_frame_t1, vicon_points_frame_t2, f)
            vicon_points_frames.append(vicon_points_frame)

        out = {}
        out['points'] = vicon_points_frames
        out['times'] = desired_times
        out['frame_ids'] = frames_id
        
        return out
    
    def get_vicon_points(self, frames_id, labels):
        vicon_points_frames = [self.get_points_dict(idx) for idx in frames_id]
        vicon_points_frames = [self.filter_dict_labels(old_dict, labels) 
                            for old_dict in vicon_points_frames]

        out = {}
        out['points'] = vicon_points_frames
        out['times'] = np.array([self.frame_times[idx] for idx in frames_id])
        out['frame_ids'] = frames_id
        
        print("out", out)
        
        return out
    
    def filter_dict_labels(self, old_dict, labels):
        return {key: old_dict[key] for key in labels if key in old_dict}
    
    def process_camera_markers(self):
        # Read position of markers on the camera system and calculate the corresponding reference frame
        camera_labels = ['stereoatis:cam_right', 'stereoatis:cam_back', 'stereoatis:cam_left']
        
        vicon_points = self.get_vicon_points(range(1, self.frame_count), camera_labels)

        camera_front = []
        camera_side = []
        camera_top = []
        for f in vicon_points['points']:
            camera_front.append(f['stereoatis:cam_right'][:3])
            camera_side.append(f['stereoatis:cam_left'][:3])
            camera_top.append(f['stereoatis:cam_back'][:3])

        camera_front = np.array(camera_front)
        camera_side = np.array(camera_side)
        camera_top = np.array(camera_top)

        # TODO: see if it is needed to introduce the low pass filter
        # if self.filter_camera_markers:
        #     self.camera_front_filt = self.filter_pose(camera_front)
        #     self.camera_side_filt = self.filter_pose(camera_side)
        #     self.camera_top_filt = self.filter_pose(camera_top)
        # else:
        self.camera_front = camera_front
        self.camera_side = camera_side
        self.camera_top = camera_top

    def world_to_camera_markers(self, vicon_points):
        # Get the transformation matrix that describes the reference frame of the 3 markers on the camera system
        # world <-> markers on camera system

        transformed_points = vicon_points.copy()
                
        for f, t, points in zip(transformed_points['frame_ids'], transformed_points['times'], transformed_points['points']):
                        
            T = self.marker_T_at_frame_vector(f, t)
            print(f"Transformation matrix (camera markers) for frame {f} at time {t:.6f}s:\n{T}")
                
            for p in points:
                points[p] = (T @ np.append(points[p], 1))[:3]
                                
        return T, transformed_points
    
    def marker_T_at_frame_vector(self, frame_id, time):
        # return transformation T to describe reference frame defined by the 3 markers placed on the camera system.
        
        # if self.camera_markers is False, use a zero T
        if not self.camera_markers:
            return np.eye(4) 
        
        # TODO: finish this function, then check if things actually work as they are supposed to
        if frame_id >= self.camera_front.shape[0]:
            return self.marker_T_at_frame_vector(frame_id-1)
        
        t1 = self.frame_times[frame_id -1]
        t2 = self.frame_times[frame_id]
        if time < 0:
            time = t2
        f = (time - t1) / (t2 - t1)
        
        camera_front = self.interpolate_point_array(
            self.camera_front[frame_id-1],
            self.camera_front[frame_id],
            f)
        camera_side = self.interpolate_point_array(
            self.camera_side[frame_id-1],
            self.camera_side[frame_id],
            f)
        camera_top = self.interpolate_point_array(
            self.camera_top[frame_id-1],
            self.camera_top[frame_id],
            f)
        
        # TODO: understand how to go from camera frame to actual markers
        
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

        self.marker_T_vector[frame_id] = np.copy(T)
        
        print("Transformation matrix for frame {} at time {:.6f}s:\n{}".format(frame_id, time, T))
        print("Markers T vector:", self.marker_T_vector)

        return np.copy(T)