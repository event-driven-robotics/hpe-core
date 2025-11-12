#functions for use with vicon processing

from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import os
import yaml
import c3d
from typing import Tuple, Optional

from scipy.spatial.transform import Rotation
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.optimize import least_squares

# dropdown menu for labeling points
import tkinter as tk
from tkinter import simpledialog

# Exceptions to pass values even after interruptions
class RotationExit(Exception):
    def __init__(self, r_vec):
        super().__init__("Rotation adjustment finished by user.")
        self.r_vec = r_vec

class DelayExit(Exception):
    def __init__(self, delay, delay_step: float = 0.01):
        super().__init__("Delay adjustment finished by user.")
        self.delay = delay
        self.delay_step = delay_step
        
class DelayReset(Exception):
    def __init__(self, new_delay: float, delay_step: float = 0.01):
        super().__init__(f"Reset requested with delay {new_delay}")
        self.new_delay = new_delay
        self.delay_step = delay_step
        
class LabelExit(Exception):
    """Raised when user quits labeling/correction early."""
    def __init__(self, labeled_dict: dict):
        super().__init__("User exited labeling/correction.")
        self.labeled_dict = labeled_dict
        
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

def marker_p(c3d_labels, c3d_points, marker_name, subject=None):
    # find markers in the c3d files
    
    candidates = []
    marker_name_clean = marker_name.strip()
    if subject:
        candidates.append(f"{subject}:{marker_name_clean}")
    candidates.append(marker_name_clean)

    i = None
    for idx, label in enumerate(c3d_labels):
        label_clean = label.strip()
        if label_clean in candidates:
            i = idx
            break
    if i is None:
        raise ValueError(
            f"Marker name '{marker_name}' (with subject '{subject}') not found in c3d_labels. "
            f"Searched for {candidates}. Available labels: {[l.strip() for l in c3d_labels]}"
        )
    out = []
    for val in c3d_points:
        out.append(np.append(val[i][0:3], 1))
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

def reprojection_error(params, Ps, pc, K, dist):
    """
    params: 6 dimensions (rotation_vector [3], translation [3]) Optimization target
    Ps: (N, 3) 3D points (system coordinates)
    pc: (N, 2) 2D points (image coordinates)
    K: (3, 3) camera matrix
    dist: (5,) distortion parameters (OpenCV format)
    """
    rvec = params[:3]
    tvec = params[3:6]

    # Project 3D system points to 2D image points
    projected_points, _ = cv2.projectPoints(Ps, rvec, tvec, K, dist)
    projected_points = projected_points.reshape(-1, 2)

    return (projected_points - pc).ravel()  # Flatten and return


def estimate_Tstoc(Ps, pc, K, dist, init_params=None):
    """
    Ps: (N, 3) 3D points (system coordinates)
    pc: (N, 2) 2D points (image coordinates)
    K: (3, 3) camera matrix
    dist: (5,) distortion parameters
    return: (4, 4) system→camera coordinate transformation matrix Tstoc
    """

    # Initial values: zero rotation, zero translation
    if init_params is None:
        # If no initial parameters are provided, set them to zero
        init_params = np.zeros(6)

    # Minimize (choose LM method, etc.)
    res = least_squares(
        reprojection_error,
        init_params,
        args=(Ps, pc, K, dist),
        method='lm'  # Levenberg-Marquardt
    )

    rvec_opt = res.x[:3]
    tvec_opt = res.x[3:6]
    R_opt, _ = cv2.Rodrigues(rvec_opt)

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_opt
    T[:3, 3] = tvec_opt

    return T

class ViconProjector:
    def __init__(self, marker_names, c3d_data, points_3d, T_system_to_camera, 
                 T_world_to_system, K, cam_res, D=None, subject=None):
        self.marker_names = marker_names
        self.c3d_data = c3d_data
        self.points_3d = points_3d
        self.T_system_to_camera = T_system_to_camera
        self.T_world_to_system = T_world_to_system
        self.K = K
        self.cam_res = cam_res
        self.D = D
        self.subject = subject
        
        # Calculate projections once during initialization
        self._calculate_projections()
    
    def _clean_marker_name(self, marker_name):
        """Remove common prefixes from marker names for display purposes."""
        if ':' in marker_name:
            prefix, suffix = marker_name.split(':', 1)
            # Remove common prefixes like 'skeleton', 'wand', etc.
            return suffix.strip()
        return marker_name.strip()
    
    def _calculate_projections(self):
        """Calculate all marker projections once"""
        self.image_points = {}
        
        for mark_name in self.marker_names:
            ps = marker_p(self.c3d_data.point_labels, self.points_3d.values(), mark_name, subject=self.subject)
            ps_trans: np.ndarray = np.empty_like(ps)
            for i in range(len(self.T_world_to_system)):    
                ps_trans[i] = (self.T_system_to_camera @ self.T_world_to_system[i] @ ps[i].transpose()).transpose()
                ps_trans[i] = ps_trans[i] / ps_trans[i, [3]]
            ps_trans = ps_trans[:, :3]

            # Project to image plane
            ps_trans = ps_trans.astype(np.float64).reshape(-1, 1, 3)
            img_pts, _ = cv2.projectPoints(ps_trans, np.zeros(3), np.zeros(3), self.K, distCoeffs=self.D)
            img_pts = img_pts.reshape(-1, 2)
            self.image_points[mark_name] = img_pts
            
    # def project_marker_at_time(self, marker_name, timestamp, vicon_helper):
    #     """
    #     Interpolate 3D marker position at the given timestamp, transform to camera frame, and project to 2D.
    #     """
    #     # Interpolate 3D marker position
    #     interp = vicon_helper.get_vicon_points_interpolated({'points': [{marker_name: 0}], 'times': [timestamp]})
    #     if not interp['points'] or marker_name not in interp['points'][0]:
    #         return np.array([np.nan, np.nan])
    #     p_3d = interp['points'][0][marker_name]
    #     p_3d_h = np.append(p_3d, 1.0)

    #     # Find the closest frame for the transformation
    #     frame_idx = np.searchsorted(vicon_helper.frame_times, timestamp)
    #     if frame_idx >= len(self.T_world_to_system):
    #         frame_idx = len(self.T_world_to_system) - 1
    #     T = self.T_system_to_camera @ self.T_world_to_system[frame_idx]
    #     p_cam = T @ p_3d_h
    #     p_cam = p_cam[:3] / p_cam[3]

    #     # Project to 2D
    #     img_pt, _ = cv2.projectPoints(p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
    #     return img_pt[0, 0]
            
    def project_vicon_to_event_plane_dynamic(self, marker_t, delay, e_ts, e_us, e_vs, period, 
                   visualize=False, video_record=False, video_writer=None, marker_time_offset=0.0, delay_step=0.01):

        # Initialize dictionary to store synchronized projections
        synced_image_points = {
            name: {
                "points": [],
                "timestamps": []
            }
            for name in self.marker_names
        }

        # if visualize or video_record:
        current_delay = delay

        i_markers = 0
        i_events = 0
                    
        # Adjust timing based on offset
        tic_markers = marker_t[0] + marker_time_offset - current_delay # + period
        tic_events = e_ts[0] # + period
        
        # Create image once outside the loop
        img = np.ones(self.cam_res, dtype=np.uint8) * 255
        
        video_segment = []
                    
        if video_record:
            fps = int(1 / period)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter('tmp.mp4', fourcc, fps, (self.cam_res[1], self.cam_res[0]), isColor=False)

        while tic_markers < marker_t[-1] and tic_events < e_ts[-1]:
            # Store valid markers and their coordinates for this frame
            current_frame_markers = {}

            while i_markers < len(marker_t) and marker_t[i_markers] < tic_markers:

                for mark_name in self.marker_names:
                    if mark_name in self.image_points and i_markers < len(self.image_points[mark_name]):
                        u_coord = self.image_points[mark_name][i_markers][0]
                        v_coord = self.image_points[mark_name][i_markers][1]

                        if np.isfinite(u_coord) and np.isfinite(v_coord):
                            current_frame_markers[mark_name] = [u_coord, v_coord] # Store marker data for this frame

                i_markers += 1

            if current_frame_markers:
                for mark_name, (u_coord, v_coord) in current_frame_markers.items():
                    u = int(u_coord); v = int(v_coord)
                    if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                        cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                        cv2.putText(img, self._clean_marker_name(mark_name), (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)

            # Only add one timestamp per frame for each marker (not for every event)
            if current_frame_markers and i_events < len(e_ts):
                # Use the current event timestamp as representative for this frame
                frame_timestamp = e_ts[i_events] if i_events < len(e_ts) else tic_events
                for mark_name, coords in current_frame_markers.items():
                    synced_image_points[mark_name]["points"].append(coords)
                    synced_image_points[mark_name]["timestamps"].append(frame_timestamp)

            # Render events up to current event time
            while i_events < len(e_ts) and e_ts[i_events] < tic_events:
                uu = int(e_us[i_events]); vv = int(e_vs[i_events])
                if 0 <= uu < self.cam_res[1] and 0 <= vv < self.cam_res[0]:
                    img[vv, uu] = 0
                i_events += 1

            # GUI keys
            cv2.putText(img, f"Delay: {current_delay:.3f}s (step: {delay_step:.3f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
            cv2.putText(img, "Keys: <-/-> adjust delay, +/- adjust step, q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            
            # Add timestamp displays in top-right corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = 128
            
            # Marker timestamp
            marker_time_text = f"Marker: {tic_markers:.3f}s"
            (text_width, text_height), _ = cv2.getTextSize(marker_time_text, font, font_scale, thickness)
            x = self.cam_res[1] - text_width - 10
            y = text_height + 10
            cv2.putText(img, marker_time_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            
            # Event timestamp
            event_time_text = f"Event: {tic_events:.3f}s"
            (text_width, text_height), _ = cv2.getTextSize(event_time_text, font, font_scale, thickness)
            x = self.cam_res[1] - text_width - 10
            y = text_height + 35  # Position below marker timestamp
            cv2.putText(img, event_time_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

            if visualize:
                cv2.imshow('Projected Points', img)
                c = cv2.waitKey(int(500 * period))
                              
                # delay GUI, TODO: leave only in fix_delay
                if c == ord('=') or c == ord('+'):  # Right arrow -> increase by step                                       
                    current_delay += delay_step
                    print(f"Delay increased to: {current_delay:.3f}s (step: {delay_step:.3f}s)")
                    # Clear local buffers defensively (optional)
                    for d in synced_image_points.values():
                        d["points"].clear()
                        d["timestamps"].clear()
                    video_segment.clear()
                    raise DelayReset(current_delay, delay_step)
                elif c == ord('-'):  # Left arrow -> decrease by step
                    current_delay -= delay_step
                    print(f"Delay decreased to: {current_delay:.3f}s (step: {delay_step:.3f}s)")
                    for d in synced_image_points.values():
                        d["points"].clear()
                        d["timestamps"].clear()
                    video_segment.clear()
                    raise DelayReset(current_delay, delay_step)
                elif c == ord('l'):
                    delay_step += 0.001  # Increase step by 1ms
                    print(f"Delay step increased to: {delay_step:.3f}s")
                elif c == ord('k'):
                    delay_step = max(0.001, delay_step - 0.001)  # Decrease step by 1ms, minimum 1ms
                    print(f"Delay step decreased to: {delay_step:.3f}s")
                if c == ord('q'):
                    cv2.destroyAllWindows()
                    raise DelayExit(current_delay, delay_step)

            # Record video
            if video_record:
                video_segment.append(img.copy())

            # prepare next frame
            img = np.ones(self.cam_res, dtype=np.uint8) * 255
            tic_markers += period
            tic_events += period
        
        # Return both the original structure, video segment, final delay, and delay step
        return synced_image_points, video_segment, current_delay, delay_step

    def manual_rotation_adjustment(self, marker_t, delay, e_ts, e_us, e_vs, period,
                                   R_init=None, tvec=None, visualize=True, video_record=True, video_writer=None,
                                   chosen_one=None, angle_step=1.0, marker_time_offset=0.0):

        own_writer = None
        H, W = self.cam_res[0], self.cam_res[1]
        if video_record and video_writer is None:
            fps = max(1, int(round(1.0 / period)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            own_writer = cv2.VideoWriter("manual_rotation_tmp.mp4", fourcc, fps, (W, H), isColor=False)
       
        # Create a copy of the current transformation for adjustment
        current_T = self.T_system_to_camera.copy()
        selected_angle = 0
       
        # if visualize:
        current_delay = delay
        paused = False
        recalc_needed = False
 
        # TODO: check and remove eventually useless stuff
        if R_init is not None:
            print(f"Using provided R_init: {R_init}")
            if isinstance(R_init, list):
                # Convert list to rotation matrix
                angles_zyx = np.array([R_init[2], R_init[1], R_init[0]], dtype=np.float64)  # [yaw, pitch, roll]
                R_matrix = Rotation.from_euler('zyx', angles_zyx, degrees=True).as_matrix()
                current_T[:3, :3] = R_matrix
                print(f"Converted to rotation matrix: {R_matrix}")
                recalc_needed = True
            else:
                current_T[:3, :3] = R_init
               
            try:
                angles_zyx = Rotation.from_matrix(current_T[:3, :3]).as_euler('zyx', degrees=True)
                print(f"Extracted Euler angles (ZYX): {angles_zyx}")
            except Exception as e:
                print(f"Error extracting Euler angles: {e}")
                angles_zyx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            angles_zyx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            print("No R_init provided, using [0.0, 0.0, 0.0]")
               
        if tvec is not None:
            current_T[:3, 3] = tvec
                   
        Rot_deg = np.array([angles_zyx[2], angles_zyx[1], angles_zyx[0]], dtype=np.float64)
 
        i_markers = 0
        i_events = 0
        tic_markers = marker_t[0] + marker_time_offset - current_delay # + period
        tic_events = e_ts[0] # + period
        img = np.ones(self.cam_res, dtype=np.uint8) * 255

        while tic_markers < marker_t[-1] and tic_events < e_ts[-1]:
            # Draw markers and events only when not paused
            if not paused:
                # Store valid markers and their coordinates for this frame
                current_frame_markers = {}

                while i_markers < len(marker_t) and marker_t[i_markers] < tic_markers:
                    for mark_name in self.marker_names:
                        if mark_name in self.image_points and i_markers < len(self.image_points[mark_name]):
                            u_coord = self.image_points[mark_name][i_markers][0]
                            v_coord = self.image_points[mark_name][i_markers][1]

                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                current_frame_markers[mark_name] = [u_coord, v_coord] # Store marker data for this frame

                    i_markers += 1

                if current_frame_markers:
                    for mark_name, (u_coord, v_coord) in current_frame_markers.items():
                        u = int(u_coord); v = int(v_coord)
                        if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                            cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                            cv2.putText(img, self._clean_marker_name(mark_name), (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)

                # Render events up to current event time
                while i_events < len(e_ts) and e_ts[i_events] < tic_events:
                    uu = int(e_us[i_events]); vv = int(e_vs[i_events])
                    if 0 <= uu < self.cam_res[1] and 0 <= vv < self.cam_res[0]:
                        img[vv, uu] = 0
                    i_events += 1                
           
            cv2.putText(img, f"Rot (deg) roll={Rot_deg[0]:+.2f} pitch={Rot_deg[1]:+.2f} yaw={Rot_deg[2]:+.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
            cv2.putText(img, "Keys: space=start/stop | enter = select roll/pitch/yaw | +/- = increase/decrease angle value | q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            cv2.putText(img, "Currently modifying: " + ['roll', 'pitch', 'yaw'][selected_angle] + " by a factor of: " + str(angle_step) + " degrees", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
 
 
            # Record video frame if enabled
            if video_record:
                if video_writer is not None:
                    video_writer.write(img)
                elif own_writer is not None:
                    own_writer.write(img)

            cv2.imshow('Manual Rotation', img)
            c = cv2.waitKey(int(1000 * period))
 
            # Handle input
            if ' ' == chr(c & 255):  # space bar
                paused = not paused
                if paused:
                    print(f"Space pressed, visualization paused at markers: {tic_markers:.3f}s, events: {tic_events:.3f}s")
                else:
                    print(f"Space pressed, visualization resumed from markers: {tic_markers:.3f}s, events: {tic_events:.3f}s")
 
            # Manual rotation adjustment GUI
            # Enter key pressed, change the angle to change
            elif c == 13:
                selected_angle = (selected_angle + 1) % 3
                print(f"Selected rotation axis: {['roll', 'pitch', 'yaw'][selected_angle]}")
               
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
 
                # First, redraw events up to current time
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                
                # Then, draw markers on top
                if 0 <= i_markers < len(marker_t):
                    for mark_name in self.marker_names:
                        if i_markers < len(self.image_points[mark_name]):
                            u_coord = self.image_points[mark_name][i_markers][0]
                            v_coord = self.image_points[mark_name][i_markers][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)
                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
 
            # - key pressed -> decrease angle by angle step
            elif c == ord('-'):
                Rot_deg[selected_angle] -= angle_step
                recalc_needed = True
            # +/= key pressed -> increase angle by angle step
            elif c == ord('+') or c == ord('='):
                Rot_deg[selected_angle] += angle_step
                recalc_needed = True
 
            # Modify angle_step
            elif c == ord('l'):
                angle_step += 0.5
                print(f"Angle step increased to: {angle_step:.3f}")
               
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
 
                # First, redraw events up to current time
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                
                # Then, draw markers on top
                if 0 <= i_markers < len(marker_t):
                    for mark_name in self.marker_names:
                        if i_markers < len(self.image_points[mark_name]):
                            u_coord = self.image_points[mark_name][i_markers][0]
                            v_coord = self.image_points[mark_name][i_markers][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)
                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
               
            elif c == ord('k'):
                angle_step = max(0.5, angle_step - 0.5)
                print(f"Angle step decreased to: {angle_step:.3f}")
               
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
 
                # First, redraw events up to current time
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                
                # Then, draw markers on top
                if 0 <= i_markers < len(marker_t):
                    for mark_name in self.marker_names:
                        if i_markers < len(self.image_points[mark_name]):
                            u_coord = self.image_points[mark_name][i_markers][0]
                            v_coord = self.image_points[mark_name][i_markers][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)
                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
 
            # quit and save current rotation
            elif c == ord('q') or c == 27:
                cv2.destroyAllWindows()
                r_vec = Rotation.from_euler('zyx', [Rot_deg[2], Rot_deg[1], Rot_deg[0]], degrees=True).as_rotvec()
                self.T_system_to_camera[:3, :3] = cv2.Rodrigues(r_vec)[0]
                self._calculate_projections()
                raise RotationExit(r_vec)
           
                # r_vec = Rotation.from_euler('zyx', [Rot_deg[2], Rot_deg[1], Rot_deg[0]], degrees=True).as_rotvec()
                # # Update the class transformation matrix
                # self.T_system_to_camera[:3, :3] = cv2.Rodrigues(r_vec)[0]
                # # Recalculate projections with new transformation
                # self._calculate_projections()
                # return r_vec
 
            # If rotation changed, recompute projections live and update frame
            if recalc_needed:
                try:
                    # Build updated rotation
                    R_new = Rotation.from_euler('zyx', [Rot_deg[2], Rot_deg[1], Rot_deg[0]], degrees=True).as_matrix()
                    current_T[:3, :3] = R_new
 
                    # Temporarily update transformation and recalculate projections
                    self.T_system_to_camera = current_T
                    self._calculate_projections()
 
                    # Redraw current frame with both markers and events
                    img = np.ones(self.cam_res, dtype=np.uint8) * 255
                   
                    # First, redraw events up to current time
                    event_time_end = tic_markers + current_delay
                    event_time_start = event_time_end - period
                   
                    temp_i_events = 0
                    while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                        temp_i_events += 1
                    while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                        if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                            img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                        temp_i_events += 1
                   
                    # Then, draw markers on top
                    if 0 <= i_markers < len(marker_t):
                        for mark_name in self.marker_names:
                            if i_markers < len(self.image_points[mark_name]):
                                u_coord = self.image_points[mark_name][i_markers][0]
                                v_coord = self.image_points[mark_name][i_markers][1]
                               
                                if np.isfinite(u_coord) and np.isfinite(v_coord):
                                    u = int(u_coord)
                                    v = int(v_coord)
                                    if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                        cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                        cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                               
                    print(f"Recomputed projections: roll={Rot_deg[0]:.2f}, pitch={Rot_deg[1]:.2f}, yaw={Rot_deg[2]:.2f}")
 
                    if chosen_one is not None and chosen_one in self.image_points:
                        idx = max(0, min(i_markers - 1, len(marker_t) - 1))
                        if idx < len(self.image_points[chosen_one]):
                            uv = self.image_points[chosen_one][idx]
                            print(f"[recalc] marker='{chosen_one}' frame_idx={idx} image_uv={tuple(uv)}")

                            clean_name = self._clean_marker_name(chosen_one)
                            position_text = f"Chosen: {clean_name} ({uv[0]:.1f}, {uv[1]:.1f})"
                            cv2.putText(img, position_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 64, 2)
 
                except Exception as e:
                    print("Error recomputing projections:", e)
 
                recalc_needed = False
 
            # Update timers (only when not paused)
            if not paused:
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                tic_markers += period
                tic_events += period

        if own_writer is not None:
            own_writer.release()        
 
        #cv2.destroyAllWindows()
        r_vec = Rotation.from_euler('zyx', [Rot_deg[2], Rot_deg[1], Rot_deg[0]], degrees=True).as_rotvec()
        # Update the class transformation matrix
        self.T_system_to_camera[:3, :3] = cv2.Rodrigues(r_vec)[0]
        # Recalculate projections with new transformation
        self._calculate_projections()
        return r_vec  

    def fix_delay(self, marker_t, delay, e_ts, e_us, e_vs, period, 
              visualize=True, marker_time_offset=0.0, video_record=False, video_writer=None, delay_step=0.01):
        # Project points from Vicon to event plane using a transformation matrix for each frame
        image_points = {}

        for mark_name in self.marker_names:
            ps = marker_p(self.c3d_data.point_labels, self.points_3d.values(), mark_name, subject=self.subject)
            ps_trans: np.ndarray = np.empty_like(ps)
            for i in range(len(self.T_world_to_system)):    
                ps_trans[i] = (self.T_system_to_camera @ self.T_world_to_system[i] @ ps[i].transpose()).transpose()
                ps_trans[i] = ps_trans[i] / ps_trans[i, [3]]
            ps_trans = ps_trans[:, :3]

            # Project to image plane
            ps_trans = ps_trans.astype(np.float64).reshape(-1, 1, 3)
            img_pts, _ = cv2.projectPoints(ps_trans, np.zeros(3), np.zeros(3), self.K, distCoeffs=self.D)
            img_pts = img_pts.reshape(-1, 2)
            image_points[mark_name] = img_pts

        # if visualize:
        current_delay = delay
        paused = False

        i_markers = 0
        i_events = 0
        
        tic_markers = marker_t[0] + marker_time_offset - current_delay # + period
        tic_events = e_ts[0] # + period
        
        img = np.ones(self.cam_res, dtype = np.uint8)*255
        
        # Loop for image update
        while tic_markers < marker_t[-1] and tic_events < e_ts[-1]:
            # Create images with projected 2D points (only when not paused)
            if not paused:
                # Store valid markers and their coordinates for this frame
                current_frame_markers = {}

                while i_markers < len(marker_t) and marker_t[i_markers] < tic_markers:
                    for mark_name in self.marker_names:
                        if mark_name in self.image_points and i_markers < len(self.image_points[mark_name]):
                            u_coord = self.image_points[mark_name][i_markers][0]
                            v_coord = self.image_points[mark_name][i_markers][1]

                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                current_frame_markers[mark_name] = [u_coord, v_coord] # Store marker data for this frame

                    i_markers += 1

                if current_frame_markers:
                    for mark_name, (u_coord, v_coord) in current_frame_markers.items():
                        u = int(u_coord); v = int(v_coord)
                        if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                            cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                            cv2.putText(img, self._clean_marker_name(mark_name), (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)

                # Render events up to current event time
                while i_events < len(e_ts) and e_ts[i_events] < tic_events:
                    uu = int(e_us[i_events]); vv = int(e_vs[i_events])
                    if 0 <= uu < self.cam_res[1] and 0 <= vv < self.cam_res[0]:
                        img[vv, uu] = 0
                    i_events += 1           

            # Add GUI text
            cv2.putText(img, f"Delay: {current_delay:.3f}s (step: {delay_step:.3f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
            cv2.putText(img, "Keys: +/- decrease/increase delay, k/l adjust step", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            cv2.putText(img, "Keys: <-/-> navigate frames, space bar stop/start, q=quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            
            # Add timestamp display
            marker_time_text = f"Marker: {tic_markers:.3f}s"
            event_time_text = f"Event: {tic_events:.3f}s"
            cv2.putText(img, marker_time_text, (self.cam_res[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            cv2.putText(img, event_time_text, (self.cam_res[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)

            # Record video frame if enabled
            if video_record and video_writer is not None:
                video_writer.write(img)

            # Visualize
            cv2.imshow('Fix Delay', img)
            c = cv2.waitKey(int(period * 1000))
            
            # Handle GUI input
            if ' ' == chr(c & 255):  # space bar
                paused = not paused
                if paused:
                    print(f"Space pressed, visualization paused at markers: {tic_markers:.3f}s, events: {tic_events:.3f}s")
                else:
                    # tic_events = tic_markers + current_delay
                    print(f"Space pressed, visualization resumed from markers: {tic_markers:.3f}s, events: {tic_events:.3f}s")

            # Navigate frames (only when paused is True)
            elif (c == 81 or c == 37) and paused:  # Left arrow -> go to previous frame
                
                tic_events = max(tic_events - delay_step, e_ts[0])
                tic_markers = tic_events - current_delay
                    
                i_events = max(0, i_events - 1)
                
                # Redraw frame immediately
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                
                # Extract markers for current frame
                if i_markers < len(marker_t) and i_markers >= 0:
                    for mark_name in self.marker_names:
                        if i_markers < len(image_points[mark_name]):
                            u_coord = image_points[mark_name][i_markers][0]
                            v_coord = image_points[mark_name][i_markers][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)
                                
                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)

                # Calculate event time window based on marker time + current delay
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                # Find and render events within the adjusted time window
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                
                # Show all events within the time window
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                
                print(f"Moved to previous frame: markers at {tic_markers:.3f}s, events at {event_time_end:.3f}s (delay: {current_delay:.3f}s)")
                
            elif (c == 83 or c == 39) and paused:  # Right arrow -> go to next frame
                
                tic_events += delay_step
                tic_markers = tic_events - current_delay
                
                # i_events += 1 # ????
                
                # Redraw frame immediately
                img = np.ones(self.cam_res, dtype=np.uint8) * 255

                # Extract markers for current frame
                if i_markers < len(marker_t) and marker_t[i_markers] < tic_markers:
                    for mark_name in self.marker_names:
                        if i_markers < len(image_points[mark_name]):
                            u_coord = image_points[mark_name][i_markers][0]
                            v_coord = image_points[mark_name][i_markers][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)
                                
                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                    i_markers += 1
                
                # Calculate event time window based on marker time + current delay
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                # Find and render events within the adjusted time window
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1

                # Show all events within the time window
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                
                print(f"Moved to next frame: markers at {tic_markers:.3f}s, events at {event_time_end:.3f}s (delay: {current_delay:.3f}s)")

            # Adjust delay - KEY FIX: Immediately redraw frame and return current_delay
            elif c == ord('+') or c == ord('='):  # l -> increase delay by step                
                current_delay += delay_step
                print(f"Delay increased to: {current_delay:.3f}s (step: {delay_step:.3f}s)")
                                
                # Immediately redraw frame with new delay
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                
                # Render markers for current frame
                current_marker_idx = max(0, min(i_markers, len(marker_t) - 1))
                if current_marker_idx < len(marker_t):
                    for mark_name in self.marker_names:
                        if current_marker_idx < len(image_points[mark_name]):
                            u_coord = image_points[mark_name][current_marker_idx][0]
                            v_coord = image_points[mark_name][current_marker_idx][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)

                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                
                # Calculate and render events with new delay
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                    
                # Update tic_events for continuous playback
                tic_events = tic_markers + current_delay
                print(f"Updated frame with new delay: markers at {tic_markers:.3f}s, events at {event_time_end:.3f}s")
                        
            elif c == ord('-'):  # k -> decrease delay by step
                                
                current_delay -= delay_step
                print(f"Delay decreased to: {current_delay:.3f}s (step: {delay_step:.3f}s)")
                                
                # Immediately redraw frame with new delay
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                
                # Render markers for current frame
                current_marker_idx = max(0, min(i_markers, len(marker_t) - 1))
                if current_marker_idx < len(marker_t):
                    for mark_name in self.marker_names:
                        if current_marker_idx < len(image_points[mark_name]):
                            u_coord = image_points[mark_name][current_marker_idx][0]
                            v_coord = image_points[mark_name][current_marker_idx][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)

                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                
                # Calculate and render events with new delay
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                                        
                # Update tic_events for continuous playback
                tic_events = tic_markers + current_delay
                print(f"Updated frame with new delay: markers at {tic_markers:.3f}s, events at {event_time_end:.3f}s")
                
            elif c == ord('l'):
                delay_step += 0.001  # Increase step by 1ms
                print(f"Delay step increased to: {delay_step:.3f}s")
                
                # Immediately redraw frame with new delay
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                
                # Render markers for current frame
                current_marker_idx = max(0, min(i_markers, len(marker_t) - 1))
                if current_marker_idx < len(marker_t):
                    for mark_name in self.marker_names:
                        if current_marker_idx < len(image_points[mark_name]):
                            u_coord = image_points[mark_name][current_marker_idx][0]
                            v_coord = image_points[mark_name][current_marker_idx][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)

                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                
                # Calculate and render events with new delay
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1
                
            elif c == ord('k'):
                delay_step = max(0.001, delay_step - 0.001)  # Decrease step by 1ms, minimum 1ms
                print(f"Delay step decreased to: {delay_step:.3f}s")
                
                # Immediately redraw frame with new delay
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                
                # Render markers for current frame
                current_marker_idx = max(0, min(i_markers, len(marker_t) - 1))
                if current_marker_idx < len(marker_t):
                    for mark_name in self.marker_names:
                        if current_marker_idx < len(image_points[mark_name]):
                            u_coord = image_points[mark_name][current_marker_idx][0]
                            v_coord = image_points[mark_name][current_marker_idx][1]
                            
                            if np.isfinite(u_coord) and np.isfinite(v_coord):
                                u = int(u_coord)
                                v = int(v_coord)

                                if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
                                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                
                # Calculate and render events with new delay
                event_time_end = tic_markers + current_delay
                event_time_start = event_time_end - period
                
                temp_i_events = 0
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_start:
                    temp_i_events += 1
                
                while temp_i_events < len(e_ts) and e_ts[temp_i_events] < event_time_end:
                    if 0 <= e_vs[temp_i_events] < self.cam_res[0] and 0 <= e_us[temp_i_events] < self.cam_res[1]:
                        img[e_vs[temp_i_events], e_us[temp_i_events]] = 0
                    temp_i_events += 1 

            elif c == ord('r'):         
                print("Resetting the sequence from the beginning...")
                raise DelayReset(current_delay, delay_step)
                 
            elif c == ord('q') or c == 27:  # quit
                print("Delay adjustment completed")
                raise DelayExit(current_delay, delay_step)
            
                # cv2.destroyAllWindows()
                # return current_delay  # Return the adjusted delay

            # Reset image and update timer (only when not paused)
            if not paused:                
                img = np.ones(self.cam_res, dtype=np.uint8) * 255
                tic_markers += period
                tic_events += period
                
        return current_delay  # Return the adjusted delay
            
class DvsLabeler:
    # functions relative to the labeling of the sequences
    
    def __init__(self, img_shape, subject=None, points_dict=None):
        self.img_shape = img_shape
        self.labels_done = False
        self.labeled_dict = None
        self.subject = subject
        self.points_dict = points_dict
        
    def _merge_window_into_accumulated(self, window_dict):
        """Merge current window results into the accumulated dictionary"""
        if not window_dict or 'points' not in window_dict or 'times' not in window_dict:
            return
            
        # Only merge frames that have actual corrections (non-empty points_dict)
        corrections_added = 0
        for i, points_dict in enumerate(window_dict['points']):
            if points_dict:  # Only add if there are actual corrections
                self.points_dict['points'].append(points_dict)
                self.points_dict['times'].append(window_dict['times'][i])
                corrections_added += 1
        
        if corrections_added > 0:
            print(f"Merged {corrections_added} corrections from window. Total accumulated: {len(self.points_dict['points'])}")

    def merge_labels(self, new_dict):
        """Legacy method - now uses _merge_window_into_accumulated"""
        if new_dict:
            self._merge_window_into_accumulated(new_dict)
    
    def label_data(self, e_ts, e_us, e_vs, period, label_tag_file: str = None):
        # Go though every event frame and call function to do the labelling.
        
        window_dict = {'points': [], 'times': []}
        # dvs_frames = []
        
        ft = e_ts[0]
        img = np.ones(self.img_shape, dtype=np.uint8) * 255
        i = 0
                
        while i < len(e_ts):
            if e_ts[i] >= ft:
                # Show the current event frame
                img[e_vs[i], e_us[i]] = 0
                
                success, process_continue, points_dict, frame = self.label_frame(img, ft, label_tag_file)
                if not success:
                    img = np.ones(self.img_shape, dtype = np.uint8)*255
                else:
                    # dvs_frames.append(frame)
                    window_dict['points'].append(points_dict)
                    window_dict['times'].append(float(ft))
                    
                    img = np.ones(self.img_shape, dtype = np.uint8)*255
                    
                if not process_continue:
                    # Before exiting, merge current window results into accumulated dictionary
                    self._merge_window_into_accumulated(window_dict)
                    raise LabelExit(self.points_dict)
                
                ft = ft + period    # maybe 2*period, just to skip some frames as they are a lot
                    
            img[e_vs[i],e_us[i]] = 0
            i += 1
            
        # cv2.destroyAllWindows()
        
        # At the end of the window, merge results into accumulated dictionary
        self._merge_window_into_accumulated(window_dict)
        
        # Update labeled_dict to point to accumulated results
        self.labeled_dict = self.points_dict
        
        # Only mark as done if we have actual labels
        has_labels = any(len(points_dict) > 0 for points_dict in self.points_dict['points'])
        if has_labels:
            self.labels_done = True
            print(f"Window completed. Total accumulated labels: {len(self.points_dict['points'])}")
        else:
            print("No labels in this window")
        
        return window_dict
    
    def save_labeled_points(self, out_file: str):
        # Save labeled points to a YAML file.
        
        assert self.labels_done is True
        
        # Check if there are actually points to save
        has_points = (self.labeled_dict and 
                     'points' in self.labeled_dict and 
                     any(len(points_dict) > 0 for points_dict in self.labeled_dict['points']))
        
        if has_points:
            with open(out_file, 'w') as yaml_file:
                yaml.dump(self.labeled_dict, yaml_file, default_flow_style=False)
            print(f"Saved corrected points at: {out_file}")
        else:
            print(f"No points to save. YAML file not created/overwritten: {out_file}")
        
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

    def select_label_terminal(self, marker_labels):
        # Terminal-based label selection
        
        print("\nAvailable labels:")
        for i, label in enumerate(marker_labels):
            print(f"  {i}: {label}")
        
        while True:
            try:
                user_input = input("Select label (enter number or exact label name): ").strip()
                
                # Try to parse as number first
                if user_input.isdigit():
                    idx = int(user_input)
                    if 0 <= idx < len(marker_labels):
                        return str(idx)
                    else:
                        print(f"Invalid index. Please enter a number between 0 and {len(marker_labels)-1}")
                        continue
                
                # Try to match exact label name
                if user_input in marker_labels:
                    return user_input
                
                # If nothing matches, ask again
                print("Invalid input. Please enter a valid number or exact label name.")
                
            except KeyboardInterrupt:
                print("\nLabeling cancelled by user.")
                return None
            except Exception as e:
                print(f"Error: {e}. Please try again.")
                continue

    #TODO: CLEAN CODE!!!!!!
    def label_frame(self, frame: np.ndarray, timestamp: float = None, label_tag_file: str = None) -> Tuple[bool, bool, dict, np.ndarray]:
        # allow user to label points in the current frame.
        
        points = []
        finished = False
        points_dict = {}
        process_continue: bool = True

        dirname = os.path.dirname(__file__)
        
        # print("dirname", dirname)
        filename: str
        if label_tag_file is not None:
            filename = os.path.join(dirname, label_tag_file)   # check later for modification of yaml file
            # print("Using custom label tag file:", filename)
        else:
            filename = os.path.join(dirname, '../scripts/config/labels_tags.yml')       # TODO: else, create one by reading all names from c3d file?
        with open(filename) as f:
            marker_labels = yaml.load(f, Loader=yaml.Loader)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # label_val = self.select_label_terminal(marker_labels)
                label_val = self.select_label_tkinter(marker_labels)
                if label_val is None:
                    print("Labeling aborted by user.")
                    self.abort_labeling = True
                    return
                try:
                    marker_name = marker_labels[int(label_val)]
                except (ValueError, IndexError):
                    marker_name = label_val
                # Only add subject prefix if the marker name doesn't already contain one
                if self.subject is not None and ':' not in marker_name:
                    marker_name = f"{self.subject}:{marker_name}"
                points.append([x, y])
                points_dict[marker_name] = {"x": int(x), "y": int(y)}
                print(f"current labels: {points_dict}")

        self.abort_labeling = False
        cv2.imshow("image", frame)
        cv2.setMouseCallback('image', on_click)

        while not finished:
            img = np.copy(frame)
            
            cv2.putText(img, "Keys: Space bar skip frame, S save labels for this frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            cv2.putText(img, "Backspace delete last label, ESC=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
            
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
                return False, process_continue, None, None
            elif c == ord('s'):     # s key
                # save labeled points for the current frame
                finished = True
                return True, process_continue, points_dict, img
            elif c == ord('q') or c == 27:   # ESC key
                print("ESC pressed, stopping labeling")
                process_continue = False
                finished = True
                return True, process_continue, points_dict, img
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

        return True, process_continue, points_dict, img

    # TODO: improve on GUI, maybe show name of the markers only when mouse hover them or is close to them or something
    def correct_data(
        self, e_ts, e_us, e_vs, period,
        marker_names, c3d_data, points_3d, marker_t,
        T_system_to_camera, T_world_to_system, K, cam_res, delay,
        D=None, marker_time_offset=0.0
    ) -> Tuple[bool, bool, dict, np.ndarray]:
        
        window_dict = {'points': [], 'times': []}
        process_continue: bool = True

        ft = e_ts[0]
        img = np.ones(self.img_shape, dtype=np.uint8) * 255
        i = 0
        frame_idx = 0

        while i < len(e_ts):
            if e_ts[i] >= ft:
                img[e_vs[i], e_us[i]] = 0

                # Find corresponding marker frame index
                while frame_idx < len(marker_t) and marker_t[frame_idx] < ft - delay:
                    frame_idx += 1

                if frame_idx < len(T_world_to_system):
                    success, process_continue, points_dict, frame = self.correct_labels(
                        img, ft, period, T_system_to_camera, T_world_to_system,
                        marker_names, c3d_data, points_3d, marker_t,
                        frame_idx, K, cam_res, D, marker_time_offset=marker_time_offset
                    )

                    if success and points_dict:
                        window_dict['points'].append(points_dict)
                        window_dict['times'].append(float(ft))

                    img = np.ones(self.img_shape, dtype=np.uint8) * 255

                    if not process_continue:
                        # Merge before exiting
                        self._merge_window_into_accumulated(window_dict)
                        raise LabelExit(self.points_dict)

                ft += period

            img[e_vs[i], e_us[i]] = 0
            i += 1

        # Merge results from this window into the accumulated global dict
        self._merge_window_into_accumulated(window_dict)

        # Only mark as done if we have actual corrections
        has_corrections = any(len(points_dict) > 0 for points_dict in window_dict['points'])
        if has_corrections:
            self.labels_done = True
            print(f"Window completed. Total accumulated labels: {len(self.points_dict['points'])}")
        else:
            print("No corrections in this window.")

        return window_dict
    
    # TODO: make GUI better, more user friendly
    def correct_labels(
        self, frame, timestamp, period, T_system_to_camera, T_world_to_system,
        marker_names, c3d_data, points_3d, marker_t, frame_idx, K, cam_res,
        D=None, marker_time_offset=0.0
    ) -> Tuple[bool, bool, dict, np.ndarray]:
        """Interactive correction GUI with improved visuals, readability, and UX."""
        
        image_points = {}
        projections_calculated = False

        corrected_points = []
        finished = False
        corrected_points_dict = {}
        process_continue: bool = True

        # Load marker labels file
        # dirname = os.path.dirname(__file__)
        # filename = os.path.join(dirname, "../scripts/config/labels_giorgia.yml")
        # with open(filename) as f:
        #     marker_labels = yaml.load(f, Loader=yaml.Loader)

        # State variables
        waiting_for_position = True
        show_projected_markers = False
        selected_position = None

        def calculate_projections():
            """Calculate projections only when needed"""
            nonlocal image_points, projections_calculated

            if projections_calculated:
                return

            for mark_name in marker_names:
                ps = marker_p(c3d_data.point_labels, points_3d.values(), mark_name)
                ps_trans: np.ndarray = np.empty_like(ps)

                for i in range(len(T_world_to_system)):
                    ps_trans[i] = (T_system_to_camera @ T_world_to_system[i] @ ps[i].transpose()).transpose()
                    ps_trans[i] = ps_trans[i] / ps_trans[i, [3]]
                ps_trans = ps_trans[:, :3]

                ps_trans = ps_trans.astype(np.float64).reshape(-1, 1, 3)
                img_pts, _ = cv2.projectPoints(ps_trans, np.zeros(3), np.zeros(3), K, distCoeffs=D)
                img_pts = img_pts.reshape(-1, 2)

                if frame_idx < len(img_pts):
                    image_points[mark_name] = img_pts[frame_idx]
                else:
                    print(f"Warning: frame_idx {frame_idx} exceeds available projections for marker {mark_name}")
                    continue

            projections_calculated = True
            print("Projections calculated.")

        def on_click(event, x, y, flags, param):
            nonlocal waiting_for_position, show_projected_markers, selected_position

            if event == cv2.EVENT_LBUTTONDOWN:
                if waiting_for_position:
                    # First click: select event position
                    selected_position = (x, y)
                    calculate_projections()
                    show_projected_markers = True
                    waiting_for_position = False
                    print(f"Position selected: ({x}, {y}). Now click on projected marker.")
                else:
                    # Second click: assign to nearest projected marker
                    if image_points:
                        min_distance = float("inf")
                        selected_marker = None

                        for mark_name, pt in image_points.items():
                            distance = np.linalg.norm(np.array([x, y]) - pt)
                            if distance < min_distance:
                                min_distance = distance
                                selected_marker = mark_name

                        if selected_marker:
                            clean_marker_name = selected_marker.strip()     # .split(":")[-1]
                            corrected_points.append([selected_position[0], selected_position[1]])
                            corrected_points_dict[clean_marker_name] = {"x": int(selected_position[0]), "y": int(selected_position[1])}
                            # print(f"Assigned marker '{clean_marker_name}' → ({selected_position[0]}, {selected_position[1]})")
                            print(f"Current labels: {corrected_points_dict}")
                        else:
                            print("No nearby marker found.")

                    waiting_for_position = True
                    show_projected_markers = False
                    selected_position = None

        cv2.imshow("Correct Markers", frame)
        cv2.setMouseCallback("Correct Markers", on_click)

        while not finished:
            img = np.copy(frame)
            
            # Add GUI text
            cv2.putText(img, "Keys: SPACE=skip | S=save | BACKSPACE=undo | q=quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)

            # Draw projected markers
            if show_projected_markers and projections_calculated:
                for mark_name, pt in image_points.items():
                    clean_name = mark_name.split(":")[-1].strip()
                    u, v = int(pt[0]), int(pt[1])
                        
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (u, v), 4, (0, 0, 255), 2)
                        cv2.putText(img, clean_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))

            # Draw corrected points, check if this is actually doing anything
            for p in corrected_points:
                cv2.circle(img, tuple(np.asarray(p, dtype=int)), 4, (255, 0, 0), -1, cv2.LINE_AA)

            # Draw selected point position
            if not waiting_for_position and selected_position:
                cv2.circle(img, selected_position, 3, (0, 255, 0), -1, cv2.LINE_AA)

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

            instruction_text = "Click where you want to place a marker" if waiting_for_position \
                            else "Click on the projected marker to assign"
            cv2.putText(img, instruction_text, (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow("Correct Markers", img)
            c = cv2.waitKey(100)
            cv2.setMouseCallback("Correct Markers", on_click)

            # if c != -1:
            if c == ord(' '):  # skip
                print("space pressed, skipping")
                finished = True
                return False, process_continue, None, None
            elif c == ord('s'):  # save
                finished = True
                return True, process_continue, corrected_points_dict, img
            elif c == 27 or c == ord('q'):  # ESC
                print("ESC pressed, stopping correction")
                process_continue = False
                finished = True
                return True, process_continue, corrected_points_dict, img
            elif c == 8:  # backspace
                if corrected_points:
                    removed_point = corrected_points.pop()
                    if corrected_points_dict:
                        last_marker = list(corrected_points_dict.keys())[-1]
                        corrected_points_dict.pop(last_marker)
                    print(f"Removed last corrected marker {removed_point}.")

        cv2.destroyAllWindows()
        return True, process_continue, corrected_points_dict, img

class ViconHelper:
    # functions relative to the extraction of the vicon data from c3d files.
    
    def __init__(self, frame_times, points_3d, delay, frame_count, point_rate, point_labels, camera_markers, filter_camera_markers, user_camera_markers=None):
        self.frame_times = frame_times
        self.points_3d = points_3d
        self.delay = delay
        self.frame_count = frame_count
        self.point_rate = point_rate
        self.point_labels = [l.strip() for l in point_labels]
        self.camera_markers = camera_markers
        self.filter_camera_markers = filter_camera_markers
        self.user_specified_camera_markers = user_camera_markers  # User-specified camera markers
        
        self.calculate_frame_times()
        
        self.marker_T_vector = {}
        
        if not self.camera_markers or not self.user_specified_camera_markers:
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

    # atual one
    # def get_frame_time(self, times):
    #     # attribute each label an id based on the timestamp.
        
    #     frame_ids = []
    #     for t in times:
    #         idx = np.searchsorted(self.frame_times, t)
    #         if idx >= len(self.frame_times):
    #             break
    #         frame_ids.append(idx)
    #     return frame_ids

    #new_Dataset_giorgia
    def get_frame_time(self, times):
        # Find closest frame for each timestamp using binary search + closest matching
        # This matches the approach used in calculate_projection_error for consistency
        
        frame_ids = []
        for t in times:
            # Use binary search to find insertion position
            insert_pos = np.searchsorted(self.frame_times, t)
            
            # Find closest frame by comparing candidates on both sides
            candidates = []
            for idx in [insert_pos - 1, insert_pos]:
                if 0 <= idx < len(self.frame_times):
                    frame_time = self.frame_times[idx]
                    time_diff = abs(frame_time - t)
                    candidates.append((time_diff, idx))
            
            if candidates:
                # Select frame with minimum time difference
                _, closest_idx = min(candidates)
                frame_ids.append(closest_idx)
            else:
                # Fallback if no valid candidates (shouldn't happen in normal cases)
                if insert_pos < len(self.frame_times):
                    frame_ids.append(insert_pos)
                else:
                    break
                    
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

        # actual:
        out['times'] = np.array([self.frame_times[idx] for idx in frames_id])
        
        # # new_dataset_giorgia:
        # ###
        # def get_frame_time_safe(idx):
        #     # Convert 1-based C3D index to 0-based frame_times index
        #     time_idx = idx - 1 if idx > 0 else 0
        #     # Clamp to valid range to avoid IndexError
        #     time_idx = max(0, min(time_idx, len(self.frame_times) - 1))
        #     return self.frame_times[time_idx]
        
        # out['times'] = np.array([get_frame_time_safe(idx) for idx in frames_id])
        # ###
        out['frame_ids'] = frames_id
                
        return out
    
    def filter_dict_labels(self, old_dict, labels):
        return {key: old_dict[key] for key in labels if key in old_dict}
    
    def filter_pose(self, x:np.ndarray, order:int = 6, fs:int = 100.0, cutoff:int = 3) -> np.ndarray:
        out = np.empty_like(x)
        
        for i in range(x.shape[1]):
            out[:, i] = butter_lowpass_filter(x[:, i], cutoff, fs, order)

        return out    

    # TODO: fix this to read camera markers in a more general way
    def process_camera_markers(self):
        """
        Read position of markers on the camera system and calculate the corresponding reference frame.
        Now handles both single and multiple marker configurations automatically.
        Uses user-specified camera markers when provided.
        """
        
        # Use user-specified camera markers if provided
        if self.user_specified_camera_markers:
            print(f"Using user-specified camera markers: {self.user_specified_camera_markers}")
            self._process_user_specified_markers()
            return
        
        # Fallback to automatic detection
        # Define possible marker configurations
        multi_marker_labels = ['camera:cam_right', 'camera:cam_back', 'camera:cam_left']
        alt_multi_marker_labels = ['stereoatis:cam_right', 'stereoatis:cam_back', 'stereoatis:cam_left']
        single_marker_labels = ['CAMERAFRONT', 'CAMERASIDE']
        
        # Check which markers are available
        available_labels = [label.strip() for label in self.point_labels]
        print(f"Available markers in dataset: {available_labels}")
        
        # Check for multi-marker setup (primary)
        multi_markers_available = all(label in available_labels for label in multi_marker_labels)
        alt_multi_markers_available = all(label in available_labels for label in alt_multi_marker_labels)
        
        if multi_markers_available:
            print("Detected: Multi-marker camera setup (camera:* format)")
            self._process_multi_marker_setup(multi_marker_labels)
            
        elif alt_multi_markers_available:
            print("Detected: Multi-marker camera setup (stereoatis:* format)")
            self._process_multi_marker_setup(alt_multi_marker_labels)
            
        else:
            # Try single marker setup
            single_marker_found = None
            for label in single_marker_labels:
                if label in available_labels:
                    single_marker_found = label
                    break
            
            if single_marker_found:
                print(f"Detected: Single-marker camera setup using '{single_marker_found}'")
                self._process_single_marker_setup(single_marker_found)
            else:
                print("⚠️  No camera markers found in dataset")
                print(f"   Searched for multi-marker: {multi_marker_labels}")
                print(f"   Searched for single-marker: {single_marker_labels}")
                print("   Using identity transforms (camera_markers=False)")
                self.camera_markers = False

    def _process_user_specified_markers(self):
        """Process user-specified camera markers."""
        # Validate that all user-specified markers exist in the dataset
        available_labels = [label.strip() for label in self.point_labels]
        missing_markers = [marker for marker in self.user_specified_camera_markers if marker not in available_labels]
        
        if missing_markers:
            print(f"⚠️  Warning: Some user-specified camera markers not found: {missing_markers}")
            print(f"   Available markers: {available_labels}")
            # Filter out missing markers
            valid_markers = [marker for marker in self.user_specified_camera_markers if marker in available_labels]
            if not valid_markers:
                print("   No valid camera markers found, disabling camera markers")
                self.camera_markers = False
                return
            self.user_specified_camera_markers = valid_markers
            print(f"   Using valid markers: {valid_markers}")
        
        # Determine setup type based on number of markers
        n_markers = len(self.user_specified_camera_markers)
        if n_markers == 1:
            print(f"Processing as single-marker setup: {self.user_specified_camera_markers[0]}")
            self._process_single_marker_setup(self.user_specified_camera_markers[0])
        elif n_markers >= 3:
            print(f"Processing as multi-marker setup: {self.user_specified_camera_markers}")
            # Use the first 3 markers for the multi-marker setup (you may want to adjust this logic)
            self._process_multi_marker_setup(self.user_specified_camera_markers[:3])
        else:
            print(f"⚠️  Warning: {n_markers} camera markers specified, but need 1 or 3+ for valid setup")
            print("   Using single-marker approach with the first marker")
            self._process_single_marker_setup(self.user_specified_camera_markers[0])

    def _process_single_marker_setup(self, marker_name):
        """Process single camera marker setup"""
        camera_labels = [marker_name]

        # actual one:
        vicon_points = self.get_vicon_points(range(1, self.frame_count), camera_labels)

        # # new_dataset_giorgia:
        # available_frames = sorted(list(self.points_3d.keys()))
        # vicon_points = self.get_vicon_points(available_frames, camera_labels)

        single_camera = []
        for f in vicon_points['points']:
            single_camera.append(f[marker_name][:3])

        single_camera = np.array(single_camera)

        if self.filter_camera_markers:
            self.single_camera = self.filter_pose(single_camera) 
        else:
            self.single_camera = single_camera
            
        self.marker_setup_type = "single_marker"
        self.single_marker_name = marker_name

    def _process_multi_marker_setup(self, camera_labels):
        """Process multi-marker camera setup"""

        # actual one:
        vicon_points = self.get_vicon_points(range(1, self.frame_count), camera_labels)

        # # new_dataset_giorgia:
        # available_frames = sorted(list(self.points_3d.keys()))
        # vicon_points = self.get_vicon_points(available_frames, camera_labels)

        # Handle different camera label configurations flexibly
        if len(camera_labels) >= 3:
            # Use the first 3 markers
            marker_right = camera_labels[0]  # First marker as "right"
            marker_back = camera_labels[1]   # Second marker as "back" 
            marker_left = camera_labels[2]   # Third marker as "left"
        else:
            raise ValueError(f"Multi-marker setup requires at least 3 markers, got {len(camera_labels)}")

        camera_right = []
        camera_left = []
        camera_back = []
        for f in vicon_points['points']:
            camera_right.append(f[marker_right][:3])  
            camera_left.append(f[marker_left][:3])   
            camera_back.append(f[marker_back][:3])   

        camera_right = np.array(camera_right)
        camera_left = np.array(camera_left)
        camera_back = np.array(camera_back)

        if self.filter_camera_markers:
            self.camera_right = self.filter_pose(camera_right)  # Use filter_camera_markers method
            self.camera_left = self.filter_pose(camera_left)    # Use filter_camera_markers method
            self.camera_back = self.filter_pose(camera_back)    # Use filter_camera_markers method
        else:
            self.camera_right = camera_right
            self.camera_left = camera_left
            self.camera_back = camera_back
        
        self.marker_setup_type = "multi_marker"
        print(f"Multi-marker setup configured with: right={marker_right}, back={marker_back}, left={marker_left}")
    
    
    def compute_camera_marker_transforms(self):
        """
        Compute transformation matrices for all frames.
        Handles both multi-marker and single-marker cases automatically.
        """
        if not self.camera_markers:
            print("Using identity transforms (no camera markers)")
            return [np.eye(4) for _ in range(self.frame_count)]
        
        # Check the type of marker setup
        setup_type = getattr(self, 'marker_setup_type', None)
        
        if setup_type == "single_marker":
            print("Computing transforms using single camera marker...")
            return self._compute_single_marker_transforms()
            
        elif setup_type == "multi_marker":
            print("Computing transforms using multi-marker setup...")
            return self._compute_multi_marker_transforms()
            
        else:
            # Fallback: try to detect based on available attributes
            if hasattr(self, 'single_camera'):
                print("Computing transforms using detected single camera marker...")
                return self._compute_single_marker_transforms()
            elif hasattr(self, 'camera_left') and hasattr(self, 'camera_right') and hasattr(self, 'camera_back'):
                print("Computing transforms using detected multi-marker setup...")
                return self._compute_multi_marker_transforms()
            else:
                print("⚠️  No camera marker data found, using identity transforms")
                return [np.eye(4) for _ in range(self.frame_count)]

    def _compute_single_marker_transforms(self):
        """Compute transformation matrices for single marker setup"""
        n_frames = self.single_camera.shape[0]
        self.Ts = []
        
        for i in range(n_frames):
            T = np.eye(4)
            T[:3, 3] = self.single_camera[i]
            # Invert for world-to-camera
            T = np.linalg.inv(T)
            self.Ts.append(T)
            
        print(f"Generated {len(self.Ts)} transformation matrices (translation-only as there is only 1 marker)")
        return self.Ts

    def _compute_multi_marker_transforms(self):
        """Compute transformation matrices for multi-marker setup"""
        n_frames = self.camera_left.shape[0]
        self.Ts = []
        
        for i in range(n_frames):
            origin = self.camera_left[i]
            x_axis = self.camera_right[i] - self.camera_left[i]
            t_axis = self.camera_back[i] - self.camera_left[i]
            z_axis = np.cross(x_axis, t_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Normalize
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Build rotation matrix
            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            T = np.eye(4)
            T[:3, :3] = R.transpose()
            T[:3, 3] = - R.transpose() @ origin
            self.Ts.append(T)
            
        print(f"Generated {len(self.Ts)} transformation matrices (full pose as there are at least 3 markers)")
        return self.Ts
    
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y