import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import c3d
import importlib
from typing import List, Optional, Dict, Tuple, Any
from collections import deque
import argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Import helpers
sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2')
import helpers

# Import bimvee
sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2/submodules/bimvee')
# from bimvee.importIitYarp import importIitYarp
from bimvee.importAe import importAe


class ViconDVSPipeline:
    """Complete pipeline for VICON-DVS calibration and projection."""
    
    def __init__(self, dvs_path: str, vicon_path: str, intrinsic_path: str, 
                 subject: Optional[str], output_path: str, camera_setup: str = "auto"):
        self.dvs_path = dvs_path
        self.vicon_path = vicon_path
        self.intrinsic_path = intrinsic_path
        self.subject = subject
        self.output_path = output_path
        self.camera_setup = camera_setup  # "single", "multi", or "auto"
        self.period = 1.0 / 100  # VICON frequency: 100Hz
        
        # Initialize data containers
        self.imp = None
        self.start_time = None
        self.end_time = None
        self.c3d_data = None
        self.points_3d = {}
        self.marker_t = None
        self.marker_names = []
        self.working_markers = []  # Subset of markers to work with
        self.K = None
        self.D = None
        self.cam_res = None
        self.T_syst_to_camera_opt = np.eye(4)
        self.delay = 0.0
        self.Ts_world_to_system = None
        
    def detect_marker_setup(self) -> str:
        """Detect the type of marker setup based on available markers."""
        available_markers = [name.strip() for name in self.c3d_data.point_labels]
        
        # Check for multi-marker camera setup
        camera_markers = ['cam_right', 'cam_back', 'cam_left']
        multi_marker_found = any(any(cam_marker in marker for cam_marker in camera_markers) 
                                for marker in available_markers)
        
        if multi_marker_found:
            return "multi"
        
        # Check for single camera marker
        single_camera_markers = ['camera', 'cam', 'stereo']
        single_marker_found = any(any(single_marker in marker.lower() for single_marker in single_camera_markers)
                                 for marker in available_markers)
        
        if single_marker_found:
            return "single"
            
        # Default to single if uncertain
        return "single"
    
    def get_working_markers(self, marker_filter: Optional[str] = None) -> List[str]:
        """Get the subset of markers to work with based on the setup."""
        available_markers = [name.strip() for name in self.c3d_data.point_labels]
        
        if marker_filter:
            # Use custom marker filter
            if marker_filter == "hpe":
                # HPE markers
                hpe_markers = ['P1:LANK', 'P1:RANK', 'P1:LKNE', 'P1:RKNE', 'P1:RSHO', 
                              'P1:LSHO', 'P1:LELB', 'P1:RELB', 'P1:LWRA', 'P1:RWRA', 
                              'P1:CLAV', 'P1:STRN', 'P1:LFHD', 'P1:RFHD']
                return [m for m in hpe_markers if m in available_markers]
            
            elif marker_filter == "simple":
                # Simple marker set
                simple_markers = ['STR', 'SHO', 'ARM', 'ELB', 'WRS', 'HAN']
                return [m for m in simple_markers if m in available_markers]
            
            elif marker_filter == "calibration":
                # Calibration markers (boxes, monitors, etc.)
                calib_keywords = ['box1', 'box2', 'monitor', 'chair']
                return [m for m in available_markers 
                       if any(keyword in m.lower() for keyword in calib_keywords)]
        
        # Auto-detect based on available markers
        print(f"Available markers: {available_markers}")
        
        # Priority order: try simple markers first, then HPE, then all
        simple_markers = ['STR', 'SHO', 'ARM', 'ELB', 'WRS', 'HAN']
        simple_found = [m for m in simple_markers if m in available_markers]
        
        if len(simple_found) >= 3:  # Need at least 3 markers
            print(f"Using simple marker set: {simple_found}")
            return simple_found
        
        # Try HPE markers
        hpe_markers = ['P1:LANK', 'P1:RANK', 'P1:LKNE', 'P1:RKNE', 'P1:RSHO', 
                      'P1:LSHO', 'P1:LELB', 'P1:RELB', 'P1:LWRA', 'P1:RWRA', 
                      'P1:CLAV', 'P1:STRN', 'P1:LFHD', 'P1:RFHD']
        hpe_found = [m for m in hpe_markers if m in available_markers]
        
        if len(hpe_found) >= 8:  # Use HPE if we have enough
            print(f"Using HPE marker set: {hpe_found}")
            return hpe_found
        
        # Use first N available markers (excluding camera markers)
        camera_keywords = ['cam', 'camera', 'stereo']
        non_camera_markers = [m for m in available_markers 
                             if not any(keyword in m.lower() for keyword in camera_keywords)]
        
        if len(non_camera_markers) >= 6:
            selected = non_camera_markers[:14]  # Use first 14
            print(f"Using first available markers: {selected}")
            return selected
        
        # Last resort: use all available markers
        print(f"Using all available markers: {available_markers}")
        return available_markers
        
    def load_event_data(self):
        """Load event data using importAe for memory efficiency."""
        print("Loading event data...")
        importers = importAe(self.dvs_path)
        self.imp = importers['data']['']['dvs']        # TODO: read middle one from folder name
        self.start_time = self.imp.get_first_ts()
        self.end_time = self.imp.get_last_ts()
        print(f"Events from {self.start_time:.3f}s to {self.end_time:.3f}s")
    
    def load_vicon_data(self):
        """Load VICON C3D data."""
        print("Loading VICON data...")
        self.c3d_data = c3d.Reader(open(self.vicon_path, 'rb'))
        for i, points, analog in self.c3d_data.read_frames():
            self.points_3d[i] = points
            
        self.marker_t = np.linspace(0.0, self.c3d_data.frame_count / self.c3d_data.point_rate, 
                                   self.c3d_data.frame_count, endpoint=False)
        
        # Get all marker names
        self.marker_names = [name.strip() for name in self.c3d_data.point_labels]
        print(f"Loaded {len(self.marker_names)} total markers")
        
        # Detect camera setup if auto
        if self.camera_setup == "auto":
            self.camera_setup = self.detect_marker_setup()
        
        print(f"Camera setup detected/specified: {self.camera_setup}")
        
    def load_calibration_data(self):
        """Load camera calibration parameters."""
        print("Loading calibration data...")
        calib = np.genfromtxt(self.intrinsic_path, delimiter=" ", skip_header=1, dtype=object)
        
        calib_dict = {key: value for key, value in zip(calib[:, 0].astype(str), calib[:, 1].astype(float))}
        
        self.cam_res = np.int64([calib_dict['h'], calib_dict['w']])
        
        # Intrinsic matrix
        self.K = np.array([
            [calib_dict['fx'], 0.0, calib_dict['cx']],
            [0.0, calib_dict['fy'], calib_dict['cy']],
            [0.0, 0.0, 1.0]
        ])
        self.D = np.array([calib_dict['k1'], calib_dict['k2'], calib_dict['p1'], calib_dict['p2']])
        
    def load_existing_calibration(self, init_file_path: str) -> bool:
        """Load existing transformation and delay if available."""
        if not os.path.exists(init_file_path):
            return False
            
        print(f"Loading existing calibration from {init_file_path}")
        with open(init_file_path, 'r') as f:
            content = f.read().splitlines()

        # Extract transformation matrix
        start = content.index("[TRANSFORMATION MATRIX SYSTEM TO CAMERA]") + 1
        end = content.index("[DELAY]")
        matrix_lines = content[start:end]

        matrix = []
        for line in matrix_lines:
            clean = line.replace("[[", "").replace("]]", "").replace("[", "").replace("]", "")
            row = [float(x) for x in clean.strip().split() if x]
            if row:
                matrix.append(row)

        self.T_syst_to_camera_opt = np.array(matrix, dtype=np.float64)

        # Extract delay
        delay_index = content.index("[DELAY]") + 1
        while delay_index < len(content) and not content[delay_index].strip():
            delay_index += 1
        self.delay = float(content[delay_index].strip())
        
        print("Loaded transformation matrix and delay")
        return True
    
    def compute_world_to_system_transforms(self):
        """Compute world to system transformation matrices."""
        print("Computing world to system transformations...")
        from helpers import ViconHelper
        
        # Use the camera setup specification
        enable_camera_markers = (self.camera_setup in ["multi", "single"])
        
        vicon_helper = ViconHelper(
            self.marker_t, self.points_3d, self.delay, 
            self.c3d_data.frame_count, self.c3d_data.point_rate, 
            self.c3d_data.point_labels, enable_camera_markers, True
        )
        
        # Print diagnostic info
        print("=== CAMERA MARKER DIAGNOSTIC ===")
        print(f"Available markers in dataset: {self.marker_names}")
        print(f"Camera setup: {self.camera_setup}")
        print(f"Camera markers enabled: {enable_camera_markers}")
        
        # Check what was detected
        if hasattr(vicon_helper, 'camera_left') and vicon_helper.camera_left is not None:
            print("✅ Multi-marker setup detected")
            print(f"   camera_left shape: {vicon_helper.camera_left.shape}")
            print(f"   camera_right shape: {vicon_helper.camera_right.shape}")
            print(f"   camera_back shape: {vicon_helper.camera_back.shape}")
        elif hasattr(vicon_helper, 'single_camera') and vicon_helper.single_camera is not None:
            print("✅ Single-marker setup detected")
            print(f"   single_camera shape: {vicon_helper.single_camera.shape}")
            print("   Note: Only translation available, rotation will be identity")
        else:
            print("❌ No camera markers detected - using identity transforms")
        print("================================")
        
        self.Ts_world_to_system = vicon_helper.compute_camera_marker_transforms()
        
    def visualize_events(self, duration: float = 10.0):
        """Visualize event data for a specified duration."""
        print(f"Visualizing events for {duration} seconds...")
        
        img = np.ones(self.cam_res, dtype=np.uint8) * 255
        ft = self.start_time
        window_size = 1000 * self.period
        window_start = self.start_time
        
        cv2.namedWindow('Event Visualization', cv2.WINDOW_NORMAL)
        
        try:
            end_vis_time = self.start_time + duration
            while ft < end_vis_time and ft < self.end_time:
                window_end = min(window_start + window_size, self.end_time)
                window_center = (window_start + window_end) / 2
                
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])
                
                for i in range(len(e_ts)):
                    if e_ts[i] >= ft:
                        cv2.imshow('Event Visualization', img)
                        k = cv2.waitKey(int(self.period * 1000))
                        
                        if k == 27 or k == ord('q'):
                            raise KeyboardInterrupt
                            
                        img = np.ones(self.cam_res, dtype=np.uint8) * 255
                        text = f"t = {ft:.6f}s"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, text, (img.shape[1] - 200, 30), font, 0.7, (0, 0, 0), 2)
                        ft += self.period
                        
                    if e_vs[i] < self.cam_res[0] and e_us[i] < self.cam_res[1]:
                        img[e_vs[i], e_us[i]] = 0
                        
                window_start = e_ts[-1] if len(e_ts) > 0 else window_start + window_size
                
        except KeyboardInterrupt:
            print("Visualization stopped by user")
        finally:
            cv2.destroyAllWindows()
        
    def manual_rotation_estimation(self, marker_filter: Optional[str] = None, 
                                  chosen_marker: Optional[str] = None) -> np.ndarray:
        """Manually estimate rotation using visual feedback with windowed approach."""
        print("Starting manual rotation estimation...")
        
        # Get working markers
        self.working_markers = self.get_working_markers(marker_filter)
        
        if not self.working_markers:
            raise RuntimeError("No suitable markers found for calibration")
            
        # Choose marker for feedback
        if chosen_marker and chosen_marker in self.working_markers:
            chosen_one = chosen_marker
        else:
            chosen_one = self.working_markers[0]
            
        print(f"Using markers: {self.working_markers}")
        print(f"Feedback marker: {chosen_one}")
        
        # Create projector for manual adjustment
        projector = helpers.ViconProjector(
            self.working_markers, self.c3d_data, self.points_3d, 
            self.T_syst_to_camera_opt, self.Ts_world_to_system, 
            self.K, self.cam_res, D=self.D, subject=self.subject
        )
        
        # Use windowed approach
        window_size = 1000 * self.period  # 10 seconds window
        window_start = self.start_time
        rvec_init = np.zeros(3)
        
        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                if len(e_ts) == 0:
                    print(f"No events in window [{window_start:.3f}, {window_end:.3f}]")
                    window_start += window_size
                    continue

                print(f"Processing {len(e_ts)} events between {e_ts[0]:.3f}s and {e_ts[-1]:.3f}s")

                # Call projector manual rotation adjustment
                rvec_init = projector.manual_rotation_adjustment(
                    self.marker_t, self.delay, e_ts, e_us, e_vs, 
                    self.period, visualize=True, chosen_one=chosen_one,
                    marker_time_offset=window_start
                )

                window_start += window_size
                
        except helpers.RotationExit as e:
            print("Visualization stopped by user with final rotation.")
            rvec_init = e.r_vec

        finally:
            cv2.destroyAllWindows()
            print("rvec", rvec_init)
            return rvec_init

    def manual_delay_correction(self) -> float:
        """Manually correct synchronization delay using windowed approach."""
        print("Starting manual delay correction...")
        
        if not self.working_markers:
            self.working_markers = self.get_working_markers()
        
        # Use windowed approach
        window_size = 1000 * self.period
        window_start = self.start_time
        
        # Create projector for delay adjustment
        projector = helpers.ViconProjector(
            self.working_markers, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )

        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                print(f"Processing {len(e_ts)} events between {e_ts[0]:.3f}s and {e_ts[-1]:.3f}s")

                # Call projector delay adjustment
                self.delay = projector.fix_delay(
                    self.marker_t, self.delay, e_ts, e_us, e_vs, self.period,
                    visualize=True, marker_time_offset=window_start
                )

                window_start += window_size
                
        except helpers.DelayExit as e:
            print("Delay adjustment stopped by user.")
            self.delay = e.delay

        finally:
            cv2.destroyAllWindows()
            print(f"Updated delay: {self.delay:.3f}s")
            return self.delay

    def label_data_interactive(self, use_projections: bool = False) -> str:
        """Interactive data labeling using windowed approach."""
        print("Starting interactive labeling...")
        
        from helpers import DvsLabeler
        
        if not self.working_markers:
            self.working_markers = self.get_working_markers()
        
        labels_path = os.path.join(os.path.dirname(self.output_path), 'labeled_points.yml')
        
        window_size = 1000 * self.period
        window_start = self.start_time
        
        # Create labeler instance
        labeler = DvsLabeler(img_shape=(self.cam_res[0], self.cam_res[1], 3), subject=self.subject)
        
        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                print(f"Processing {len(e_ts)} events between {e_ts[0]:.3f}s and {e_ts[-1]:.3f}s")
                
                if use_projections:
                    labeler.correct_data(
                        e_ts, e_us, e_vs, self.period,
                        self.working_markers, self.c3d_data, self.points_3d, self.marker_t,
                        self.T_syst_to_camera_opt, self.Ts_world_to_system,
                        self.K, self.cam_res, self.delay, D=self.D,
                        marker_time_offset=window_start
                    )
                else:
                    input_label_tag_file = '../scripts/config/labels_tags.yml'
                    labeler.label_data(
                        e_ts, e_us, e_vs,
                        self.period, input_label_tag_file
                    )

                window_start += window_size
                
        except helpers.LabelExit as e:
            print("Labeling stopped early by user, saving partial results.")
            if hasattr(e, 'labeled_dict') and e.labeled_dict:
                labeler.labeled_dict = e.labeled_dict
                labeler.labels_done = True

        finally:
            cv2.destroyAllWindows()
            
            if hasattr(labeler, 'labels_done') and labeler.labels_done:
                labeler.save_labeled_points(labels_path)
                print(f"Saved labeled points to: {labels_path}")
            else:
                print("No labels were created - labeling process was incomplete")
        
        return labels_path

    def create_projection_video(self, output_video: str = 'projection_video.mp4'):
        """Create video with projected markers using windowed approach."""
        print("Creating projection video...")
        
        if not self.working_markers:
            self.working_markers = self.get_working_markers()

        # Create projector
        projector = helpers.ViconProjector(
            self.working_markers, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )
        
        collected_video_segments = []
        window_size = 1000 * self.period
        window_start = self.start_time
        
        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                print(f"Processing {len(e_ts)} events between {e_ts[0]:.3f}s and {e_ts[-1]:.3f}s")

                # Call projector
                image_points, video_segment = projector.project_vicon_to_event_plane_dynamic(
                    self.marker_t, self.delay,
                    e_ts, e_us, e_vs, self.period,
                    visualize=True, video_record=True,
                    marker_time_offset=window_start
                )
                
                if video_segment is not None:
                    collected_video_segments.append(video_segment)

                window_start += window_size

        except KeyboardInterrupt:
            print("Video creation stopped by user")

        finally:
            cv2.destroyAllWindows()
            
            # Merge video segments
            if collected_video_segments:
                print(f"Merging {len(collected_video_segments)} video segments...")
                
                try:
                    fps = int(1 / self.period)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        output_video, fourcc, fps,
                        (self.cam_res[1], self.cam_res[0]), isColor=False
                    )
                    
                    if video_writer.isOpened():
                        total_frames = 0
                        
                        for i, segment_frames in enumerate(collected_video_segments):
                            if segment_frames:
                                print(f"Merging segment {i+1}/{len(collected_video_segments)} with {len(segment_frames)} frames")
                                
                                for frame in segment_frames:
                                    if len(frame.shape) == 3:
                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    video_writer.write(frame)
                                    total_frames += 1
                        
                        video_writer.release()
                        
                        if total_frames > 0:
                            print(f"Successfully created video: {output_video}")
                            print(f"Total frames: {total_frames}")
                            
                except Exception as e:
                    print(f"Error creating video: {e}")
            else:
                print("No video segments were created")

    def estimate_transformation(self, system_points: np.ndarray, image_points: np.ndarray, init_params: np.ndarray) -> np.ndarray:
        """
        Estimate system-to-camera transformation using optimization.
        """
        print(f"Starting optimization with {len(system_points)} correspondences")
        print(f"Initial parameters: {init_params}")
        
        # Use the existing helpers function
        T_optimized = helpers.estimate_Tstoc(
            system_points, 
            image_points, 
            self.K, 
            self.D, 
            init_params
        )
        
        print("Optimization completed")
        print(f"Optimized transformation matrix:\n{T_optimized}")
        
        return T_optimized
        
    def optimize_calibration(self, labels_path: str):
        """Optimize calibration using labeled points."""
        print("Optimizing calibration...")
        
        from helpers import ViconHelper
        
        # Load labeled points
        labeled_points = helpers.read_points_labels(labels_path)
        
        vicon_helper = ViconHelper(
            self.marker_t, self.points_3d, self.delay,
            self.c3d_data.frame_count, self.c3d_data.point_rate,
            self.c3d_data.point_labels, True, True
        )
        
        # Get interpolated VICON points
        frames_id = vicon_helper.get_frame_time(labeled_points['times'])
        vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)
        
        # Collect correspondences
        world_points = []
        image_points_clean = []
        
        for idx, (dvs_frame, vicon_frame) in enumerate(zip(labeled_points['points'], vicon_points['points'])):
            for label in dvs_frame.keys():
                try:
                    if label not in vicon_frame:
                        continue
                        
                    w_p = vicon_frame[label]
                    if w_p is None or np.any(np.isnan(w_p)):
                        continue
                        
                    i_p = [dvs_frame[label]['x'], dvs_frame[label]['y']]
                    world_points.append(w_p)
                    image_points_clean.append(i_p)
                    
                except Exception as e:
                    print(f"Error processing marker '{label}': {e}")
                    
        world_points = np.array(world_points, dtype=np.float64)
        image_points_clean = np.array(image_points_clean, dtype=np.float64)
        
        print(f"Collected {len(world_points)} valid correspondences")
        
        # Solve PnP for initial estimate
        success, rvec, tvec = cv2.solvePnP(world_points, image_points_clean, self.K, self.D)
        
        if success:
            R_mat, _ = cv2.Rodrigues(rvec)
            T_world_to_cam = np.eye(4)
            T_world_to_cam[:3, :3] = R_mat
            T_world_to_cam[:3, 3] = tvec.flatten()
            
            # Create initial estimate for system-to-camera transformation
            init_T = T_world_to_cam @ np.linalg.inv(self.Ts_world_to_system[0])
            r_vec = Rotation.from_matrix(init_T[:3, :3]).as_rotvec()
            t_vec = init_T[:3, 3]
            init_param = np.concatenate((r_vec, t_vec))
            
            # Create system points for optimization
            system_points = []
            image_points = []
            
            for idx, (dvs_frame, vicon_frame) in enumerate(zip(labeled_points['points'], vicon_points['points'])):
                for label in dvs_frame.keys():
                    try:
                        w_p = vicon_frame[label]
                        w_ph = np.append(w_p, 1.0)
                        i_p = [dvs_frame[label]['x'], dvs_frame[label]['y']]
                        frame_id = vicon_points['frame_ids'][idx]
                        p_sys = self.Ts_world_to_system[frame_id] @ w_ph
                        system_points.append(p_sys[:3])
                        image_points.append(i_p)
                    except Exception:
                        continue
                        
            system_points = np.array(system_points, dtype=np.float64)
            image_points = np.array(image_points, dtype=np.float64)
            
            # Optimize transformation
            self.T_syst_to_camera_opt = self.estimate_transformation(system_points, image_points, init_param)
            
            print("Calibration optimization completed")
        else:
            print("PnP solution failed - using manual estimates")
            
    def save_calibration(self, output_file: str):
        """Save transformation matrix and delay."""
        def format_matrix_block(T: np.ndarray) -> str:
            rows = []
            for i, row in enumerate(T):
                row_str = " ".join(f"{val: .8e}" for val in row).lstrip()
                if i == 0:
                    rows.append(f"[[ {row_str}]")
                elif i == T.shape[0] - 1:
                    rows.append(f" [ {row_str}]]")
                else:
                    rows.append(f" [ {row_str}]")
            return "\n".join(rows)
        
        with open(output_file, "w") as f:
            f.write("[TRANSFORMATION MATRIX SYSTEM TO CAMERA]\n\n")
            f.write(format_matrix_block(self.T_syst_to_camera_opt) + "\n\n")
            f.write("[DELAY]\n\n")
            f.write(f"{self.delay}\n")
            
        print(f"Saved calibration to {output_file}")
        
    def save_transforms_txt(self, output_file: str):
        """Save all transformation matrices to a text file."""
        with open(output_file, 'w') as f:
            f.write("# Transformation matrices from world to system coordinates\n")
            f.write("# Format: timestamp: array([4x4 transformation matrix])\n\n")
            
            for timestamp, transform_matrix in zip(self.marker_t, self.Ts_world_to_system):
                f.write(f"{timestamp:.6f}: array({transform_matrix.tolist()})\n")
                
        print(f"Saved transformation matrices to {output_file}")
        
    def run_full_pipeline(self, manual_calibration: bool = True, 
        use_projections: bool = False, create_video: bool = True,
        init_file_path: str = None, labels_path: str = None,
        marker_filter: str = 'auto', chosen_marker: str = None):
        
        """Run the complete pipeline."""
        print("Starting VICON-DVS pipeline...")
        
        # 1. Load all data
        self.load_event_data()
        self.load_vicon_data()
        self.load_calibration_data()
        
        # 2. Try to load existing calibration
        if init_file_path is None:
            init_file = os.path.join(os.path.dirname(self.vicon_path), "init_file.txt")
        else:
            init_file = init_file_path
            
        calibration_exists = self.load_existing_calibration(init_file)
        
        # 3. Compute world to system transforms
        self.compute_world_to_system_transforms()
        
        if not calibration_exists and manual_calibration:
            print("No existing calibration found. Starting manual calibration...")
            
            # 4. Manual rotation estimation with marker filter
            tvec_init = np.array([0.0, 0.0, 0.0])
            rvec_init = self.manual_rotation_estimation(
                marker_filter=marker_filter if marker_filter != 'auto' else None,
                chosen_marker=chosen_marker
            )
            
            # Update transformation matrix
            self.T_syst_to_camera_opt[:3, :3] = cv2.Rodrigues(rvec_init)[0]
            self.T_syst_to_camera_opt[:3, 3] = tvec_init
            
            # 5. Manual delay correction
            self.delay = self.manual_delay_correction()
            
            # 6. Interactive labeling or use existing labels
            if labels_path and os.path.exists(labels_path):
                print(f"Using existing labeled points from: {labels_path}")
                final_labels_path = labels_path
            else:
                print("Starting interactive labeling...")
                final_labels_path = self.label_data_interactive(use_projections=use_projections)
            
            # 7. Optimize calibration
            self.optimize_calibration(final_labels_path)
            
            # 8. Save calibration
            self.save_calibration(init_file)
            
        # 9. Save transformation matrices
        transforms_file = os.path.join(os.path.dirname(self.vicon_path), "transformation_matrices.txt")
        self.save_transforms_txt(transforms_file)
        
        # 10. Create projection video
        if create_video:
            video_file = os.path.join(os.path.dirname(self.vicon_path), "projection_video.mp4")
            self.create_projection_video(video_file)
            
            # 11. Ask user for confirmation
            print("\n" + "="*60)
            print("CALIBRATION RESULTS REVIEW")
            print("="*60)
            print(f"Projection video has been created: {video_file}")
            print("Please review the video to check the quality of marker projections.")
            
            while True:
                response = input("\nAre you satisfied with the calibration results? (y/n): ").lower().strip()
                
                if response in ['y', 'yes']:
                    print("Pipeline completed successfully!")
                    return
                elif response in ['n', 'no']:
                    print("\nRestarting calibration process...")
                    print("Previous calibration will be ignored.")
                    
                    # Reset calibration parameters
                    self.T_syst_to_camera_opt = np.eye(4)
                    self.delay = 0.0
                    
                    # Restart the full pipeline with no init file and forced manual calibration
                    return self.run_full_pipeline(
                        manual_calibration=True,
                        use_projections=use_projections,
                        create_video=create_video,
                        init_file_path=None,  # Force no init file on restart
                        labels_path=labels_path,  # Keep the same labels path for restart
                        marker_filter=marker_filter,
                        chosen_marker=chosen_marker
                    )
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
        else:
            # If no video creation, just complete the pipeline
            print("Pipeline completed successfully!")
            return
    
def main():
    parser = argparse.ArgumentParser(
        prog='VICON markers projection',
        description='Project VICON markers onto DVS frames using time-synchronized windows'
    )
    
    parser.add_argument('--dvs_path', required=True,
                       help='Path to the YARP folder containing DVS recording')
    parser.add_argument('--vicon_path', required=True,
                       help='Path to the .c3d file containing VICON recording')
    parser.add_argument('--intrinsic', required=True,
                       help='Intrinsic calibration file for the camera')
    parser.add_argument('--init-file', default=None,
                       help='Path to initialization file containing transformation matrix and delay')
    parser.add_argument('--output_path', required=True,
                       help='Output path for labeled points (YAML file)')
    parser.add_argument('--subject', default=None,
                       help='Subject name for labels (e.g., P1, P11)')
    parser.add_argument('--labels_path', default=None,
                       help='Path to existing labeled points YAML file')
    parser.add_argument('--camera-setup', choices=['single', 'multi', 'auto'], default='auto',
                       help='Camera marker setup: single marker, multi-marker, or auto-detect')
    parser.add_argument('--marker-filter', choices=['hpe', 'simple', 'calibration', 'auto'], default='auto',
                       help='Marker set to use: hpe (human pose), simple (STR/SHO/etc), calibration (boxes/monitors), or auto-detect')
    parser.add_argument('--chosen-marker', default=None,
                       help='Specific marker to use for rotation adjustment feedback')
    parser.add_argument('--list', action='store_true',
                       help='Use list-based labeling interface')
    parser.add_argument('--projections', action='store_true',
                       help='Use projection-based labeling interface')
    parser.add_argument('--no-manual', action='store_true',
                       help='Skip manual calibration if existing calibration found')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video creation')
    parser.add_argument('--visualize-events', type=float, default=0,
                       help='Visualize events for specified duration (seconds)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ViconDVSPipeline(
        dvs_path=args.dvs_path,
        vicon_path=args.vicon_path,
        intrinsic_path=args.intrinsic,
        subject=args.subject,
        output_path=args.output_path,
        camera_setup=args.camera_setup
    )
    
    # Visualize events if requested
    if args.visualize_events > 0:
        pipeline.load_event_data()
        pipeline.load_calibration_data()
        pipeline.visualize_events(args.visualize_events)
        return
    
    # Run full pipeline
    pipeline.run_full_pipeline(
        manual_calibration=not args.no_manual,
        use_projections=args.projections,
        create_video=not args.no_video,
        init_file_path=args.init_file,
        labels_path=args.labels_path,
        marker_filter=args.marker_filter,
        chosen_marker=args.chosen_marker
    )
    
    # # Load data first to detect markers
    # pipeline.load_event_data()
    # pipeline.load_vicon_data()
    # pipeline.load_calibration_data()
    
    # # Try to load existing calibration
    # if args.init_file is None:
    #     init_file = os.path.join(os.path.dirname(pipeline.vicon_path), "init_file.txt")
    # else:
    #     init_file = args.init_file
        
    # calibration_exists = pipeline.load_existing_calibration(init_file)
    
    # # Compute world to system transforms
    # pipeline.compute_world_to_system_transforms()
    
    # if not calibration_exists and not args.no_manual:
    #     print("No existing calibration found. Starting manual calibration...")
        
    #     # Manual rotation estimation with marker filter
    #     tvec_init = np.array([0.0, 0.0, 0.0])
    #     rvec_init = pipeline.manual_rotation_estimation(
    #         marker_filter=args.marker_filter if args.marker_filter != 'auto' else None,
    #         chosen_marker=args.chosen_marker
    #     )
        
    #     # Update transformation matrix
    #     pipeline.T_syst_to_camera_opt[:3, :3] = cv2.Rodrigues(rvec_init)[0]
    #     pipeline.T_syst_to_camera_opt[:3, 3] = tvec_init
        
    #     # Manual delay correction
    #     pipeline.delay = pipeline.manual_delay_correction()
        
    #     # Interactive labeling
    #     if args.labels_path and os.path.exists(args.labels_path):
    #         print(f"Using existing labeled points from: {args.labels_path}")
    #         final_labels_path = args.labels_path
    #     else:
    #         print("Starting interactive labeling...")
    #         final_labels_path = pipeline.label_data_interactive(use_projections=args.projections)
        
    #     # Optimize calibration
    #     pipeline.optimize_calibration(final_labels_path)
        
    #     # Save calibration
    #     pipeline.save_calibration(init_file)
        
    # # Save transformation matrices
    # transforms_file = os.path.join(os.path.dirname(pipeline.vicon_path), "transformation_matrices.txt")
    # pipeline.save_transforms_txt(transforms_file)
    
    # # Create projection video
    # if not args.no_video:
    #     video_file = os.path.join(os.path.dirname(pipeline.vicon_path), "projection_video.mp4")
    #     pipeline.create_projection_video(video_file)
        
    #     # User confirmation loop (keep existing)
    #     print("\n" + "="*60)
    #     print("CALIBRATION RESULTS REVIEW")
    #     print("="*60)
    #     print(f"Projection video has been created: {video_file}")
    #     print("Please review the video to check the quality of marker projections.")
        
    #     while True:
    #         response = input("\nAre you satisfied with the calibration results? (y/n): ").lower().strip()
            
    #         if response in ['y', 'yes']:
    #             print("Pipeline completed successfully!")
    #             return
    #         elif response in ['n', 'no']:
    #             print("\nRestarting calibration process...")
    #             print("Previous calibration will be ignored.")
                
    #             # Reset and restart
    #             pipeline.T_syst_to_camera_opt = np.eye(4)
    #             pipeline.delay = 0.0
                
    #             return main()  # Restart with same arguments
    #         else:
    #             print("Please enter 'y' for yes or 'n' for no.")
    # else:
    #     print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()