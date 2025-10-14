import sys
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import c3d
import importlib
from typing import List, Optional
import argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# # Import helpers
# sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2')
# import helpers
 
# # Import bimvee
# sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2/submodules/bimvee')
# # from bimvee.importIitYarp import importIitYarp
# from bimvee.importAe import importAe
 
# Get the absolute path to the current file (pipeline.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
 
# Add needed paths dynamically
HELPERS_PATH = os.path.join(CURRENT_DIR)
BIMVEE_PATH = os.path.join(CURRENT_DIR, "submodules/bimvee")
 
for path in [HELPERS_PATH, BIMVEE_PATH]:
    if path not in sys.path:
        sys.path.append(path)
 
# Now safely import your modules
import helpers
from bimvee.importAe import importAe


class ViconDVSPipeline:
    """Complete pipeline for VICON-DVS calibration and projection."""
    
    def __init__(self, dvs_path: str, vicon_path: str, intrinsic_path: str, 
                 subject: Optional[str], output_path: str, camera_setup: str = "auto", marker_list_path: Optional[str] = None):
        self.dvs_path = dvs_path
        self.vicon_path = vicon_path
        self.intrinsic_path = intrinsic_path
        self.subject = subject
        self.output_path = output_path
        self.camera_setup = camera_setup  # "single", "multi", or "auto"
        self.marker_list_path = marker_list_path
        self.period = 1.0 / 100  # VICON frequency: 100Hz
        
        # Initialize data containers
        self.imp = None
        self.start_time = None
        self.end_time = None
        self.c3d_data = None
        self.points_3d = {}
        self.marker_t = None
        self.marker_names = []
        self.markers_names = []  # Subset of markers to work with
        self.K = None
        self.D = None
        self.cam_res = None
        self.T_syst_to_camera_opt = np.eye(4)
        self.delay = 0.0
        self.Ts_world_to_system = None    

###
    def _detect_subject_from_c3d(self) -> Optional[str]:
        subject_counts = {}
        
        for marker_name in self.marker_names:
            marker_name = marker_name.strip()
            if ':' in marker_name:
                # Extract the part before the colon as potential subject
                potential_subject = marker_name.split(':', 1)[0].strip()
                
                # Check if it matches common subject patterns (P1, P2, etc. or S1, S2, etc.)
                if potential_subject and (
                    potential_subject.startswith('P') or
                    potential_subject.startswith('S') or
                    potential_subject.startswith('Subject')
                ):
                    subject_counts[potential_subject] = subject_counts.get(potential_subject, 0) + 1
        
        if subject_counts:
            # Return the subject with the most markers
            most_common_subject = max(subject_counts.keys(), key=lambda x: subject_counts[x])
            print(f"Found subject patterns: {dict(subject_counts)}")
            return most_common_subject
        
        return None

    # TODO: make it so that i don't use list of known markers but look for 'cam', 'camera', 'dvs', etc...
    def prompt_camera_setup(self) -> str:
        """Ask user how many markers are attached to the camera to understand setup."""
        print("\nCamera setup configuration (manual input)")
        print("Enter the number of markers attached to the camera.")
        # print("  1 -> Single marker (only translation known, rotation = identity initially)")
        # print("  3 or more -> Multi-marker rigid body (full pose)")
        
        while True:
            try:
                n = int(input("Number of camera markers: "))
                if n < 1:
                    print("Please enter a positive integer.")
                    continue
                setup = "multi" if n >= 3 else "single"
                print(f"Selected camera setup: {setup} (from {n} markers)")
                return setup
            except ValueError:
                print("Invalid input. Please enter an integer (e.g. 1 or 3).")
    
    
    # TODO: check this function and fix it
    def get_markers_names(self) -> List[str]:
        """Return list of markers to use based on optional user label file"""
        
        available_markers_raw = [name.strip() for name in self.c3d_data.point_labels]
        # Build a case-insensitive map of aliases -> original c3d label
        avail_map = {}
        subj = (self.subject or "").strip()
        subj_prefix = f"{subj}:" if subj else ""
        subj_upper = subj_prefix.upper()

        for m in available_markers_raw:
            orig = m.strip()
            key_full = orig.upper()
            if key_full not in avail_map:
                avail_map[key_full] = orig

            # Alias without subject prefix if present in C3D
            if ":" in orig:
                no_subj = orig.split(":", 1)[1].strip()
                key_no_subj = no_subj.upper()
                if key_no_subj not in avail_map:
                    avail_map[key_no_subj] = orig
            # Alias with subject prefix if C3D labels lack it and subject is provided
            if subj and not orig.upper().startswith(subj_upper):
                key_with_subj = f"{subj}:{orig}".upper()
                if key_with_subj not in avail_map:
                    avail_map[key_with_subj] = orig

        # Build a case-insensitive map: UPPER -> original
        avail_map = {}
        for m in available_markers_raw:
            key = m.upper()
            # Keep the first occurrence to preserve stable output
            if key not in avail_map:
                avail_map[key] = m

        def parse_yaml_structure(data) -> List[str]:
            acc = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        acc.append(item)
            elif isinstance(data, dict):
                # Try common keys first
                for k in ['labels', 'markers', 'tags']:
                    v = data.get(k, None)
                    if isinstance(v, list):
                        acc.extend([x for x in v if isinstance(x, str)])
                # Fallback: scan all values
                if not acc:
                    for v in data.values():
                        if isinstance(v, list):
                            acc.extend([x for x in v if isinstance(x, str)])
            return acc

        def load_label_file(path: str) -> List[str]:
            labels = []
            if path.lower().endswith(('.yml', '.yaml')):
                try:
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f)
                    labels = parse_yaml_structure(data)
                except Exception as e:
                    print(f"Warning: YAML parse failed ({e}); falling back to line parsing.")
            if not labels:
                # Plain / fallback parsing
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.startswith('-'):
                            line = line[1:].strip()
                        if '#' in line:
                            line = line.split('#', 1)[0].strip()
                        line = line.strip().strip("'").strip('"')
                        if line:
                            labels.append(line)
            # Normalize whitespace / quotes
            clean = []
            seen = set()
            for lab in labels:
                c = lab.strip().strip("'").strip('"')
                if c and c not in seen:
                    seen.add(c)
                    clean.append(c)
            return clean

        # If user provided a file
        if self.marker_list_path and os.path.isfile(self.marker_list_path):
            requested = load_label_file(self.marker_list_path)
            if not requested:
                print(f"Label file {self.marker_list_path} produced no labels. Using all markers.")
                return available_markers_raw

            # Case-insensitive matching with subject-aware aliases
            matched = []
            missing = []
            seen_upper = set()
            for r in requested:
                r_clean = r.strip()
                candidates = []
                ru = r_clean.upper()
                # Prefer subject-prefixed candidate when subject is set and r has no subject
                if subj and ":" not in r_clean:
                    candidates.append(f"{subj}:{r_clean}".upper())
                # Also try exactly as provided (case-insensitive)
                candidates.append(ru)

                hit = None
                for cand in candidates:
                    if cand in avail_map:
                        # Avoid duplicates via original's upper
                        orig_hit = avail_map[cand]
                        orig_key = orig_hit.upper()
                        if orig_key not in seen_upper:
                            seen_upper.add(orig_key)
                            hit = orig_hit
                            break
                if hit is not None:
                    matched.append(hit)
                else:
                    missing.append(r_clean)

            # print(f"Using {len(requested)} labels; matched {len(matched)}; missing {len(missing)}.")
            # if missing:
            #     print(f"Missing (ignored): {missing}")

            if len(matched) > 0:
                # print(f"Using {len(matched)} markers from {os.path.basename(self.marker_list_path)}.")
                return matched

        # Fallback
        # print(f"Using all {len(available_markers_raw)} available markers.")
        return available_markers_raw   
###
        
    def load_event_data(self):
        """Load event data efficiently using importAe, with interactive camera stream selection."""
        print("Loading event data...")

        importers = importAe(self.dvs_path)

        # from what i understand, importAe always returns a dict with 'data' key containing available streams
        if 'data' not in importers:
            raise KeyError("Missing 'data' in importAe output — check that dvs_path is correct.")

        data_keys = list(importers['data'].keys())
        if not data_keys:
            raise KeyError("No keys found under importers['data'] — event data not loaded properly.")

        if len(data_keys) == 1:
            middle_key = data_keys[0]
            print(f"Only one event stream found, using: '{middle_key}'")
        else:
            print("\nMultiple event streams detected:")
            for i, key in enumerate(data_keys):
                print(f"  [{i}] {key}")
            print()

            # Loop until valid input
            while True:
                user_input = input("Select event stream (type index number or name): ").strip()

                if user_input.isdigit():
                    # Number input
                    idx = int(user_input)
                    if 0 <= idx < len(data_keys):
                        middle_key = data_keys[idx]
                        break
                    else:
                        print("Invalid index. Please try again.")
                else:
                    # String input
                    if user_input in data_keys:
                        middle_key = user_input
                        break
                    else:
                        print("Invalid name. Please type one of:", data_keys)

        # Load event stream
        self.imp = importers['data'][middle_key]['dvs']

        self.start_time = self.imp.get_first_ts()
        self.end_time = self.imp.get_last_ts()

        print(f"\n✅ Loaded event stream: '{middle_key}'")
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
        
        # Auto-detect subject from C3D file if not provided
        if self.subject is None:
            self.subject = self._detect_subject_from_c3d()
            if self.subject:
                print(f"Auto-detected subject from C3D file: {self.subject}")
            else:
                print("No subject pattern detected in C3D marker names")
        
        # # Detect camera setup if auto
        # if self.camera_setup == "auto":
        #     self.camera_setup = self.detect_marker_setup()
        
        self.camera_setup = self.prompt_camera_setup()
        
        print(f"Camera setup specified: {self.camera_setup}")
        
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
        if init_file_path is None or not os.path.exists(init_file_path):
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
        
        self.Ts_world_to_system = vicon_helper.compute_camera_marker_transforms()
        
    def visualize_events(self):
        """Visualize event data for a specified duration."""
        
        print(f"Visualizing events")
        print("\n" + "="*60)
        print("EVENT VISUALIZATION - GUI INSTRUCTIONS")
        print("="*60)
        print("A window will show the raw event data stream.")
        print("This helps you understand the data before calibration.")
        print("\nControls:")
        print("  • q or ESC: Stop visualization")
        print("="*60)    
        
        img = np.ones(self.cam_res, dtype=np.uint8) * 255
        ft = self.start_time
        window_size = 1000 * self.period
        window_start = self.start_time
        
        cv2.namedWindow('Event Visualization', cv2.WINDOW_NORMAL)
        
        try:
            while ft < float("%.1f" % self.end_time):
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
        
    def manual_rotation_estimation(self, chosen_marker: Optional[str] = None) -> np.ndarray:
        """Manually estimate rotation using visual feedback with windowed approach."""
        print("Starting manual rotation estimation...")
        print("\n" + "="*60)
        print("MANUAL ROTATION ESTIMATION - GUI INSTRUCTIONS")
        print("="*60)
        print("A window will open showing event data with projected markers.")
        print("Use the following controls to manually adjust the camera rotation:")
        print("  • SPACE: Pause/resume event visualization")
        print("  • ENTER: Select rotation axis (roll/pitch/yaw)")
        print("  • +/-: Increase/decrease angle of selected axis by the current step size")
        print("  • k/l: Increase/decrease angle step size (default 0.5 degrees)")
        print("  • q or ESC: Finish rotation adjustment")
        print("\nGoal: Align the projected markers with the events as closely as possible.")
        print("Look for the feedback marker to get an idea of where on the event plane the markers are being projected:\n")

        # TODO: ask user input for initial rotation values and translations??
        
        # TODO: user input for chosen marker to track???

        # Get working markers
        self.markers_names = self.get_markers_names()
        
        if not self.markers_names:
            raise RuntimeError("No suitable markers found for calibration")
            
        # Choose marker for feedback
        if chosen_marker and chosen_marker in self.markers_names:
            chosen_one = chosen_marker
        else:
            chosen_one = self.markers_names[0]
            
        print(f"Using markers: {self.markers_names}")
        print(f"Feedback marker: {chosen_one}")
        
        # Create projector for manual adjustment
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d, 
            self.T_syst_to_camera_opt, self.Ts_world_to_system, 
            self.K, self.cam_res, D=self.D, subject=self.subject
        )
        
        # Use windowed approach
        window_size = 1000 * self.period  # 10 seconds window
        window_start = self.start_time
        rvec_init = np.zeros(3)
                        
        # TODO: find better solution than .1f 
        try:
            while window_start < self.end_time:  # float("%.1f" % self.end_time):
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
                
                R_init = Rotation.from_rotvec(rvec_init).as_matrix()

                # Call projector manual rotation adjustment
                rvec_init = projector.manual_rotation_adjustment(
                    self.marker_t, self.delay, e_ts, e_us, e_vs, 
                    self.period, R_init=R_init, visualize=True, 
                    chosen_one=chosen_one, marker_time_offset=window_start
                )
                
                window_start = window_end

                print("window_start updated to:", window_start)

        except helpers.RotationExit as e:
            print("Visualization stopped by user with final rotation.")
            rvec_init = e.r_vec

        finally:
            cv2.destroyAllWindows()
            print("rvec", rvec_init)
            return rvec_init

    def manual_delay_correction(self) -> float:
        """Manually correct synchronization delay using windowed approach."""
        print("Starting manual ...")
        
        print("Starting manual delay correction...")
        print("\n" + "="*60)
        print("MANUAL DELAY CORRECTION - GUI INSTRUCTIONS")
        print("="*60)
        print("A window will open showing event data with projected markers.")
        print("Use the following controls to manually adjust the delay to best synchronize the delay between events and vicon data:")
        print("  • SPACE: Pause/resume event visualization")
        print("  • +/-: Increase/decrease the delay by the current step size")
        print("  • k/l: Increase/decrease step size (default 0.1 seconds)")
        print("  • q or ESC: Finish delay adjustment")
        print("\nGoal: Time align the projected markers with the events as closely as possible.\n")
        
        if not self.markers_names:
            self.markers_names = self.get_markers_names()
        
        # Use windowed approach
        window_size = 1000 * self.period
        window_start = self.start_time
        
        # Create projector for delay adjustment
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )

        try:
            while window_start < float("%.1f" % self.end_time):
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
                
                print("e_ts final:", e_ts[-1], "window_start:", window_start, "window_size:", window_size)

                window_start = window_end
                
                print("window_start updated to:", window_start)
                
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
        
        if use_projections:
            print("\n" + "="*60)
            print("INTERACTIVE PROJECTION-BASED LABELING - GUI INSTRUCTIONS")
            print("="*60)
            print("A window will open showing event data frames.")
            print("After clicking a point on the screen, all the projections of the markers will appear.")
            print("You can label the marker by clicking on the projected points.")
            print("  • Left click: Select position / assign projected marker")
            print("  • SPACE: Skip current frame")
            print("  • S: Save labels for current frame and continue")
            print("  • BACKSPACE: Delete last labeled marker")
            print("  • q or ESC: Finish labeling process, save the results and exit")
            print("\nGoal: Label the marker using projections to have a faster interaction. These labels will then be used to construct 2D-3D correspondences")
        else:
            print("\n" + "="*60)
            print("INTERACTIVE MANUAL LABELING - GUI INSTRUCTIONS")
            print("="*60)
            print("A window will open showing event data frames.")
            print("After clicking a point on the screen, a list of markers will appear.")
            print("You will manually label marker positions in each frame by selecting the correct marker by inputting the relative number.")
            print("  • Left click: Select a point on the screen")
            print("  • Input the number corresponding to the marker to assign it")
            print("  • SPACE: Skip current frame")
            print("  • S: Save labels for current frame and continue")
            print("  • BACKSPACE: Delete last labeled marker")
            print("  • q or ESC: Finish labeling process, save the results and exit")
            print("\nGoal: Label the marker using a list containing the possible labels. These labels will then be used to construct 2D-3D correspondences")
        
        print("="*60)
        print()
        
        from helpers import DvsLabeler
        
        if not self.markers_names:
            self.markers_names = self.get_markers_names()
        
        # labels_path = os.path.join(os.path.dirname(self.output_path), 'labeled_points.yml')
        
        
        
        window_size = 1000 * self.period
        window_start = self.start_time
        
        # Create labeler instance
        labeler = DvsLabeler(img_shape=(self.cam_res[0], self.cam_res[1], 3), subject=self.subject)
        
        labeler.points_dict = {'points': [], 'times': []}
        
        try:
            while window_start < float("%.1f" % self.end_time):
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
                        self.markers_names, self.c3d_data, self.points_3d, self.marker_t,
                        self.T_syst_to_camera_opt, self.Ts_world_to_system,
                        self.K, self.cam_res, self.delay, D=self.D,
                        marker_time_offset=window_start
                    )
                    # labeler._merge_window_into_accumulated(window_dict)

                else:
                    input_label_tag_file = self.marker_list_path       # TODO: read from user input in parser
                    print("Using label tag file:", input_label_tag_file)
                    labeler.label_data(
                        e_ts, e_us, e_vs,
                        self.period, input_label_tag_file
                    )
                    # labeler._merge_window_into_accumulated(window_dict)

                window_start = window_end
                
        except helpers.LabelExit as e:
            print("Labeling stopped early by user, saving partial results.")
            # if hasattr(e, 'labeled_dict') and e.labeled_dict:
            #     labeler.labeled_dict = e.labeled_dict
            #     labeler.labels_done = True

        finally:
            cv2.destroyAllWindows()

            # Always merge final accumulated results
            if hasattr(labeler, "points_dict") and labeler.points_dict:
                labeler.labeled_dict = labeler.points_dict

                # Check if we actually have labels
                has_labels = (
                    'points' in labeler.labeled_dict
                    and any(len(p) > 0 for p in labeler.labeled_dict['points'])
                )

                if has_labels:
                    labeler.labels_done = True

                    # Define default save path if not provided
                    # labels_path = self.output_path

                    labeler.save_labeled_points(self.output_path)
                    print(f"Saved labeled points to: {self.output_path}")
                else:
                    print("No labeled points found — nothing saved.")
            else:
                print("Labeler has no labeled_dict or points_dict.")


        return self.output_path

    def create_projection_video(self, output_video: str = 'projection_video.mp4'):
        """Create video with projected markers using windowed approach and collect all image points."""
        print("Creating projection video...")
        
        if not self.markers_names:
            self.markers_names = self.get_markers_names()
            
        print("Using markers:", self.markers_names)

        # Create projector
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )
        
        collected_video_segments = []
        all_projected_points = {}  # Dictionary to collect all projected points
        window_size = 1000 * self.period
        window_start = self.start_time
        
        # Initialize projected points dictionary for each marker
        for marker_name in self.markers_names:
            all_projected_points[marker_name] = []
        
        try:
            while window_start < float("%.1f" % self.end_time): # TODO: properly fix this, as it is pretty bad like this
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                if len(e_ts) == 0:
                    window_start += window_size
                    continue

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

                # Collect projected points from this window
                if image_points is not None:
                    self._collect_projected_points_from_window(
                        image_points, all_projected_points, window_start
                    )

                window_start = window_end
                
                print(e_ts[-1], "window_start updated to:", window_start)

        except KeyboardInterrupt:
            print("Video creation stopped by user")

        finally:
            cv2.destroyAllWindows()
            
            # Save all collected projected points
            if all_projected_points and any(len(points) > 0 for points in all_projected_points.values()):
                points_file = os.path.join(os.path.dirname(output_video), "projected_points.txt")
                self._save_projected_points_to_file(all_projected_points, points_file)
            
            # Merge video segments (existing code)
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

###
    # TODO: check and fix
    def _collect_projected_points_from_window(self, image_points, all_projected_points, window_start):
        """Collect projected points from a processing window."""
        
        # Calculate the starting frame index for this window
        tic_markers = self.marker_t[0] + window_start - self.delay
        
        # Find the starting marker frame index
        start_frame_idx = 0
        while start_frame_idx < len(self.marker_t) and self.marker_t[start_frame_idx] < tic_markers:
            start_frame_idx += 1
        
        # Process each marker's projected points
        for marker_name in self.markers_names:
            if marker_name in image_points:
                marker_data = image_points[marker_name]
                
                if isinstance(marker_data, np.ndarray) and len(marker_data) > 0:
                    # Each point in marker_data corresponds to a frame starting from start_frame_idx
                    for i, coords in enumerate(marker_data):
                        frame_idx = start_frame_idx + i
                        
                        # Make sure we don't exceed the marker timeline
                        if frame_idx >= len(self.marker_t):
                            break
                        
                        marker_timestamp = self.marker_t[frame_idx]
                        
                        # Store the point with its timestamp
                        point_data = {
                            'timestamp': marker_timestamp,
                            'frame_idx': frame_idx,
                            'coords': coords.copy() if hasattr(coords, 'copy') else list(coords)
                        }
                        
                        all_projected_points[marker_name].append(point_data)

    # TODO: do this so that the user can choose the format, i.e.: .csv, .txt, .h5, etc...
    def _save_projected_points_to_file(self, all_projected_points, output_file):
        """Save all collected projected points to a text file in the requested format."""
        
        points_written = 0
        total_markers = len([marker for marker in all_projected_points.keys() if len(all_projected_points[marker]) > 0])
        
        print(f"Saving projected points for {total_markers} markers...")
        
        with open(output_file, 'w') as f:
            f.write("# Projected VICON Marker Points\n")
            f.write("# Generated by VICON-DVS Pipeline\n")
            f.write(f"# Camera resolution: {self.cam_res[1]}x{self.cam_res[0]}\n")
            f.write(f"# Delay: {self.delay:.6f}s\n")
            f.write(f"# Period: {self.period:.6f}s\n")
            f.write("#\n")
            f.write("# Format: marker_timestamp: x, y # marker_name\n")
            f.write("#\n")
            
            # Collect all points with timestamps and sort them
            all_timestamped_points = []
                        
            # TODO: use event timestamps or marker timestamps?

            for marker_name, points_list in all_projected_points.items():
                for point_data in points_list:
                    coords = point_data['coords']
                    timestamp = point_data['timestamp']
                    
                    # Check if coordinates are valid (not NaN or inf)
                    if (len(coords) >= 2 and 
                        np.isfinite(coords[0]) and np.isfinite(coords[1])):
                        
                        all_timestamped_points.append({
                            'timestamp': timestamp,
                            'marker_name': marker_name,
                            'x': coords[0],
                            'y': coords[1]
                        })
            
            all_timestamped_points.sort(key=lambda x: x['timestamp'])
            
            # Write all points
            for point in all_timestamped_points:
                f.write(f"{point['timestamp']:.6f}: {point['x']:.2f}, {point['y']:.2f} # {point['marker_name']}\n")
                points_written += 1
        
        print(f"Saved {points_written} projected points to: {output_file}")
###

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

###        
    # TODO: check this and fix it
    def optimize_calibration(self, labels_path: str):
        """Optimize calibration using dynamic multi-frame PnP + robust averaging + global refinement."""
        print("Optimizing calibration (multi-frame robust)...")
        from helpers import ViconHelper

        # 1. Load labeled points
        labeled_points = helpers.read_points_labels(labels_path)

        # 2. Prepare VICON helper & interpolated 3D points at label times
        vicon_helper = ViconHelper(
            self.marker_t, self.points_3d, self.delay,
            self.c3d_data.frame_count, self.c3d_data.point_rate,
            self.c3d_data.point_labels, True, True
        )
        vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)

        # TODO: check these out, as i don't know if it really makes sense

        ### Internal functions for optimization steps
        def collect_frame_correspondences(labeled_points, vicon_points):
            per_frame = []
            for idx, (dvs_frame, v_frame) in enumerate(zip(labeled_points['points'], vicon_points['points'])):
                world_pts = []
                image_pts = []
                for m, xy in dvs_frame.items():
                    if m in v_frame:
                        w = v_frame[m]
                        if w is None or np.any(np.isnan(w)):
                            continue
                        world_pts.append(w)
                        image_pts.append([xy['x'], xy['y']])
                if len(world_pts) >= 4:
                    per_frame.append({
                        "frame_idx": vicon_points['frame_ids'][idx],
                        "world": np.asarray(world_pts, dtype=np.float64),
                        "image": np.asarray(image_pts, dtype=np.float64),
                        "timestamp": labeled_points['times'][idx]
                    })
            return per_frame

        def solve_pnp_candidates(per_frame, Ts_world_to_system, K, D,
                                 use_ransac=True, refine=True,
                                 ransac_reproj_err=3.0, ransac_iters=200):
            candidates = []
            for rec in per_frame:
                wpts = rec["world"]
                ipts = rec["image"]
                if use_ransac:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        wpts, ipts, K, D,
                        iterationsCount=ransac_iters,
                        reprojectionError=ransac_reproj_err,
                        flags=cv2.SOLVEPNP_EPNP
                    )
                    if not ok:
                        continue
                    if refine and inliers is not None and len(inliers) >= 4:
                        w_in = wpts[inliers.flatten()]
                        i_in = ipts[inliers.flatten()]
                        ok2, rvec, tvec = cv2.solvePnP(
                            w_in, i_in, K, D,
                            rvec=rvec, tvec=tvec,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        if not ok2:
                            continue
                else:
                    ok, rvec, tvec = cv2.solvePnP(
                        wpts, ipts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if not ok:
                        continue

                R_wc, _ = cv2.Rodrigues(rvec)
                T_world_to_cam = np.eye(4)
                T_world_to_cam[:3, :3] = R_wc
                T_world_to_cam[:3, 3] = tvec.flatten()

                frame_sys = self.Ts_world_to_system[rec["frame_idx"]]
                T_sys_to_cam_candidate = T_world_to_cam @ np.linalg.inv(frame_sys)

                proj, _ = cv2.projectPoints(wpts, rvec, tvec, K, D)
                proj = proj.reshape(-1, 2)
                err = np.linalg.norm(proj - ipts, axis=1)
                mean_err = float(np.mean(err))

                candidates.append({
                    "frame_idx": rec["frame_idx"],
                    "timestamp": rec["timestamp"],
                    "T_world_to_cam": T_world_to_cam,
                    "T_sys_to_cam": T_sys_to_cam_candidate,
                    "mean_err": mean_err,
                    "n_points": len(wpts)
                })
            return candidates

        def robust_average_system_to_camera(candidates,
                                            rot_thresh_deg=5.0,
                                            trans_thresh=0.10,
                                            min_inliers=5,
                                            eps=1e-6):
            if not candidates:
                raise RuntimeError("No candidates to average.")

            Rs = np.stack([c["T_sys_to_cam"][:3, :3] for c in candidates], axis=0)
            ts = np.stack([c["T_sys_to_cam"][:3, 3] for c in candidates], axis=0)
            errs = np.asarray([c["mean_err"] for c in candidates], dtype=np.float64)

            w = 1.0 / (errs**2 + eps)
            w /= np.sum(w)

            quats = Rotation.from_matrix(Rs).as_quat()
            q_mean = np.average(quats, axis=0, weights=w)
            q_mean /= np.linalg.norm(q_mean)
            R_mean = Rotation.from_quat(q_mean).as_matrix()

            def rot_geodesic_deg(Ra, Rb):
                dR = Ra.T @ Rb
                angle = np.clip((np.trace(dR) - 1) / 2, -1, 1)
                return np.degrees(np.arccos(angle))

            rot_errs = np.array([rot_geodesic_deg(R_mean, Ri) for Ri in Rs])
            trans_center = np.average(ts, axis=0, weights=w)
            trans_dists = np.linalg.norm(ts - trans_center, axis=1)

            inliers = (rot_errs <= rot_thresh_deg) & (trans_dists <= trans_thresh)
            if np.sum(inliers) < min_inliers:
                order = np.argsort(errs)
                inliers = np.zeros_like(errs, dtype=bool)
                inliers[order[:min_inliers]] = True

            Rs_in = Rs[inliers]
            ts_in = ts[inliers]
            errs_in = errs[inliers]
            w_in = 1.0 / (errs_in**2 + eps)
            w_in /= np.sum(w_in)

            quats_in = Rotation.from_matrix(Rs_in).as_quat()
            q_mean2 = np.average(quats_in, axis=0, weights=w_in)
            q_mean2 /= np.linalg.norm(q_mean2)

            R_final = Rotation.from_quat(q_mean2).as_matrix()
            t_final = np.average(ts_in, axis=0, weights=w_in)

            T_avg = np.eye(4)
            T_avg[:3, :3] = R_final
            T_avg[:3, 3] = t_final

            summary = {
                "num_candidates": len(candidates),
                "num_inliers": int(np.sum(inliers)),
                "rot_outlier_thresh_deg": rot_thresh_deg,
                "trans_outlier_thresh": trans_thresh,
                "inlier_mean_error": float(np.mean(errs_in)),
                "inlier_std_error": float(np.std(errs_in))
            }
            return T_avg, inliers, summary

        # 3. Per-frame correspondences
        print("Collecting per-frame correspondences...")
        per_frame = collect_frame_correspondences(labeled_points, vicon_points)
        print(f"Frames with >=4 correspondences: {len(per_frame)}")

        # 4. Solve per-frame PnP
        print("Solving per-frame PnP...")
        candidates = solve_pnp_candidates(per_frame, self.Ts_world_to_system, self.K, self.D)
        print(f"Valid PnP candidates: {len(candidates)}")
        if not candidates:
            print("No candidates produced. Falling back to legacy single-batch PnP.")
            # (Fallback: original approach)
            world_points = []
            image_points_clean = []
            for dvs_frame, v_frame in zip(labeled_points['points'], vicon_points['points']):
                for label, xy in dvs_frame.items():
                    if label not in v_frame:
                        continue
                    w_p = v_frame[label]
                    if w_p is None or np.any(np.isnan(w_p)):
                        continue
                    world_points.append(w_p)
                    image_points_clean.append([xy['x'], xy['y']])
            world_points = np.asarray(world_points, dtype=np.float64)
            image_points_clean = np.asarray(image_points_clean, dtype=np.float64)
            print(f"Collected {len(world_points)} correspondences (fallback).")
            if len(world_points) >= 4:
                ok, rvec, tvec = cv2.solvePnP(world_points, image_points_clean, self.K, self.D)
                if ok:
                    R_mat, _ = cv2.Rodrigues(rvec)
                    T_world_to_cam = np.eye(4)
                    T_world_to_cam[:3, :3] = R_mat
                    T_world_to_cam[:3, 3] = tvec.flatten()
                    init_T = T_world_to_cam @ np.linalg.inv(self.Ts_world_to_system[0])
                    r_vec = Rotation.from_matrix(init_T[:3, :3]).as_rotvec()
                    t_vec = init_T[:3, 3]
                    init_param = np.concatenate((r_vec, t_vec))
                    # Build system/image lists
                    system_points = []
                    image_points = []
                    for dvs_frame, v_frame, fid in zip(labeled_points['points'],
                                                       vicon_points['points'],
                                                       vicon_points['frame_ids']):
                        for label, xy in dvs_frame.items():
                            if label not in v_frame:
                                continue
                            w_p = v_frame[label]
                            if w_p is None or np.any(np.isnan(w_p)):
                                continue
                            w_ph = np.append(w_p, 1.0)
                            p_sys = self.Ts_world_to_system[fid] @ w_ph
                            system_points.append(p_sys[:3])
                            image_points.append([xy['x'], xy['y']])
                    system_points = np.asarray(system_points, dtype=np.float64)
                    image_points = np.asarray(image_points, dtype=np.float64)
                    self.T_syst_to_camera_opt = self.estimate_transformation(system_points, image_points, init_param)
                    print("Calibration optimization (fallback) completed.")
                else:
                    print("Fallback PnP failed.")
            else:
                print("Insufficient correspondences for fallback.")
            return

        # 5. Robust averaging
        print("Robust averaging system->camera candidates...")
        T_avg, inliers_mask, summary = robust_average_system_to_camera(candidates)
        print("Averaging summary:", summary)
        print("Averaged T_sys->cam:\n", T_avg)

        # 6. Build global correspondences in system coordinates
        system_points = []
        image_points = []
        for idx, (dvs_frame, v_frame) in enumerate(zip(labeled_points['points'], vicon_points['points'])):
            frame_id = vicon_points['frame_ids'][idx]
            for label, xy in dvs_frame.items():
                if label not in v_frame:
                    continue
                w_p = v_frame[label]
                if w_p is None or np.any(np.isnan(w_p)):
                    continue
                w_ph = np.append(w_p, 1.0)
                p_sys = self.Ts_world_to_system[frame_id] @ w_ph
                system_points.append(p_sys[:3])
                image_points.append([xy['x'], xy['y']])
        system_points = np.asarray(system_points, dtype=np.float64)
        image_points = np.asarray(image_points, dtype=np.float64)
        print(f"Global correspondences: {len(system_points)}")

        # 7. Refinement (least squares)
        print("Refining with global least-squares...")
        r0 = Rotation.from_matrix(T_avg[:3, :3]).as_rotvec()
        t0 = T_avg[:3, 3]
        init_param = np.concatenate([r0, t0])
        self.T_syst_to_camera_opt = self.estimate_transformation(system_points, image_points, init_param)

        # 8. Diagnostics
        rotations = [c["T_sys_to_cam"][:3, :3] for c in candidates]
        translations = [c["T_sys_to_cam"][:3, 3] for c in candidates]
        R_ref = self.T_syst_to_camera_opt[:3, :3]
        rot_dists = []
        for Ri in rotations:
            dR = Ri.T @ R_ref
            ang = np.clip((np.trace(dR) - 1) / 2, -1, 1)
            rot_dists.append(np.degrees(np.arccos(ang)))
        rot_dists = np.array(rot_dists)
        trans_errs = np.linalg.norm(np.stack(translations) - self.T_syst_to_camera_opt[:3, 3], axis=1)

        print(f"Rotation dispersion (deg): mean={rot_dists.mean():.3f} std={rot_dists.std():.3f} max={rot_dists.max():.3f}")
        print(f"Translation dispersion: mean={trans_errs.mean():.4f} std={trans_errs.std():.4f} max={trans_errs.max():.4f}")

        print("Calibration optimization completed (robust multi-frame).")
###           
            
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
        
    # def save_transforms_txt(self, output_file: str):
    #     """Save all transformation matrices to a text file."""
    #     with open(output_file, 'w') as f:
    #         f.write("# Transformation matrices from world to system coordinates\n")
    #         f.write("# Format: timestamp: array([4x4 transformation matrix])\n\n")
            
    #         for timestamp, transform_matrix in zip(self.marker_t, self.Ts_world_to_system):
    #             f.write(f"{timestamp:.6f}: array({transform_matrix.tolist()})\n")
                
    #     print(f"Saved transformation matrices to {output_file}")
        
    def run_full_pipeline(self, manual_calibration: bool = True, 
        use_projections: bool = False, create_video: bool = True,
        init_file_path: str = None, chosen_marker: str = None):
        
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
            tvec_init = np.array([0.0, 0.0, 0.0])       # TODO: make it user-input
            rvec_init = self.manual_rotation_estimation(
                chosen_marker=chosen_marker
            )
            
            # Update transformation matrix
            self.T_syst_to_camera_opt[:3, :3] = cv2.Rodrigues(rvec_init)[0]
            self.T_syst_to_camera_opt[:3, 3] = tvec_init
            
            # 5. Manual delay correction
            self.delay = self.manual_delay_correction()
            
            # 6. Interactive labeling or use existing labels
            self.label_data_interactive(use_projections=use_projections)
            
            # 7. Optimize calibration
            self.optimize_calibration(self.output_path)
            
            # 8. Save calibration
            self.save_calibration(init_file)
            
        # 9. Save transformation matrices
        # transforms_file = os.path.join(os.path.dirname(self.vicon_path), "transformation_matrices.txt")
        # self.save_transforms_txt(transforms_file)
        
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
            
            # Projected points are automatically saved during video creation
            projected_points_file = os.path.join(os.path.dirname(self.vicon_path), "projected_points.txt")
            print(f"Projected points saved to: {projected_points_file}")
                
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
                        chosen_marker=chosen_marker
                    )
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
        else:
            # If no video creation, just complete the pipeline
            print("Pipeline completed successfully!")
            return
        
    # TODO: step 11: calibration error and report of accuracy, check notebook for 2d-2d error check
    # TODO: if depth is provided, get 2d projections put them in 3d and check with vicon data as to have a more accurate analysis
    
def main():
    parser = argparse.ArgumentParser(
        prog='VICON markers projection',
        description='Project VICON markers onto DVS frames using time-synchronized windows'
    )
    
    parser.add_argument('--dvs_path', required=True,
                       help='REQUIRED: Path to the YARP folder containing DVS recording')
    parser.add_argument('--vicon_path', required=True,
                       help='REQUIRED: Path to the .c3d file containing VICON recording')
    parser.add_argument('--intrinsic', required=True,
                       help='REQUIRED: path directing to the intrinsic calibration file for the camera')
    parser.add_argument('--init_file', default=None,
                       help='Path to initialization file containing transformation matrix and delay')
    parser.add_argument('--output_path', required=True,
                       help='REQUIRED: Output path for labeled points (YAML file)')
    parser.add_argument('--subject', default=None,      # TODO: needed only for hpe, maybe read the subject from the c3d file?
                       help='Subject name for labels (e.g., P1, P11)')
    parser.add_argument('--marker_list_path', default=None,
                       help='Path to a text or YAML file listing desired marker labels')
    parser.add_argument('--camera_setup', choices=['single', 'multi', 'auto'], default='auto',      # TODO: remove this and try to look for known words in the c3d file
                       help='Camera marker setup: single marker, multi-marker, or auto-detect')
    parser.add_argument('--chosen_marker', default=None,            # TODO: not really that usefult as it is read to be the first element of array of markers, but can be useful depending on c3d structure
                       help='Specific marker to use for rotation adjustment feedback')
    parser.add_argument('--list', action='store_true',
                       help='Use list-based labeling interface, N.B. there are two ways to do this, by apt installing tkinter or by using the terminal interface, default is tkinter, modify line 1085 in helpers.py to change between the two')
    parser.add_argument('--projections', action='store_true',
                       help='Use projection-based labeling interface')
    parser.add_argument('--no-manual', action='store_true',
                       help='Skip manual calibration if existing calibration found')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video creation')
    parser.add_argument('--visualize_events', action='store_true',
                       help='Visualize events')
    
    # TODO: add argument to let the user choose the size of the time windows????
    
    args = parser.parse_args()
    
    # Require --marker_list_path when --list is used
    if args.list and args.marker_list_path is None:
        parser.error("--marker_list_path is required when using --list")
    
    # TODO: check process_camera_markers, as it uses known camera names that aren't actually a thing, so maybe let the user input a file in which it states the camera names and their setups?
    # TODO: otherwise look for some known words in markers as 'cam', 'cammera', 'dvs', etc and if they are not found ask the user to input the file with their name
    # TODO: same also for camera_setup and stuff
    
    # TODO: add a button R to restart sequences from the beginning
    
    # Create pipeline
    pipeline = ViconDVSPipeline(
        dvs_path=args.dvs_path,
        vicon_path=args.vicon_path,
        intrinsic_path=args.intrinsic,
        subject=args.subject,
        output_path=args.output_path,
        camera_setup=args.camera_setup,
        marker_list_path=args.marker_list_path
    )
    
    # Visualize events if requested
    if args.visualize_events:
        pipeline.load_event_data()
        pipeline.load_calibration_data()
        pipeline.visualize_events()
        return
    
    # Run full pipeline
    pipeline.run_full_pipeline(
        manual_calibration=not args.no_manual,
        use_projections=args.projections,
        create_video=not args.no_video,
        init_file_path=args.init_file,
        chosen_marker=args.chosen_marker
    )
    

if __name__ == "__main__":
    main()