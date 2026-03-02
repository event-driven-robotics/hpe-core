# Different version of the pipeline to work specifically on the static camera setup from MoveIIT dataset
# Please tell me if you need any specific changes or additions to this code

import sys
import os
import yaml
import csv          # TODO: switch to pandas?
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect
import c3d
from typing import List, Optional
import argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
 
# Get the absolute path to the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
 
# Add needed paths dynamically
HELPERS_PATH = os.path.join(CURRENT_DIR)
BIMVEE_PATH = os.path.join(CURRENT_DIR, "submodules/bimvee")
 
for path in [HELPERS_PATH, BIMVEE_PATH]:
    if path not in sys.path:
        sys.path.append(path)
 
import helpers

from bimvee.importAe import importAe


class ViconDVSPipeline:
    """Complete pipeline for VICON-DVS calibration and projection."""
    
    def __init__(self, dvs_path: str, vicon_path: str, intrinsic_path: str, 
                 subject: Optional[str], output_path: str, camera_setup: str = "auto", marker_list_path: Optional[str] = None):
        self.dvs_path = dvs_path
        self.vicon_path = vicon_path
        self.intrinsic_path = intrinsic_path
        self.subject = subject                  # actually changed to remove this parameter, TODO: check usage
        self.output_path = output_path          # path for yaml file, will be removed after method works
        self.camera_setup = camera_setup        # either single or multiple markers for camera
        self.marker_list_path = marker_list_path
        self.period = 1.0 / 100                 # VICON frequency: 100Hz
        
        # Initialize data containers
        self.imp = None
        self.start_time = None
        self.end_time = None
        self.c3d_data = None
        self.points_3d = {}
        self.marker_t = None
        self.marker_names = []
        self.markers_names = []                 # Markers list from yaml file
        self.camera_markers = []                # Camera markers identified by user
        self.camera_marker_patterns = []        # User input patterns for camera markers
        self.K = None
        self.D = None
        self.cam_res = None
        self.T_syst_to_camera_opt = np.eye(4)
        self.delay = 0.0
        self.current_delay_step = 0.01          # Default delay step for manual adjustments
        self.Ts_world_to_system = None    

    def _detect_subject_from_c3d(self, all_markers: Optional[list] = None) -> Optional[str]:
        """Use provided markers or fall back to self.marker_names if available"""

        subject_counts = {}
        
        for marker_name in all_markers:
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

    def _find_matching_markers(self, pattern: str) -> list:
        """Find markers that match a given pattern (exact name or prefix)."""

        matches = []
        pattern_upper = pattern.upper()
        
        for marker in self.marker_names:
            marker_upper = marker.upper()
            
            # Exact match
            if marker_upper == pattern_upper:
                matches.append(marker)
            # Prefix match (if pattern ends with ':' or is a substring)
            elif (pattern.endswith(':') and marker_upper.startswith(pattern_upper)) or \
                 (not pattern.endswith(':') and pattern_upper in marker_upper):
                matches.append(marker)
        
        return matches

    #TODO: double check
    def _find_matching_markers_cleaned(self, name, all_c3d_markers) -> Optional[str]:
        """Find C3D marker that matches the joint name using cleaned label matching logic."""

        clean_name = name.strip()
        
        candidates = []
        
        # Add subject-prefixed version if subject exists
        if self.subject:
            candidates.append(f"{self.subject}:{clean_name}")
        
        candidates.append(clean_name)
        
        # Search through C3D markers using case-insensitive matching
        for marker in all_c3d_markers:
            marker_clean = marker.strip()
            
            # Check all candidate names
            for candidate in candidates:
                if marker_clean.upper() == candidate.upper():
                    return marker_clean
                
                # Check if C3D marker ends with the joint name (for subject prefixes)
                if ":" in marker_clean and marker_clean.upper().endswith(f":{clean_name.upper()}"):
                    return marker_clean
        
        # No match found fallback
        print(f"No match for: '{clean_name}' (tried candidates: {candidates})")
        return None

    #TODO: change so that args takes as input the folder directly
    def _get_output_directory(self) -> str:
        """Get the output directory from the output_path."""

        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _extract_sequence_name(self) -> str:
        """Extract sequence name from DVS or VICON path for file naming."""

        # Try to extract from vicon_path first (C3D file)
        if self.vicon_path:
            vicon_basename = os.path.basename(self.vicon_path)
            # Remove .c3d extension and use as sequence name
            sequence_name = os.path.splitext(vicon_basename)[0]
            if sequence_name:
                print(f"Extracted sequence name from VICON path: {sequence_name}")
                return sequence_name
        
        # Fallback to DVS path
        if self.dvs_path:
            dvs_basename = os.path.basename(self.dvs_path.rstrip('/'))
            if dvs_basename:
                print(f"Extracted sequence name from DVS path: {dvs_basename}")
                return dvs_basename
        
        # Final fallback
        return "sequence"

    #TODO: check again if it is more useful or confusing
    def _generate_unique_init_file_path(self, base_dir: str = None) -> str:
        """Generate a unique init file path using sequence name and avoiding overwrites."""

        if base_dir is None:
            base_dir = self._get_output_directory() #TODO: remove and let user input via args
        
        # Get sequence name for the file
        sequence_name = self._extract_sequence_name()
        
        base_filename = f"{sequence_name}_init_file.txt"
        init_file_path = os.path.join(base_dir, base_filename)
        
        # If file doesn't exist, use it as is
        if not os.path.exists(init_file_path):
            return init_file_path
        
        # If file exists, add number of iterations at the end
        i = 1
        while True:
            filename_with_iter = f"{sequence_name}_init_file_{i}.txt"
            init_file_path = os.path.join(base_dir, filename_with_iter)
            if not os.path.exists(init_file_path):
                return init_file_path
            i += 1

    #TODO: check if it is needed or if i can do it directly in the different steps
    def _track_delay_change(self, initial_delay: float, init_file: str) -> tuple[bool, str]:
        """Track and save delay changes during different pipeline phases."""

        delay_changed = abs(self.delay - initial_delay) > 1e-6
        updated_file_path = init_file  # Default to original file
        
        if delay_changed:
            print(f" Delay adjusted: {initial_delay:.6f}s → {self.delay:.6f}s")
            # Save to sequence-specific file instead of overwriting original
            calibration_file = self._generate_unique_init_file_path()
            self.save_calibration(calibration_file)
            updated_file_path = calibration_file
            print(f" Saved sequence-specific calibration: {os.path.basename(calibration_file)}")
            return True, updated_file_path
        
        return False, updated_file_path

    #TODO: again check for user input in args
    def _generate_unique_video_path(self, base_path: str) -> str:
        """Generate unique video file path to avoid overwrites."""

        if not os.path.exists(base_path):
            return base_path
        
        # Extract directory, filename, and extension
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        
        # Add iteration number until we find a unique name
        i = 1
        while True:
            new_filename = f"{name}_{i}{ext}"
            new_path = os.path.join(directory, new_filename)
            if not os.path.exists(new_path):
                return new_path
            i += 1

    def _visualize_marker_trajectory(self, marker_name: str):
        """Visualize 3D marker trajectory with filtering and statistics.
        
        Creates a 2x2 plot showing:
        - 3D trajectory plot with start/end markers
        - Position vs time (raw + filtered)  
        - Velocity components and magnitude     # TODO: check if it's actually useful
        - XY projection of trajectory
        
        Uses gaussian filtering if scipy available, else moving average fallback.
        """
        
        print(f"Visualizing trajectory for marker: {marker_name}")
        
        # Extract and validate marker data
        marker_points = helpers.marker_p(self.c3d_data.point_labels, 
                                       list(self.points_3d.values()), 
                                       marker_name)
        if marker_points is None or len(marker_points) == 0:
            print(f"No data found for marker '{marker_name}'")
            return
            
        # Convert to numpy arrays for easier processing
        pts = np.asarray(marker_points)
        times = np.asarray(self.marker_t[:len(pts)])
        
        # Setup figure and subplots
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(f"Marker Trajectory Analysis: {marker_name}", fontsize=14)
        
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_pos = fig.add_subplot(2, 2, 2)  
        ax_vel = fig.add_subplot(2, 2, 3)
        ax_xy = fig.add_subplot(2, 2, 4)
        
        # Apply smoothing filter (prefer gaussian, fallback to moving average)
        pts_filtered, filter_type = self._apply_position_filter(pts)
        
        # Plot 3D trajectory
        self._plot_3d_trajectory(ax_3d, pts)
        
        # Plot position vs time (raw + filtered)
        self._plot_position_time(ax_pos, times, pts, pts_filtered)
        
        # Calculate and plot velocities  
        self._plot_velocity_analysis(ax_vel, times, pts, pts_filtered, filter_type)
        
        # Plot XY projection
        self._plot_xy_projection(ax_xy, pts)
        
        # Finalize and display
        plt.tight_layout()
        plt.show()

    def _apply_position_filter(self, pts):
        """Apply position filtering - gaussian if available, moving average otherwise."""
        try:
            from scipy.ndimage import gaussian_filter1d
            sigma = 2.0
            pts_filtered = np.column_stack([
                gaussian_filter1d(pts[:, i], sigma=sigma) for i in range(3)
            ])
            return pts_filtered, "gaussian"
        except ImportError:
            print("  Note: Using moving average filter (scipy not available)")
            window = 5
            
            def moving_average(data, window=window):
                return np.convolve(data, np.ones(window)/window, mode='same')
                
            pts_filtered = np.column_stack([
                moving_average(pts[:, i]) for i in range(3)
            ])
            return pts_filtered, "moving_average"
    
    def _plot_3d_trajectory(self, ax, pts):
        """Plot 3D trajectory with start/end markers."""
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=1)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], 
                  c='green', s=50, label='Start')
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], 
                  c='red', s=50, label='End')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')  
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Trajectory')
        ax.legend()
    
    def _plot_position_time(self, ax, times, pts_raw, pts_filtered):
        """Plot position components vs time (raw and filtered)."""
        colors = ['red', 'green', 'blue'] 
        labels = ['X', 'Y', 'Z']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax.plot(times, pts_raw[:, i], color=color, alpha=0.3, 
                   linewidth=0.6, label=f'{label} raw')
            ax.plot(times, pts_filtered[:, i], color=color, 
                   linewidth=1.5, label=f'{label} filtered')
                   
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (mm)')
        ax.set_title('Position vs Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_velocity_analysis(self, ax, times, pts_raw, pts_filtered, filter_type):
        """Plot velocity components and magnitude."""
        # Calculate velocity magnitudes
        v_raw = np.linalg.norm(np.diff(pts_raw, axis=0), axis=1) / self.period
        v_filtered = np.linalg.norm(np.diff(pts_filtered, axis=0), axis=1) / self.period
        
        # Plot velocity magnitudes
        ax.plot(times[1:], v_raw, 'gray', alpha=0.4, linewidth=0.6, 
               label='|V| raw')
        ax.plot(times[1:], v_filtered, 'purple', linewidth=1.5, 
               label='|V| filtered')
               
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (mm/s)')
        ax.set_title('Velocity Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_xy_projection(self, ax, pts):
        """Plot XY projection with start/end markers."""
        ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1)
        ax.scatter(pts[0, 0], pts[0, 1], c='green', s=40, label='Start')
        ax.scatter(pts[-1, 0], pts[-1, 1], c='red', s=40, label='End')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('XY Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def prompt_camera_setup(self) -> tuple:
        """Hard-coded information for the staticDVS sequences in MoveIIT dataset."""
       
        # Specific for marker of static camera
        setup = "single"
        n = 1
 
        # Show available markers for reference
        print(f"\nAvailable markers in C3D file ({len(self.marker_names)} total):")
        for i, marker in enumerate(self.marker_names):
            print(f"  {i+1:2d}. {marker}")
       
        camera_markers = []
        identified_markers = []
       
        i = 0
        while i < n:
            print(f"\n--- Camera marker {i+1} of {n} ---")
            while True:
                marker_input = "*118" # "*118" # "oweuinhvuihiouivnr"      # Specific marker for static camera in moveIIT
               
                # Find matching markers
                matches = self._find_matching_markers(marker_input)
               
               # TODO: check after everything fixing also the markers plotting functions
                ###
                if not matches:
                    print(f" No markers found matching '{marker_input}'")
                    print(f" Expected marker '*118' not found in dataset. Switching to interactive mode.")
                    print(f"\nAll available markers in C3D file:")
                    for idx, marker in enumerate(self.marker_names):
                        print(f"  {idx+1:3d}. {marker}")
                    
                    print(f"\nPlease select the camera marker:")
                    while True:
                        try:
                            user_input = input("Enter marker number or exact marker name: ").strip()
                            
                            # Check if it's a number (index)
                            if user_input.isdigit():
                                marker_idx = int(user_input) - 1  # Convert to 0-based index
                                if 0 <= marker_idx < len(self.marker_names):
                                    selected_marker = self.marker_names[marker_idx]
                                    print(f" Selected: {selected_marker}")
                                    
                                    # Visualize the marker trajectory
                                    self._visualize_marker_trajectory(selected_marker)
                                    
                                    # Confirm selection after optional visualization
                                    while True:
                                        confirm_choice = input(f"Confirm '{selected_marker}' as camera marker? (y/n): ").strip().lower()
                                        if confirm_choice in ['y', 'yes']:
                                            identified_markers.append(selected_marker)
                                            camera_markers.append(selected_marker)  # Use actual marker name instead of pattern
                                            i += 1
                                            break
                                        elif confirm_choice in ['n', 'no']:
                                            print("Please select a different marker.")
                                            break
                                        else:
                                            print("Please enter 'y' or 'n'")
                                    
                                    if confirm_choice in ['y', 'yes']:
                                        break  # Exit marker selection loop
                                else:
                                    print(f" Invalid index. Please enter a number between 1 and {len(self.marker_names)}")
                            else:
                                # Check if it's an exact marker name
                                if user_input in self.marker_names:
                                    print(f" Selected: {user_input}")
                                    
                                    # Visualize the marker trajectory
                                    self._visualize_marker_trajectory(user_input)
                                    
                                    # Confirm selection after optional visualization
                                    while True:
                                        confirm_choice = input(f"Confirm '{user_input}' as camera marker? (y/n): ").strip().lower()
                                        if confirm_choice in ['y', 'yes']:
                                            identified_markers.append(user_input)
                                            camera_markers.append(user_input)  # Use actual marker name
                                            i += 1
                                            break
                                        elif confirm_choice in ['n', 'no']:
                                            print("Please select a different marker.")
                                            break
                                        else:
                                            print("Please enter 'y' or 'n'")
                                    
                                    if confirm_choice in ['y', 'yes']:
                                        break  # Exit marker selection loop
                                else:
                                    print(f" Marker '{user_input}' not found. Please try again.")
                        except KeyboardInterrupt:
                            print("\n User interrupted camera marker selection.")
                            raise  # Re-raise to exit properly
                        except ValueError:
                            print(" Invalid input. Please enter a number or exact marker name.")
                    break  # Exit the inner while loop after successful selection
                ###

                # Show matches and let user confirm
                if len(matches) == 1:
                    print(f" Found: {matches[0]}")
                    identified_markers.extend(matches)
                    camera_markers.append(marker_input)
                    i += 1
                    break
                else:
                    print(f" Found {len(matches)} matches: {matches}")
                    if len(matches) <= 10:  # Show all if reasonable number
                        confirm = input("Use all these markers? (y/n): ").lower().strip()
                        if confirm in ['y', 'yes']:
                            markers_to_take = min(len(matches), n - i)
                            selected_matches = matches[:markers_to_take]
                            identified_markers.extend(selected_matches)
                            camera_markers.append(marker_input)
                           
                            if markers_to_take > 1:
                                print(f"Using {markers_to_take} markers:")
                                for j, match in enumerate(selected_matches):
                                    print(f"  Camera marker {i+j+1}: {match}")
                           
                            i += markers_to_take
                            break
                        else:
                            print("Please provide a more specific name or prefix.")
                    else:
                        print("Too many matches! Please be more specific.")
                        confirm = input("Show all matches? (y/n): ").lower().strip()
                        if confirm in ['y', 'yes']:
                            for match in matches:
                                print(f"  - {match}")
       
        print(f"\n Camera setup summary:")
        print(f"   Setup type: {setup}")
        print(f"   Identified markers: {identified_markers}")
        print(f"   Search patterns used: {camera_markers}")
       
        return setup, camera_markers

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # TODO: check fuctioning after the cleaning
    # the yaml file is expected to have a specific configuration following the format of labels_tags.yml, TODO: generalize it
    def get_markers_names(self) -> List[str]:
        """Return list of markers to use from user input yaml file."""

        available = [m.strip() for m in self.c3d_data.point_labels]

        subj = (self.subject or "").strip()
        subj_prefix = f"{subj}:" if subj else ""
        subj_upper = subj_prefix.upper()

        alias_map = {}
        for orig in available:
            up = orig.upper()
            alias_map.setdefault(up, orig)

            # If C3D label has subject prefix, also allow bare name
            if ":" in orig:
                bare = orig.split(":", 1)[1].strip()
                alias_map.setdefault(bare.upper(), orig)

            # If subject is defined but C3D label lacks prefix, allow SUBJECT:label
            if subj and not up.startswith(subj_upper):
                alias_map.setdefault(f"{subj}:{orig}".upper(), orig)

        # If no label file -> return all markers
        if not self.marker_list_path or not os.path.isfile(self.marker_list_path):
            print(f"Using all {len(available)} markers from C3D file.")
            return available

        # Load YAML labels (expecting a list of strings)
        try:
            with open(self.marker_list_path, "r") as f:
                data = yaml.safe_load(f)
            requested = [s.strip() for s in data if isinstance(s, str)]
        except Exception as e:
            print(f"Error reading {self.marker_list_path}: {e}")
            return available

        print(f"Requested markers: {requested}")

        if not requested:
            print("No usable labels in file → using all markers.")
            return available

        # Match labels
        matched = []
        missing = []
        used = set()

        for r in requested:
            r_up = r.upper()

            candidates = []
            if subj and ":" not in r:
                candidates.append(f"{subj}:{r}".upper())
            candidates.append(r_up)

            hit = next((alias_map[c] for c in candidates if c in alias_map), None)

            if hit and hit.upper() not in used:
                used.add(hit.upper())
                matched.append(hit)
            else:
                missing.append(r)

        print(f"Matched {len(matched)} / {len(requested)}")
        if missing:
            print(f"Missing markers: {missing}")

        return matched if matched else available

    def load_event_data(self):
        """Load event data efficiently using importAe, with interactive stream selection as sometimes there are multiple saved inside??."""
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
                print(f" [{i}] {key}")
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

        #self.start_time = 0.0

        # Get actual first event timestamp for more accurate start time
        self.start_time = self.imp.get_first_ts()  # Use actual first event timestamp instead of 0.0
        self.end_time = self.imp.get_last_ts()

        print(f"\n Loaded event stream: '{middle_key}'")
        print(f"Events from {self.start_time:.3f}s to {self.end_time:.3f}s")

    def load_vicon_data(self):
        """Load VICON C3D data."""
        print("Loading VICON data...")

        self.c3d_data = c3d.Reader(open(self.vicon_path, 'rb'))
        for i, points, _ in self.c3d_data.read_frames():
            self.points_3d[i] = points
            
        self.marker_t = np.linspace(0.0, self.c3d_data.frame_count / self.c3d_data.point_rate, self.c3d_data.frame_count, endpoint=False)
        
        # Get all marker names from C3D file - keep all markers available for now
        self.marker_names = [name.strip() for name in self.c3d_data.point_labels]
        print(f"Loaded {len(self.marker_names)} total markers from C3D file")
        
        # Auto-detect subject from C3D file if not provided
        if self.subject is None:
            self.subject = self._detect_subject_from_c3d(self.marker_names)
            if self.subject:
                print(f"Auto-detected subject from C3D file: {self.subject}")
            else:
                print("No subject pattern detected in C3D marker names")
        
        # Configure camera setup and identify camera markers, hard-coded for MoveIIT dataset
        self.camera_setup, self.camera_marker_patterns = self.prompt_camera_setup()
        
        # Extract all camera markers based on user patterns
        self.camera_markers = []
        for pattern in self.camera_marker_patterns:
            matches = self._find_matching_markers(pattern)
            self.camera_markers.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        self.camera_markers = [x for x in self.camera_markers if not (x in seen or seen.add(x))]
        
        print(f"Camera setup: {self.camera_setup}")
        print(f"Camera markers: {self.camera_markers}")
        print(f"Total camera markers found: {len(self.camera_markers)}")
        
        # Validate camera marker count
        if len(self.camera_markers) == 0:
            print(" Warning: No camera markers identified! This will cause issues with transformation computation.")
        elif self.camera_setup == "multi" and len(self.camera_markers) < 2:
            print(f" Warning: Multi-marker setup specified but only {len(self.camera_markers)} markers found.")
            print(" This may affect estimation accuracy.")
        
        # Now filter markers based on marker_list_path if provided (after camera selection)
        if self.marker_list_path:
            filtered_markers = self.get_markers_names()     # only use the markers specified in the file by the user
            print(f"Filtered markers: {filtered_markers}")
            self.marker_names = filtered_markers
        else:
            print(f"Using all {len(self.marker_names)} markers from C3D file")
        
    def load_calibration_data(self):
        """Load camera intrinsic parameters."""

        print("Loading camera intrinsic parameters...")
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
        """Compute world to system transformation matrices using user-specified camera markers."""

        print("Computing world to system transformations...")
        
        # Use the camera setup specification
        enable_camera_markers = (self.camera_setup in ["multi", "single"])
        
        # Create ViconHelper with user-specified camera markers
        vicon_helper = helpers.ViconHelper(
            self.marker_t, self.points_3d, self.delay, 
            self.c3d_data.frame_count, self.c3d_data.point_rate, 
            self.c3d_data.point_labels, enable_camera_markers, True,
            user_camera_markers=self.camera_markers  # Pass user-specified camera markers
        )   #TODO: check after cleaning helpers functions
        
        self.Ts_world_to_system = vicon_helper.compute_camera_marker_transforms()
        
        # # DEBUG: Identify and print unique transformation matrices with timestamps
        # unique_transforms = []
        # unique_timestamps = []  # Store first occurrence timestamp for each unique transform
        # tolerance = 1e-6
        
        # for i, T in enumerate(self.Ts_world_to_system):
        #     is_unique = True
        #     for unique_T in unique_transforms:
        #         if np.allclose(T, unique_T, atol=tolerance):
        #             is_unique = False
        #             break
        #     if is_unique:
        #         unique_transforms.append(T)
        #         # Get timestamp for this frame (frame i corresponds to marker_t[i])
        #         timestamp = self.marker_t[i] if i < len(self.marker_t) else i * self.period
        #         unique_timestamps.append(timestamp)
        
        # if len(unique_transforms) == 1:
        #     print(f"Transformation (constant across all frames):")
        #     print(f"First occurrence at t={unique_timestamps[0]:.3f}s")
        #     print(unique_transforms[0])
        # else:
        #     print(f"Unique transformations ({len(unique_transforms)} different matrices found):")
        #     for i, (T, timestamp) in enumerate(zip(unique_transforms, unique_timestamps)):
        #         print(f"Transformation {i+1} (first occurrence at t={timestamp:.3f}s):")
        #         print(T)
        # # *DEBUG
        
        print(f"Computed transformations using camera setup: {self.camera_setup}")
        
    def visualize_events(self):
        """Visualize event data for a specified duration."""
        
        print(f"Visualizing events")
        print("\n" + "="*60)
        print("EVENT VISUALIZATION - GUI INSTRUCTIONS")
        print("="*60)
        print("A window will show the raw event data stream.")
        print("This helps understand the data before calibration.")
        print("\nControls:")
        print("  • q or ESC: Stop visualization")
        print("="*60)    
        
        img = np.ones(self.cam_res, dtype=np.uint8) * 255
        ft = self.start_time
        window_size = 500 * self.period
        window_start = self.start_time
        
        cv2.namedWindow('Event Visualization', cv2.WINDOW_NORMAL)
        
        try:
            while ft < self.end_time:
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
                        
                        # press ESC or q to quit thw window
                        if k == 27 or k == ord('q'):
                            raise KeyboardInterrupt
                            
                        img = np.ones(self.cam_res, dtype=np.uint8) * 255
                        text = f"t = {ft:.6f}s"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, text, (img.shape[1] - 200, 30), font, 0.7, (0, 0, 0), 2)
                        ft += self.period
                        
                    if e_vs[i] < self.cam_res[0] and e_us[i] < self.cam_res[1]:
                        img[e_vs[i], e_us[i]] = 0
                        
                window_start = window_end
                
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
        #TODO: check after cleaning helpers
        
        # Load events one window after the other
        window_size = 500 * self.period  # 10 seconds window
        window_start = self.start_time
        # rvec_init = np.zeros(3) 

        """hard-coded initial rotation for MoveIIT staticDVS sequences"""
        roll, pitch, yaw = -85.0, 180.0, -80.0
        rvec_init = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_rotvec()
                        
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
                
                R_init = Rotation.from_rotvec(rvec_init).as_matrix()    # needed to keep the info going from one window to the next

                rvec_init = projector.manual_rotation_adjustment(
                    self.marker_t, self.delay, e_ts, e_us, e_vs, 
                    self.period, R_init=R_init, visualize=True, 
                    chosen_one=chosen_one, marker_time_offset=window_start
                )
                
                window_start = window_end

                print("window_start updated to:", window_start)

        except helpers.RotationExit as e:
            print("Visualization stopped, saved rotation vector.")
            rvec_init = e.r_vec

        finally:
            cv2.destroyAllWindows()
            print("rvec", rvec_init)
            return rvec_init

    def manual_delay_correction(self) -> float:
        """Manually correct synchronization delay using windowed approach."""
        
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
        
        # Load events one window after the other
        window_size = 500 * self.period
        window_start = self.start_time
        
        # Create projector for delay adjustment
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d,
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

                try:
                    self.delay, self.current_delay_step = projector.fix_delay(
                        self.marker_t, self.delay, e_ts, e_us, e_vs, self.period,
                        visualize=True, marker_time_offset=window_start, delay_step=self.current_delay_step
                    )
                except helpers.DelayReset as e:
                    # Reset requested by user -> save data and begin from first timestamp
                    print("Reset requested by user, saving data and start over from the first timestamp.")
                    # Save the last values for delay and delay_step
                    self.delay = e.new_delay
                    self.current_delay_step = e.delay_step  # Preserve the delay step
                    # Start over from the first event timestamp
                    window_start = self.start_time   # or self.imp.first_event_time
                    continue
                
                print("e_ts final:", e_ts[-1], "window_start:", window_start, "window_size:", window_size)

                window_start = window_end
                
                print("window_start updated to:", window_start)
                
        except helpers.DelayExit as e:
            print("Visualization stopped, saved delay.")            
            self.delay = e.delay
            self.current_delay_step = e.delay_step

        finally:
            cv2.destroyAllWindows()
            print(f"Updated delay: {self.delay:.3f}s")
            return self.delay

    def label_data_interactive(self, use_projections: bool = False) -> str:
        """Interactive data labeling using windowed approach."""

        #TODO: add a way to navigate through frames for the user?? not so useful i think

        print("Starting interactive labeling...")
        
        # DEBUG: if label file already exists give the possibility to choose to use them, start over or append new labels
        # Check if labels already exist
        user_choice = None
        existing_labels = None
        
        if os.path.exists(self.output_path):
            print(f"\nFound existing label file: {self.output_path}")
            try:
                # Try to load existing labels
                existing_labels = helpers.read_points_labels(self.output_path)
                
                # Check if the file contains actual labels
                has_labels = (
                    existing_labels and 
                    'points' in existing_labels and 
                    'times' in existing_labels and
                    len(existing_labels['points']) > 0 and
                    any(len(point_dict) > 0 for point_dict in existing_labels['points'])
                )
                
                if has_labels:
                    print(f"Found {len(existing_labels['points'])} labeled frames with {sum(len(p) for p in existing_labels['points'])} total labels")
                    
                    while True:
                        user_choice = input("\nExisting labels found. Choose action:\n"
                                          "  [u] Use existing labels (skip labeling)\n"
                                          "  [r] Re-label from scratch (overwrite existing)\n"
                                          "  [c] Continue/append to existing labels\n"
                                          "Enter choice (u/r/c): ").strip().lower()
                        
                        if user_choice == 'u':
                            print("Using existing labels, skipping labeling process")
                            return self.output_path
                        elif user_choice == 'r':
                            print("Re-labeling from scratch...")
                            break  # Continue with normal labeling process
                        elif user_choice == 'c':
                            print("Continuing with existing labels (append mode)")
                            print(" You can add new labels to supplement the existing ones")
                            # We'll initialize the labeler with existing data
                            break
                        else:
                            print("Invalid choice. Please enter 'u', 'r', or 'c'.")
                else:
                    print("Label file exists but contains no valid labels. Starting fresh labeling process.")
            except Exception as e:
                print(f"Warning: Could not read existing label file ({e}). Starting fresh labeling process.")
        # *DEBUG
        
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
        
        if not self.markers_names:
            self.markers_names = self.get_markers_names()
        
        window_size = 500 * self.period
        window_start = self.start_time
        
        # Create labeler instance
        labeler = helpers.DvsLabeler(img_shape=(self.cam_res[0], self.cam_res[1], 3), subject=self.subject)
        
        # DEBUG: take existing labels from the yaml file
        # Initialize with existing labels if continuing, or empty if starting fresh
        if user_choice == 'c' and existing_labels:
            try:
                if existing_labels and 'points' in existing_labels and 'times' in existing_labels:
                    labeler.points_dict = existing_labels
                    print(f"✓ Initialized with {len(existing_labels['points'])} existing labels")
                else:
                    labeler.points_dict = {'points': [], 'times': []}
            except Exception as e:
                print(f"Could not load existing labels for continuation: {e}")
                labeler.points_dict = {'points': [], 'times': []}
        else:
            labeler.points_dict = {'points': [], 'times': []}
        # *DEBUG
        
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
                        self.markers_names, self.c3d_data, self.points_3d, self.marker_t,
                        self.T_syst_to_camera_opt, self.Ts_world_to_system,
                        self.K, self.cam_res, self.delay, D=self.D,
                        marker_time_offset=window_start
                    )

                else:
                    input_label_tag_file = self.marker_list_path       # TODO: read from user input in parser, could be hardcoded in
                    print("Using label tag file:", input_label_tag_file)
                    labeler.label_data(
                        e_ts, e_us, e_vs,
                        self.period, input_label_tag_file
                    )

                window_start = window_end
                
        except helpers.LabelExit as e:
            print("Labeling stopped early by user, saving labels.")

        finally:
            cv2.destroyAllWindows()

            # Merge accumulated labels
            if hasattr(labeler, "points_dict") and labeler.points_dict:
                labeler.labeled_dict = labeler.points_dict

                # Check if there are labels
                has_labels = (
                    'points' in labeler.labeled_dict
                    and any(len(p) > 0 for p in labeler.labeled_dict['points'])
                )

                if has_labels:
                    labeler.labels_done = True

                    labeler.save_labeled_points(self.output_path)
                    
                    # Provide feedback on what was saved
                    total_labels = sum(len(p) for p in labeler.labeled_dict['points'])
                    total_frames = len(labeler.labeled_dict['points'])
                    
                    # DEBUG: append labels to the file
                    if user_choice == 'c':
                        print(f" Updated labeled points file: {self.output_path}")
                        print(f" Total: {total_frames} frames with {total_labels} labels")
                    else:
                        print(f" Saved labeled points to: {self.output_path}")
                        print(f" Created: {total_frames} frames with {total_labels} labels")
                else:
                    print("No labeled points found — nothing saved.")
            else:
                print("Labeler has no labeled_dict or points_dict.")

        return self.output_path # TODO: this could be set to no returns
    
    def create_projection_video(self) -> List[dict]:
        """Create video and continuously store projected marker points with timestamps."""
        print("Creating projection video with live point collection...")

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
        all_projected_points = []  # list of dicts: {'timestamp': t, 'x': x, 'y': y, 'marker': name}
        window_size = 500 * self.period
        window_start = self.start_time

        print(f"Processing time range: {self.start_time:.3f}s to {self.end_time:.3f}s")

        window_count = 0

        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2
                window_count += 1

                print(f"Processing window {window_count}, from {window_start:.3f}s to {window_end:.3f}s")

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                print(f"Loaded {len(e_ts)} events from {e_ts[0]:.3f}s to {e_ts[-1]:.3f}s")

                try:
                    synced_image_points, video_segment, current_delay, current_delay_step = projector.project_vicon_to_event_plane_dynamic(
                        self.marker_t, self.delay,
                        e_ts, e_us, e_vs, self.period,
                        visualize=True, video_record=True,
                        marker_time_offset=window_start,
                        delay_step=self.current_delay_step
                    )
                except helpers.DelayReset as e:
                    # Save data and reset from the first timestamp
                    print("Delay changed: full reset requested.")
                    # Save the last values for delay and delay_step
                    self.delay = e.new_delay
                    self.current_delay_step = e.delay_step
                    # Clear dictionaries of projections
                    all_projected_points.clear()
                    collected_video_segments.clear()
                    # Start over from the first event timestamp
                    window_start = self.start_time   # or self.imp.first_event_time
                    window_count = 0
                    continue

                # Update delay if it was adjusted during projection
                self.delay = current_delay
                self.current_delay_step = current_delay_step

                # TODO: fix when it actually creates empty video
                # Collect frames for video
                if video_segment is not None and video_segment != []:
                    collected_video_segments.append(video_segment)

                # Collect all projected points with event timestamps
                if synced_image_points:
                    for marker_name, marker_data in synced_image_points.items():
                        if marker_data and "points" in marker_data and "timestamps" in marker_data:
                            points = marker_data["points"]
                            timestamps = marker_data["timestamps"]
                                                        
                            # Ensure matching points and timestamps
                            n = min(len(points), len(timestamps))
                            points_added = 0
                            for i in range(n):
                                if len(points[i]) >= 2:  # Ensure to have x, y coordinates
                                    x, y = map(float, points[i][:2])
                                    all_projected_points.append({
                                        "timestamp": float(timestamps[i]),
                                        "x": x,
                                        "y": y,
                                        "marker": marker_name
                                    })
                                    points_added += 1
                        else:
                            print(f" {marker_name}: Invalid marker_data structure")
                else:
                    print(f" Window {window_count}: No synced_image_points returned")

                # Advance to next window
                window_start = window_end

        except KeyboardInterrupt:
            print(" Projection stopped early by user.")

        finally:
            cv2.destroyAllWindows()

        # # DEBUG: summary of points and video segments
        # if all_projected_points:
        #     marker_counts = {}
        #     for entry in all_projected_points:
        #         marker_name = entry.get('marker', 'UNKNOWN')
        #         marker_counts[marker_name] = marker_counts.get(marker_name, 0) + 1
        #     print(f"Projected points (preview): total={len(all_projected_points)}  breakdown={marker_counts}")
        # else:
        #     print("No projected points collected in this preview session.")

        # if collected_video_segments:
        #     total_frames = sum(len(seg) for seg in collected_video_segments)
        #     print(f"Collected video frames (preview): {total_frames} (across {len(collected_video_segments)} segments)")
        # else:
        #     print("No video segments collected in this preview session.")
        # # *DEBUG

        return {
            "segments": collected_video_segments,
            "points": all_projected_points,
        }  

    # TODO: instead of output_video ask the user for the output_path in the args, chek usefullness as the joint method also exists
    def _save_projected_points_csv(self, all_projected_points, output_video):
        """
        Save projected points to CSV in format: event_timestamp, marker1_x, marker1_y, marker2_x, marker2_y, ...
        Markers are saved in the same order as defined in the YAML file (if provided) or in the order they appear
        """

        csv_path = os.path.join(os.path.dirname(output_video), "projected_points.csv")
        csv_path = self._generate_unique_video_path(csv_path)

        if not all_projected_points:
            print(" No projected points to save in CSV.")
            return

        # Sort by timestamp for consistent order
        all_projected_points.sort(key=lambda d: d['timestamp'])

        # Use self.marker_names to preserve the order from YAML file (if provided)
        projected_markers = set(p['marker'] for p in all_projected_points)
        all_markers = [m for m in self.marker_names if m in projected_markers]
        timestamps = sorted(set(p['timestamp'] for p in all_projected_points))

        # Format -> {timestamp: {marker: (x, y)}}
        frame_dict = {t: {} for t in timestamps}
        for p in all_projected_points:
            frame_dict[p['timestamp']][p['marker']] = (p['x'], p['y'])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header row
            header = ["event_timestamp"]
            for m in all_markers:
                header.extend([f"{m}_x", f"{m}_y"])
            writer.writerow(header)

            for t in timestamps:
                row = [f"{t:.6f}"]
                for m in all_markers:
                    if m in frame_dict[t]:
                        x, y = frame_dict[t][m]
                        row.extend([f"{x:.2f}", f"{y:.2f}"])
                    else:
                        row.extend(["", ""])
                writer.writerow(row)

        print(f" Saved projected points CSV: {csv_path}")
        # print(f" Marker order matches YAML definition: {[m for m in self.marker_names[:5]]}{'...' if len(self.marker_names) > 5 else ''}")

    #TODO: check modifications
    def extract_joint_depths(self, event_timestamp: float, joint_name: str, joint_indices: dict) -> dict:
        """Extract depth and 2D position for a specific joint at a specific event timestamp."""

        # Find closest VICON frame
        time_diffs = np.abs(self.marker_t - event_timestamp)
        closest_idx = np.argmin(time_diffs)
        
        # Check if time difference is reasonable
        frame_period = 1.0 / self.c3d_data.point_rate
        if time_diffs[closest_idx] > frame_period:
            return None
            
        # Get the closest C3D frame (convert from 0-based to 1-based indexing)
        frame_idx = closest_idx + 1
        if frame_idx not in self.points_3d:
            return None
            
        # World → camera transformation at this time (uses 0-based indexing)
        marker_t_idx = max(0, min(closest_idx, len(self.Ts_world_to_system) - 1))
        T_ws = self.Ts_world_to_system[marker_t_idx]
        T_wc = self.T_syst_to_camera_opt @ T_ws
        
        # Get joint data
        col_idx = joint_indices.get(joint_name)
        if col_idx is None:
            return None
            
        frame_points = self.points_3d[frame_idx]
        if col_idx >= frame_points.shape[0]:
            return None
            
        p = np.asarray(frame_points[col_idx], dtype=float)
        
        # Validate point data - check for missing, invalid, or zero coordinates
        if p.shape[0] < 3:
            return None
        p = p[:3]  # Use first 3 coordinates
        
        # Check for NaN, infinity, or all-zero coordinates (invalid data)
        if not np.all(np.isfinite(p)) or np.allclose(p, 0, atol=1e-6):
            return None
            
        # Transform to camera coordinates
        p_world_h = np.array([p[0], p[1], p[2], 1.0], dtype=float)
        p_cam_h = T_wc @ p_world_h
        p_cam = p_cam_h[:3] / p_cam_h[3]
        z_depth = float(p_cam[2])
        
        return {
            "event_timestamp": event_timestamp,
            f"{joint_name}_depth": z_depth
        }

    #TODO: remove max_samples and modify so that args is output_path
    def project_joints_position(self, output_csv_path: str = None, max_samples: int = None, output_projections_csv: str = None, output_video: str = None, extract_depth: bool = True):
        """Unified method to compute joint positions and depths with event-synchronized timestamps."""

        #TODO: remove max_samples after finishing debugging

        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        JOINT_CONFIG_PATH = os.path.join(CURRENT_DIR, "../scripts/config/labels_joints.yml")

        if not os.path.exists(JOINT_CONFIG_PATH):
            raise FileNotFoundError(f"Joint config file not found: {JOINT_CONFIG_PATH}")

        with open(JOINT_CONFIG_PATH, "r") as f:
            joint_labels = yaml.safe_load(f)

        if not isinstance(joint_labels, list):
            raise ValueError(f"Expected list of joint labels in {JOINT_CONFIG_PATH}, got {type(joint_labels)}")

        c3d_labels = [n.strip() for n in self.c3d_data.point_labels]

        joint_markers: list[str] = []
        joint_indices: dict[str, int] = {}

        for j in joint_labels:
            m = self._find_matching_markers_cleaned(j, c3d_labels)
            if m:
                joint_markers.append(m)
                joint_indices[m] = c3d_labels.index(m)

        if not joint_markers:
            raise RuntimeError("No joint markers from YAML matched the C3D labels.")

        print(f"Found {len(joint_markers)} matching joint markers: {joint_markers}")

        joint_projector = helpers.ViconProjector(
            joint_markers, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )

        # Use event-based processing instead of C3D frame-based
        depth_rows = []
        all_projected_joints_points = []
        collected_joint_video_segments = []
        
        window_size = 500 * self.period  # 500ms windows
        window_start = self.start_time
        window_count = 0
        
        print(f"Processing event windows from {self.start_time:.3f}s to {self.end_time:.3f}s")

        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2
                window_count += 1
                
                print(f"Processing event window {window_count}: {window_start:.3f}s to {window_end:.3f}s")
                
                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                #TODO: after debugging remove max_samples
                if max_samples is not None and len(e_ts) > max_samples:
                    indices = np.linspace(0, len(e_ts) - 1, max_samples, dtype=int)
                    e_ts = e_ts[indices]
                    e_us = e_us[indices]  
                    e_vs = e_vs[indices]
                    print(f"Sampled {len(e_ts)} events from window {window_count}")
                
                try:
                    synced_joint_points, joint_video_segment, current_delay, current_delay_step = joint_projector.project_vicon_to_event_plane_dynamic(
                        self.marker_t, self.delay,
                        e_ts, e_us, e_vs, self.period,
                        visualize=True, video_record=True,
                        marker_time_offset=window_start,
                        delay_step=self.current_delay_step
                    )
                except helpers.DelayReset as e:
                    # Save data and reset from the first timestamp                    
                    print("Delay changed: full reset requested.")
                    # Save the last values for delay and delay_step
                    self.delay = e.new_delay
                    self.current_delay_step = e.delay_step
                    # Clear dictionaries of projections
                    depth_rows.clear()
                    all_projected_joints_points.clear()
                    collected_joint_video_segments.clear()
                    # Start over from the first event timestamp
                    window_start = self.start_time      # or self.imp.first_event_time
                    window_count = 0
                    continue

                # Update delay if it was adjusted during projection
                self.delay = current_delay
                self.current_delay_step = current_delay_step

                # TODO: fix when it actually creates empty video
                # Collect frames for video
                if joint_video_segment is not None and joint_video_segment != []:
                    collected_joint_video_segments.append(joint_video_segment)

                # Collect all projected points with event timestamps AND extract depths if requested
                if synced_joint_points:
                    for joint_name, joint_data in synced_joint_points.items():
                        if joint_data and "points" in joint_data and "timestamps" in joint_data:
                            points = joint_data["points"]
                            timestamps = joint_data["timestamps"]

                            n = min(len(points), len(timestamps))
                            for i in range(n):
                                if len(points[i]) >= 2:  # Ensure to have x, y coordinates
                                    x, y = map(float, points[i][:2])
                                    event_timestamp = float(timestamps[i])
                                    
                                    # Add projection data
                                    all_projected_joints_points.append({
                                        "timestamp": event_timestamp,
                                        "x": x,
                                        "y": y,
                                        "marker": joint_name
                                    })
                                    
                                    # Extract depth for the same timestamp if requested
                                    if extract_depth:
                                        depth_data = self.extract_joint_depths(event_timestamp, joint_name, joint_indices)
                                        if depth_data is not None:
                                            # Check if we already have a row for this timestamp
                                            existing_row = None
                                            for row in depth_rows:
                                                if row["event_timestamp"] == event_timestamp:
                                                    existing_row = row
                                                    break
                                            
                                            if existing_row is None:
                                                # Create new row with this timestamp
                                                existing_row = {"event_timestamp": event_timestamp}
                                                depth_rows.append(existing_row)
                                            
                                            # Use the projected coordinates (x, y) from projector, only add depth from depth_data
                                            existing_row[f"{joint_name}_x"] = x
                                            existing_row[f"{joint_name}_y"] = y
                                            existing_row[f"{joint_name}_depth"] = depth_data[f"{joint_name}_depth"]
                else:
                    print(f" Window {window_count}: No synced_joint_points returned")

                # Depth extraction is now done inline with projection collection above, TODO: check
                
                window_start = window_end

        except KeyboardInterrupt:
            print("Joint analysis interrupted by user")

        # TODO: check validity of extracted depth data

        # Create depth DataFrame with columns in joint config file order
        if depth_rows:
            # Create ordered column list: timestamp + joints in config order
            ordered_columns = ["event_timestamp"]
            for joint in joint_markers:  # joint_markers preserves config file order
                ordered_columns.extend([f"{joint}_x", f"{joint}_y", f"{joint}_depth"])
            
            depth_df = pd.DataFrame(depth_rows)
            # Reorder columns to match config file order, add missing columns as empty
            depth_df = depth_df.reindex(columns=ordered_columns)
        else:
            depth_df = pd.DataFrame()
        
        print(f"✓ Extracted positions and depths for {len(depth_rows)} event timestamps")

        # Create projections DataFrame if requested
        projections_df = None
        if output_projections_csv:
            if all_projected_joints_points:
                # Sort by timestamp for consistent order
                all_projected_joints_points.sort(key=lambda d: d['timestamp'])
                
                # Create projections DataFrame with columns for ALL joint markers (not just projected ones)
                timestamps = sorted(set(p['timestamp'] for p in all_projected_joints_points))
                
                # Build {timestamp: {marker: (x, y)}} - only for valid projections
                frame_dict = {t: {} for t in timestamps}
                for p in all_projected_joints_points:
                    frame_dict[p['timestamp']][p['marker']] = (p['x'], p['y'])
                
                projection_rows = []
                for t in timestamps:
                    row = {"event_timestamp": t}
                    # Include ALL joint markers in config file order, but leave empty if no valid data
                    for m in joint_markers:  # joint_markers preserves config file order
                        if m in frame_dict[t]:
                            x, y = frame_dict[t][m]
                            row[f"{m}_x"] = f"{x:.6f}"
                            row[f"{m}_y"] = f"{y:.6f}"
                        else:
                            # Leave empty (pandas will handle as NaN) for missing/invalid data
                            row[f"{m}_x"] = ""
                            row[f"{m}_y"] = ""
                    projection_rows.append(row)
                
                # Create DataFrame with ordered columns
                ordered_proj_columns = ["event_timestamp"]
                for joint in joint_markers:  # joint_markers preserves config file order
                    ordered_proj_columns.extend([f"{joint}_x", f"{joint}_y"])
                
                projections_df = pd.DataFrame(projection_rows)
                # Reorder columns to match config file order
                projections_df = projections_df.reindex(columns=ordered_proj_columns)
                print(f"✓ Generated projections for {len(timestamps)} event timestamps")
            else:
                projections_df = pd.DataFrame()  # Empty DataFrame
                print("No projection points collected")

        # Format timestamps consistently for both DataFrames before saving
        if not depth_df.empty:
            depth_df['event_timestamp'] = depth_df['event_timestamp'].apply(lambda x: f"{x:.6f}")
        
        # Save outputs
        if output_csv_path:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            depth_df.to_csv(output_csv_path, index=False)
            print(f"✓ Combined positions+depths saved: {os.path.basename(output_csv_path)}")

        if output_projections_csv:
            os.makedirs(os.path.dirname(output_projections_csv), exist_ok=True)
            if projections_df is not None and not projections_df.empty:
                # Format timestamps consistently
                projections_df['event_timestamp'] = projections_df['event_timestamp'].apply(lambda x: f"{x:.6f}")
                projections_df.to_csv(output_projections_csv, index=False)
                print(f"✓ Projections saved: {os.path.basename(output_projections_csv)}")
            else:
                print("No projections to save")

        # Save video if requested
        if output_video and collected_joint_video_segments:
            print(f"Merging {len(collected_joint_video_segments)} video segments...")
            try:
                fps = int(1 / self.period)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_video, fourcc, fps,
                    (self.cam_res[1], self.cam_res[0]), isColor=False
                )
                
                total_frames = 0
                for segment_frames in collected_joint_video_segments:
                    for frame in segment_frames:
                        if frame.ndim == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        video_writer.write(frame)
                        total_frames += 1
                
                video_writer.release()
                print(f"✓ Joint projection video created: {os.path.basename(output_video)} ({total_frames} frames)")
            except Exception as e:
                print(f"Error creating video: {e}")

        # Return results
        if output_csv_path:
            return output_csv_path
        else:
            return depth_df, projections_df
        
###
    # def analyze_depth_accuracy(self, depth_csv_path: str = None, max_checks: int = 200):
    #     """
    #     Check if extracted depths are consistent with VICON 3D + extrinsics.

    #     For a subset of timestamps and joints, this does:
    #       1) VICON 3D -> world -> camera  => 'true' camera Z
    #       2) Project to pixels using K, D
    #       3) Back-project pixels + saved depth -> 3D world
    #       4) Compare back-projected 3D vs original VICON 3D

    #     Prints summary of:
    #       - |saved_depth - true_Z|
    #       - 3D reconstruction error in mm
    #     """
    #     import pandas as pd
    #     import numpy as np
    #     import cv2

    #     print("\n" + "=" * 60)
    #     print("DEPTH CONSISTENCY CHECK")
    #     print("=" * 60)

    #     # ------------------------------------------------------------
    #     # 1) Load depth CSV
    #     # ------------------------------------------------------------
    #     if depth_csv_path is None:
    #         sequence_name = self._extract_sequence_name()
    #         output_dir = self._get_output_directory()
    #         depth_csv_path = os.path.join(output_dir, f"{sequence_name}_joint_depths.csv")

    #     if not os.path.exists(depth_csv_path):
    #         print(f"❌ Depth CSV not found: {depth_csv_path}")
    #         return None

    #     df = pd.read_csv(depth_csv_path)
    #     if 'event_timestamp' in df.columns:
    #         time_col = 'event_timestamp'
    #     elif 'timestamp' in df.columns:
    #         time_col = 'timestamp'
    #     else:
    #         print("❌ No timestamp column ('event_timestamp' or 'timestamp') in depth CSV.")
    #         return None

    #     # Might be stored as string '0.123456' → convert to float
    #     df[time_col] = df[time_col].astype(float)

    #     # Identify joint depth columns and corresponding marker names
    #     depth_cols = [c for c in df.columns if c.endswith('_depth')]
    #     if not depth_cols:
    #         print("❌ No *_depth columns found in depth CSV.")
    #         return None

    #     joint_names = [c[:-6] for c in depth_cols]  # remove '_depth'
    #     c3d_labels = [n.strip() for n in self.c3d_data.point_labels]

    #     # Map each joint name (full C3D label) to its index in points_3d frames
    #     joint_indices = {}
    #     for j in joint_names:
    #         if j in c3d_labels:
    #             joint_indices[j] = c3d_labels.index(j)
    #         else:
    #             print(f"  ⚠️ Joint '{j}' not found in C3D labels; skipping in accuracy check.")
    #     joint_names = [j for j in joint_names if j in joint_indices]
    #     if not joint_names:
    #         print("❌ None of the depth joints were found in C3D labels.")
    #         return None

    #     print(f"Checking depth for {len(joint_names)} joints: {joint_names}")

    #     # ------------------------------------------------------------
    #     # 2) Decide which rows to sample
    #     # ------------------------------------------------------------
    #     n_rows = len(df)
    #     if n_rows == 0:
    #         print("❌ Depth CSV has no rows.")
    #         return None

    #     if max_checks >= n_rows:
    #         row_indices = list(range(n_rows))
    #     else:
    #         step = max(1, n_rows // max_checks)
    #         row_indices = list(range(0, n_rows, step))[:max_checks]

    #     # Map "time index" -> actual C3D frame id
    #     available_frames = sorted(self.points_3d.keys())  # usually 1..frame_count

    #     # ------------------------------------------------------------
    #     # 3) Loop and accumulate errors
    #     # ------------------------------------------------------------
    #     depth_diff_errors = []   # |saved_depth - true_Z|
    #     recon_3d_errors = []     # ||backprojected_world - original_world||
    #     negative_depths = 0
    #     total_depths = 0

    #     frame_period = 1.0 / self.c3d_data.point_rate

    #     for row_idx in row_indices:
    #         row = df.iloc[row_idx]
    #         t_event = float(row[time_col])

    #         # Find closest VICON frame index
    #         time_diffs = np.abs(self.marker_t - t_event)
    #         closest_idx = int(np.argmin(time_diffs))

    #         if time_diffs[closest_idx] > frame_period:
    #             # Too far from any VICON sample; skip
    #             continue

    #         # marker_t index (0-based) vs points_3d frame keys (1-based)
    #         if closest_idx >= len(available_frames):
    #             continue
    #         frame_id = available_frames[closest_idx]
    #         if frame_id not in self.points_3d:
    #             continue

    #         # World -> system -> camera
    #         marker_t_idx = max(0, min(closest_idx, len(self.Ts_world_to_system) - 1))
    #         T_ws = self.Ts_world_to_system[marker_t_idx]
    #         T_wc = self.T_syst_to_camera_opt @ T_ws

    #         frame_points = self.points_3d[frame_id]  # shape (num_markers, >=3)

    #         for j in joint_names:
    #             col_idx = joint_indices[j]
    #             depth_col = f"{j}_depth"
    #             if depth_col not in df.columns:
    #                 continue

    #             saved_depth = row[depth_col]
    #             total_depths += 1
    #             if not np.isfinite(saved_depth):
    #                 continue

    #             if saved_depth < 0:
    #                 negative_depths += 1

    #             if col_idx >= frame_points.shape[0]:
    #                 continue

    #             p = np.asarray(frame_points[col_idx], dtype=float)
    #             if p.shape[0] >= 3:
    #                 p = p[:3]
    #             else:
    #                 continue

    #             if not np.all(np.isfinite(p)) or np.allclose(p, 0):
    #                 continue

    #             # --- 3.1 "True" camera coords from extrinsics ---
    #             p_world_h = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    #             p_cam_h = T_wc @ p_world_h
    #             p_cam = p_cam_h[:3] / p_cam_h[3]
    #             x_c, y_c, z_true = p_cam

    #             # Compare saved depth vs true Z
    #             depth_diff = abs(saved_depth - z_true)
    #             depth_diff_errors.append(depth_diff)

    #             # --- 3.2 Project to pixels ---
    #             # Using cv2.projectPoints to be consistent with your code
    #             p_cam_reshaped = p_cam.reshape(1, 1, 3)
    #             proj, _ = cv2.projectPoints(
    #                 p_cam_reshaped,
    #                 np.zeros(3), np.zeros(3),
    #                 self.K, self.D
    #             )
    #             u, v = proj.reshape(2)

    #             # --- 3.3 Back-project (u, v, saved_depth) to world ---
    #             K_inv = np.linalg.inv(self.K)
    #             ray = K_inv @ np.array([u, v, 1.0])
    #             p_cam_recon = ray * saved_depth

    #             T_cw = np.linalg.inv(T_wc)
    #             p_cam_recon_h = np.array([p_cam_recon[0], p_cam_recon[1], p_cam_recon[2], 1.0])
    #             p_world_recon_h = T_cw @ p_cam_recon_h
    #             p_world_recon = p_world_recon_h[:3]

    #             err_3d = np.linalg.norm(p_world_recon - p)
    #             recon_3d_errors.append(err_3d)

    #     # ------------------------------------------------------------
    #     # 4) Summarize results
    #     # ------------------------------------------------------------
    #     if not depth_diff_errors or not recon_3d_errors:
    #         print("⚠️ Not enough valid samples to evaluate depth consistency.")
    #         return None

    #     depth_diff_errors = np.array(depth_diff_errors)
    #     recon_3d_errors = np.array(recon_3d_errors)

    #     print(f"\nChecked {len(depth_diff_errors)} joint depths over {len(row_indices)} rows.")
    #     print(f"  Negative depths: {negative_depths} / {total_depths} (should ideally be 0)")
    #     print("\nDepth vs true camera Z (mm):")
    #     print(f"  mean  |ΔZ| = {depth_diff_errors.mean():.3f}")
    #     print(f"  median|ΔZ| = {np.median(depth_diff_errors):.3f}")
    #     print(f"  max   |ΔZ| = {depth_diff_errors.max():.3f}")

    #     print("\n3D reconstruction error (backprojected vs VICON) [mm]:")
    #     print(f"  mean  = {recon_3d_errors.mean():.3f}")
    #     print(f"  median= {np.median(recon_3d_errors):.3f}")
    #     print(f"  max   = {recon_3d_errors.max():.3f}")

    #     print("\nRule of thumb:")
    #     print("  • If |ΔZ| is very small (≈ 0–1 mm) and 3D error is small (few mm–cm),")
    #     print("    then the depth extraction + transforms are consistent.")
    #     print("  • Many negative depths or huge 3D errors ⇒ something is wrong with T_ws / T_syst_to_camera_opt / units.")

    #     return {
    #         "num_samples": len(depth_diff_errors),
    #         "depth_diff_errors": depth_diff_errors,
    #         "recon_3d_errors": recon_3d_errors,
    #         "negative_depths": negative_depths,
    #         "total_depths": total_depths,
    #     }

    # # TODO: fix timestamps issues
    # def visualize_depth_extraction(self, depth_csv_path: str = None, projections_csv_path: str = None):
    #     """
    #     Create depth visualization by overlaying depth-colored joint projections on event frames.
        
    #     Args:
    #         depth_csv_path: Path to the combined CSV file with depth and position data. If None, will use default naming.
    #         projections_csv_path: DEPRECATED - no longer needed as data is in single file.
    #     """
        
    #     print("\n" + "=" * 60)
    #     print("DEPTH OVERLAY VISUALIZATION")
    #     print("=" * 60)
    #     print("Creating depth-colored overlays on event frames...")
        
    #     # Load combined CSV file with depth and position data
    #     if depth_csv_path is None:
    #         sequence_name = self._extract_sequence_name()
    #         output_dir = self._get_output_directory()
    #         depth_csv_path = os.path.join(output_dir, f"{sequence_name}_joint_depths.csv")
        
    #     # Check if file exists
    #     if not os.path.exists(depth_csv_path):
    #         print(f"❌ Combined CSV not found: {depth_csv_path}")
    #         return
        
    #     # Load CSV data - single file now contains both depth and position data
    #     combined_df = pd.read_csv(depth_csv_path)
        
    #     print(f"✓ Loaded combined data: {len(combined_df)} timestamps")
    #     print(f"✓ Columns found: {list(combined_df.columns)[:10]}...")  # Show first 10 columns
        
        
        
    #     # Identify timestamp column
    #     time_col = 'event_timestamp'
    #     if time_col not in combined_df.columns:
    #         print("❌ No event_timestamp column found in CSV.")
    #         return
        
    #     # Show timestamp range comparison
    #     csv_time_range = (combined_df[time_col].min(), combined_df[time_col].max())
    #     event_time_range = (self.start_time, self.end_time) if hasattr(self, 'start_time') else ("N/A", "N/A")
    #     print(f"CSV timestamp range: {csv_time_range[0]:.6f}s to {csv_time_range[1]:.6f}s")
    #     print(f"Event data range: {event_time_range[0]:.6f}s to {event_time_range[1]:.6f}s" if event_time_range[0] != "N/A" else "Event data range: Not loaded")

    #     # Convert timestamps to float
    #     combined_df[time_col] = combined_df[time_col].astype(float)
        
    #     # Get joint information from depth columns
    #     depth_cols = [c for c in combined_df.columns if c.endswith('_depth')]
    #     joint_names = [c[:-6] for c in depth_cols]  # Remove '_depth' suffix
        
    #     print(f"Found {len(joint_names)} joints: {joint_names}")
        
    #     # Use the combined dataframe directly (no merging needed)
    #     merged_df = combined_df
    #     print(f"✓ Using combined data: {len(merged_df)} timestamps")
        
    #     if len(merged_df) == 0:
    #         print("❌ No synchronized timestamps between depth and projection data")
    #         return
        
    #     # Create output directory
    #     output_dir = os.path.dirname(depth_csv_path)
    #     viz_dir = os.path.join(output_dir, "depth_overlay_frames")
    #     os.makedirs(viz_dir, exist_ok=True)
        
    #     # Setup for event frame generation
    #     window_size = 500 * self.period  # 500ms windows
    #     frame_count = 0
    #     max_frames = 50  # Limit number of frames to generate
        
    #     # Calculate depth color mapping ranges
    #     all_depths = []
    #     for joint in joint_names:
    #         depth_col = f"{joint}_depth"
    #         valid_depths = merged_df[depth_col].dropna()
    #         all_depths.extend(valid_depths.values)
        
    #     if not all_depths:
    #         print("❌ No valid depth data found")
    #         return
            
    #     depth_min, depth_max = np.percentile(all_depths, [5, 95])  # Use 5th-95th percentile for better color mapping
    #     print(f"Depth range for coloring: {depth_min:.1f}mm to {depth_max:.1f}mm")
        
    #     # Sample timestamps evenly across the data
    #     timestamps = sorted(merged_df[time_col].values)
    #     if len(timestamps) > max_frames:
    #         indices = np.linspace(0, len(timestamps)-1, max_frames, dtype=int)
    #         selected_timestamps = [timestamps[i] for i in indices]
    #     else:
    #         selected_timestamps = timestamps
        
    #     print(f"Creating {len(selected_timestamps)} depth overlay frames...")
        
    #     # Process each selected timestamp
    #     for i, target_time in enumerate(selected_timestamps):
    #         try:
    #             # Get the exact row for this timestamp
    #             time_diffs = np.abs(merged_df[time_col] - target_time)
    #             closest_idx = time_diffs.idxmin()
    #             row = merged_df.loc[closest_idx]
    #             actual_time = row[time_col]
                
    #             print(f"Processing frame {i+1}/{len(selected_timestamps)}: t={actual_time:.6f}s")
                
    #             # Check if this timestamp is within the event data range
    #             if hasattr(self, 'start_time') and hasattr(self, 'end_time'):
    #                 if actual_time < self.start_time or actual_time > self.end_time:
    #                     print(f"  ⚠️  Timestamp {actual_time:.6f}s is outside event range [{self.start_time:.6f}s, {self.end_time:.6f}s]")
                
    #             # Load event data for the EXACT timestamp when depth was extracted
    #             window_center = actual_time
                
    #             # Use a fixed 10ms window to get events right around the depth extraction time
    #             event_time_window = 0.01  # 10ms window
    #             e_data = self.imp.get_data_at_time(window_center, event_time_window)
    #             e_ts = np.array(e_data['ts'])
    #             e_us = np.array(e_data['x'])
    #             e_vs = np.array(e_data['y'])
                                
    #             # Create event frame (grayscale background)
    #             event_frame = np.zeros((self.cam_res[0], self.cam_res[1], 3), dtype=np.uint8)
                
    #             # Draw events as white dots
    #             for u, v in zip(e_us, e_vs):
    #                 if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
    #                     event_frame[int(v), int(u)] = [255, 255, 255]  # White events for contrast
                
    #             # Overlay depth-colored dots at exact joint positions from this timestamp
    #             joint_count = 0
    #             valid_joints = []
                
    #             for joint in joint_names:
    #                 depth_col = f"{joint}_depth"
    #                 x_col = f"{joint}_x"
    #                 y_col = f"{joint}_y"
                    
    #                 # Check if projection and depth data exist for this exact timestamp
    #                 if (x_col in row and y_col in row and depth_col in row and 
    #                     pd.notna(row[x_col]) and pd.notna(row[y_col]) and pd.notna(row[depth_col]) and
    #                     row[x_col] != '' and row[y_col] != ''):
                        
    #                     try:
    #                         x = float(row[x_col])
    #                         y = float(row[y_col])
    #                         depth = float(row[depth_col])
                            
    #                         # Skip if coordinates are out of bounds
    #                         if not (0 <= x < self.cam_res[1] and 0 <= y < self.cam_res[0]):
    #                             print(f"    {joint}: coordinates ({x:.1f}, {y:.1f}) out of bounds, skipping")
    #                             continue
                            
    #                         # Map depth to color (blue = close, red = far)
    #                         normalized_depth = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
                            
    #                         # Create color: Blue (close) -> Cyan -> Green -> Yellow -> Red (far)
    #                         # Using HSV-like mapping for better depth perception
    #                         if normalized_depth < 0.25:
    #                             # Deep blue to cyan
    #                             ratio = normalized_depth * 4
    #                             color = [int(255 * ratio), int(255 * ratio), 255]  # BGR format
    #                         elif normalized_depth < 0.5:
    #                             # Cyan to green
    #                             ratio = (normalized_depth - 0.25) * 4
    #                             color = [int(255 * (1 - ratio)), 255, int(255 * (1 - ratio))]
    #                         elif normalized_depth < 0.75:
    #                             # Green to yellow
    #                             ratio = (normalized_depth - 0.5) * 4
    #                             color = [0, 255, int(255 * ratio)]
    #                         else:
    #                             # Yellow to red
    #                             ratio = (normalized_depth - 0.75) * 4
    #                             color = [0, int(255 * (1 - ratio)), 255]
                            
    #                         # Draw joint marker as colored dot
    #                         center = (int(x), int(y))
                            
    #                         # Draw larger outer circle with black border for visibility
    #                         cv2.circle(event_frame, center, 12, (0, 0, 0), 2)  # Black border
    #                         cv2.circle(event_frame, center, 10, color, -1)     # Filled colored circle
                            
    #                         # Add depth value as text
    #                         font = cv2.FONT_HERSHEY_SIMPLEX
    #                         font_scale = 0.4
    #                         thickness = 1
    #                         depth_text = f"{depth:.0f}"
    #                         text_size = cv2.getTextSize(depth_text, font, font_scale, thickness)[0]
    #                         text_x = int(x - text_size[0] // 2)
    #                         text_y = int(y - 18)
                            
    #                         # Black background for depth value
    #                         cv2.rectangle(event_frame, (text_x-2, text_y-text_size[1]-2), 
    #                                     (text_x + text_size[0]+2, text_y+2), (0, 0, 0), -1)
    #                         cv2.putText(event_frame, depth_text, (text_x, text_y), font, font_scale, 
    #                                   (255, 255, 255), thickness)
                            
    #                         # Add joint name below
    #                         joint_text_y = int(y + 25)
    #                         joint_text_size = cv2.getTextSize(joint, font, font_scale, thickness)[0]
    #                         joint_text_x = int(x - joint_text_size[0] // 2)
                            
    #                         cv2.rectangle(event_frame, (joint_text_x-2, joint_text_y-joint_text_size[1]-2), 
    #                                     (joint_text_x + joint_text_size[0]+2, joint_text_y+2), (0, 0, 0), -1)
    #                         cv2.putText(event_frame, joint, (joint_text_x, joint_text_y), font, font_scale, 
    #                                   (255, 255, 255), thickness)
                            
    #                         valid_joints.append((joint, depth, x, y))
    #                         joint_count += 1
    #                         print(f"    {joint}: ({x:.1f}, {y:.1f}) depth={depth:.1f}mm")
                            
    #                     except (ValueError, TypeError) as e:
    #                         print(f"    {joint}: Error processing data - {e}")
    #                         continue
                
    #             print(f"  Overlaid {joint_count} joints on event frame")
                
    #             # Add timestamp and info overlay
    #             info_text = [
    #                 f"Time: {actual_time:.6f}s",
    #                 f"Joints: {joint_count}/{len(joint_names)}",
    #                 f"Events: {len(e_ts)}",
    #                 f"Depth Range: {depth_min:.0f}-{depth_max:.0f}mm",
    #                 "Color: Blue=Close → Red=Far"
    #             ]
                
    #             y_offset = 30
    #             for text in info_text:
    #                 cv2.rectangle(event_frame, (10, y_offset-20), (10 + len(text)*8, y_offset+5), (0, 0, 0), -1)
    #                 cv2.putText(event_frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #                 y_offset += 25
                
    #             # Add depth colorbar legend
    #             colorbar_width = 200
    #             colorbar_height = 20
    #             colorbar_x = self.cam_res[1] - colorbar_width - 20
    #             colorbar_y = 30
                
    #             # Create colorbar
    #             for j in range(colorbar_width):
    #                 normalized_pos = j / colorbar_width
    #                 if normalized_pos < 0.5:
    #                     color = [0, int(255 * normalized_pos * 2), int(255 * (1 - normalized_pos * 2))]
    #                 else:
    #                     color = [int(255 * (normalized_pos - 0.5) * 2), int(255 * (1 - (normalized_pos - 0.5) * 2)), 0]
                    
    #                 cv2.line(event_frame, (colorbar_x + j, colorbar_y), 
    #                        (colorbar_x + j, colorbar_y + colorbar_height), color, 1)
                
    #             # Colorbar labels
    #             cv2.putText(event_frame, f"{depth_min:.0f}mm", (colorbar_x, colorbar_y - 5), 
    #                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    #             cv2.putText(event_frame, f"{depth_max:.0f}mm", (colorbar_x + colorbar_width - 40, colorbar_y - 5), 
    #                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
    #             # Save frame
    #             frame_filename = f"depth_overlay_{i:04d}_t{target_time:.3f}s.png"
    #             frame_path = os.path.join(viz_dir, frame_filename)
    #             cv2.imwrite(frame_path, event_frame)
    #             frame_count += 1
                
    #             if (i + 1) % 10 == 0:
    #                 print(f"  Generated {i + 1}/{len(selected_timestamps)} frames...")
                
    #         except Exception as e:
    #             print(f"Error processing timestamp {target_time:.3f}s: {e}")
    #             continue
        
    #     print(f"\n✓ Generated {frame_count} depth overlay frames")
    #     print(f"✓ Frames saved in: {viz_dir}")
        
    #     # Create a summary plot showing depth distribution
    #     plt.figure(figsize=(12, 8))
        
    #     # Plot depth over time for each joint
    #     for joint in joint_names:
    #         depth_col = f"{joint}_depth"
    #         valid_data = merged_df[merged_df[depth_col].notna()]
            
    #         if len(valid_data) > 0:
    #             plt.plot(valid_data[time_col], valid_data[depth_col], 
    #                     'o-', label=joint, alpha=0.7, markersize=4)
        
    #     plt.xlabel('Time (seconds)')
    #     plt.ylabel('Depth (mm)')
    #     plt.title('Joint Depth Evolution Over Time\n(Matching the color-coded overlay frames)')
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True, alpha=0.3)
        
    #     # Add color reference
    #     plt.axhline(y=depth_min, color='blue', linestyle='--', alpha=0.5, label=f'Min depth ({depth_min:.0f}mm)')
    #     plt.axhline(y=depth_max, color='red', linestyle='--', alpha=0.5, label=f'Max depth ({depth_max:.0f}mm)')
        
    #     plt.tight_layout()
        
    #     # Save depth plot
    #     depth_plot_path = os.path.join(viz_dir, "depth_timeline_summary.png")
    #     plt.savefig(depth_plot_path, dpi=300, bbox_inches='tight')
    #     print(f"✓ Depth timeline plot saved: {os.path.basename(depth_plot_path)}")
    #     plt.show()
        
    #     print("\n" + "=" * 60)
    #     print("DEPTH OVERLAY VISUALIZATION COMPLETED")
    #     print("=" * 60)
    #     print("The generated frames show:")
    #     print("• Gray dots: DVS events") 
    #     print("• Colored circles: Joint positions with depth-based coloring")
    #     print("• Blue circles: Closer joints (smaller depth values)")
    #     print("• Red circles: Farther joints (larger depth values)")
    #     print("• White labels: Joint names")
    #     print(f"\nFrames can be used to validate depth extraction accuracy!")
    #     print(f"All outputs saved in: {viz_dir}")

###

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #

    def _generate_joint_projections_and_depths(self, extract_depth: bool = False, output_dir: str = None, sequence_name: str = None) -> str:
        """Unified function to generate joint projections and optionally depths."""

        print("\n" + "="*60)
        if extract_depth:
            print("JOINT PROJECTIONS & DEPTH EXTRACTION")
        else:
            print("JOINT PROJECTIONS")
        print("="*60)
        
        # TODO: check for args
        if output_dir is None:
            output_dir = os.path.dirname(self.output_path)
        if sequence_name is None:
            sequence_name = self._extract_sequence_name()
            
        # Generate output file paths
        projections_csv_path = os.path.join(output_dir, f"{sequence_name}_joint_projections.csv")
        joint_video_path = os.path.join(output_dir, f"{sequence_name}_joint_projections.mp4")
        depth_csv_path = None
        
        if extract_depth:
            depth_csv_path = os.path.join(output_dir, f"{sequence_name}_joint_projections_and_depths.csv")
            # joint_video_path = os.path.join(output_dir, f"{sequence_name}_joint_analysis_video.mp4")
            print("Generating joint projections, depths, and analysis video...")
        else:
            print("Generating joint projections and video...")
        
        # Use unified method to generate both depths and projections
        depth_csv_path = self.project_joints_position(
            output_csv_path=depth_csv_path,
            output_projections_csv=projections_csv_path,
            output_video=joint_video_path,
            extract_depth=extract_depth
        )
        print(f"Joint analysis completed and video saved")

        return depth_csv_path   #TODO: kinda useless

###
    # TODO: check, debug and fix
    def plot_per_marker_error_boxplot(self, marker_errors: dict, save_path: str = None):
        """Display per-marker error distribution with stats."""

        valid_markers = [m for m, errs in marker_errors.items() if len(errs) > 0]
        if not valid_markers:
            print(" No valid markers with errors to plot.")
            return

        errors = [marker_errors[m] for m in valid_markers]
        stats = {
            m: {
                "mean": np.mean(errs),
                "std": np.std(errs),
                "min": np.min(errs),
                "max": np.max(errs)
            }
            for m, errs in zip(valid_markers, errors)
        }

        # Sort markers by mean error for clarity
        valid_markers = sorted(valid_markers, key=lambda m: stats[m]["mean"])
        errors = [marker_errors[m] for m in valid_markers]

        plt.figure(figsize=(12, 6))
        box = plt.boxplot(errors, patch_artist=True, labels=valid_markers, showmeans=True)

        # Styling
        for patch in box['boxes']:
            patch.set(facecolor='#b3cde3', alpha=0.8, edgecolor='black')
        for median in box['medians']:
            median.set(color='orange', linewidth=2)
        for mean in box['means']:
            mean.set(marker='o', color='red', markersize=4)

        plt.ylabel("Error (pixels)")
        plt.title("Per-Marker Error Distribution")
        plt.grid(axis='y', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(facecolor='#b3cde3', alpha=0.8, edgecolor='black', label='Error Distribution'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='Median'),
            plt.Line2D([0], [0], marker='o', color='green', markersize=4, linestyle='None', label='Mean'),
            plt.Line2D([0], [0], marker='o', color='black', markersize=3, linestyle='None', 
                      markerfacecolor='white', markeredgecolor='black', label='Outliers')
        ]
        plt.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()

        # --- Print comprehensive stats ---
        print("\n=== PER-MARKER ERROR SUMMARY ===")
        print(f"{'Marker':15s} {'Count':>6s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Q1':>8s} {'Median':>8s} {'Q3':>8s} {'Max':>8s}")
        print("-" * 85)
        
        for i, m in enumerate(valid_markers):
            errs = errors[i]
            s = stats[m]
            q1, median, q3 = np.percentile(errs, [25, 50, 75])
            
            print(f"{m:15s} {len(errs):6d} {s['mean']:8.2f} {s['std']:8.2f} {s['min']:8.2f} {q1:8.2f} {median:8.2f} {q3:8.2f} {s['max']:8.2f}")
            
            # Check for potential outliers (beyond 1.5*IQR from quartiles)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers = [e for e in errs if e < lower_fence or e > upper_fence]
            
            if outliers:
                print(f"               Outliers (>{upper_fence:.2f} or <{lower_fence:.2f}): {len(outliers)} points, max={max(outliers):.2f}")

        print("\n PLOT EXPLANATION:")
        print("• Light Blue Box = Interquartile Range (Q1 to Q3, contains middle 50% of data)")
        print("• Orange Line = Median (50th percentile)")
        print("• Red Dot = Mean (average)")
        print("• Whiskers = Extend to 1.5×IQR beyond box, or to min/max if closer")
        print("• Circles = Outliers (beyond whiskers)")

        # Removed the μ and σ annotations as requested

        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f" Error analysis plot saved to: {save_path}")
        
        plt.show()

    # def save_frames_with_gt_and_projections(self, labels_path: str, create_video: bool = False):
    #     """Save every frame showing ground truth and projected points - one frame per GT timestamp."""
        
    #     print("Creating frame-by-frame visualization with ground truth and projected points...")
        
    #     # --- Load labeled points from YAML ---
    #     labeled_points = helpers.read_points_labels(labels_path)
    #     print(f"Loaded {len(labeled_points['times'])} labeled timestamps from YAML")
    #     print(f"Available markers in labels: {set().union(*[frame.keys() for frame in labeled_points['points']])}")

    #     # --- Load projected points from CSV ---
    #     sequence_name = self._extract_sequence_name()
    #     base_csv_name = f"{sequence_name}_projection_points.csv"
    #     projected_points_csv = os.path.join(self._get_output_directory(), base_csv_name)
        
    #     if not os.path.exists(projected_points_csv):
    #         import glob
    #         csv_pattern = os.path.join(self._get_output_directory(), "projected_points*.csv")
    #         csv_files = glob.glob(csv_pattern)
    #         if csv_files:
    #             projected_points_csv = max(csv_files, key=os.path.getmtime)
    #             print(f"Using most recent projected points CSV: {os.path.basename(projected_points_csv)}")
    #         else:
    #             print(f"Error: No projected points CSV files found in {self._get_output_directory()}")
    #             return None

    #     # Load CSV data
    #     projected_data = {}  # marker_name -> [(timestamp, x, y), ...]
    #     try:
    #         with open(projected_points_csv, 'r') as f:
    #             reader = csv.DictReader(f)
    #             for row in reader:
    #                 timestamp = float(row['event_timestamp'])
                    
    #                 # Extract marker data from CSV columns
    #                 for column in reader.fieldnames:
    #                     if column == 'event_timestamp':
    #                         continue
    #                     if column.endswith('_x'):
    #                         marker_name = column[:-2]  # Remove '_x' suffix
    #                         y_column = f"{marker_name}_y"
                            
    #                         if row[column] and row.get(y_column):
    #                             try:
    #                                 x = float(row[column])
    #                                 y = float(row[y_column])
    #                                 projected_data.setdefault(marker_name, []).append((timestamp, x, y))
    #                             except ValueError:
    #                                 continue
                                    
    #     except Exception as e:
    #         print(f"Error reading CSV file {projected_points_csv}: {e}")
    #         return None

    #     print(f"Loaded projected data for {len(projected_data)} markers from CSV.")

    #     # Create output directory for frame images
    #     frames_dir = os.path.join(self._get_output_directory(), "gt_projection_frames")
    #     os.makedirs(frames_dir, exist_ok=True)
        
    #     print(f"Frame images will be saved to: {frames_dir}")
    #     print(f"Note: CSV projections already include delay correction - using direct timestamp matching")

    #     frame_count = 0
    #     video_frames = []
        
    #     # Process each GT timestamp individually (one frame per GT timestamp)
    #     for frame_idx, (gt_timestamp, labeled_frame) in enumerate(zip(labeled_points['times'], labeled_points['points'])):
            
    #         # Create image once for this GT timestamp (like original method)
    #         img = np.ones((self.cam_res[0], self.cam_res[1], 3), dtype=np.uint8) * 255
            
    #         # Load events around this specific GT timestamp (like original method pattern)
    #         try:
    #             event_window = 0.02  # 20ms window around GT timestamp
    #             e_data = self.imp.get_data_at_time(gt_timestamp, event_window)
    #             e_ts = np.array(e_data['ts'])
    #             e_us = np.array(e_data['x'])
    #             e_vs = np.array(e_data['y'])
                
    #             # Render events as background (like original method)
    #             for i in range(len(e_ts)):
    #                 uu = int(e_us[i])
    #                 vv = int(e_vs[i])
    #                 if 0 <= uu < self.cam_res[1] and 0 <= vv < self.cam_res[0]:
    #                     img[vv, uu] = [80, 80, 80]  # Dark gray events for background
                        
    #         except Exception as e:
    #             print(f"Warning: Could not load events for timestamp {gt_timestamp:.3f}s: {e}")
    #             # Keep white background if events cannot be loaded
            
    #         points_drawn = 0
    #         frame_errors = []  # Track pixel errors for this frame
            
    #         # Process each marker in this GT frame
    #         for marker, label in labeled_frame.items():
    #             gt_x, gt_y = int(label['x']), int(label['y'])
                
    #             if marker not in projected_data:
    #                 # Draw only ground truth if no projection available - single small circle
    #                 cv2.circle(img, (gt_x, gt_y), 5, (0, 255, 0), -1)  # Green circle for GT
    #                 points_drawn += 1
    #                 continue

    #             # Find closest projected point (no additional delay correction needed - already in CSV)
    #             proj_list = sorted(projected_data[marker], key=lambda x: x[0])
    #             timestamps = [p[0] for p in proj_list]
    #             insert_pos = bisect.bisect_left(timestamps, gt_timestamp)

    #             candidates = []
    #             for idx in [insert_pos - 1, insert_pos]:
    #                 if 0 <= idx < len(proj_list):
    #                     proj_timestamp, proj_x, proj_y = proj_list[idx]
    #                     time_diff = abs(proj_timestamp - gt_timestamp)
    #                     candidates.append((time_diff, proj_x, proj_y, proj_timestamp))
                
    #             if candidates:
    #                 _, proj_x, proj_y, proj_timestamp = min(candidates)
                    
    #                 # Draw ground truth point (green) - single small circle
    #                 cv2.circle(img, (gt_x, gt_y), 5, (0, 255, 0), -1)
                    
    #                 # Draw projected point (red) - single small circle  
    #                 cv2.circle(img, (int(proj_x), int(proj_y)), 5, (0, 0, 255), -1)
                    
    #                 # Draw connection line (yellow)
    #                 cv2.line(img, (gt_x, gt_y), (int(proj_x), int(proj_y)), (0, 255, 255), 2)
                    
    #                 # Calculate and display error
    #                 error_magnitude = np.sqrt((proj_x - gt_x)**2 + (proj_y - gt_y)**2)
    #                 frame_errors.append(error_magnitude)  # Track for frame statistics
                    
    #                 # Position text above the line with better visibility
    #                 mid_x, mid_y = (gt_x + int(proj_x)) // 2, (gt_y + int(proj_y)) // 2
    #                 text_y = mid_y - 15  # Position 15 pixels above the line for better spacing
                    
    #                 # Create distance text with marker name for clarity
    #                 distance_text = f"{marker}: {error_magnitude:.1f}px"
                    
    #                 # Add black background rectangle for better text visibility
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 font_scale = 0.45
    #                 thickness = 1
    #                 (text_width, text_height), baseline = cv2.getTextSize(distance_text, font, font_scale, thickness)
                    
    #                 # Draw black background rectangle
    #                 cv2.rectangle(img, (mid_x - 2, text_y - text_height - 2), 
    #                              (mid_x + text_width + 2, text_y + baseline + 2), (0, 0, 0), -1)
                    
    #                 # Draw white text on black background
    #                 cv2.putText(img, distance_text, (mid_x, text_y), font, font_scale, (255, 255, 255), thickness)
                    
    #                 points_drawn += 1
    #             else:
    #                 # Draw only ground truth if no matching projection found - single small circle
    #                 cv2.circle(img, (gt_x, gt_y), 5, (0, 255, 0), -1)
    #                 points_drawn += 1
            
    #         if points_drawn > 0:
    #             # Calculate frame statistics
    #             frame_stats_text = ""
    #             if frame_errors:
    #                 mean_error = np.mean(frame_errors)
    #                 max_error = np.max(frame_errors)
    #                 frame_stats_text = f" | Errors: avg={mean_error:.1f}px, max={max_error:.1f}px"
                
    #             # Add frame info (CSV data note) with error statistics
    #             cv2.putText(img, f"GT: {gt_timestamp:.3f}s (CSV projections){frame_stats_text}", (10, 30), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    #             cv2.putText(img, "Legend: GT=Green, PROJ=Red, Error=Yellow line + distance", (10, 60), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
    #             # Add timestamp display in top-right corner
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             font_scale = 0.5
    #             thickness = 1
    #             color = (128, 128, 128)
                
    #             # GT timestamp (same as CSV lookup timestamp)
    #             gt_time_text = f"Timestamp: {gt_timestamp:.3f}s"
    #             (text_width, text_height), _ = cv2.getTextSize(gt_time_text, font, font_scale, thickness)
    #             x = self.cam_res[1] - text_width - 10
    #             y = text_height + 10
    #             cv2.putText(img, gt_time_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                
    #             # Save frame
    #             frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}_t{gt_timestamp:.3f}.png")
    #             cv2.imwrite(frame_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
    #             if create_video:
    #                 video_frames.append(img.copy())
                
    #             frame_count += 1
                
    #             # Print progress every 10 frames
    #             if frame_count % 10 == 0:
    #                 print(f"Saved {frame_count} frames...")
        
    #     print(f"Generated {frame_count} frame visualization images")
        
    #     # Create video if requested
    #     if create_video and video_frames:
    #         video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_gt_projection_comparison.mp4")
    #         self._save_video_from_frames(video_frames, video_path, fps=10)
    #         print(f"Saved comparison video: {video_path}")
        
    #     return frames_dir

    # def _save_video_from_frames(self, frames, output_path, fps=10):
    #     """Save a list of frames as a video file."""
    #     if not frames:
    #         print("No frames to save for video")
    #         return
        
    #     height, width = frames[0].shape[:2]
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        
    #     for frame in frames:
    #         video_writer.write(frame)
        
    #     video_writer.release()

    # def create_vicon_marker_projection_video(self, output_video_path: str = None) -> str:
    #     """Create and save a video showing VICON marker projections on event frames.
        
    #     Args:
    #         output_video_path: Optional output path. If None, generates automatic path.
            
    #     Returns:
    #         str: Path to the saved video file
    #     """
    #     print("Creating VICON marker projection video...")
        
    #     if not self.markers_names:
    #         self.markers_names = self.get_markers_names()
        
    #     print(f"Projecting {len(self.markers_names)} VICON markers: {self.markers_names[:5]}{'...' if len(self.markers_names) > 5 else ''}")
        
    #     # Generate output path if not provided
    #     if output_video_path is None:
    #         sequence_name = self._extract_sequence_name()
    #         output_video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_vicon_marker_projections.mp4")
    #         output_video_path = self._generate_unique_video_path(output_video_path)
        
    #     # Create projector for VICON markers
    #     projector = helpers.ViconProjector(
    #         self.markers_names, self.c3d_data, self.points_3d,
    #         self.T_syst_to_camera_opt, self.Ts_world_to_system,
    #         self.K, self.cam_res, D=self.D, subject=self.subject
    #     )
        
    #     # Create video writer
    #     fps = int(1 / self.period)  # VICON frame rate
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (self.cam_res[1], self.cam_res[0]), isColor=True)
        
    #     if not video_writer.isOpened():
    #         print(f"Error: Could not open video writer for {output_video_path}")
    #         return None
        
    #     # Process time windows
    #     window_size = 500 * self.period  # 500ms windows
    #     window_start = self.start_time
    #     frame_count = 0
    #     total_frames = 0
        
    #     try:
    #         while window_start < self.end_time:
    #             window_end = window_start + window_size
    #             window_center = (window_start + window_end) / 2
                
    #             # Get events for this window
    #             e_data = self.imp.get_data_at_time(window_center, window_size)
    #             e_ts = np.array(e_data['ts'])
    #             e_us = np.array(e_data['x'])
    #             e_vs = np.array(e_data['y'])
                
    #             if len(e_ts) == 0:
    #                 window_start = window_end
    #                 continue
                
    #             # Create event frame
    #             event_frame = np.zeros((self.cam_res[0], self.cam_res[1], 3), dtype=np.uint8)
                
    #             # Draw events
    #             for u, v in zip(e_us, e_vs):
    #                 if 0 <= u < self.cam_res[1] and 0 <= v < self.cam_res[0]:
    #                     event_frame[int(v), int(u)] = [255, 255, 255]  # White events
                
    #             # Project and draw VICON markers
    #             try:
    #                 projections, valid_markers = projector.project_markers_at_time(window_center)
                    
    #                 for marker_name, (x, y) in projections.items():
    #                     if 0 <= x < self.cam_res[1] and 0 <= y < self.cam_res[0]:
    #                         # Draw marker as colored circle
    #                         cv2.circle(event_frame, (int(x), int(y)), 6, (0, 255, 0), -1)  # Green circle
    #                         cv2.circle(event_frame, (int(x), int(y)), 8, (0, 255, 0), 2)   # Green outline
                            
    #                         # Add marker name
    #                         cv2.putText(event_frame, marker_name, (int(x) + 10, int(y) - 10),
    #                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
    #             except Exception as e:
    #                 print(f"Warning: Could not project markers at time {window_center:.3f}s: {e}")
                
    #             # Add timestamp overlay
    #             cv2.putText(event_frame, f"t={window_center:.3f}s", (10, 30),
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    #             # Write frame to video
    #             video_writer.write(event_frame)
    #             frame_count += 1
    #             total_frames += 1
                
    #             # Progress update
    #             if frame_count % 100 == 0:
    #                 print(f"Processed {frame_count} frames...")
                
    #             window_start = window_end
                
    #     except KeyboardInterrupt:
    #         print("Video creation interrupted by user")
    #     except Exception as e:
    #         print(f"Error during video creation: {e}")
    #     finally:
    #         video_writer.release()
        
    #     if total_frames > 0:
    #         print(f"✓ VICON marker projection video created: {os.path.basename(output_video_path)} ({total_frames} frames)")
    #         return output_video_path
    #     else:
    #         print("❌ No frames were written to video")
    #         if os.path.exists(output_video_path):
    #             os.remove(output_video_path)
    #         return None

    # TODO: check debug and fix
    def calculate_projection_error(self, labels_path: str, visualize: bool = True):
        """Calculate 2D-2D error between manual labels and projected markers (TXT file)."""
        
        print("Comparing manually labeled points with projected marker positions from TXT file...")

        # --- Load labeled points from YAML ---
        labeled_points = helpers.read_points_labels(labels_path)
        print(f"Loaded {len(labeled_points['times'])} labeled timestamps from YAML")
        print(f"Available markers in labels: {set().union(*[frame.keys() for frame in labeled_points['points']])}")

        # --- Load projected points from CSV ---
        # Look for CSV files with projected points (sequence-based naming)
        sequence_name = self._extract_sequence_name()
        base_csv_name = f"{sequence_name}_projection_points.csv"
        projected_points_csv = os.path.join(self._get_output_directory(), base_csv_name)
        
        #TODO: cehck 
        # If sequence-specific CSV doesn't exist, look for generic ones
        if not os.path.exists(projected_points_csv):
            # Look for any projected_points*.csv files
            import glob
            csv_pattern = os.path.join(self._get_output_directory(), "projected_points*.csv")
            csv_files = glob.glob(csv_pattern)
            if csv_files:
                # Use the most recent CSV file
                projected_points_csv = max(csv_files, key=os.path.getmtime)
                print(f"Using most recent projected points CSV: {os.path.basename(projected_points_csv)}")
            else:
                print(f"Error: No projected points CSV files found in {self._get_output_directory()}")
                return None
        
        # Load CSV data
        projected_data = {}  # marker_name -> [(timestamp, x, y), ...]
        try:
            with open(projected_points_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = float(row['event_timestamp'])
                    
                    # Extract marker data from CSV columns
                    for column in reader.fieldnames:
                        if column == 'event_timestamp':
                            continue
                        if column.endswith('_x'):
                            marker_name = column[:-2]  # Remove '_x' suffix
                            y_column = f"{marker_name}_y"
                            
                            # Check if both x and y values exist and are not empty
                            if row[column] and row.get(y_column):
                                try:
                                    x = float(row[column])
                                    y = float(row[y_column])
                                    projected_data.setdefault(marker_name, []).append((timestamp, x, y))
                                except ValueError:
                                    continue
                                    
        except Exception as e:
            print(f" Error reading CSV file {projected_points_csv}: {e}")
            return None

        print(f"Loaded projected data for {len(projected_data)} markers from CSV.")

        # --- Compute errors ---
        comparison_results = {'marker_errors': {}}
        print(f"\nAnalyzing {len(labeled_points['times'])} labeled frames...")

        for event_timestamp, labeled_frame in zip(labeled_points['times'], labeled_points['points']):
            for marker, label in labeled_frame.items():
                label_x, label_y = int(label['x']), int(label['y'])
                if marker not in projected_data:
                    continue

                proj_list = sorted(projected_data[marker], key=lambda x: x[0])
                timestamps = [p[0] for p in proj_list]
                insert_pos = bisect.bisect_left(timestamps, event_timestamp)

                # Find closest projection by timestamp
                candidates = []
                for idx in [insert_pos - 1, insert_pos]:
                    if 0 <= idx < len(proj_list):
                        proj_timestamp, proj_x, proj_y = proj_list[idx]
                        time_diff = abs(proj_timestamp - event_timestamp)
                        candidates.append((time_diff, proj_x, proj_y))
                if not candidates:
                    continue

                _, proj_x, proj_y = min(candidates)
                error = np.linalg.norm(np.array([label_x, label_y]) - np.array([proj_x, proj_y]))
                comparison_results['marker_errors'].setdefault(marker, []).append(error)

        # --- Plot per-marker error distribution ---
        # Generate save path for the error plot
        # import datetime
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # plot_save_path = os.path.join(self._get_output_directory(), f"error_analysis.png")
        # self.plot_per_marker_error_boxplot(comparison_results['marker_errors'], save_path=plot_save_path)

        return comparison_results
###

###
    # def optimize_calibration(self, labels_path: str, ransac_reproj_err: float = 3.0, ransac_iters: int = 300, min_points: int = 6):
    #     """ Optimize a single, fixed T_sys->cam using multi-frame labels and per-frame world -> system (per frame, known) -> camera (unknown, fixed)."""

    #     print("Optimizing calibration with OpenCV (global multi-frame PnP + refine)...")

    #     # Load labels + interpolate 3D
    #     labeled_points = helpers.read_points_labels(labels_path)
    #     vicon_helper = helpers.ViconHelper(
    #         self.marker_t, self.points_3d, self.delay,
    #         self.c3d_data.frame_count, self.c3d_data.point_rate,
    #         self.c3d_data.point_labels, True, True,
    #         user_camera_markers=getattr(self, "camera_markers", None)
    #     )
    #     vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)

    #     # 2) Build global correspondences in SYSTEM frame
    #     system_points, image_points = [], []

    #     n_vp = len(vicon_points.get('points', []))
    #     n_lp = len(labeled_points.get('points', []))
    #     n_fi = len(vicon_points.get('frame_ids', []))
    #     n_frames = min(n_vp, n_lp, n_fi)

    #     if n_frames == 0:
    #         raise RuntimeError("No overlapping labeled frames and VICON points.")

    #     if (n_vp != n_lp) or (n_vp != n_fi):
    #         print(f"[warn] Length mismatch: vicon_points.points={n_vp}, "
    #             f"labeled_points.points={n_lp}, frame_ids={n_fi}. "
    #             f"Using first {n_frames} aligned entries.")

    #     valid_pairs = 0
    #     skipped_frames = 0

    #     for idx in range(n_frames):
    #         if idx >= len(vicon_points['frame_ids']):
    #             skipped_frames += 1
    #             continue
    #         frame_id = int(vicon_points['frame_ids'][idx])
    #         if not (0 <= frame_id < len(self.Ts_world_to_system)):
    #             skipped_frames += 1
    #             continue

    #         w_frame = vicon_points['points'][idx] or {}
    #         d_frame = labeled_points['points'][idx] or {}
    #         if not w_frame or not d_frame:
    #             skipped_frames += 1
    #             continue

    #         T_w2s = self.Ts_world_to_system[frame_id]  # 4x4

    #         for label, px in d_frame.items():
    #             if label not in w_frame:
    #                 continue
    #             w = np.asarray(w_frame[label], dtype=np.float64)
    #             if w is None or np.any(~np.isfinite(w)):
    #                 continue

    #             p_sys = (T_w2s @ np.append(w, 1.0))[:3]
    #             if np.any(~np.isfinite(p_sys)):
    #                 continue

    #             u = float(px['x']); v = float(px['y'])
    #             if not (np.isfinite(u) and np.isfinite(v)):
    #                 continue

    #             system_points.append(p_sys)
    #             image_points.append([u, v])
    #             valid_pairs += 1

    #     print(f"Collected correspondences: {valid_pairs} (skipped frames: {skipped_frames})")
    #     if valid_pairs < max(4, min_points):
    #         raise RuntimeError("Not enough valid correspondences to run PnP.")

    #     # 3) Prepare arrays for OpenCV
    #     obj_pts = np.ascontiguousarray(np.asarray(system_points), dtype=np.float32).reshape(-1, 3)
    #     img_pts = np.ascontiguousarray(np.asarray(image_points),  dtype=np.float32).reshape(-1, 2)
    #     K_cv    = np.ascontiguousarray(self.K, dtype=np.float32)

    #     if self.D is None:
    #         D_cv = None
    #     else:
    #         D_flat = np.asarray(self.D, dtype=np.float32).ravel()
    #         # OpenCV accepts (k,) just fine; avoid forcing (k,1) on stricter builds
    #         D_cv = None if D_flat.size == 0 else np.ascontiguousarray(D_flat)

    #     # 4) Initial pose with RANSAC (try legacy 8-arg first)
    #     rvec = None; tvec = None; inliers = None
    #     try:
    #         # Legacy signature (8 args): (obj, img, K, D, iterationsCount, reprojErr, confidence, flags)
    #         retval, rvec, tvec, inliers = cv2.solvePnPRansac(
    #             obj_pts, img_pts, K_cv, D_cv,
    #             int(ransac_iters), float(ransac_reproj_err), 0.99, cv2.SOLVEPNP_EPNP
    #         )
    #     except cv2.error:
    #         try:
    #             # Newer signature with useExtrinsicGuess (9 args): add False after D
    #             retval, rvec, tvec, inliers = cv2.solvePnPRansac(
    #                 obj_pts, img_pts, K_cv, D_cv,
    #                 False, int(ransac_iters), float(ransac_reproj_err), 0.99, cv2.SOLVEPNP_EPNP
    #             )
    #         except cv2.error:
    #             # Last resort: keyword flags only (some builds insist on this)
    #             retval, rvec, tvec, inliers = cv2.solvePnPRansac(
    #                 obj_pts, img_pts, K_cv, D_cv, flags=cv2.SOLVEPNP_EPNP
    #             )

    #     if not bool(retval) or inliers is None or len(inliers) < 4:
    #         print("[warn] RANSAC failed or too few inliers; trying iterative PnP...")
    #         ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_cv, D_cv, flags=cv2.SOLVEPNP_ITERATIVE)
    #         if not bool(ok):
    #             raise RuntimeError("PnP failed to produce an initial estimate.")
    #         inlier_idx = np.arange(len(obj_pts))
    #     else:
    #         inlier_idx = inliers.ravel()

    #     print(f"Initial inliers: {len(inlier_idx)} / {len(obj_pts)}")

    #     # 5) Refine with LM on inliers
    #     obj_in = obj_pts[inlier_idx]
    #     img_in = img_pts[inlier_idx]
    #     rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K_cv, D_cv, rvec, tvec)

    #     # 6) Build T_sys->cam
    #     R_sc, _ = cv2.Rodrigues(rvec)
    #     T_sc = np.eye(4, dtype=np.float64)
    #     T_sc[:3, :3] = R_sc.astype(np.float64)
    #     T_sc[:3, 3]  = tvec.reshape(3).astype(np.float64)

    #     # 7) Evaluate & store
    #     proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_cv, D_cv)
    #     proj = proj.reshape(-1, 2)
    #     errs = np.linalg.norm(proj - img_pts, axis=1)
    #     print(f"Reproj error: mean={errs.mean():.3f}px, median={np.median(errs):.3f}px, 95%={np.percentile(errs,95):.3f}px")

    #     self.T_syst_to_camera_opt = T_sc
    #     print("Optimized T_sys->cam:\n", T_sc)
    #     return T_sc


    def optimize_calibration(self, labels_path: str,
                         min_points_per_frame: int = 4,
                         ransac_reproj_err: float = 3.0,
                         ransac_iters: int = 300,
                         pnp_reproj_clip: float = 5.0):
        """
        Robust system→camera calibration using:
        • per-frame PnP (RANSAC + LM)
        • frame rejection
        • global Levenberg–Marquardt reprojection minimization
        """

        import numpy as np, cv2
        import helpers
        from helpers import ViconHelper
        from scipy.spatial.transform import Rotation

        print("Optimizing calibration using global LM reprojection minimization ...")

        # ------------------ helpers ------------------
        def invert_Rt(R, t):
            Rinv = R.T
            tinv = -Rinv @ t.reshape(3, 1)
            return Rinv, tinv

        def per_frame_reproj_err(R, t, P3, P2, K, D):
            rvec, _ = cv2.Rodrigues(R)
            proj, _ = cv2.projectPoints(P3, rvec, t.reshape(3,1), K, D)
            return np.linalg.norm(proj.reshape(-1,2) - P2, axis=1)

        K_cv = np.ascontiguousarray(self.K, dtype=np.float64)
        D_cv = None if self.D is None else np.ascontiguousarray(self.D, dtype=np.float64)

        # ------------------ load labels + VICON ------------------
        labeled_points = helpers.read_points_labels(labels_path)

        vicon_helper = ViconHelper(
            self.marker_t, self.points_3d, self.delay,
            self.c3d_data.frame_count, self.c3d_data.point_rate,
            self.c3d_data.point_labels, True, True,
            user_camera_markers=getattr(self, "camera_markers", None)
        )

        vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)

        n_frames = min(len(labeled_points['points']),
                    len(vicon_points['points']),
                    len(vicon_points['frame_ids']))

        # ------------------ per-frame PnP ------------------
        R_wc_list, t_wc_list = [], []
        kept = 0

        for idx in range(n_frames):
            dvs_frame = labeled_points['points'][idx] or {}
            w_frame   = vicon_points['points'][idx] or {}
            if not dvs_frame or not w_frame:
                continue

            W3, I2 = [], []
            for lab, pix in dvs_frame.items():
                if lab not in w_frame:
                    continue
                W = np.asarray(w_frame[lab], dtype=np.float64)
                if np.any(~np.isfinite(W)):
                    continue
                W3.append(W)
                I2.append([pix['x'], pix['y']])

            if len(W3) < min_points_per_frame:
                continue

            W3 = np.asarray(W3, dtype=np.float64)
            I2 = np.asarray(I2, dtype=np.float64)

            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    W3, I2, K_cv, D_cv,
                    iterationsCount=ransac_iters,
                    reprojectionError=ransac_reproj_err,
                    flags=cv2.SOLVEPNP_EPNP
                )
                if not ok:
                    continue

                rvec, tvec = cv2.solvePnPRefineLM(
                    W3[inl[:,0]], I2[inl[:,0]],
                    K_cv, D_cv, rvec, tvec
                )

            except cv2.error:
                continue

            R_wc, _ = cv2.Rodrigues(rvec)
            t_wc = tvec.reshape(3,1)

            err = per_frame_reproj_err(R_wc, t_wc, W3, I2, K_cv, D_cv)
            if np.median(err) > pnp_reproj_clip:
                continue

            R_wc_list.append(R_wc)
            t_wc_list.append(t_wc)
            kept += 1

        print(f"Frames kept after PnP filtering: {kept}")
        if kept < 3:
            raise RuntimeError("Not enough valid frames for optimization")

        # ------------------ build global correspondences ------------------
        system_points = []
        image_points  = []

        for idx in range(n_frames):
            dvs_frame = labeled_points['points'][idx] or {}
            w_frame   = vicon_points['points'][idx] or {}
            if not dvs_frame or not w_frame:
                continue

            frame_id = int(vicon_points['frame_ids'][idx])
            if not (0 <= frame_id < len(self.Ts_world_to_system)):
                continue

            T_w2s = self.Ts_world_to_system[frame_id]

            for lab, pix in dvs_frame.items():
                if lab not in w_frame:
                    continue
                W = np.asarray(w_frame[lab], dtype=np.float64)
                if np.any(~np.isfinite(W)):
                    continue

                p_sys = (T_w2s @ np.append(W, 1.0))[:3]
                system_points.append(p_sys)
                image_points.append([pix['x'], pix['y']])

        Ps = np.asarray(system_points, dtype=np.float64)
        pc = np.asarray(image_points, dtype=np.float64)

        print(f"Global correspondences: {len(Ps)}")

        # ------------------ initial guess ------------------
        # Use average of per-frame estimates
        Rs = np.stack(R_wc_list, axis=0)
        ts = np.stack([t.reshape(3) for t in t_wc_list], axis=0)

        R_init = Rotation.from_matrix(Rs).mean().as_matrix()
        t_init = np.mean(ts, axis=0)

        rvec_init = Rotation.from_matrix(R_init).as_rotvec()
        init_params = np.hstack([rvec_init, t_init])

        # ------------------ LM optimization ------------------
        print("Running global Levenberg–Marquardt optimization ...")

        T_sc = helpers.estimate_Tstoc(
            Ps, pc, K_cv, D_cv, init_params
        )

        # ------------------ diagnostics ------------------
        rvec_sc, _ = cv2.Rodrigues(T_sc[:3, :3])
        tvec_sc = T_sc[:3, 3].reshape(3,1)
        proj, _ = cv2.projectPoints(Ps, rvec_sc, tvec_sc, K_cv, D_cv)
        err = np.linalg.norm(proj.reshape(-1,2) - pc, axis=1)

        print(f"[LM] reprojection error: mean={err.mean():.3f}px, "
            f"median={np.median(err):.3f}px, 95%={np.percentile(err,95):.3f}px")

        self.T_syst_to_camera_opt = T_sc
        print("Estimated T_sys→cam (LM):\n", T_sc)

        return T_sc

###           
            
    def save_calibration(self, output_file: str):
        """Save transformation matrix and delay."""
        
        def format_matrix_block(T: np.ndarray) -> str:
            rows = []
            for i, row in enumerate(T):
                row_str = " ".join(f"{val:.6f}" for val in row).lstrip()
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

        print(f"Saved calibration to {os.path.basename(output_file)} (delay: {self.delay}s)")
        
    def plot_3d_marker_trajectory(self, marker_name: str = None, save_path: str = None):
        """Plot 3D trajectory of a specific marker over time."""
        
        if marker_name is None:
            # Ask user to select a marker
            available_markers = [label.strip() for label in self.c3d_data.point_labels]
            print(f"\nAvailable markers: {available_markers[:10]}{'...' if len(available_markers) > 10 else ''}")
            while True:
                marker_name = input("Enter marker name for 3D trajectory plot: ").strip()
                if marker_name in available_markers:
                    break
                print(f"Marker '{marker_name}' not found. Available markers: {available_markers}")
        
        try:
            # Get 3D marker positions over time
            marker_positions = []
            valid_times = []
            
            for i, frame_data in enumerate(self.points_3d.values()):
                if marker_name in [label.strip() for label in self.c3d_data.point_labels]:
                    marker_idx = [label.strip() for label in self.c3d_data.point_labels].index(marker_name)
                    if marker_idx < len(frame_data):
                        pos = frame_data[marker_idx][:3]  # x, y, z
                        if np.all(np.isfinite(pos)) and not np.allclose(pos, 0):
                            marker_positions.append(pos)
                            valid_times.append(self.marker_t[i] if i < len(self.marker_t) else i * (1/100))
            
            if not marker_positions:
                print(f"No valid positions found for marker '{marker_name}'")
                return
                
            positions = np.array(marker_positions)
            times = np.array(valid_times)
            
            fig = plt.figure(figsize=(15, 5), num=f"3D Marker Trajectory: {marker_name}")
            
            # 3D trajectory
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7, linewidth=2)
            ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
            ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            ax1.set_zlabel('Z (mm)')
            ax1.set_title(f'3D Trajectory: {marker_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # X, Y, Z vs time
            ax2 = fig.add_subplot(132)
            ax2.plot(times, positions[:, 0], 'r-', label='X', alpha=0.8)
            ax2.plot(times, positions[:, 1], 'g-', label='Y', alpha=0.8)
            ax2.plot(times, positions[:, 2], 'b-', label='Z', alpha=0.8)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position (mm)')
            ax2.set_title(f'Position vs Time: {marker_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Velocity magnitude
            ax3 = fig.add_subplot(133)
            if len(positions) > 1:
                dt = np.diff(times)
                velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
                ax3.plot(times[1:], velocity, 'purple', linewidth=2)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Velocity (mm/s)')
                ax3.set_title(f'Velocity Magnitude: {marker_name}')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"3D marker plot saved: {save_path}")
            
            plt.show(block=False)  # Non-blocking show to allow multiple plots
            
        except Exception as e:
            print(f"Error plotting 3D marker trajectory: {e}")
    
    def plot_2d_marker_projections(self, marker_name: str = None, save_path: str = None):
        """Plot 2D projections of a specific marker over time."""
        
        if marker_name is None:
            # Ask user to select a marker
            available_markers = [label.strip() for label in self.c3d_data.point_labels]
            print(f"\nAvailable markers: {available_markers[:10]}{'...' if len(available_markers) > 10 else ''}")
            while True:
                marker_name = input("Enter marker name for 2D projection plot: ").strip()
                if marker_name in available_markers:
                    break
                print(f"Marker '{marker_name}' not found. Available markers: {available_markers}")
        
        try:
            # Create projector to get 2D projections
            if not hasattr(self, 'T_syst_to_camera_opt') or self.T_syst_to_camera_opt is None:
                print("No optimized calibration found. Cannot compute 2D projections.")
                return
                
            projector = helpers.ViconProjector(
                [marker_name], self.c3d_data, self.points_3d,
                self.T_syst_to_camera_opt, self.Ts_world_to_system,
                self.K, self.cam_res, D=self.D, subject=self.subject
            )
            
            # Get 2D projections
            if marker_name not in projector.image_points:
                print(f"Could not compute projections for marker '{marker_name}'")
                return
                
            projections = projector.image_points[marker_name]
            times = self.marker_t[:len(projections)]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), num=f"2D Marker Projections: {marker_name}")
            
            # 2D trajectory on image plane
            valid_mask = (projections[:, 0] >= 0) & (projections[:, 0] < self.cam_res[1]) & \
                        (projections[:, 1] >= 0) & (projections[:, 1] < self.cam_res[0])
            valid_projs = projections[valid_mask]
            valid_times = times[valid_mask]
            
            if len(valid_projs) > 0:
                ax1.plot(valid_projs[:, 0], valid_projs[:, 1], 'b-', alpha=0.7, linewidth=2)
                ax1.scatter(valid_projs[0, 0], valid_projs[0, 1], color='green', s=100, label='Start')
                ax1.scatter(valid_projs[-1, 0], valid_projs[-1, 1], color='red', s=100, label='End')
                ax1.set_xlim(0, self.cam_res[1])
                ax1.set_ylim(self.cam_res[0], 0)  # Invert Y axis for image coordinates
                ax1.set_xlabel('X (pixels)')
                ax1.set_ylabel('Y (pixels)')
                ax1.set_title(f'2D Trajectory on Image Plane: {marker_name}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
            
            # X pixel position vs time
            ax2.plot(valid_times, valid_projs[:, 0], 'r-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('X Position (pixels)')
            ax2.set_title(f'X Projection vs Time: {marker_name}')
            ax2.grid(True, alpha=0.3)
            
            # Y pixel position vs time
            ax3.plot(valid_times, valid_projs[:, 1], 'g-', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Y Position (pixels)')
            ax3.set_title(f'Y Projection vs Time: {marker_name}')
            ax3.grid(True, alpha=0.3)
            
            # 2D velocity
            if len(valid_projs) > 1:
                dt = np.diff(valid_times)
                velocity_2d = np.linalg.norm(np.diff(valid_projs, axis=0), axis=1) / dt
                ax4.plot(valid_times[1:], velocity_2d, 'purple', linewidth=2)
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('2D Velocity (pixels/s)')
                ax4.set_title(f'2D Velocity: {marker_name}')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"2D marker plot saved: {save_path}")
            
            plt.show(block=False)  # Non-blocking show to allow multiple plots
            
        except Exception as e:
            print(f"Error plotting 2D marker projections: {e}")
    
    def plot_camera_motion(self, save_path: str = None):
        """Plot camera motion trajectory and orientation over time."""
        
        try:
            if not hasattr(self, 'T_syst_to_camera_opt') or self.T_syst_to_camera_opt is None:
                print("No optimized calibration found. Cannot compute camera motion.")
                return
                
            if not hasattr(self, 'Ts_world_to_system') or self.Ts_world_to_system is None:
                print("No world-to-system transforms found. Cannot compute camera motion.")
                return
            
            # Compute camera poses in world frame
            camera_positions = []
            camera_orientations = []
            times = self.marker_t[:len(self.Ts_world_to_system)]
            
            T_sc = self.T_syst_to_camera_opt  # System to camera (fixed)
            
            for i, T_ws in enumerate(self.Ts_world_to_system):
                # World to camera transform
                T_wc = T_sc @ T_ws
                
                # Camera position in world frame (inverse transform)
                T_cw = np.linalg.inv(T_wc)
                camera_pos = T_cw[:3, 3]
                camera_positions.append(camera_pos)
                
                # Camera orientation (rotation matrix to Euler angles)
                R_cw = T_cw[:3, :3]
                rot = Rotation.from_matrix(R_cw)
                euler_angles = rot.as_euler('xyz', degrees=True)
                camera_orientations.append(euler_angles)
            
            positions = np.array(camera_positions)
            orientations = np.array(camera_orientations)
            
            fig = plt.figure(figsize=(18, 10), num="Camera Motion Analysis")
            
            # Camera trajectory in 3D
            ax1 = fig.add_subplot(231, projection='3d')
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.8, linewidth=2)
            ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
            ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            ax1.set_zlabel('Z (mm)')
            ax1.set_title('Camera 3D Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Camera position vs time
            ax2 = fig.add_subplot(232)
            ax2.plot(times, positions[:, 0], 'r-', label='X', alpha=0.8)
            ax2.plot(times, positions[:, 1], 'g-', label='Y', alpha=0.8)
            ax2.plot(times, positions[:, 2], 'b-', label='Z', alpha=0.8)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position (mm)')
            ax2.set_title('Camera Position vs Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Camera orientation vs time
            ax3 = fig.add_subplot(233)
            ax3.plot(times, orientations[:, 0], 'r-', label='Roll (X)', alpha=0.8)
            ax3.plot(times, orientations[:, 1], 'g-', label='Pitch (Y)', alpha=0.8)
            ax3.plot(times, orientations[:, 2], 'b-', label='Yaw (Z)', alpha=0.8)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Orientation (degrees)')
            ax3.set_title('Camera Orientation vs Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Camera velocity
            ax4 = fig.add_subplot(234)
            if len(positions) > 1:
                dt = np.diff(times)
                velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
                ax4.plot(times[1:], velocity, 'purple', linewidth=2)
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('Velocity (mm/s)')
                ax4.set_title('Camera Linear Velocity')
                ax4.grid(True, alpha=0.3)
            
            # Camera angular velocity
            ax5 = fig.add_subplot(235)
            if len(orientations) > 1:
                dt = np.diff(times)
                angular_velocity = np.linalg.norm(np.diff(orientations, axis=0), axis=1) / dt
                ax5.plot(times[1:], angular_velocity, 'orange', linewidth=2)
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Angular Velocity (deg/s)')
                ax5.set_title('Camera Angular Velocity')
                ax5.grid(True, alpha=0.3)
            
            # Top-down view of trajectory
            ax6 = fig.add_subplot(236)
            ax6.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.8, linewidth=2)
            ax6.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
            ax6.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
            ax6.set_xlabel('X (mm)')
            ax6.set_ylabel('Y (mm)')
            ax6.set_title('Camera Trajectory (Top View)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_aspect('equal')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Camera motion plot saved: {save_path}")
            
            plt.show(block=False)  # Non-blocking show to allow multiple plots
            
            # Print camera motion statistics
            print(f"\n=== CAMERA MOTION SUMMARY ===")
            print(f"Total recording time: {times[-1] - times[0]:.2f} seconds")
            print(f"Camera position range:")
            print(f"  X: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f} mm (range: {positions[:, 0].max() - positions[:, 0].min():.1f} mm)")
            print(f"  Y: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f} mm (range: {positions[:, 1].max() - positions[:, 1].min():.1f} mm)")
            print(f"  Z: {positions[:, 2].min():.1f} to {positions[:, 2].max():.1f} mm (range: {positions[:, 2].max() - positions[:, 2].min():.1f} mm)")
            
            if len(positions) > 1:
                total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                avg_velocity = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times))
                print(f"Total distance traveled: {total_distance:.1f} mm")
                print(f"Average velocity: {avg_velocity:.1f} mm/s")
            
        except Exception as e:
            print(f"Error plotting camera motion: {e}")
    
    def run_full_pipeline(self, use_projections: bool = False, init_file_path: str = None,
        chosen_marker: str = None, perform_error_analysis: bool = False,
        extract_depth: bool = False, visualize_depth: bool = False, 
        save_frames: bool = False, frames_video: bool = False):
        
        """Run the complete pipeline."""
        print("Starting VICON-DVS pipeline...")
        
        # 1. Load all data
        self.load_event_data()
        self.load_vicon_data()
        self.load_calibration_data()
        
        # 2. Try to load existing calibration
        if init_file_path is None:
            # Generate sequence-specific init file name that avoids overwriting an already existing one
            init_file = self._generate_unique_init_file_path()
        else:
            init_file = init_file_path
            
        calibration_exists = self.load_existing_calibration(init_file)
        
        # 3. Compute world to system transforms
        self.compute_world_to_system_transforms()
        
        if not calibration_exists:
            print("No existing calibration found. Starting manual calibration...")
            
            # 4. Manual rotation estimation with marker filter
            tvec_init = np.array([0.0, 0.0, 0.0])       # TODO: make it user-input, as well as rvec_init, this can also be hard-coded for the specfic dataset
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
        
        # 9. Create projection video
        # Extract sequence name for consistent naming
        sequence_name = self._extract_sequence_name()
        base_video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_projection_video.mp4")   # TODO: change so that args gets as input the path to the directory
        video_file = self._generate_unique_video_path(base_video_path)      # TODO: change so that args gets as input the path to the directory

        # Store initial delay to check if it changed during projection
        initial_delay = self.delay

        # Run the projection session but DO NOT save yet — just collect buffers
        result = self.create_projection_video()  # now returns dict with segments & points
        collected_video_segments = result["segments"]
        all_projected_points = result["points"]

        # Check if delay was modified during projection
        delay_changed = abs(self.delay - initial_delay) > 1e-6            
            
        while True:
            response = input("\nAre you satisfied with the calibration results? (y/n): ").lower().strip()

            if response in ['y', 'yes']:
                # Save updated calibration if delay was changed during projection
                if delay_changed:
                    print(f"\n Delay was adjusted during projection: {initial_delay:.6f}s → {self.delay:.6f}s")
                    _, new_init_file = self._track_delay_change(initial_delay, init_file)
                    print(f"New calibration file: {os.path.basename(new_init_file)}")

                # Save points in CSV (if error check)
                if all_projected_points and perform_error_analysis:
                    print("Saving projected points CSV...")
                    self._save_projected_points_csv(all_projected_points, video_file)

                #TODO: check to avoid empty video files
                # Merge segments into video
                if collected_video_segments:
                    print("Merging video segments...")
                    try:
                        fps = int(1 / self.period)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            video_file, fourcc, fps,
                            (self.cam_res[1], self.cam_res[0]), isColor=False
                        )

                        total_frames = 0
                        for segment_frames in collected_video_segments:
                            for frame in segment_frames:
                                if frame.ndim == 3:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                video_writer.write(frame)
                                total_frames += 1

                        video_writer.release()
                        print(f"Projection video created: {video_file}")

                    except Exception as e:
                        print(f"Error merging video: {e}")

                # Optional: 2D–2D error analysis
                # TODO: remove try except blocks??
                if perform_error_analysis:
                    try:
                        self.calculate_projection_error(self.output_path, visualize=True)
                    except Exception as e:
                        print(f"Error during projection error analysis: {e}")
                        print("Pipeline completed with calibration but error analysis failed.")

                # Optional: Frame-by-frame GT vs projected visualization
                if save_frames:
                    try:
                        print("\n" + "="*60)
                        print("FRAME-BY-FRAME GT VS PROJECTED VISUALIZATION")
                        print("="*60)
                        print("Creating frame-by-frame visualization with delay compensation...")
                        
                        frames_dir = self.save_frames_with_gt_and_projections(
                            self.output_path, 
                            create_video=frames_video
                        )
                        print(f"Frame visualization completed! Check: {frames_dir}")
                        
                    except Exception as e:
                        print(f"Error during frame-by-frame visualization: {e}")
                        print("Pipeline completed but frame visualization failed.")

                # Step 10: Extract joint projections + Optional joint depths
                #TODO: remove try-except blocks???
                if extract_depth:
                    try:
                        print("\n" + "="*60)
                        print("AUTOMATIC JOINT PROJECTION")
                        print("="*60)
                        print("Generating joint projections and video using joint configuration...")

                        depth_csv_path = self._generate_joint_projections_and_depths(extract_depth=extract_depth)

                        # # Optional: depth visualization (only if depth was actually extracted)
                        # if visualize_depth:
                        #     print("\nCreating depth visualization...")
                        #     self.analyze_depth_accuracy(depth_csv_path)
                        #     self.visualize_depth_extraction(depth_csv_path)
                        #     print("Depth visualization completed")

                    except Exception as e:
                        print(f"Error during joint depth analysis: {e}")
                        print("Pipeline completed but joint depth analysis failed.")

                # Step 10.5: Create VICON marker projection video
                try:
                    print("\n" + "="*60)
                    print("VICON MARKER PROJECTION VIDEO")
                    print("="*60)
                    print("Creating VICON marker projection video...")
                    
                    vicon_video_path = self.create_vicon_marker_projection_video()
                    if vicon_video_path:
                        print(f"✓ VICON marker video saved: {os.path.basename(vicon_video_path)}")
                    else:
                        print("⚠ VICON marker video creation failed")
                        
                except Exception as e:
                    print(f"Error during VICON marker video creation: {e}")
                    print("Pipeline continued but VICON marker video creation failed.")

                # Step 11: Visualization plots
                print("\n" + "="*60)
                print("STEP 11: TRAJECTORY & MOTION VISUALIZATION")
                print("="*60)
                
                # Generate output directory for plots
                output_dir = self._get_output_directory()
                sequence_name = self._extract_sequence_name()
                
                try:
                    # Ask user if they want to create visualization plots
                    viz_response = input("\nWould you like to create visualization plots? (y/n): ").lower().strip()
                    
                    if viz_response in ['y', 'yes']:
                        print("Creating visualization plots...")
                        
                        # Ask user to select a marker for both 3D and 2D plots
                        available_markers = [label.strip() for label in self.c3d_data.point_labels]
                        print(f"\nAvailable markers: {available_markers[:10]}{'...' if len(available_markers) > 10 else ''}")
                        
                        while True:
                            marker_name = input("Enter marker name for 3D and 2D trajectory plots: ").strip()
                            if marker_name in available_markers:
                                break
                            print(f"Marker '{marker_name}' not found. Available markers: {available_markers}")
                        
                        # Create all plots simultaneously with the selected marker
                        print(f"\nGenerating all plots for marker: {marker_name}")
                        
                        # 3D marker trajectory plot
                        plot_3d_path = os.path.join(output_dir, f"{sequence_name}_3d_marker_trajectory.png")
                        print("--- Creating 3D Marker Trajectory Plot ---")
                        self.plot_3d_marker_trajectory(marker_name=marker_name, save_path=plot_3d_path)
                        
                        # 2D marker projection plot
                        plot_2d_path = os.path.join(output_dir, f"{sequence_name}_2d_marker_projections.png")
                        print("--- Creating 2D Marker Projection Plot ---")
                        self.plot_2d_marker_projections(marker_name=marker_name, save_path=plot_2d_path)
                        
                        # Camera motion plot
                        camera_plot_path = os.path.join(output_dir, f"{sequence_name}_camera_motion.png")
                        print("--- Creating Camera Motion Plot ---")
                        self.plot_camera_motion(save_path=camera_plot_path)
                        
                        print(f"\n✓ All visualization plots saved in: {output_dir}")
                        print(f"  • 3D marker trajectory: {os.path.basename(plot_3d_path)}")
                        print(f"  • 2D marker projections: {os.path.basename(plot_2d_path)}")
                        print(f"  • Camera motion: {os.path.basename(camera_plot_path)}")
                        
                        # Keep all plots open simultaneously
                        print(f"\n🎯 All three plots are now displayed simultaneously!")
                        print("📊 You can interact with each plot window independently:")
                        print("   • Zoom, pan, rotate (for 3D plots)")
                        print("   • Compare trajectories across different views")
                        print("   • Close individual windows when done")
                        
                        # Block until user closes all plot windows or presses enter
                        input("\nPress Enter to continue (plots will remain open)...")
                    
                    else:
                        print("Skipping visualization plots.")
                        
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    print("Pipeline completed but visualization failed.")

                # Step 12: Final summary
                print("\n" + "="*60)
                print("STEP 12: PIPELINE COMPLETION")
                print("="*60)
                print("Pipeline completed successfully!")
                print("All outputs saved to:", self.output_path)
                return

            elif response in ['n', 'no']:
                print("Discarding the results and starting over.")
                # You can loop back or exit depending on your pipeline design
                return

            else:
                print("Please answer 'y' or 'n'.")
            
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
                       help='Path to initialization file containing transformation matrix and delay, or path to save new one if not existing')
    parser.add_argument('--output_path', required=True,
                       help='REQUIRED: Output path for labeled points (YAML file). All outputs (videos, CSV files, init files) will be saved in the same directory.')
    parser.add_argument('--subject', default=None,      # TODO: needed only for hpe, maybe read the subject from the c3d file?
                       help='Subject name for labels (e.g., P1, P11), it is read from the .c3d file if not provided')
    parser.add_argument('--marker_list_path', default=None,
                       help='Path to a text or YAML file listing desired marker labels')    # TODO: could be hardcoded as the markers are always the same
    parser.add_argument('--chosen_marker', default=None,            # TODO: not really that usefult as it is read to be the first element of array of markers, but can be useful depending on c3d structure
                       help='Specific marker to use for rotation adjustment feedback')
    parser.add_argument('--list', action='store_true',
                       help='Use list-based labeling interface, N.B. there are two ways to do this, by apt installing tkinter or by using the terminal interface, default is tkinter, modify line 1085 in helpers.py to change between the two')
    parser.add_argument('--projections', action='store_true',
                       help='Use projection-based labeling interface')
    parser.add_argument('--visualize_events', action='store_true',
                       help='Visualize events')
    parser.add_argument('--error', action='store_true',
                       help='Perform 2D-2D error analysis between manual labels and projected markers (Step 11)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save frame-by-frame visualization showing ground truth and projected points with delay compensation')
    parser.add_argument('--frames_video', action='store_true',
                       help='Create video from frame-by-frame comparison (use with --save_frames)')
    parser.add_argument('--extract_depth', action='store_true',
                       help='Extract positions and depths from camera to joints and save to CSV file')
    parser.add_argument('--visualize_depth', action='store_true',
                       help='Create visualization plots of depth extraction results (requires --extract_depth)')
    
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
        marker_list_path=args.marker_list_path
    )
    
    # Visualize events if requested
    if args.visualize_events:
        pipeline.load_event_data()
        pipeline.load_calibration_data()
        pipeline.visualize_events()
        return
    
    # Visualization of depth and analysis require depth extraction
    if args.visualize_depth and not args.extract_depth:
        parser.error("--visualize_depth requires --extract_depth to be enabled")

    # Frame video requires frame saving
    if args.frames_video and not args.save_frames:
        parser.error("--frames_video requires --save_frames to be enabled")

    # Run full pipeline
    pipeline.run_full_pipeline(
        use_projections=args.projections,
        init_file_path=args.init_file,
        chosen_marker=args.chosen_marker,
        perform_error_analysis=args.error,
        extract_depth=args.extract_depth,
        visualize_depth=args.visualize_depth,
        save_frames=args.save_frames,
        frames_video=args.frames_video
    )
    

if __name__ == "__main__":
    main()