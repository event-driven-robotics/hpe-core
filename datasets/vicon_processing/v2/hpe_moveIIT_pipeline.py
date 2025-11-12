# Different version of the pipeline to work specifically on the static camera setup from MoveIIT dataset
# Please tell me if you need any specific changes or additions to this code

import sys
import os
import yaml
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import bisect
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

###
    def _detect_subject_from_c3d(self, all_markers: Optional[list] = None) -> Optional[str]:
        # Use provided markers or fall back to self.marker_names if available
        markers_to_check = all_markers or getattr(self, 'marker_names', [])
        subject_counts = {}
        
        for marker_name in markers_to_check:
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

    # ???
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

    def _find_cleaned_marker_match(self, joint_name, all_c3d_markers):
        """Find C3D marker that matches the joint name using cleaned label matching logic."""
        joint_name_clean = joint_name.strip()
        
        # Create candidate names to search for
        candidates = []
        
        # Add subject-prefixed version if subject exists
        if self.subject:
            candidates.append(f"{self.subject}:{joint_name_clean}")
        
        # Add plain joint name
        candidates.append(joint_name_clean)
        
        # Search through C3D markers using case-insensitive matching
        for marker in all_c3d_markers:
            marker_clean = marker.strip()
            
            # Check against all candidate names
            for candidate in candidates:
                if marker_clean.upper() == candidate.upper():
                    # print(f"  Matched '{joint_name}' -> '{marker_clean}' (exact match)")
                    return marker_clean
                
                # Check if C3D marker ends with the joint name (for subject prefixes)
                if ":" in marker_clean and marker_clean.upper().endswith(f":{joint_name_clean.upper()}"):
                    # print(f"  Matched '{joint_name}' -> '{marker_clean}' (subject prefix)")
                    return marker_clean
        
        # No match found
        print(f"  No match found for joint: '{joint_name}'")
        return None

    def _get_output_directory(self):
        """Get the output directory from the output_path."""
        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _extract_sequence_name(self):
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

    def _generate_unique_init_file_path(self, base_dir: str = None) -> str:
        """Generate a unique init file path using sequence name and avoiding overwrites."""
        if base_dir is None:
            base_dir = self._get_output_directory()
        
        # Get sequence name for the file
        sequence_name = self._extract_sequence_name()
        
        # Base filename with sequence name
        base_filename = f"{sequence_name}_init_file.txt"
        init_file_path = os.path.join(base_dir, base_filename)
        
        # If file doesn't exist, use it as is
        if not os.path.exists(init_file_path):
            print(f"Generated init file path: {init_file_path}")
            return init_file_path
        
        # If file exists, add iteration number
        i = 1
        while True:
            filename_with_iter = f"{sequence_name}_init_file_{i}.txt"
            init_file_path = os.path.join(base_dir, filename_with_iter)
            if not os.path.exists(init_file_path):
                print(f"Generated unique init file path: {init_file_path} (iteration {i})")
                return init_file_path
            i += 1

    def _track_delay_change(self, initial_delay: float, phase: str, init_file: str) -> bool:
        """Track and save delay changes during different pipeline phases."""
        delay_changed = abs(self.delay - initial_delay) > 1e-6
        if delay_changed:
            print(f" Delay adjusted during {phase}: {initial_delay:.6f}s → {self.delay:.6f}s")
            print(f" Updating calibration file: {os.path.basename(init_file)}")
            self.save_calibration(init_file)
            return True
        return False

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

    def prompt_camera_setup(self) -> tuple:
        """Ask user how many markers are attached to the camera and their names/prefixes."""
        # print("\nCamera setup configuration (manual input)")
        # print("Configure your camera marker setup:")
        # print("  • Single marker: Only translation known, rotation = identity initially")
        # print("  • Multi-marker (2+): Full pose estimation with rigid body")
       
        # # Get number of markers
        # while True:
        #     try:
        #         n = int(input("\nNumber of markers attached to the camera: "))
        #         if n < 1:
        #             print("Please enter a positive integer.")
        #             continue
        #         setup = "multi" if n >= 2 else "single"
        #         print(f"Selected camera setup: {setup} (from {n} markers)")
        #         break
        #     except ValueError:
        #         print("Invalid input. Please enter an integer equal to the amount of markers attached to the camera in question.")
       
        # Specific for marker of static camera
        setup = "single"
        n = 1
 
        # Show available markers for reference
        print(f"\nAvailable markers in C3D file ({len(self.marker_names)} total):")
        for i, marker in enumerate(self.marker_names):
            print(f"  {i+1:2d}. {marker}")
           
        # # ???
        # # Provide helpful examples based on actual marker names
        # print("\n Pattern matching examples:")
        # example_patterns = []
        # for marker in self.marker_names[:5]:  # Show examples from first few markers
        #     if ':' in marker:
        #         prefix = marker.split(':')[0] + ':'
        #         example_patterns.append(f"'{prefix}' (matches all markers starting with '{prefix}')")
        #     if any(word in marker.lower() for word in ['cam', 'camera', 'dvs']):
        #         for word in ['cam', 'camera', 'dvs']:
        #             if word in marker.lower():
        #                 example_patterns.append(f"'{word}' (matches markers containing '{word}')")
        #                 break
       
        # if example_patterns:
        #     for example in example_patterns[:3]:  # Show max 3 examples
        #         print(f"  • {example}")
        # else:
        #     print("  • 'cam' (matches markers containing 'cam')")
        #     print("  • 'camera:' (matches markers starting with 'camera:')")
        #     print("  • 'marker1' (exact match for 'marker1')")
       
        # # Get camera marker identification
        # print(f"\nNow specify how to identify your {n} camera markers:")
        # print("You can provide:")
        # print("  • Exact marker names (e.g., 'cam1', 'camera_front')")
        # print("  • Prefixes to match multiple markers (e.g., 'cam:', 'camera')")
        # print("  • Partial names that appear in marker names (e.g., 'cam' matches 'cam1', 'mycam')")
        # print("\nTip: Use ':' at the end for exact prefix matching (e.g., 'cam:' only matches markers starting with 'cam:')")
       
        camera_markers = []
        identified_markers = []
       
        # ???
        i = 0
        while i < n:
            print(f"\n--- Camera marker {i+1} of {n} ---")
            while True:
                marker_input = "*118"       # Specific marker for static camera in moveIIT
               
                # input(f"Enter name/pattern: ").strip()
                # if not marker_input:
                #     print("Please enter a marker name or prefix.")
                #     continue
               
                # Find matching markers
                matches = self._find_matching_markers(marker_input)
               
                if not matches:
                    print(f" No markers found matching '{marker_input}'")
                    remaining = [m for m in self.marker_names if m not in identified_markers]
                    if remaining:
                        print(f"Available markers: {remaining[:10]}{'...' if len(remaining) > 10 else ''}")
                    continue
               
                # ???
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
    
    # ???
    # TODO: check this function and fix it
    # function generated to read labels from specified file, will be changed
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
            print(f"Loading marker list from: {self.marker_list_path}")
            print(f"Available C3D markers: {available_markers_raw[:10]}{'...' if len(available_markers_raw) > 10 else ''}")
            requested = load_label_file(self.marker_list_path)
            print(f"Requested markers from YAML: {requested}")
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

            print(f"Requested {len(requested)} labels from YAML; matched {len(matched)}; missing {len(missing)}.")
            if missing:
                print(f"Missing markers (ignored): {missing}")

            if len(matched) > 0:
                print(f"Using {len(matched)} markers from {os.path.basename(self.marker_list_path)}: {matched}")
                return matched

        # Fallback
        print(f"Using all {len(available_markers_raw)} available markers from C3D file.")
        return available_markers_raw   
###
        
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

        self.start_time = 0.0 # self.imp.get_first_ts()
        self.end_time = self.imp.get_last_ts()

        print(f"\n Loaded event stream: '{middle_key}'")
        print(f"Events from {self.start_time:.3f}s to {self.end_time:.3f}s")

    # Check camera stuff
    def load_vicon_data(self):
        """Load VICON C3D data."""
        print("Loading VICON data...")
        self.c3d_data = c3d.Reader(open(self.vicon_path, 'rb'))
        for i, points, analog in self.c3d_data.read_frames():
            self.points_3d[i] = points
            
        self.marker_t = np.linspace(0.0, self.c3d_data.frame_count / self.c3d_data.point_rate, 
                                   self.c3d_data.frame_count, endpoint=False)
        
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
        
        # Configure camera setup and identify camera markers (using all available markers)
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
            print(" Warning: No camera markers identified! This may cause issues with transformation computation.")
        elif self.camera_setup == "multi" and len(self.camera_markers) < 3:
            print(f" Warning: Multi-marker setup specified but only {len(self.camera_markers)} markers found.")
            print(" This may affect pose estimation accuracy.")
        
        # Now filter markers based on marker_list_path if provided (after camera selection)
        if self.marker_list_path:
            filtered_markers = self.get_markers_names()
            print(f"Filtering to {len(filtered_markers)} markers from marker list file")
            print(f"Filtered markers: {filtered_markers}")
            self.marker_names = filtered_markers
        else:
            print(f"Using all {len(self.marker_names)} markers from C3D file")
        
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
        """Compute world to system transformation matrices using user-specified camera markers."""

        print("Computing world to system transformations...")
        from helpers import ViconHelper
        
        # Use the camera setup specification
        enable_camera_markers = (self.camera_setup in ["multi", "single"])
        
        # Create ViconHelper with user-specified camera markers
        vicon_helper = ViconHelper(
            self.marker_t, self.points_3d, self.delay, 
            self.c3d_data.frame_count, self.c3d_data.point_rate, 
            self.c3d_data.point_labels, enable_camera_markers, True,
            user_camera_markers=self.camera_markers  # Pass user-specified camera markers
        )
        
        self.Ts_world_to_system = vicon_helper.compute_camera_marker_transforms()
        print(f"Computed transformations using camera setup: {self.camera_setup}")
        
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
        window_size = 500 * self.period
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

###    
    # def plot_marker_positions(self, plot_type: str = 'all'):
    #     """Plot marker positions over time using helpers.marker_p function.
        
    #     Args:
    #         plot_type: Type of plot to generate ('markers', 'joints', 'cameras', 'all')
    #     """
    #     print(f"Plotting marker positions over time: {plot_type}")
        
    #     if not self.c3d_data:
    #         print("❌ Error: C3D data not loaded. Run load_c3d_data() first.")
    #         return
        
    #     # Use matplotlib tk backend for separate windows
    #     plt.matplotlib.use('TkAgg')
        
    #     # Load joint names and marker tags from config files
    #     joint_names = self._load_config_markers('joints')
    #     marker_tags = self._load_config_markers('tags')
        
    #     # Identify different marker types
    #     all_markers = self.marker_names
    #     joint_markers = []
    #     tag_markers = []
    #     camera_markers = []
        
    #     # Categorize markers based on config files
    #     for marker in all_markers:
    #         marker_clean = marker.split(':')[-1] if ':' in marker else marker  # Remove subject prefix like P9:
    #         marker_clean = marker_clean.strip()  # Remove any whitespace
            
    #         if marker_clean in joint_names:
    #             joint_markers.append(marker)
    #         elif marker_clean in marker_tags:
    #             tag_markers.append(marker)
    #         elif any(cam_word in marker.lower() for cam_word in ['cam', 'camera', 'dvs', 'lens']):
    #             camera_markers.append(marker)
    #         else:
    #             # If not found in config files, treat as tag marker by default
    #             tag_markers.append(marker)
        
    #     print(f"Marker categorization: {len(tag_markers)} tags, {len(joint_markers)} joints, {len(camera_markers)} cameras")
        
    #     # Create plots based on requested type
    #     if plot_type in ['markers', 'all']:
    #         if tag_markers:
    #             self._plot_markers_window(tag_markers, "Tag Markers")
    #         else:
    #             print("No tag markers found to plot.")
            
    #     if plot_type in ['joints', 'all']:
    #         if joint_markers:
    #             self._plot_markers_window(joint_markers, "Joint Markers")
    #         else:
    #             print("No joint markers found to plot.")
            
    #     if plot_type in ['cameras', 'all']:
    #         if camera_markers:
    #             self._plot_markers_window(camera_markers, "Camera Markers")
    #         else:
    #             print("No camera markers found to plot.")
        
    #     plt.show()
    
    # def _plot_markers_window(self, markers, window_title):
    #     """Plot markers in a separate window using helpers.marker_p function."""
    #     plt.figure(figsize=(12, 8))
        
    #     for marker in markers:
    #         try:
    #             # Use helpers.marker_p to extract marker data
    #             # Note: self.c3d_data is a c3d.Reader object, so we use dot notation
    #             marker_points = helpers.marker_p(self.c3d_data.point_labels, list(self.points_3d.values()), marker)
                
    #             # Plot X, Y, Z coordinates
    #             plt.plot(self.marker_t, marker_points[:, 0:3])
                
    #         except Exception as e:
    #             print(f"Could not plot marker {marker}: {e}")
    #             continue
        
    #     # Add proper labels and title
    #     plt.xlabel('Time (seconds)')
    #     plt.ylabel('Position (mm)')
    #     plt.title(f'{window_title} - 3D Position Over Time')
    #     plt.legend(['X', 'Y', 'Z'])
    #     plt.grid(True)
    
    # def _load_config_markers(self, config_type):
    #     """Load marker names from config files.
        
    #     Args:
    #         config_type: 'joints' or 'tags'
        
    #     Returns:
    #         List of marker names from the config file
    #     """
    #     try:
    #         # Get the absolute path to the config directory
    #         current_dir = os.path.dirname(os.path.abspath(__file__))
    #         config_dir = os.path.join(current_dir, '..', 'scripts', 'config')
            
    #         if config_type == 'joints':
    #             config_path = os.path.join(config_dir, 'labels_joints.yml')
    #         elif config_type == 'tags':
    #             config_path = os.path.join(config_dir, 'labels_tags.yml')
    #         else:
    #             print(f"❌ Unknown config type: {config_type}")
    #             return []
            
    #         # Load YAML file
    #         with open(config_path, 'r') as file:
    #             markers = yaml.safe_load(file)
                
    #         if markers is None:
    #             print(f"❌ Empty config file: {config_path}")
    #             return []
                
    #         print(f"✅ Loaded {len(markers)} {config_type} from {os.path.basename(config_path)}")
    #         return markers
            
    #     except FileNotFoundError:
    #         print(f"❌ Config file not found: {config_path}")
    #         return []
    #     except Exception as e:
    #         print(f"❌ Error loading config file: {e}")
    #         return []
###

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
        # rvec_init = np.zeros(3) 

        # Define known initial rotation (in degrees)
        roll, pitch, yaw = -85.0, 180.0, -80.0

        # Convert to rotation vector (Rodrigues form)
        rvec_init = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_rotvec()
                        
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
        window_size = 500 * self.period
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
                    visualize=True, marker_time_offset=window_start, delay_step=self.current_delay_step
                )
                
                print("e_ts final:", e_ts[-1], "window_start:", window_start, "window_size:", window_size)

                window_start = window_end
                
                print("window_start updated to:", window_start)
                
        except helpers.DelayExit as e:
            print("Delay adjustment stopped by user.")
            self.delay = e.delay
            self.current_delay_step = e.delay_step  # Preserve the delay step from user adjustment
            print(f"Final delay step from manual adjustment: {self.current_delay_step:.3f}s")

        finally:
            cv2.destroyAllWindows()
            print(f"Updated delay: {self.delay:.3f}s")
            return self.delay

    # Check the added stuff with the reading of the existing file
    def label_data_interactive(self, use_projections: bool = False) -> str:
        """Interactive data labeling using windowed approach."""

        print("Starting interactive labeling...")
        
        # Check if labels already exist
        user_choice = None
        existing_labels = None
        
        if os.path.exists(self.output_path):
            print(f"\n✓ Found existing label file: {self.output_path}")
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
                            print("✓ Using existing labels, skipping labeling process")
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
                print(f"⚠️  Warning: Could not read existing label file ({e}). Starting fresh labeling process.")
        
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
        
        window_size = 500 * self.period
        window_start = self.start_time
        
        # Create labeler instance
        labeler = DvsLabeler(img_shape=(self.cam_res[0], self.cam_res[1], 3), subject=self.subject)
        
        # Initialize with existing labels if continuing, or empty if starting fresh
        if user_choice == 'c' and existing_labels:
            try:
                if existing_labels and 'points' in existing_labels and 'times' in existing_labels:
                    labeler.points_dict = existing_labels
                    print(f"✓ Initialized with {len(existing_labels['points'])} existing labels")
                else:
                    labeler.points_dict = {'points': [], 'times': []}
            except Exception as e:
                print(f"⚠️  Could not load existing labels for continuation: {e}")
                labeler.points_dict = {'points': [], 'times': []}
        else:
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
                    
                    # Provide feedback on what was saved
                    total_labels = sum(len(p) for p in labeler.labeled_dict['points'])
                    total_frames = len(labeler.labeled_dict['points'])
                    
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


        return self.output_path
    
    def create_projection_video(self, output_video: str = 'projection_video.mp4'):
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
        print(f"Window size: {window_size/1000:.1f}s, Period: {self.period:.3f}s")

        window_count = 0

        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2
                window_count += 1

                print(f"\n--- Processing window {window_count} ---")
                print(f"Window range: {window_start:.3f}s to {window_end:.3f}s")

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
                        visualize=True, video_record=False,
                        marker_time_offset=window_start,
                        delay_step=self.current_delay_step
                    )
                except helpers.DelayReset as e:
                    # ---- FULL RESTART FROM THE FIRST EVER EVENT TIMESTAMP ----
                    print("Delay changed with arrow key: full reset requested.")
                    # 1) adopt the new delay and preserve delay_step
                    self.delay = e.new_delay
                    self.current_delay_step = e.delay_step  # Preserve the delay step
                    print(f"Preserved delay step: {self.current_delay_step:.3f}s")
                    # 2) clear all accumulators
                    all_projected_points.clear()
                    collected_video_segments.clear()
                    # 3) reset scanning window to the beginning
                    window_start = self.start_time   # or self.imp.first_event_time if you expose it
                    window_count = 0
                    # 4) close any windows and restart the loop
                    # cv2.destroyAllWindows()
                    continue

                # Update delay if it was adjusted during projection
                if current_delay != self.delay:
                    print(f"Delay updated during projection window {window_count}: {self.delay:.6f}s → {current_delay:.6f}s")
                self.delay = current_delay
                self.current_delay_step = current_delay_step

                # Collect frames for video
                if video_segment is not None:
                    collected_video_segments.append(video_segment)

                # Collect all projected points with event timestamps using synced data
                if synced_image_points:
                    # print(f" Window {window_count}: synced_image_points keys = {list(synced_image_points.keys())}")
                    for marker_name, marker_data in synced_image_points.items():
                        if marker_data and "points" in marker_data and "timestamps" in marker_data:
                            points = marker_data["points"]
                            timestamps = marker_data["timestamps"]
                            
                            # print(f" {marker_name}: {len(points)} points, {len(timestamps)} timestamps")
                            
                            # Ensure we have matching points and timestamps
                            n = min(len(points), len(timestamps))
                            points_added = 0
                            for i in range(n):
                                if len(points[i]) >= 2:  # Ensure we have x, y coordinates
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

        # DO NOT SAVE HERE — return buffers to caller
        # Optional: quick summary for debugging
        if all_projected_points:
            marker_counts = {}
            for entry in all_projected_points:
                marker_name = entry.get('marker', 'UNKNOWN')
                marker_counts[marker_name] = marker_counts.get(marker_name, 0) + 1
            print(f"Projected points (preview): total={len(all_projected_points)}  breakdown={marker_counts}")
        else:
            print("No projected points collected in this preview session.")

        if collected_video_segments:
            total_frames = sum(len(seg) for seg in collected_video_segments)
            print(f"Collected video frames (preview): {total_frames} (across {len(collected_video_segments)} segments)")
        else:
            print("No video segments collected in this preview session.")

        return {
            "segments": collected_video_segments,
            "points": all_projected_points,
        }

    # def _save_projected_points_txt(self, all_projected_points, output_video):
    #     """
    #     Save projected points to TXT in format:
    #     event_timestamp, x, y, marker_name
    #     """
    #     txt_path = os.path.join(os.path.dirname(output_video), "projected_points.txt")

    #     if not all_projected_points:
    #         print(" No projected points to save in TXT.")
    #         return

    #     # Sort by timestamp for consistency
    #     all_projected_points.sort(key=lambda d: d['timestamp'])

    #     with open(txt_path, "w") as f:
    #         f.write("# Projected marker points\n")
    #         f.write("# Format: event_timestamp, x, y, marker_name\n")
    #         f.write("#\n")
    #         for entry in all_projected_points:
    #             f.write(f"{entry['timestamp']:.6f}, {entry['x']:.2f}, {entry['y']:.2f}, {entry['marker']}\n")

    #     print(f" Saved projected points TXT: {txt_path} ({len(all_projected_points)} points)")    

    def _save_projected_points_csv(self, all_projected_points, output_video):
        """
        Save projected points to CSV in format:
        event_timestamp, marker1_x, marker1_y, marker2_x, marker2_y, ...
        
        Note: Markers are saved in the same order as defined in the YAML file (if provided)
        or in the order they appear in self.marker_names to maintain consistency.
        """
        csv_path = os.path.join(os.path.dirname(output_video), "projected_points.csv")
        if os.path.exists(csv_path):
            i = 0
            while os.path.exists(csv_path):
                csv_path = os.path.join(os.path.dirname(output_video), "projected_points_" + str(i) + ".csv")
                i += 1

        if not all_projected_points:
            print(" No projected points to save in CSV.")
            return

        # Sort by timestamp for consistent order
        all_projected_points.sort(key=lambda d: d['timestamp'])

        # Use self.marker_names to preserve the order from YAML file (if provided)
        # instead of sorting alphabetically
        projected_markers = set(p['marker'] for p in all_projected_points)
        all_markers = [m for m in self.marker_names if m in projected_markers]
        timestamps = sorted(set(p['timestamp'] for p in all_projected_points))

        # Build {timestamp: {marker: (x, y)}}
        frame_dict = {t: {} for t in timestamps}
        for p in all_projected_points:
            frame_dict[p['timestamp']][p['marker']] = (p['x'], p['y'])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
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

        print(f" Saved projected points CSV: {csv_path} ({len(timestamps)} timestamps, {len(all_markers)} markers)")
        print(f" Marker order matches YAML definition: {[m for m in self.marker_names[:5]]}{'...' if len(self.marker_names) > 5 else ''}")

    def _save_joint_projections_and_video(self):
        """Generate projections and video specifically for joint markers defined in joint config file."""
        import yaml
        
        JOINT_CONFIG_PATH = os.path.join(CURRENT_DIR, "../scripts/config/labels_joints.yml")
 
        if JOINT_CONFIG_PATH not in sys.path:
            sys.path.append(JOINT_CONFIG_PATH)
        
        # Load joint labels from config file
        if not os.path.exists(JOINT_CONFIG_PATH):
            raise FileNotFoundError(f"Joint config file not found: {JOINT_CONFIG_PATH}")

        with open(JOINT_CONFIG_PATH, 'r') as f:
            joint_labels = yaml.safe_load(f)
            
        if not isinstance(joint_labels, list):
            raise ValueError(f"Expected list of joint labels in {JOINT_CONFIG_PATH}, got {type(joint_labels)}")

        print(f"Loaded {len(joint_labels)} joint labels from config: {joint_labels}")
        
        # Get all available C3D markers with cleaned names
        all_c3d_markers = [name.strip() for name in self.c3d_data.point_labels]
        
        # Filter available markers using cleaned label matching logic
        available_joints = []
        missing_joints = []
        
        for joint in joint_labels:
            found_marker = self._find_cleaned_marker_match(joint, all_c3d_markers)
            if found_marker:
                available_joints.append(found_marker)
            else:
                missing_joints.append(joint)
        
        if missing_joints:
            print(f"Warning: {len(missing_joints)} joints not found in C3D data: {missing_joints}")
        
        if not available_joints:
            raise RuntimeError("No joint markers found in C3D data matching the joint config")
            
        print(f"Found {len(available_joints)} matching joint markers: {available_joints}")
        
        # Create projector specifically for joints
        joint_projector = helpers.ViconProjector(
            available_joints, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )
        
        # Extract sequence name for file naming
        sequence_name = self._extract_sequence_name()
        
        # Generate joint projections with video
        base_joint_video = os.path.join(self._get_output_directory(), f"{sequence_name}_joint_projections.mp4")
        joint_video_file = self._generate_unique_video_path(base_joint_video)
        
        joint_csv_file = os.path.join(self._get_output_directory(), f"{sequence_name}_joint_projections.csv")
        
        print(f"Generating joint projections...")
        print(f"Joint video output: {joint_video_file}")
        print(f"Joint CSV output: {joint_csv_file}")
        
        # Collect joint projection data
        collected_joint_segments = []
        all_joint_points = []
        window_size = 500 * self.period
        window_start = self.start_time
        window_count = 0
        
        try:
            while window_start < self.end_time:
                window_end = window_start + window_size
                window_center = (window_start + window_end) / 2
                window_count += 1
                
                print(f"Processing joint window {window_count}: {window_start:.3f}s to {window_end:.3f}s")
                
                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])
                
                if len(e_ts) == 0:
                    print(f"No events in joint window {window_count}, skipping...")
                    window_start = window_end
                    continue
                
                # Project joints for this window
                try:
                    synced_joint_points, joint_video_segment, current_delay, current_delay_step = joint_projector.project_vicon_to_event_plane_dynamic(
                        self.marker_t, self.delay,
                        e_ts, e_us, e_vs, self.period,
                        visualize=False, video_record=True,
                        marker_time_offset=window_start,
                        delay_step=self.current_delay_step
                    )
                    
                    # Update delay if it was adjusted during projection
                    if current_delay != self.delay:
                        print(f"Delay updated during joint projection window {window_count}: {self.delay:.6f}s → {current_delay:.6f}s")
                        self.delay = current_delay
                        self.current_delay_step = current_delay_step  # Update instance variable
                    
                    # Debug: Show delay_step is preserved between joint windows
                    if window_count > 0:  # Don't show for first window
                        print(f"Joint Window {window_count}: Using delay step {self.current_delay_step:.3f}s (preserved from previous window)")
                        
                except helpers.DelayExit as e:
                    print(f"Delay adjustment detected during joint projection, using delay: {e.delay}")
                    self.delay = e.delay
                    self.current_delay_step = e.delay_step  # Preserve the delay step from user adjustment
                    print(f"Preserved delay step from joint projection: {self.current_delay_step:.3f}s")
                    break  # Exit the window loop if user exits delay adjustment
                except helpers.DelayReset as e:
                    print("Delay reset requested during joint projection: restarting from beginning")
                    self.delay = e.new_delay
                    self.current_delay_step = e.delay_step  # Preserve the delay step
                    print(f"Preserved delay step: {self.current_delay_step:.3f}s")
                    # Reset joint projection from the beginning
                    collected_joint_segments.clear()
                    all_joint_points.clear()
                    window_start = self.start_time
                    window_count = 0
                    continue
                
                # Collect video frames
                if joint_video_segment is not None:
                    collected_joint_segments.append(joint_video_segment)
                
                # Collect point data
                if synced_joint_points:
                    for joint_name, joint_data in synced_joint_points.items():
                        if joint_data and "points" in joint_data and "timestamps" in joint_data:
                            points = joint_data["points"]
                            timestamps = joint_data["timestamps"]
                            
                            n = min(len(points), len(timestamps))
                            for i in range(n):
                                point = points[i]
                                timestamp = timestamps[i]
                                if len(point) >= 2 and np.isfinite(point[0]) and np.isfinite(point[1]):
                                    all_joint_points.append({
                                        'timestamp': timestamp,
                                        'x': point[0],
                                        'y': point[1],
                                        'marker': joint_name
                                    })
                
                window_start = window_end
                
        except KeyboardInterrupt:
            print("Joint projection generation interrupted by user")
        
        # Save joint CSV
        if all_joint_points:
            print(f"Saving {len(all_joint_points)} joint points to CSV...")
            self._save_joint_points_csv(all_joint_points, joint_csv_file, available_joints)
        else:
            print("No joint points collected")
        
        # Save joint video
        if collected_joint_segments:
            print(f"Merging {len(collected_joint_segments)} joint video segments...")
            try:
                fps = int(1 / self.period)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                joint_video_writer = cv2.VideoWriter(
                    joint_video_file, fourcc, fps,
                    (self.cam_res[1], self.cam_res[0]), isColor=False
                )
                
                total_joint_frames = 0
                for segment_frames in collected_joint_segments:
                    for frame in segment_frames:
                        if frame.ndim == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        joint_video_writer.write(frame)
                        total_joint_frames += 1
                
                joint_video_writer.release()
                print(f"Joint projection video created: {joint_video_file} ({total_joint_frames} frames)")
            except Exception as e:
                print(f"Error creating joint video: {e}")
        else:
            print("No joint video segments collected")
        
        print("Joint projection generation completed!")

    def _save_joint_points_csv(self, all_joint_points, csv_path, joint_markers):
        """Save joint points to CSV with consistent marker ordering."""
        if os.path.exists(csv_path):
            i = 0
            while os.path.exists(csv_path):
                base_path = csv_path.replace('.csv', '')
                csv_path = f"{base_path}_{i}.csv"
                i += 1
        
        if not all_joint_points:
            print("No joint points to save in CSV.")
            return
        
        # Sort by timestamp for consistent order
        all_joint_points.sort(key=lambda d: d['timestamp'])
        
        # Use joint_markers order for consistent column ordering
        projected_joints = set(p['marker'] for p in all_joint_points)
        ordered_joints = [m for m in joint_markers if m in projected_joints]
        timestamps = sorted(set(p['timestamp'] for p in all_joint_points))
        
        # Build {timestamp: {marker: (x, y)}}
        frame_dict = {t: {} for t in timestamps}
        for p in all_joint_points:
            frame_dict[p['timestamp']][p['marker']] = (p['x'], p['y'])
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # header = ["event_timestamp"]
            # for m in ordered_joints:
            #     header.extend([f"{m}_x", f"{m}_y"])
            # writer.writerow(header)
            
            for t in timestamps:
                row = [f"{t:.6f}"]
                for m in ordered_joints:
                    if m in frame_dict[t]:
                        x, y = frame_dict[t][m]
                        row.extend([f"{x:.2f}", f"{y:.2f}"])
                    else:
                        row.extend(["", ""])
                writer.writerow(row)
        
        print(f"Saved joint points CSV: {csv_path} ({len(timestamps)} timestamps, {len(ordered_joints)} joints)")
        print(f"Joint order: {ordered_joints}")
###

###
    def plot_per_marker_error_boxplot(self, marker_errors: dict, save_path: str = None):
        """Display per-marker error distribution with stats (mean, std, min, max).
        
        Args:
            marker_errors: Dictionary mapping marker names to error lists
            save_path: Optional path to save the plot image (e.g., 'error_analysis.png')
        """
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
        import csv
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_save_path = os.path.join(self._get_output_directory(), f"error_analysis.png")
        self.plot_per_marker_error_boxplot(comparison_results['marker_errors'], save_path=plot_save_path)

        return comparison_results

###
    def optimize_calibration(self, labels_path: str,
                         ransac_reproj_err: float = 3.0,
                         ransac_iters: int = 300,
                         min_points: int = 6):
        """
        Optimize a single, fixed T_sys->cam using multi-frame labels:
        world -> system (per frame, known) -> camera (unknown, fixed).
        Robust to length mismatches and OpenCV PnP overload differences.
        """
        import numpy as np, cv2
        from helpers import ViconHelper

        print("Optimizing calibration with OpenCV (global multi-frame PnP + refine)...")

        # 1) Load labels + interpolate 3D
        labeled_points = helpers.read_points_labels(labels_path)
        vicon_helper = ViconHelper(
            self.marker_t, self.points_3d, self.delay,
            self.c3d_data.frame_count, self.c3d_data.point_rate,
            self.c3d_data.point_labels, True, True,
            user_camera_markers=getattr(self, "camera_markers", None)
        )
        vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)

        # 2) Build global correspondences in SYSTEM frame
        system_points, image_points = [], []

        n_vp = len(vicon_points.get('points', []))
        n_lp = len(labeled_points.get('points', []))
        n_fi = len(vicon_points.get('frame_ids', []))
        n_frames = min(n_vp, n_lp, n_fi)

        if n_frames == 0:
            raise RuntimeError("No overlapping labeled frames and VICON points.")

        if (n_vp != n_lp) or (n_vp != n_fi):
            print(f"[warn] Length mismatch: vicon_points.points={n_vp}, "
                f"labeled_points.points={n_lp}, frame_ids={n_fi}. "
                f"Using first {n_frames} aligned entries.")

        valid_pairs = 0
        skipped_frames = 0

        for idx in range(n_frames):
            if idx >= len(vicon_points['frame_ids']):
                skipped_frames += 1
                continue
            frame_id = int(vicon_points['frame_ids'][idx])
            if not (0 <= frame_id < len(self.Ts_world_to_system)):
                skipped_frames += 1
                continue

            w_frame = vicon_points['points'][idx] or {}
            d_frame = labeled_points['points'][idx] or {}
            if not w_frame or not d_frame:
                skipped_frames += 1
                continue

            T_w2s = self.Ts_world_to_system[frame_id]  # 4x4

            for label, px in d_frame.items():
                if label not in w_frame:
                    continue
                w = np.asarray(w_frame[label], dtype=np.float64)
                if w is None or np.any(~np.isfinite(w)):
                    continue

                p_sys = (T_w2s @ np.append(w, 1.0))[:3]
                if np.any(~np.isfinite(p_sys)):
                    continue

                u = float(px['x']); v = float(px['y'])
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue

                system_points.append(p_sys)
                image_points.append([u, v])
                valid_pairs += 1

        print(f"Collected correspondences: {valid_pairs} (skipped frames: {skipped_frames})")
        if valid_pairs < max(4, min_points):
            raise RuntimeError("Not enough valid correspondences to run PnP.")

        # 3) Prepare arrays for OpenCV
        obj_pts = np.ascontiguousarray(np.asarray(system_points), dtype=np.float32).reshape(-1, 3)
        img_pts = np.ascontiguousarray(np.asarray(image_points),  dtype=np.float32).reshape(-1, 2)
        K_cv    = np.ascontiguousarray(self.K, dtype=np.float32)

        if self.D is None:
            D_cv = None
        else:
            D_flat = np.asarray(self.D, dtype=np.float32).ravel()
            # OpenCV accepts (k,) just fine; avoid forcing (k,1) on stricter builds
            D_cv = None if D_flat.size == 0 else np.ascontiguousarray(D_flat)

        # 4) Initial pose with RANSAC (try legacy 8-arg first)
        rvec = None; tvec = None; inliers = None
        try:
            # Legacy signature (8 args): (obj, img, K, D, iterationsCount, reprojErr, confidence, flags)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, K_cv, D_cv,
                int(ransac_iters), float(ransac_reproj_err), 0.99, cv2.SOLVEPNP_EPNP
            )
        except cv2.error:
            try:
                # Newer signature with useExtrinsicGuess (9 args): add False after D
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K_cv, D_cv,
                    False, int(ransac_iters), float(ransac_reproj_err), 0.99, cv2.SOLVEPNP_EPNP
                )
            except cv2.error:
                # Last resort: keyword flags only (some builds insist on this)
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K_cv, D_cv, flags=cv2.SOLVEPNP_EPNP
                )

        if not bool(retval) or inliers is None or len(inliers) < 4:
            print("[warn] RANSAC failed or too few inliers; trying iterative PnP...")
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_cv, D_cv, flags=cv2.SOLVEPNP_ITERATIVE)
            if not bool(ok):
                raise RuntimeError("PnP failed to produce an initial estimate.")
            inlier_idx = np.arange(len(obj_pts))
        else:
            inlier_idx = inliers.ravel()

        print(f"Initial inliers: {len(inlier_idx)} / {len(obj_pts)}")

        # 5) Refine with LM on inliers
        obj_in = obj_pts[inlier_idx]
        img_in = img_pts[inlier_idx]
        rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K_cv, D_cv, rvec, tvec)

        # 6) Build T_sys->cam
        R_sc, _ = cv2.Rodrigues(rvec)
        T_sc = np.eye(4, dtype=np.float64)
        T_sc[:3, :3] = R_sc.astype(np.float64)
        T_sc[:3, 3]  = tvec.reshape(3).astype(np.float64)

        # 7) Evaluate & store
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_cv, D_cv)
        proj = proj.reshape(-1, 2)
        errs = np.linalg.norm(proj - img_pts, axis=1)
        print(f"Reproj error: mean={errs.mean():.3f}px, median={np.median(errs):.3f}px, 95%={np.percentile(errs,95):.3f}px")

        self.T_syst_to_camera_opt = T_sc
        print("Optimized T_sys->cam:\n", T_sc)
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

    # def save_transforms_txt(self, output_file: str):
    #     """Save all transformation matrices to a text file."""
    #     with open(output_file, 'w') as f:
    #         f.write("# Transformation matrices from world to system coordinates\n")
    #         f.write("# Format: timestamp: array([4x4 transformation matrix])\n\n")
            
    #         for timestamp, transform_matrix in zip(self.marker_t, self.Ts_world_to_system):
    #             f.write(f"{timestamp:.6f}: array({transform_matrix.tolist()})\n")
                
    #     print(f"Saved transformation matrices to {output_file}")
        
    def run_full_pipeline(self, use_projections: bool = False, create_video: bool = True,
        init_file_path: str = None, chosen_marker: str = None, perform_error_analysis: bool = False):
        
        """Run the complete pipeline."""
        print("Starting VICON-DVS pipeline...")
        
        # 1. Load all data
        self.load_event_data()
        self.load_vicon_data()
        self.load_calibration_data()
        
        # 2. Try to load existing calibration
        if init_file_path is None:
            # Generate sequence-specific init file name that avoids overwrites
            init_file = self._generate_unique_init_file_path()
        else:
            init_file = init_file_path
            
        calibration_exists = self.load_existing_calibration(init_file)
        
        # 3. Compute world to system transforms
        self.compute_world_to_system_transforms()
        
        if not calibration_exists:
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
            # Extract sequence name for consistent naming
            sequence_name = self._extract_sequence_name()
            base_video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_projection_video.mp4")
            video_file = self._generate_unique_video_path(base_video_path)

            # Store initial delay to check if it changed during projection
            initial_delay = self.delay

            # Run the projection session but DO NOT save yet — just collect buffers
            result = self.create_projection_video(video_file)  # now returns dict with segments & points
            collected_video_segments = result["segments"]
            all_projected_points = result["points"]

            # Check if delay was modified during projection
            delay_changed = abs(self.delay - initial_delay) > 1e-6
            if delay_changed:
                print(f"\n Delay was adjusted during projection: {initial_delay:.6f}s → {self.delay:.6f}s")

            print("\n" + "="*60)
            print("CALIBRATION RESULTS REVIEW")
            print("="*60)
            print("A preview session has finished.")
            print("Please review the on-screen projection during the run you just did.")
            print("Nothing has been saved yet; we'll only save if you confirm.")
            
            # # Projected points are automatically saved during video creation
            # projected_points_file = os.path.join(os.path.dirname(self.vicon_path), "projected_points.txt")
            # print(f"Projected points saved to: {projected_points_file}")
                
            while True:
                response = input("\nAre you satisfied with the calibration results? (y/n): ").lower().strip()

                if response in ['y', 'yes']:
                    # Save updated calibration if delay was changed during projection
                    if delay_changed:
                        self._track_delay_change(initial_delay, "projection video creation", init_file)

                    # Save points CSV (if any)
                    if all_projected_points:
                        print("Saving projected points CSV...")
                        self._save_projected_points_csv(all_projected_points, video_file)
                    else:
                        print("No projected points to save.")

                    # Merge segments into video (if any)
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
                            print(f"Projection video created: {video_file} ({total_frames} frames)")
                        except Exception as e:
                            print(f"Error merging video: {e}")
                    else:
                        print("No video segments were created, so no video file will be saved.")

                    # Automatically save joint locations using joint config
                    try:
                        print("\n" + "="*60)
                        print("AUTOMATIC JOINT PROJECTION")
                        print("="*60)
                        print("Generating joint projections and video using joint configuration...")
                        
                        # Store delay before joint projection to check for changes
                        joint_initial_delay = self.delay
                        self._save_joint_projections_and_video()
                        
                        # Check if delay was changed during joint projection and save if needed
                        self._track_delay_change(joint_initial_delay, "joint projection", init_file)
                            
                    except Exception as e:
                        print(f"Error during joint projection generation: {e}")
                        print("Main pipeline completed but joint projection failed.")

                    # Optional: error analysis after saving
                    if perform_error_analysis:
                        labels_path = self.output_path
                        try:
                            self.calculate_projection_error(labels_path, visualize=True)
                        except Exception as e:
                            print(f"Error during projection error analysis: {e}")
                            print("Pipeline completed with calibration but error analysis failed.")

                    print("Pipeline completed successfully!")
                    return

                elif response in ['n', 'no']:
                    print("Okay — discarding this run (no video, no CSV saved).")
                    # You can loop back or exit depending on your pipeline design
                    return

                else:
                    print("Please answer 'y' or 'n'.")
        else:
            # If no video creation, still generate joint projections and track delay changes
            try:
                print("\n" + "="*60)
                print("AUTOMATIC JOINT PROJECTION")
                print("="*60)
                print("Generating joint projections using joint configuration...")
                
                # Store delay before joint projection to check for changes
                joint_initial_delay = self.delay
                self._save_joint_projections_and_video()
                
                # Check if delay was changed during joint projection and save if needed
                self._track_delay_change(joint_initial_delay, "joint projection", init_file)
                    
            except Exception as e:
                print(f"Error during joint projection generation: {e}")
                print("Pipeline proceeding without joint projections.")
            
            # Step 11: Perform error analysis if requested
            if perform_error_analysis:
                labels_path = self.output_path
                try:
                    self.calculate_projection_error(labels_path, visualize=True)
                except Exception as e:
                    print(f"Error during projection error analysis: {e}")
                    print("Pipeline completed with calibration but error analysis failed.")
            
            print("Pipeline completed successfully!")
            return
        
    # Step 11 implemented: 2D-2D projection error analysis available via --error flag
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
                       help='Path to initialization file containing transformation matrix and delay, or path to save new one if not existing')
    parser.add_argument('--output_path', required=True,
                       help='REQUIRED: Output path for labeled points (YAML file). All outputs (videos, CSV files, init files) will be saved in the same directory.')
    parser.add_argument('--subject', default=None,      # TODO: needed only for hpe, maybe read the subject from the c3d file?
                       help='Subject name for labels (e.g., P1, P11), it is read from the .c3d file if not provided')
    parser.add_argument('--marker_list_path', default=None,
                       help='Path to a text or YAML file listing desired marker labels')
    parser.add_argument('--chosen_marker', default=None,            # TODO: not really that usefult as it is read to be the first element of array of markers, but can be useful depending on c3d structure
                       help='Specific marker to use for rotation adjustment feedback')
    parser.add_argument('--list', action='store_true',
                       help='Use list-based labeling interface, N.B. there are two ways to do this, by apt installing tkinter or by using the terminal interface, default is tkinter, modify line 1085 in helpers.py to change between the two')
    parser.add_argument('--projections', action='store_true',
                       help='Use projection-based labeling interface')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video creation')
    parser.add_argument('--visualize_events', action='store_true',
                       help='Visualize events')
    parser.add_argument('--error', action='store_true',
                       help='Perform 2D-2D error analysis between manual labels and projected markers (Step 11)')
    # parser.add_argument('--plot_markers', choices=['markers', 'joints', 'cameras', 'all'], default=None,
    #                    help='Plot marker positions over time. Options: markers (with tags), joints (all joints), cameras (camera markers only), all (all three plots)')
    
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
    
    # # Plot marker positions if requested
    # if args.plot_markers:
    #     pipeline.load_vicon_data()
    #     pipeline.plot_marker_positions(plot_type=args.plot_markers)
    #     return
    
    # Run full pipeline
    pipeline.run_full_pipeline(
        use_projections=args.projections,
        create_video=not args.no_video,
        init_file_path=args.init_file,
        chosen_marker=args.chosen_marker,
        perform_error_analysis=args.error
    )
    

if __name__ == "__main__":
    main()