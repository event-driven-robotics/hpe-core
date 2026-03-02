import sys
import os
import yaml
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import c3d
from typing import List, Optional
import argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# TODO: add undistortion to the things maybe??????

# Get the absolute path to the current file (pipeline.py)
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
                 subject: Optional[str], output_path: str, marker_list_path: Optional[str] = None):
        self.dvs_path = dvs_path
        self.vicon_path = vicon_path
        self.intrinsic_path = intrinsic_path
        self.subject = subject                  # actually changed to remove this parameter, TODO: check usage
        self.output_path = output_path          # path for yaml file, will be removed after method works
        self.camera_setup = "multiple"          # DEBUG
        self.marker_list_path = marker_list_path
        self.period = 1.0 / 100                 # VICON frequency: 100Hz
        self.window_size = 100 * self.period
        
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

    def _get_output_directory(self):
        """Get the output directory from the output_path."""
        # Handle case where output_path is already a directory
        if os.path.isdir(self.output_path):
            output_dir = os.path.abspath(self.output_path)
        else:
            output_dir = os.path.dirname(os.path.abspath(self.output_path))
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

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

    def _track_delay_change(self, initial_delay: float, phase: str, init_file: str) -> tuple[bool, str]:
        """Track and save delay changes during different pipeline phases."""
        delay_changed = abs(self.delay - initial_delay) > 1e-6
        updated_file_path = init_file  # Default to original file
        
        if delay_changed:
            print(f" Delay adjusted during {phase}: {initial_delay:.6f}s → {self.delay:.6f}s")
            # Save to sequence-specific file instead of overwriting original
            updated_file_path = self.save_sequence_specific_calibration()
            print(f" Created sequence-specific calibration (preserving original: {os.path.basename(init_file)})")
            return True, updated_file_path
        return False, updated_file_path

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
            for candidate in candidates:
                if marker_clean.upper() == candidate.upper():
                    print(f"  Found match: '{joint_name}' → '{marker_clean}'")
                    return marker_clean
        
        # No match found
        print(f"  No match found for joint: '{joint_name}'")
        return None

    def save_sequence_specific_calibration(self) -> str:
        """Save transformation matrix and delay to a sequence-specific file."""
        # Generate sequence-specific calibration file path
        calibration_file = self._generate_unique_init_file_path()
        
        # Use existing save_calibration method
        self.save_calibration(calibration_file)
        
        print(f"✓ Saved sequence-specific calibration: {os.path.basename(calibration_file)}")
        return calibration_file
    
    def save_calibration_for_sequence(self, sequence_name: str = None) -> str:
        """Save calibration with a custom sequence name."""
        if sequence_name:
            # Temporarily override the sequence name extraction
            original_extract_method = self._extract_sequence_name
            self._extract_sequence_name = lambda: sequence_name
            
        try:
            return self.save_sequence_specific_calibration()
        finally:
            if sequence_name:
                # Restore the original method
                self._extract_sequence_name = original_extract_method

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
        
        # Get camera marker identification
        print(f"\nNow specify how to identify your {n} camera markers:")
        print("You can provide:")
        print("  • Exact marker names (e.g., 'cam1', 'camera_front')")
        print("  • Prefixes to match multiple markers (e.g., 'cam:', 'camera')")
        print("  • Partial names that appear in marker names (e.g., 'cam' matches 'cam1', 'mycam')")
        print("\nTip: Use ':' at the end for exact prefix matching (e.g., 'cam:' only matches markers starting with 'cam:')")
        
        camera_markers = []
        identified_markers = []
        
        # ???
        i = 0
        while i < n:
            print(f"\n--- Camera marker {i+1} of {n} ---")
            while True:
                marker_input = input(f"Enter name/pattern: ").strip()
                if not marker_input:
                    print("Please enter a marker name or prefix.")
                    continue
                
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
        """Return list of markers to use based on optional user label file handling prefixes"""
        
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
        """Load event data efficiently using importAe, with interactive stream selection as sometimes there are multiple saved inside?? (depends on the folder structure)."""
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

        # Use actual event timestamps, not hardcoded 0.0
        self.start_time = self.imp.get_first_ts()
        self.end_time = self.imp.get_last_ts()

        print(f"\n Loaded event stream: '{middle_key}'")
        print(f"Events from {self.start_time:.3f}s to {self.end_time:.3f}s")

    # TODO: Check camera stuff
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
        
        try:
            # First, let's try to understand the file format
            with open(self.intrinsic_path, 'r') as f:
                lines = f.readlines()
            
            print(f"Calibration file has {len(lines)} lines")
            print("First few lines:")
            for i, line in enumerate(lines[:10]):
                print(f"Line {i+1}: {repr(line.strip())}")
            
            # Try different parsing approaches
            if self.intrinsic_path.endswith('.yaml') or self.intrinsic_path.endswith('.yml'):
                # YAML format
                with open(self.intrinsic_path, 'r') as f:
                    calib_dict = yaml.safe_load(f)
            else:
                # Try to parse as key-value pairs, handling various formats
                calib_dict = {}
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('%'):
                        continue
                    
                    # Try different separators
                    if '=' in line:
                        key, value = line.split('=', 1)
                        calib_dict[key.strip()] = float(value.strip())
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        calib_dict[key.strip()] = float(value.strip())
                    elif len(line.split()) == 2:
                        parts = line.split()
                        calib_dict[parts[0]] = float(parts[1])
                    else:
                        print(f"Warning: Could not parse line {line_num}: {line}")
        
        except Exception as e:
            print(f"Error reading calibration file: {e}")
            print(f"File path: {self.intrinsic_path}")
            # Try the original approach as fallback
            try:
                calib = np.genfromtxt(self.intrinsic_path, delimiter=" ", skip_header=1, dtype=object)
                calib_dict = {key: value for key, value in zip(calib[:, 0].astype(str), calib[:, 1].astype(float))}
            except Exception as e2:
                print(f"Fallback parsing also failed: {e2}")
                raise
        
        print("Parsed calibration parameters:")
        for key, value in calib_dict.items():
            print(f"  {key}: {value}")
        
        # Extract parameters
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
        
        # Print the first T_world_to_system transformation matrix
        if self.Ts_world_to_system is not None and len(self.Ts_world_to_system) > 0:
            print(f"\nFirst T_world_to_system (frame 0):")
            print(self.Ts_world_to_system[0])
        else:
            print("No T_world_to_system transformations available")
        
    def visualize_events(self):
        """Visualize event data for a specified duration with optional video recording."""
        
        print(f"Visualizing events (recording enabled)")
        print("\n" + "="*60)
        print("EVENT VISUALIZATION - GUI INSTRUCTIONS")
        print("="*60)
        print("A window will show the raw event data stream.")
        print("This helps you understand the data before calibration.")
        print("\nControls:")
        print("  • q or ESC: Stop visualization")
        print("="*60)    
        
        # Generate unique output video path
        sequence_name = self._extract_sequence_name()
        base_video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_event_visualization.mp4")
        output_video = self._generate_unique_video_path(base_video_path)
        
        # Prepare VideoWriter for the whole session
        H, W = self.cam_res[0], self.cam_res[1]
        fps = max(1, int(round(1.0 / self.period)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H), isColor=False)
        
        if not video_writer.isOpened():
            print(f"[warn] Could not open video writer at '{output_video}'. Continuing without recording.")
            video_writer = None
        else:
            print(f"Recording video to: {output_video}")
            print(f"Video settings: {W}x{H} @ {fps} FPS")
        
        img = np.ones(self.cam_res, dtype=np.uint8) * 255
        ft = self.start_time
        # self.window_size = 100 * self.period
        window_start = self.start_time
        frame_count = 0
        total_frames_expected = int((self.end_time - self.start_time) / self.period)
        
        cv2.namedWindow('Event Visualization', cv2.WINDOW_NORMAL)
        
        try:
            print(f"Starting visualization from {self.start_time:.3f}s to {self.end_time:.3f}s")
            print(f"Expected total frames: {total_frames_expected}")
            if video_writer:
                print(f"Video will be saved to: {output_video}")
            
            while ft < self.end_time:
                window_end = min(window_start + self.window_size, self.end_time)
                window_center = (window_start + window_end) / 2
                
                e_data = self.imp.get_data_at_time(window_center, self.window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])
                
                for i in range(len(e_ts)):
                    if e_ts[i] >= ft:
                        # Create display image with enhanced information
                        display_img = img.copy()
                        
                        # Add timestamp
                        text = f"t = {ft:.6f}s"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(display_img, text, (display_img.shape[1] - 200, 30), font, 0.7, (0, 0, 0), 2)
                        
                        # Add progress information
                        progress = (ft - self.start_time) / (self.end_time - self.start_time) * 100
                        progress_text = f"Progress: {progress:.1f}%"
                        cv2.putText(display_img, progress_text, (10, 30), font, 0.5, (0, 0, 0), 1)
                        
                        # Frame counter
                        frame_text = f"Frame: {frame_count+1}/{total_frames_expected}"
                        cv2.putText(display_img, frame_text, (10, 50), font, 0.5, (0, 0, 0), 1)
                        
                        # Record frame to video if recording enabled
                        if video_writer is not None:
                            video_writer.write(display_img)
                        
                        cv2.imshow('Event Visualization', display_img)
                        k = cv2.waitKey(int(self.period * 1000))
                        
                        if k == 27 or k == ord('q'):
                            print(f"\nStopped by user at frame {frame_count+1}")
                            raise KeyboardInterrupt
                            
                        img = np.ones(self.cam_res, dtype=np.uint8) * 255
                        ft += self.period
                        frame_count += 1
                        
                        # Progress indicator (every 5%)
                        if frame_count % max(1, total_frames_expected // 20) == 0:
                            total_progress = (ft - self.start_time) / (self.end_time - self.start_time) * 100
                            print(f"Progress: {total_progress:.1f}% | Frames: {frame_count}/{total_frames_expected}")
                        
                    if e_vs[i] < self.cam_res[0] and e_us[i] < self.cam_res[1]:
                        img[e_vs[i], e_us[i]] = 0
                        
                window_start = e_ts[-1] if len(e_ts) > 0 else window_start + self.window_size
                
        except KeyboardInterrupt:
            print(f"\nVisualization stopped by user at frame {frame_count}")
        
        except Exception as e:
            print(f"\nError during visualization: {str(e)}")
            
        finally:
            cv2.destroyAllWindows()
            
            # Clean up and provide summary
            if video_writer is not None:
                video_writer.release()
            
            print(f"\nVisualization completed. Total frames processed: {frame_count}")

    def manual_rotation_estimation(self, chosen_marker: Optional[str] = None) -> np.ndarray:
        """Manually estimate rotation using visual feedback with windowed approach."""

        print("Starting manual rotation estimation (recording enabled)...")
        print("\n" + "="*60)
        print("MANUAL ROTATION ESTIMATION - GUI INSTRUCTIONS")
        print("="*60)
        print("SPACE pause/resume  |  ENTER select axis  |  +/- change angle  |  k/l change step  |  q/ESC finish")

        # Get working markers
        self.markers_names = self.get_markers_names()
        if not self.markers_names:
            raise RuntimeError("No suitable markers found for calibration")

        # Choose feedback marker
        chosen_one = chosen_marker if (chosen_marker and chosen_marker in self.markers_names) else self.markers_names[0]
        print(f"Using markers: {self.markers_names}")
        print(f"Feedback marker: {chosen_one}")

        # Projector
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )

        # # Prepare one VideoWriter for the whole session
        # H, W = self.cam_res[0], self.cam_res[1]
        # fps = max(1, int(round(1.0 / self.period)))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # vw = cv2.VideoWriter(output_video, fourcc, fps, (W, H), isColor=False)
        # if not vw.isOpened():
        #     print(f"[warn] Could not open video writer at '{output_video}'. Continuing without recording.")
        #     vw = None

        # self.window_size = 100 * self.period  # 10 seconds window
        window_start = self.start_time
        # rvec_init = np.zeros(3)

        # Define known initial rotation (in degrees)
        roll, pitch, yaw = -118.0, 13.0, -27.0

        # Convert to rotation vector (Rodrigues form)
        rvec_init = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_rotvec()

        try:
            while window_start < self.end_time:
                window_end = window_start + self.window_size
                window_center = 0.5 * (window_start + window_end)

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, self.window_size)
                e_ts = np.array(e_data['ts']); e_us = np.array(e_data['x']); e_vs = np.array(e_data['y'])

                if len(e_ts) == 0:
                    print(f"No events in window [{window_start:.3f}, {window_end:.3f}]")
                    window_start += self.window_size
                    continue

                print(f"Window {window_start:.3f}–{window_end:.3f}  ({len(e_ts)} events)")
                R_init = Rotation.from_rotvec(rvec_init).as_matrix()

                # Run interactive adjuster and RECORD frames into the same writer
                rvec_init = projector.manual_rotation_adjustment(
                    self.marker_t, self.delay, e_ts, e_us, e_vs, self.period,
                    R_init=R_init, visualize=True, chosen_one=chosen_one,
                    marker_time_offset=window_start, video_record=False
                )

                # next window
                window_start = window_end
                print("window_start updated to:", window_start)

        except helpers.RotationExit as e:
            print("Visualization finished by user with final rotation.")
            rvec_init = e.r_vec

        finally:
            cv2.destroyAllWindows()
            # if vw is not None:
            #     vw.release()
            #     print(f"Saved manual-rotation video → {output_video}")

        return rvec_init

    def manual_delay_correction(self) -> float:
        """Manually correct synchronization delay using windowed approach."""
        
        print("Starting manual delay correction (recording enabled)...")
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
        # self.window_size = 100 * self.period
        window_start = self.start_time
        
        # Create projector for delay adjustment
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )

        # # Prepare VideoWriter for the whole session
        # H, W = self.cam_res[0], self.cam_res[1]
        # fps = max(1, int(round(1.0 / self.period)))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # vw = cv2.VideoWriter(output_video, fourcc, fps, (W, H), isColor=False)
        # if not vw.isOpened():
        #     print(f"[warn] Could not open video writer at '{output_video}'. Continuing without recording.")
        #     vw = None

        try:
            while window_start < float("%.1f" % self.end_time):
                window_end = window_start + self.window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, self.window_size)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])

                print(f"Processing {len(e_ts)} events between {e_ts[0]:.3f}s and {e_ts[-1]:.3f}s")

                try:
                    # Call projector delay adjustment
                    self.delay, self.current_delay_step = projector.fix_delay(
                        self.marker_t, self.delay, e_ts, e_us, e_vs, self.period,
                        visualize=True, marker_time_offset=window_start, delay_step=self.current_delay_step
                    )
                except helpers.DelayReset as e:
                    # ---- FULL RESTART FROM THE FIRST EVER EVENT TIMESTAMP ----
                    print("Delay changed with arrow key: full reset requested.")
                    # 1) adopt the new delay and preserve delay_step
                    self.delay = e.new_delay
                    self.current_delay_step = e.delay_step  # Preserve the delay step
                    print(f"Preserved delay step: {self.current_delay_step:.3f}s")
                    # 2) reset scanning window to the beginning
                    window_start = self.start_time   # or self.imp.first_event_time if you expose it
                    # cv2.destroyAllWindows()
                    continue
                
                print("e_ts final:", e_ts[-1], "window_start:", window_start, "self.window_size:", self.window_size)

                window_start = window_end
                
                print("window_start updated to:", window_start)
                
        except helpers.DelayExit as e:
            print("Delay adjustment stopped by user.")
            self.delay = e.delay
            self.current_delay_step = e.delay_step  # Preserve the delay step from user adjustment
            print(f"Final delay step from manual adjustment: {self.current_delay_step:.3f}s")

        finally:
            cv2.destroyAllWindows()
            # if vw is not None:
            #     vw.release()
            #     print(f"Saved delay-correction video → {output_video}")
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
                            print("📝 You can add new labels to supplement the existing ones")
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
        
        # self.window_size = 100 * self.period
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
                window_end = window_start + self.window_size
                window_center = (window_start + window_end) / 2

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, self.window_size)
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
        # self.window_size = 100 * self.period
        window_start = self.start_time

        print(f"Processing time range: {self.start_time:.3f}s to {self.end_time:.3f}s")
        print(f"Window size: {self.window_size/1000:.1f}s, Period: {self.period:.3f}s")

        window_count = 0

        try:
            while window_start < self.end_time:
                window_end = window_start + self.window_size
                window_center = (window_start + window_end) / 2
                window_count += 1

                print(f"\n--- Processing window {window_count} ---")
                print(f"Window range: {window_start:.3f}s to {window_end:.3f}s")

                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, self.window_size)
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
                self.current_delay_step = current_delay_step  # Update instance variable
                
                # Debug: Show delay_step is preserved between windows
                if window_count > 0:  # Don't show for first window
                    print(f"Window {window_count}: Using delay step {self.current_delay_step:.3f}s (preserved from previous window)")

                # Collect frames for video
                if video_segment is not None:
                    collected_video_segments.append(video_segment)

                # Collect all projected points with event timestamps using synced data
                if synced_image_points:
                    print(f" Window {window_count}: synced_image_points keys = {list(synced_image_points.keys())}")
                    for marker_name, marker_data in synced_image_points.items():
                        if marker_data and "points" in marker_data and "timestamps" in marker_data:
                            points = marker_data["points"]
                            timestamps = marker_data["timestamps"]
                            
                            print(f" {marker_name}: {len(points)} points, {len(timestamps)} timestamps")
                            
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
            print("⚠️ Projection stopped early by user.")

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

###        
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

### Automatization of joint extraction, TODO: check after debug

    def _save_joint_projections_and_video(self):
        """Generate projections and video specifically for joint markers defined in joint config file."""
        
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
        # self.window_size = 100 * self.period
        window_start = self.start_time
        window_count = 0
        
        try:
            while window_start < self.end_time:
                window_end = window_start + self.window_size
                window_center = (window_start + window_end) / 2
                window_count += 1
                
                print(f"Processing joint window {window_count}: {window_start:.3f}s to {window_end:.3f}s")
                
                # Load events for this window
                e_data = self.imp.get_data_at_time(window_center, self.window_size)
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
                        visualize=True, video_record=True,
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

    def run_joint_extraction_automation(self, skip_init_file_creation: bool = False):
        """Automated joint extraction: process projections, create init file, and analyze errors."""
        print("Starting automated joint extraction process...")
        
        # 1. Generate joint projections and video
        try:
            print("Step 1/4: Generating joint projections and video...")
            self._save_joint_projections_and_video()
        except Exception as e:
            print(f"Error during joint projection generation: {e}")
            return False
        
        # 2. Create init file with current parameters (skip if already using existing init file)
        if not skip_init_file_creation:
            try:
                print("Step 2/4: Creating init file...")
                self.save_current_init_file()
            except Exception as e:
                print(f"Error creating init file: {e}")
                return False
        else:
            print("Step 2/4: Skipping init file creation (using existing init file)")
        
        # 3. Save calibration file with sequence-specific naming
        try:
            print("Step 3/4: Saving calibration file...")
            self.save_sequence_specific_calibration()
        except Exception as e:
            print(f"Error saving calibration file: {e}")
            return False
        
        # 4. Run projection error analysis if CSV files are available
        try:
            print("Step 4/4: Running projection error analysis...")
            self.calculate_projection_error(self.output_path, visualize=True)
        except Exception as e:
            print(f"Warning: Error analysis failed: {e}")
            print("Joint extraction completed successfully, but error analysis was skipped")
        
        print("Joint extraction automation completed successfully!")
        return True
    
    def save_current_init_file(self):
        """Save current calibration parameters to a uniquely named init file."""
        # Generate a unique init file path
        init_file = self._generate_unique_init_file_path()
        
        # Save the calibration using the existing method
        self.save_calibration(init_file)
        print(f"Saved current calibration to: {init_file}")
        return init_file
###

###
    def plot_per_marker_error_boxplot(self, marker_errors: dict, save_path: str = None):
        """Display per-marker error distribution with stats (mean, std, min, max)."""
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

        print("\n📊 PLOT EXPLANATION:")
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

    def calculate_projection_error(self, labels_path: str, visualize: bool = True, use_undistorted: bool = False):
        """Enhanced error analysis with frame-by-frame visualization using live projections.
        
        Args:
            labels_path: Path to YAML file with ground truth labels
            visualize: Whether to visualize the errors
            use_undistorted: Whether to undistort ground truth points for fairer comparison
        """
        
        print("Enhanced error analysis: Frame-by-frame GT vs live projected points with directional analysis...")

        # --- Load labeled points from YAML ---
        labeled_points = helpers.read_points_labels(labels_path)
        print(f"Loaded {len(labeled_points['times'])} labeled timestamps from YAML")
        print(f"Available markers in labels: {set().union(*[frame.keys() for frame in labeled_points['points']])}")

        # --- Load projected points from CSV (delay already included in these projections) ---
        sequence_name = self._extract_sequence_name()
        base_csv_name = f"{sequence_name}_projection_points.csv"
        projected_points_csv = os.path.join(self._get_output_directory(), base_csv_name)
        
        # If sequence-specific CSV doesn't exist, look for generic ones
        if not os.path.exists(projected_points_csv):
            import glob
            csv_pattern = os.path.join(self._get_output_directory(), "projected_points*.csv")
            csv_files = glob.glob(csv_pattern)
            if csv_files:
                projected_points_csv = max(csv_files, key=os.path.getmtime)
                print(f"Using most recent projected points CSV: {os.path.basename(projected_points_csv)}")
            else:
                print(f"Error: No projected points CSV files found in {self._get_output_directory()}")
                return None

        # Load CSV data (projections already include delay correction)
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
                            
                            if row[column] and row.get(y_column):
                                try:
                                    x = float(row[column])
                                    y = float(row[y_column])
                                    projected_data.setdefault(marker_name, []).append((timestamp, x, y))
                                except ValueError:
                                    continue
                                    
        except Exception as e:
            print(f"Error reading CSV file {projected_points_csv}: {e}")
            return None

        print(f"Loaded projected data for {len(projected_data)} markers from CSV.")

        # --- Enhanced error analysis with frame-by-frame visualization ---
        comparison_results = {
            'marker_errors': {},
            'directional_errors': {},  # x and y components
            'spatial_distribution': {}  # error patterns across image
        }
        
        # Create output directory for frame images
        frames_dir = os.path.join(self._get_output_directory(), "error_analysis_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"\nGenerating frame-by-frame error analysis...")
        print(f"Frame images will be saved to: {frames_dir}")

        all_frame_data = []  # Store data for comprehensive analysis
        
        for frame_idx, (event_timestamp, labeled_frame) in enumerate(zip(labeled_points['times'], labeled_points['points'])):
            # Create frame image for visualization
            import cv2
            frame_img = np.ones((self.cam_res[0], self.cam_res[1], 3), dtype=np.uint8) * 255  # White background
            
            frame_errors = {}
            frame_positions = {'gt': {}, 'proj': {}}
            
            for marker, label in labeled_frame.items():
                gt_x, gt_y = int(label['x']), int(label['y'])
                
                # Apply undistortion to ground truth points if requested
                if use_undistorted and hasattr(self, 'K') and hasattr(self, 'D') and self.D is not None:
                    try:
                        temp_projector = helpers.ViconProjector(None, None, None, self.K, self.D)
                        undistorted_gt = temp_projector.undistort_image_points(np.array([[gt_x, gt_y]], dtype=np.float64))
                        gt_x, gt_y = int(undistorted_gt[0, 0]), int(undistorted_gt[0, 1])
                    except Exception as e:
                        print(f"⚠️ Could not undistort GT point for {marker}: {e}")
                
                label_x, label_y = gt_x, gt_y
                
                if marker not in projected_data:
                    continue

                # Find closest projected point (no additional delay correction needed - already in CSV)
                proj_list = sorted(projected_data[marker], key=lambda x: x[0])
                timestamps = [p[0] for p in proj_list]
                import bisect
                insert_pos = bisect.bisect_left(timestamps, event_timestamp)

                candidates = []
                for idx in [insert_pos - 1, insert_pos]:
                    if 0 <= idx < len(proj_list):
                        proj_timestamp, proj_x, proj_y = proj_list[idx]
                        time_diff = abs(proj_timestamp - event_timestamp)
                        candidates.append((time_diff, proj_x, proj_y))
                
                if not candidates:
                    continue
                    
                _, proj_x, proj_y = min(candidates)
                
                
                # Calculate errors
                error_x = float(proj_x - label_x)
                error_y = float(proj_y - label_y) 
                error_magnitude = np.sqrt(error_x**2 + error_y**2)
                
                # Store results for analysis
                if marker not in comparison_results['marker_errors']:
                    comparison_results['marker_errors'][marker] = []
                    comparison_results['directional_errors'][marker] = {'x': [], 'y': []}
                
                comparison_results['marker_errors'][marker].append(error_magnitude)
                comparison_results['directional_errors'][marker]['x'].append(error_x)
                comparison_results['directional_errors'][marker]['y'].append(error_y)
                
                frame_errors[marker] = {
                    'magnitude': error_magnitude,
                    'x_error': error_x, 
                    'y_error': error_y
                }
                frame_positions['gt'][marker] = (label_x, label_y)
                
                if not candidates:
                    continue

                _, proj_x, proj_y = min(candidates)
                
                # Calculate error components
                error_x = proj_x - label_x  # Positive = projected is right of GT
                error_y = proj_y - label_y  # Positive = projected is below GT
                error_magnitude = np.sqrt(error_x**2 + error_y**2)
                
                # Store error data
                comparison_results['marker_errors'].setdefault(marker, []).append(error_magnitude)
                comparison_results['directional_errors'].setdefault(marker, {'x': [], 'y': []})
                comparison_results['directional_errors'][marker]['x'].append(error_x)
                comparison_results['directional_errors'][marker]['y'].append(error_y)
                
                frame_errors[marker] = {
                    'error_x': error_x,
                    'error_y': error_y,
                    'error_mag': error_magnitude,
                    'gt_pos': (label_x, label_y),
                    'proj_pos': (int(proj_x), int(proj_y))
                }
                
                frame_positions['gt'][marker] = (label_x, label_y)
                frame_positions['proj'][marker] = (int(proj_x), int(proj_y))
                
                # Draw on frame image
                # Ground truth point (green circle)
                cv2.circle(frame_img, (label_x, label_y), 8, (0, 255, 0), -1)
                cv2.putText(frame_img, f"GT-{marker}", (label_x + 10, label_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Projected point (red circle)
                cv2.circle(frame_img, (int(proj_x), int(proj_y)), 8, (0, 0, 255), -1)
                cv2.putText(frame_img, f"PROJ-{marker}", (int(proj_x) + 10, int(proj_y) + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Error vector (yellow arrow)
                cv2.arrowedLine(frame_img, (label_x, label_y), (int(proj_x), int(proj_y)), 
                              (0, 255, 255), 2, tipLength=0.3)
                
                # Error magnitude text
                mid_x, mid_y = (label_x + int(proj_x)) // 2, (label_y + int(proj_y)) // 2
                cv2.putText(frame_img, f"{error_magnitude:.1f}px", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Add frame info
            cv2.putText(frame_img, f"Frame {frame_idx+1}/{len(labeled_points['times'])} | t={event_timestamp:.3f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}_t{event_timestamp:.3f}.png")
            cv2.imwrite(frame_path, frame_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            # Store frame data for analysis
            all_frame_data.append({
                'timestamp': event_timestamp,
                'frame_idx': frame_idx,
                'errors': frame_errors,
                'positions': frame_positions
            })
        
        print(f"Generated {len(all_frame_data)} frame analysis images")
        
        # --- Comprehensive error analysis and visualization ---
        # self._create_comprehensive_error_dashboard(comparison_results, all_frame_data)
        
        # --- Time-series plots: per-marker X and Y vs time (timestamps in seconds with ms resolution) ---
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FormatStrFormatter
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Build labeled per-marker time series from labeled_points
            labeled_times = labeled_points['times']
            labeled_frames = labeled_points['points']

            markers_union = set(list(projected_data.keys()))
            for frame in labeled_frames:
                if isinstance(frame, dict):
                    markers_union.update(frame.keys())

            out_dir = self._get_output_directory()

            # Determine global right-hand x-limit: last labeled timestamp (show plots only up to this time)
            last_labeled_time = None
            try:
                last_labeled_time = max([float(t) for t in labeled_times]) if labeled_times else None
            except Exception:
                last_labeled_time = None

            for marker in sorted(markers_union):
                # collect labeled times/x/y
                lab_t = []
                lab_x = []
                lab_y = []
                for t, frame in zip(labeled_times, labeled_frames):
                    if not isinstance(frame, dict):
                        continue
                    if marker in frame:
                        try:
                            lx = float(frame[marker]['x'])
                            ly = float(frame[marker]['y'])
                        except Exception:
                            continue
                        lab_t.append(float(t))
                        lab_x.append(lx)
                        lab_y.append(ly)

                # collect projected times/x/y
                proj_entries = projected_data.get(marker, [])
                proj_t = [float(p[0]) for p in proj_entries]
                proj_x = [float(p[1]) for p in proj_entries]
                proj_y = [float(p[2]) for p in proj_entries]

                # Skip markers with no data
                if not lab_t and not proj_t:
                    continue

                # Create figure with two subplots: X vs time (top), Y vs time (bottom)
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

                    # Plot labeled and projected points. Use translucency for better overlay visibility.
                    if lab_t:
                        ax1.plot(lab_t, lab_x, marker='o', linestyle='-', color='tab:blue', markersize=4, label='Labeled X', alpha=0.6, linewidth=1)
                        ax2.plot(lab_t, lab_y, marker='o', linestyle='-', color='tab:blue', markersize=4, label='Labeled Y', alpha=0.6, linewidth=1)
                    if proj_t:
                        ax1.plot(proj_t, proj_x, marker='s', linestyle='None', color='tab:red', markersize=4, label='Projected X', alpha=0.45)
                        ax2.plot(proj_t, proj_y, marker='s', linestyle='None', color='tab:red', markersize=4, label='Projected Y', alpha=0.45)

                    ax1.set_ylabel('X (pixels)')
                    ax2.set_ylabel('Y (pixels)')
                    ax2.set_xlabel('Time (s)')

                    ax1.legend(loc='best', fontsize='small')
                    ax2.legend(loc='best', fontsize='small')

                    # Format x-axis to show milliseconds (3 decimal places)
                    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

                    # Limit x-axis to the last labeled timestamp if available
                    try:
                        combined_times = []
                        if lab_t:
                            combined_times.extend(lab_t)
                        if proj_t:
                            combined_times.extend(proj_t)
                        if combined_times:
                            xmin = min(combined_times)
                            xmax = last_labeled_time if last_labeled_time is not None else max(combined_times)
                            # Ensure xmin < xmax
                            if xmin < xmax:
                                ax2.set_xlim(xmin, xmax)
                    except Exception:
                        pass

                    fig.suptitle(f"Marker: {marker} — Labeled vs Projected (X/Y over time)")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                    # sanitize marker name for filenames
                    safe_marker = marker.replace(':', '_').replace('/', '_').replace(' ', '_')
                    save_name = os.path.join(out_dir, f"timeseries_{safe_marker}_{timestamp}.png")
                    fig.savefig(save_name, dpi=200)
                    plt.close(fig)
                    print(f" Saved time-series plot for {marker}: {save_name}")
                except Exception as e:
                    print(f"Failed to plot time series for marker {marker}: {e}")

        except Exception as e:
            print(f"Failed to generate time-series plots: {e}")

        # --- Traditional error plot ---
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_save_path = os.path.join(self._get_output_directory(), f"error_analysis_{timestamp}.png")
        self.plot_per_marker_error_boxplot(comparison_results['marker_errors'], save_path=plot_save_path)

        return comparison_results
    
    def _create_comprehensive_error_dashboard(self, comparison_results, all_frame_data):
        """Create comprehensive error analysis dashboard with directional analysis."""
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy import stats
        
        # Setup figure with multiple subplots
        fig = plt.figure(figsize=(20, 12), num="Comprehensive Error Analysis Dashboard")
        
        # 1. Error magnitude distribution per marker
        ax1 = fig.add_subplot(2, 4, 1)
        markers = list(comparison_results['marker_errors'].keys())
        if markers:
            errors_list = [comparison_results['marker_errors'][m] for m in markers]
            ax1.boxplot(errors_list, labels=markers)
            ax1.set_title("Error Magnitude Distribution")
            ax1.set_ylabel("Error (pixels)")
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. Directional error analysis (X vs Y components)
        ax2 = fig.add_subplot(2, 4, 2)
        colors = plt.cm.tab10(np.linspace(0, 1, len(markers)))
        for i, marker in enumerate(markers):
            if marker in comparison_results['directional_errors']:
                x_errors = comparison_results['directional_errors'][marker]['x']
                y_errors = comparison_results['directional_errors'][marker]['y']
                ax2.scatter(x_errors, y_errors, alpha=0.6, color=colors[i], label=marker, s=30)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel("X Error (pixels)")
        ax2.set_ylabel("Y Error (pixels)")
        ax2.set_title("Directional Error Pattern")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 3. Error trends over time
        ax3 = fig.add_subplot(2, 4, 3)
        timestamps = [frame['timestamp'] for frame in all_frame_data]
        for marker in markers:
            marker_error_timeline = []
            marker_timestamps = []
            for frame in all_frame_data:
                if marker in frame['errors']:
                    marker_error_timeline.append(frame['errors'][marker]['error_mag'])
                    marker_timestamps.append(frame['timestamp'])
            if marker_error_timeline:
                ax3.plot(marker_timestamps, marker_error_timeline, 
                        label=marker, alpha=0.7, linewidth=1.5)
        
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Error (pixels)")
        ax3.set_title("Error Evolution Over Time")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Spatial distribution of errors
        ax4 = fig.add_subplot(2, 4, 4)
        all_gt_x, all_gt_y, all_errors = [], [], []
        for frame in all_frame_data:
            for marker, error_data in frame['errors'].items():
                gt_x, gt_y = error_data['gt_pos']
                all_gt_x.append(gt_x)
                all_gt_y.append(gt_y)
                all_errors.append(error_data['error_mag'])
        
        if all_gt_x:
            scatter = ax4.scatter(all_gt_x, all_gt_y, c=all_errors, cmap='viridis', 
                                 alpha=0.7, s=50)
            plt.colorbar(scatter, ax=ax4, label='Error (pixels)')
            ax4.set_xlabel("X Position (pixels)")
            ax4.set_ylabel("Y Position (pixels)")
            ax4.set_title("Spatial Error Distribution")
            ax4.set_xlim(0, self.cam_res[1])
            ax4.set_ylim(self.cam_res[0], 0)  # Invert Y for image coordinates
            ax4.grid(True, alpha=0.3)
        
        # 5-8. Individual marker analysis (X/Y error components)
        for i, marker in enumerate(markers[:4]):  # Show up to 4 markers
            ax = fig.add_subplot(2, 4, 5 + i)
            if marker in comparison_results['directional_errors']:
                x_errors = comparison_results['directional_errors'][marker]['x']
                y_errors = comparison_results['directional_errors'][marker]['y']
                
                # Histogram of directional errors
                ax.hist([x_errors, y_errors], bins=15, alpha=0.7, 
                       label=['X Error', 'Y Error'], color=['red', 'blue'])
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.set_xlabel("Error (pixels)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{marker} Error Components")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Statistical analysis
                x_mean, x_std = np.mean(x_errors), np.std(x_errors)
                y_mean, y_std = np.mean(y_errors), np.std(y_errors)
                
                # Test for systematic bias
                t_stat_x, p_val_x = stats.ttest_1samp(x_errors, 0)
                t_stat_y, p_val_y = stats.ttest_1samp(y_errors, 0)
                
                # Add text with statistics
                stats_text = f"X: μ={x_mean:.2f}±{x_std:.2f}\n"
                stats_text += f"Y: μ={y_mean:.2f}±{y_std:.2f}\n"
                stats_text += f"Bias X: p={'<0.05' if p_val_x < 0.05 else '≥0.05'}\n"
                stats_text += f"Bias Y: p={'<0.05' if p_val_y < 0.05 else '≥0.05'}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive dashboard
        dashboard_path = os.path.join(self._get_output_directory(), "comprehensive_error_dashboard.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive error dashboard saved: {dashboard_path}")
        
        plt.show(block=False)
        
        # Print statistical summary
        self._print_directional_error_summary(comparison_results)
    
    def _print_directional_error_summary(self, comparison_results):
        """Print comprehensive directional error statistics."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE DIRECTIONAL ERROR ANALYSIS")
        print("="*80)
        
        from scipy import stats
        
        for marker in comparison_results['directional_errors'].keys():
            x_errors = comparison_results['directional_errors'][marker]['x']
            y_errors = comparison_results['directional_errors'][marker]['y']
            magnitudes = comparison_results['marker_errors'][marker]
            
            print(f"\n📍 MARKER: {marker}")
            print("-" * 50)
            
            # Basic statistics
            print(f"Sample size: {len(x_errors)} points")
            print(f"Error magnitude: {np.mean(magnitudes):.2f} ± {np.std(magnitudes):.2f} pixels")
            
            # X-direction analysis
            x_mean, x_std = np.mean(x_errors), np.std(x_errors)
            x_median = np.median(x_errors)
            t_stat_x, p_val_x = stats.ttest_1samp(x_errors, 0)
            
            print(f"\nX-Direction (horizontal) errors:")
            print(f"  Mean: {x_mean:+.3f} pixels (std: {x_std:.3f})")
            print(f"  Median: {x_median:+.3f} pixels")
            print(f"  Range: {np.min(x_errors):+.2f} to {np.max(x_errors):+.2f} pixels")
            print(f"  Systematic bias test: t={t_stat_x:.3f}, p={p_val_x:.4f}")
            if p_val_x < 0.05:
                direction = "RIGHT" if x_mean > 0 else "LEFT"
                print(f"  ⚠️  SIGNIFICANT BIAS detected: projected points tend to be {direction} of ground truth")
            else:
                print(f"  ✓ No significant systematic bias in X direction")
            
            # Y-direction analysis
            y_mean, y_std = np.mean(y_errors), np.std(y_errors)
            y_median = np.median(y_errors)
            t_stat_y, p_val_y = stats.ttest_1samp(y_errors, 0)
            
            print(f"\nY-Direction (vertical) errors:")
            print(f"  Mean: {y_mean:+.3f} pixels (std: {y_std:.3f})")
            print(f"  Median: {y_median:+.3f} pixels")
            print(f"  Range: {np.min(y_errors):+.2f} to {np.max(y_errors):+.2f} pixels")
            print(f"  Systematic bias test: t={t_stat_y:.3f}, p={p_val_y:.4f}")
            if p_val_y < 0.05:
                direction = "BELOW" if y_mean > 0 else "ABOVE"
                print(f"  ⚠️  SIGNIFICANT BIAS detected: projected points tend to be {direction} ground truth")
            else:
                print(f"  ✓ No significant systematic bias in Y direction")
            
            # Spatial pattern analysis
            print(f"\nSpatial distribution:")
            # Divide screen into quadrants and analyze
            screen_center_x, screen_center_y = self.cam_res[1] // 2, self.cam_res[0] // 2
            
            quadrant_errors = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
            for frame_data in comparison_results.get('frame_data', []):
                if marker in frame_data.get('errors', {}):
                    gt_pos = frame_data['errors'][marker]['gt_pos']
                    error_mag = frame_data['errors'][marker]['error_mag']
                    
                    if gt_pos[0] >= screen_center_x and gt_pos[1] <= screen_center_y:
                        quadrant_errors['Q1'].append(error_mag)  # Top-right
                    elif gt_pos[0] < screen_center_x and gt_pos[1] <= screen_center_y:
                        quadrant_errors['Q2'].append(error_mag)  # Top-left
                    elif gt_pos[0] < screen_center_x and gt_pos[1] > screen_center_y:
                        quadrant_errors['Q3'].append(error_mag)  # Bottom-left
                    else:
                        quadrant_errors['Q4'].append(error_mag)  # Bottom-right
            
            for quad, errors in quadrant_errors.items():
                if errors:
                    print(f"  {quad}: {len(errors)} points, mean error: {np.mean(errors):.2f}px")
        
        print("\n" + "="*80)
        print("📊 INTERPRETATION GUIDE:")
        print("• X Error: Positive = projected point is RIGHT of ground truth")
        print("• Y Error: Positive = projected point is BELOW ground truth") 
        print("• Systematic bias (p<0.05) suggests calibration issues")
        print("• Random errors suggest labeling precision or temporal sync issues")
        print("="*80)

    def save_frames_with_gt_and_csv_projections(self, labels_path: str, create_video: bool = False):
        """Save every frame showing ground truth and projected points - one frame per GT timestamp."""
        
        print("Creating frame-by-frame visualization with ground truth and projected points...")
        
        # --- Load labeled points from YAML ---
        labeled_points = helpers.read_points_labels(labels_path)
        print(f"Loaded {len(labeled_points['times'])} labeled timestamps from YAML")
        print(f"Available markers in labels: {set().union(*[frame.keys() for frame in labeled_points['points']])}")

        # --- Load projected points from CSV ---
        sequence_name = self._extract_sequence_name()
        base_csv_name = f"{sequence_name}_projection_points.csv"
        projected_points_csv = os.path.join(self._get_output_directory(), base_csv_name)
        
        if not os.path.exists(projected_points_csv):
            import glob
            csv_pattern = os.path.join(self._get_output_directory(), "projected_points*.csv")
            csv_files = glob.glob(csv_pattern)
            if csv_files:
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
                            
                            if row[column] and row.get(y_column):
                                try:
                                    x = float(row[column])
                                    y = float(row[y_column])
                                    projected_data.setdefault(marker_name, []).append((timestamp, x, y))
                                except ValueError:
                                    continue
                                    
        except Exception as e:
            print(f"Error reading CSV file {projected_points_csv}: {e}")
            return None

        print(f"Loaded projected data for {len(projected_data)} markers from CSV.")
        
        # Debug: Print sample of projected data structure
        if projected_data:
            sample_marker = list(projected_data.keys())[0]
            print(f"Debug - Sample marker '{sample_marker}' has {len(projected_data[sample_marker])} projected points")
            if projected_data[sample_marker]:
                print(f"Debug - First projected point: timestamp={projected_data[sample_marker][0][0]:.6f}, x={projected_data[sample_marker][0][1]:.2f}, y={projected_data[sample_marker][0][2]:.2f}")
        else:
            print("Debug - No projected data loaded!")

        # Create output directory for frame images
        frames_dir = os.path.join(self._get_output_directory(), "gt_projection_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Frame images will be saved to: {frames_dir}")
        print(f"Note: CSV projections already include delay correction - using direct timestamp matching")

        frame_count = 0
        video_frames = []
        
        for frame_idx, (gt_timestamp, labeled_frame) in enumerate(zip(labeled_points['times'], labeled_points['points'])):
            # Create frame image with white background and event data
            frame_img = np.ones((self.cam_res[0], self.cam_res[1], 3), dtype=np.uint8) * 255  # White background
            
            # Load events around this specific GT timestamp for background
            try:
                event_window = 0.02  # 20ms window around GT timestamp
                e_data = self.imp.get_data_at_time(gt_timestamp, event_window)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])
                
                # Render events as background (like original method)
                for i in range(len(e_ts)):
                    uu = int(e_us[i])
                    vv = int(e_vs[i])
                    if 0 <= uu < self.cam_res[1] and 0 <= vv < self.cam_res[0]:
                        frame_img[vv, uu] = [80, 80, 80]  # Dark gray events for background
                        
            except Exception as e:
                print(f"Warning: Could not load events for timestamp {gt_timestamp:.3f}s: {e}")
                # Keep white background if events cannot be loaded
            
            points_drawn = 0
            frame_errors = []  # Track pixel errors for this frame
            
            for marker, label in labeled_frame.items():
                gt_x, gt_y = int(label['x']), int(label['y'])
                
                if marker not in projected_data:
                    print(f"Debug - No projected data found for marker: {marker}")
                    # Draw only ground truth if no projection available - single small circle
                    cv2.circle(frame_img, (gt_x, gt_y), 5, (0, 255, 0), -1)  # Green circle for GT
                    points_drawn += 1
                    continue

                # Find closest projected point (no additional delay correction needed - already in CSV)
                proj_list = sorted(projected_data[marker], key=lambda x: x[0])
                timestamps = [p[0] for p in proj_list]
                import bisect
                insert_pos = bisect.bisect_left(timestamps, gt_timestamp)

                candidates = []
                for idx in [insert_pos - 1, insert_pos]:
                    if 0 <= idx < len(proj_list):
                        proj_timestamp, proj_x, proj_y = proj_list[idx]
                        time_diff = abs(proj_timestamp - gt_timestamp)
                        candidates.append((time_diff, proj_x, proj_y, proj_timestamp))
                
                if candidates:
                    _, proj_x, proj_y, proj_timestamp = min(candidates)
                    #print(f"Debug - Found projection for {marker}: gt_time={gt_timestamp:.6f}, proj_time={proj_timestamp:.6f}, proj=({proj_x:.1f},{proj_y:.1f})")
                    
                    # Draw ground truth point (green) - single small circle
                    cv2.circle(frame_img, (gt_x, gt_y), 5, (0, 255, 0), -1)
                    
                    # Draw projected point (red) - single small circle  
                    cv2.circle(frame_img, (int(proj_x), int(proj_y)), 5, (0, 0, 255), -1)
                    
                    # Draw connection line (yellow)
                    cv2.line(frame_img, (gt_x, gt_y), (int(proj_x), int(proj_y)), (0, 255, 255), 2)
                    
                    # Calculate and display error
                    error_magnitude = np.sqrt((proj_x - gt_x)**2 + (proj_y - gt_y)**2)
                    frame_errors.append(error_magnitude)  # Track for frame statistics
                    
                    # Position text above the line with better visibility
                    mid_x, mid_y = (gt_x + int(proj_x)) // 2, (gt_y + int(proj_y)) // 2
                    text_y = mid_y - 15  # Position 15 pixels above the line for better spacing
                    
                    # Create distance text with marker name for clarity
                    distance_text = f"{marker}: {error_magnitude:.1f}px"
                    
                    # Add black background rectangle for better text visibility
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.45
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(distance_text, font, font_scale, thickness)
                    
                    # Draw black background rectangle
                    cv2.rectangle(frame_img, (mid_x - 2, text_y - text_height - 2), 
                                 (mid_x + text_width + 2, text_y + baseline + 2), (0, 0, 0), -1)
                    
                    # Draw white text on black background
                    cv2.putText(frame_img, distance_text, (mid_x, text_y), font, font_scale, (255, 255, 255), thickness)
                    
                    points_drawn += 1
                else:
                    print(f"Debug - No matching projection candidates found for marker: {marker}")
                    # Draw only ground truth if no matching projection found - single small circle
                    cv2.circle(frame_img, (gt_x, gt_y), 5, (0, 255, 0), -1)
                    points_drawn += 1
            
            if points_drawn > 0:
                # Calculate frame statistics
                frame_stats_text = ""
                if frame_errors:
                    mean_error = np.mean(frame_errors)
                    max_error = np.max(frame_errors)
                    frame_stats_text = f" | Errors: avg={mean_error:.1f}px, max={max_error:.1f}px"
                
                # Add frame info (CSV data note) with error statistics
                info_y = 25
                cv2.putText(frame_img, f"GT: {gt_timestamp:.3f}s (CSV projections){frame_stats_text}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                # Add enhanced legend
                legend_y = self.cam_res[0] - 60
                cv2.putText(frame_img, "Legend: GT=Green, PROJ=Red, Error=Yellow line + distance", 
                           (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                legend_y += 20
                cv2.putText(frame_img, f"Points shown: {points_drawn} | Frame {frame_idx+1}/{len(labeled_points['times'])}", 
                           (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                # Add timestamp display in top-right corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                color = (128, 128, 128)
                
                # GT timestamp (same as CSV lookup timestamp)
                gt_time_text = f"Timestamp: {gt_timestamp:.3f}s"
                (text_width, text_height), _ = cv2.getTextSize(gt_time_text, font, font_scale, thickness)
                x = self.cam_res[1] - text_width - 10
                y = text_height + 10
                cv2.putText(frame_img, gt_time_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                # Draw camera direction arrow in top-right corner (if camera transforms available)
                try:
                    # Prefer orientation derived from camera markers if available
                    used_forward = None
                    if hasattr(self, 'camera_forwards') and getattr(self, 'camera_forwards') is not None:
                        # find nearest vicon frame index for this gt timestamp
                        if hasattr(self, 'marker_t'):
                            frame_idx_v = int(np.argmin(np.abs(self.marker_t - gt_timestamp)))
                            if 0 <= frame_idx_v < len(self.camera_forwards):
                                used_forward = np.array(self.camera_forwards[frame_idx_v], dtype=np.float64)

                    if used_forward is None:
                        # compute camera world position at current and previous vicon frames as fallback
                        if hasattr(self, 'marker_t') and getattr(self, 'Ts_world_to_system', None) is not None and getattr(self, 'T_syst_to_camera_opt', None) is not None:
                            frame_idx_v = int(np.argmin(np.abs(self.marker_t - gt_timestamp)))

                            def invert_SE3(Tmat):
                                Rm = Tmat[:3, :3]
                                tm = Tmat[:3, 3].reshape(3,1)
                                Rinv = Rm.T
                                tinv = -Rinv.dot(tm)
                                Tinv = np.eye(4, dtype=np.float64)
                                Tinv[:3, :3] = Rinv
                                Tinv[:3, 3] = tinv.ravel()
                                return Tinv

                            def camera_world_pos_for_frame(fid):
                                if not (0 <= fid < len(self.Ts_world_to_system)):
                                    return None
                                T_w2s = np.array(self.Ts_world_to_system[fid], dtype=np.float64)
                                T_s2w = invert_SE3(T_w2s)
                                T_s2c = np.array(self.T_syst_to_camera_opt, dtype=np.float64)
                                T_c2s = invert_SE3(T_s2c)
                                cam_in_s = (T_c2s @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
                                cam_in_w = (T_s2w @ np.append(cam_in_s, 1.0))[:3]
                                return cam_in_w

                            p_curr = camera_world_pos_for_frame(frame_idx_v)
                            p_prev = camera_world_pos_for_frame(max(0, frame_idx_v - 1))
                            if p_curr is not None and p_prev is not None:
                                delta = p_curr - p_prev
                                used_forward = np.array([float(delta[0]), float(delta[1]), 0.0], dtype=np.float64)

                    if used_forward is None:
                        # final fallback: optical axis from estimated T_syst_to_camera_opt
                        Tsc = np.array(self.T_syst_to_camera_opt, dtype=np.float64)
                        R = Tsc[:3, :3]
                        cam_forward_sys = R.T.dot(np.array([0.0, 0.0, 1.0], dtype=np.float64))
                        used_forward = np.array([float(cam_forward_sys[0]), float(cam_forward_sys[1]), 0.0], dtype=np.float64)

                    vx, vy = float(used_forward[0]), float(used_forward[1])
                    vec_norm = np.hypot(vx, vy)
                    if vec_norm >= 1e-6:
                        vx /= vec_norm
                        vy /= vec_norm
                        arrow_len = max(24, min(self.cam_res) // 12)
                        cx = self.cam_res[1] - 60
                        cy = 60
                        ex = int(cx + vx * arrow_len)
                        ey = int(cy - vy * arrow_len)
                        cv2.arrowedLine(frame_img, (cx, cy), (ex, ey), (80, 80, 80), 2, tipLength=0.3)
                        cv2.putText(frame_img, "Cam dir", (cx - 36, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
                except Exception:
                    pass
                
                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}_t{gt_timestamp:.3f}.png")
                cv2.imwrite(frame_path, frame_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                if create_video:
                    video_frames.append(frame_img.copy())
                
                frame_count += 1
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"Saved {frame_count} frames...")
        
        print(f"Generated {frame_count} frame visualization images")
        
        # Create video if requested
        if create_video and video_frames:
            video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_gt_projection_comparison.mp4")
            self._save_video_from_frames(video_frames, video_path, fps=10)
            print(f"Saved comparison video: {video_path}")
        
        return frames_dir

    def _save_video_from_frames(self, frames, output_path, fps=10):
        """Save a list of frames as a video file."""
        if not frames:
            print("No frames to save for video")
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        
        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()

    def save_frames_with_gt_and_live_projections(self, labels_path: str, create_video: bool = False, use_undistorted: bool = True):
        """Save every frame showing ground truth and live projected points using project_vicon_to_event_plane_dynamic.
        
        Args:
            labels_path: Path to YAML file with ground truth labels
            create_video: Whether to create a video from the frames
            use_undistorted: Whether to use undistorted projections for better accuracy
        """
        
        print("Creating frame-by-frame visualization with ground truth and live projected points...")
        
        # --- Load labeled points from YAML ---
        labeled_points = helpers.read_points_labels(labels_path)
        print(f"Loaded {len(labeled_points['times'])} labeled timestamps from YAML")
        print(f"Available markers in labels: {set().union(*[frame.keys() for frame in labeled_points['points']])}")

        # --- Setup live projection using ViconProjector ---
        if not self.markers_names:
            self.markers_names = self.get_markers_names()
        
        print(f"Using {len(self.markers_names)} markers for live projection")
        
        # Create projector for live projections
        projector = helpers.ViconProjector(
            self.markers_names, self.c3d_data, self.points_3d,
            self.T_syst_to_camera_opt, self.Ts_world_to_system,
            self.K, self.cam_res, D=self.D, subject=self.subject
        )
        
        # Option to use undistorted projections for better accuracy
        if use_undistorted and projector.D is not None:
            print(f"🔧 Computing undistorted projections for better accuracy...")
            projector._calculate_projections_undistorted()
            print(f"✅ Using undistorted projections for visualization")
        else:
            print(f"ℹ️ Using regular (distorted) projections")

        # Create output directory for frame images
        frames_dir = os.path.join(self._get_output_directory(), "gt_projection_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Frame images will be saved to: {frames_dir}")
        print(f"Using delay: {self.delay:.6f}s for projection matching")

        frame_count = 0
        video_frames = []
        current_delay = self.delay
        
        # Process each GT timestamp individually
        for frame_idx, (gt_timestamp, labeled_frame) in enumerate(zip(labeled_points['times'], labeled_points['points'])):
            
            # Create image once for this GT timestamp
            img = np.ones((self.cam_res[0], self.cam_res[1], 3), dtype=np.uint8) * 255
            
            # Load events around this specific GT timestamp
            try:
                event_window = 0.02  # 20ms window around GT timestamp
                e_data = self.imp.get_data_at_time(gt_timestamp, event_window)
                e_ts = np.array(e_data['ts'])
                e_us = np.array(e_data['x'])
                e_vs = np.array(e_data['y'])
                
                # Render events as background
                for i in range(len(e_ts)):
                    uu = int(e_us[i])
                    vv = int(e_vs[i])
                    if 0 <= uu < self.cam_res[1] and 0 <= vv < self.cam_res[0]:
                        img[vv, uu] = [80, 80, 80]  # Dark gray events for background
                        
            except Exception as e:
                print(f"Warning: Could not load events for timestamp {gt_timestamp:.3f}s: {e}")
                # Keep white background if events cannot be loaded
            
            # Get live projections at this timestamp using the projector
            live_projections = {}
            try:
                # Project markers at the corrected timestamp (accounting for delay)
                target_proj_timestamp = gt_timestamp - current_delay
                projected_points = projector.project_markers_at_time(target_proj_timestamp)
                
                # Convert to our expected format: marker_name -> (x, y)
                for marker_name in self.markers_names:
                    if (marker_name in projected_points and 
                        projected_points[marker_name] is not None and 
                        len(projected_points[marker_name]) >= 2):
                        x, y = projected_points[marker_name][:2]
                        if np.isfinite(x) and np.isfinite(y):
                            # Check if projection is within image bounds
                            if 0 <= x < self.cam_res[1] and 0 <= y < self.cam_res[0]:
                                live_projections[marker_name] = (float(x), float(y))
                                
            except Exception as e:
                print(f"Warning: Could not get live projections for timestamp {gt_timestamp:.3f}s: {e}")
                live_projections = {}
            
            points_drawn = 0
            
            # Process each marker in this GT frame
            for marker, label in labeled_frame.items():
                gt_x, gt_y = int(label['x']), int(label['y'])
                
                if marker not in live_projections:
                    # Draw only ground truth if no projection available - single small circle
                    cv2.circle(img, (gt_x, gt_y), 5, (0, 255, 0), -1)  # Green circle for GT
                    points_drawn += 1
                    continue

                # Get live projected point
                proj_x, proj_y = live_projections[marker]
                
                # Draw ground truth point (green) - single small circle
                cv2.circle(img, (gt_x, gt_y), 5, (0, 255, 0), -1)
                
                # Draw projected point (red) - single small circle  
                cv2.circle(img, (int(proj_x), int(proj_y)), 5, (0, 0, 255), -1)
                
                # Draw connection line (yellow)
                cv2.line(img, (gt_x, gt_y), (int(proj_x), int(proj_y)), (0, 255, 255), 2)
                
                # Calculate and display error
                error_magnitude = np.sqrt((proj_x - gt_x)**2 + (proj_y - gt_y)**2)
                mid_x, mid_y = (gt_x + int(proj_x)) // 2, (gt_y + int(proj_y)) // 2
                cv2.putText(img, f"{error_magnitude:.1f}px", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                points_drawn += 1
            
            if points_drawn > 0:
                # Add frame info and timing details
                cv2.putText(img, f"GT: {gt_timestamp:.3f}s (delay: {current_delay:.3f}s)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                cv2.putText(img, "Legend: GT=Green, PROJ=Red, Error=Yellow line", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                # Add timestamp displays in top-right corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                color = (128, 128, 128)
                
                # GT timestamp
                gt_time_text = f"GT: {gt_timestamp:.3f}s"
                (text_width, text_height), _ = cv2.getTextSize(gt_time_text, font, font_scale, thickness)
                x = self.cam_res[1] - text_width - 10
                y = text_height + 10
                cv2.putText(img, gt_time_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                
                # Target projection timestamp
                target_proj_timestamp = gt_timestamp - current_delay
                target_time_text = f"Target: {target_proj_timestamp:.3f}s"
                (text_width, text_height), _ = cv2.getTextSize(target_time_text, font, font_scale, thickness)
                x = self.cam_res[1] - text_width - 10
                y = text_height + 35  # Position below GT timestamp
                cv2.putText(img, target_time_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                # Draw camera direction arrow in top-right corner (updated per timestamp using vicon frames)
                try:
                    # Prefer per-frame camera forward if computed from camera markers
                    used_forward = None
                    if hasattr(self, 'camera_forwards') and getattr(self, 'camera_forwards') is not None:
                        if hasattr(self, 'marker_t'):
                            frame_idx_v = int(np.argmin(np.abs(self.marker_t - gt_timestamp)))
                            if 0 <= frame_idx_v < len(self.camera_forwards):
                                used_forward = np.array(self.camera_forwards[frame_idx_v], dtype=np.float64)

                    if used_forward is None:
                        # fallback to motion delta or optical axis
                        if hasattr(self, 'marker_t') and getattr(self, 'Ts_world_to_system', None) is not None and getattr(self, 'T_syst_to_camera_opt', None) is not None:
                            frame_idx_v = int(np.argmin(np.abs(self.marker_t - gt_timestamp)))

                            def invert_SE3(Tmat):
                                Rm = Tmat[:3, :3]
                                tm = Tmat[:3, 3].reshape(3,1)
                                Rinv = Rm.T
                                tinv = -Rinv.dot(tm)
                                Tinv = np.eye(4, dtype=np.float64)
                                Tinv[:3, :3] = Rinv
                                Tinv[:3, 3] = tinv.ravel()
                                return Tinv

                            def camera_world_pos_for_frame(fid):
                                if not (0 <= fid < len(self.Ts_world_to_system)):
                                    return None
                                T_w2s = np.array(self.Ts_world_to_system[fid], dtype=np.float64)
                                T_s2w = invert_SE3(T_w2s)
                                T_s2c = np.array(self.T_syst_to_camera_opt, dtype=np.float64)
                                T_c2s = invert_SE3(T_s2c)
                                cam_in_s = (T_c2s @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
                                cam_in_w = (T_s2w @ np.append(cam_in_s, 1.0))[:3]
                                return cam_in_w

                            p_curr = camera_world_pos_for_frame(frame_idx_v)
                            p_prev = camera_world_pos_for_frame(max(0, frame_idx_v - 1))
                            if p_curr is not None and p_prev is not None:
                                delta = p_curr - p_prev
                                used_forward = np.array([float(delta[0]), float(delta[1]), 0.0], dtype=np.float64)

                    if used_forward is None:
                        Tsc = np.array(self.T_syst_to_camera_opt, dtype=np.float64)
                        R = Tsc[:3, :3]
                        cam_forward_sys = R.T.dot(np.array([0.0, 0.0, 1.0], dtype=np.float64))
                        used_forward = np.array([float(cam_forward_sys[0]), float(cam_forward_sys[1]), 0.0], dtype=np.float64)

                    vx, vy = float(used_forward[0]), float(used_forward[1])
                    vec_norm = np.hypot(vx, vy)
                    if vec_norm >= 1e-6:
                        vx /= vec_norm
                        vy /= vec_norm
                        arrow_len = max(24, min(self.cam_res) // 12)
                        cx = self.cam_res[1] - 60
                        cy = 50
                        ex = int(cx + vx * arrow_len)
                        ey = int(cy - vy * arrow_len)
                        cv2.arrowedLine(img, (cx, cy), (ex, ey), (80, 80, 80), 2, tipLength=0.3)
                        cv2.putText(img, "Cam dir", (cx - 36, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
                except Exception:
                    pass
                
                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}_t{gt_timestamp:.3f}.png")
                cv2.imwrite(frame_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                if create_video:
                    video_frames.append(img.copy())
                
                frame_count += 1
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"Saved {frame_count} frames...")
        
        print(f"Generated {frame_count} frame visualization images")
        
        # Create video if requested
        if create_video and video_frames:
            sequence_name = self._extract_sequence_name()
            video_path = os.path.join(self._get_output_directory(), f"{sequence_name}_gt_projection_comparison.mp4")
            self._save_video_from_frames(video_frames, video_path, fps=10)
            print(f"Saved comparison video: {video_path}")
        
        return frames_dir

###
    # def optimize_calibration(self, labels_path: str,
    #                      ransac_reproj_err: float = 3.0,
    #                      ransac_iters: int = 300,
    #                      min_points: int = 6):
    #     """
    #     Optimize a single, fixed T_sys->cam using multi-frame labels:
    #     world -> system (per frame, known) -> camera (unknown, fixed).
    #     Robust to length mismatches and OpenCV PnP overload differences.
    #     """
    #     import numpy as np, cv2
    #     from helpers import ViconHelper

    #     print("Optimizing calibration with OpenCV (global multi-frame PnP + refine)...")

    #     # 1) Load labels + interpolate 3D
    #     labeled_points = helpers.read_points_labels(labels_path)
    #     vicon_helper = ViconHelper(
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

    # def optimize_calibration(self, labels_path: str,
    #                      min_points_per_frame: int = 4,
    #                      ransac_reproj_err: float = 3.0,
    #                      ransac_iters: int = 300,
    #                      pnp_reproj_clip: float = 5.0,
    #                      handeye_method: int = None):
    #     """
    #     Robust Hand–Eye using OpenCV calibrateHandEye with:
    #     • per-frame PnP (RANSAC) + LM refinement
    #     • outlier frame rejection by per-frame reprojection error
    #     • returns single fixed T_sys->cam
    #     """
    #     import numpy as np, cv2
    #     import helpers
    #     from helpers import ViconHelper

    #     print("Optimizing calibration with OpenCV calibrateHandEye() ...")

    #     # -------- small helpers --------
    #     def _cv_distortion(D):
    #         if D is None: return None
    #         D_flat = np.asarray(D, dtype=np.float64).ravel()
    #         return None if D_flat.size == 0 else np.ascontiguousarray(D_flat, dtype=np.float64)

    #     def to_SE3(R, t):
    #         T = np.eye(4, dtype=np.float64)
    #         T[:3, :3] = R; T[:3, 3] = t.reshape(3)
    #         return T

    #     def invert_Rt(R, t):
    #         Rinv = R.T
    #         tinv = -Rinv @ t.reshape(3, 1)
    #         return Rinv, tinv

    #     def per_frame_reproj_err(R, t, P3, P2, K, D):
    #         rvec, _ = cv2.Rodrigues(R)
    #         proj, _ = cv2.projectPoints(P3, rvec, t.reshape(3,1), K, D)
    #         return np.linalg.norm(proj.reshape(-1,2) - P2, axis=1)

    #     K_cv = np.ascontiguousarray(self.K, dtype=np.float64)
    #     D_cv = _cv_distortion(self.D)

    #     # -------- 1) Load labels + interpolate VICON --------
    #     labeled_points = helpers.read_points_labels(labels_path)
    #     vicon_helper = ViconHelper(
    #         self.marker_t, self.points_3d, self.delay,
    #         self.c3d_data.frame_count, self.c3d_data.point_rate,
    #         self.c3d_data.point_labels, True, True,
    #         user_camera_markers=getattr(self, "camera_markers", None)
    #     )
    #     vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)

    #     n_frames = min(len(labeled_points.get('points', [])),
    #                 len(vicon_points.get('points', [])),
    #                 len(vicon_points.get('frame_ids', [])))

    #     # Hand–Eye inputs (aligned by index)
    #     R_gripper2base, t_gripper2base = [], []   # system -> world   (g->b)
    #     R_target2cam,   t_target2cam   = [], []   # world  -> camera  (t->c)

    #     kept = 0
    #     skip_few, skip_pnp, skip_higherr, skip_bounds = 0, 0, 0, 0

    #     # -------- 2) Per-frame PnP (robust) --------
    #     for idx in range(n_frames):
    #         dvs_frame = labeled_points['points'][idx] or {}
    #         w_frame   = vicon_points['points'][idx]   or {}
    #         if not dvs_frame or not w_frame:
    #             skip_few += 1
    #             continue

    #         # Collect 2D-3D correspondences in *world* frame
    #         W3, I2 = [], []
    #         for lab, pix in dvs_frame.items():
    #             if lab not in w_frame: continue
    #             W = np.asarray(w_frame[lab], dtype=np.float64)
    #             if W is None or np.any(~np.isfinite(W)): continue
    #             u, v = float(pix['x']), float(pix['y'])
    #             if not (np.isfinite(u) and np.isfinite(v)): continue
    #             W3.append(W); I2.append([u, v])

    #         if len(W3) < max(4, min_points_per_frame):
    #             skip_few += 1
    #             continue

    #         W3 = np.ascontiguousarray(np.asarray(W3, dtype=np.float64))
    #         I2 = np.ascontiguousarray(np.asarray(I2, dtype=np.float64))

    #         # RANSAC PnP (use P3P if exactly 4 pts, else EPNP)
    #         try:
    #             if len(W3) == 4:
    #                 ok, rvec, tvec, inl = cv2.solvePnPRansac(
    #                     W3, I2, K_cv, D_cv,
    #                     iterationsCount=int(ransac_iters),
    #                     reprojectionError=float(ransac_reproj_err),
    #                     flags=cv2.SOLVEPNP_P3P
    #                 )
    #                 if not ok or inl is None or len(inl) < 4:
    #                     ok, rvec, tvec, inl = cv2.solvePnPRansac(
    #                         W3, I2, K_cv, D_cv,
    #                         iterationsCount=int(ransac_iters),
    #                         reprojectionError=float(ransac_reproj_err),
    #                         flags=cv2.SOLVEPNP_EPNP
    #                     )
    #             else:
    #                 ok, rvec, tvec, inl = cv2.solvePnPRansac(
    #                     W3, I2, K_cv, D_cv,
    #                     iterationsCount=int(ransac_iters),
    #                     reprojectionError=float(ransac_reproj_err),
    #                     flags=cv2.SOLVEPNP_EPNP
    #                 )
    #         except cv2.error:
    #             ok = False

    #         if (not ok) and len(W3) >= 6:
    #             ok, rvec, tvec = cv2.solvePnP(W3, I2, K_cv, D_cv, flags=cv2.SOLVEPNP_ITERATIVE)

    #         if not ok:
    #             skip_pnp += 1
    #             continue

    #         # LM refine
    #         try:
    #             rvec, tvec = cv2.solvePnPRefineLM(W3, I2, K_cv, D_cv, rvec, tvec)
    #         except cv2.error:
    #             pass

    #         R_wc, _ = cv2.Rodrigues(rvec)               # world -> camera
    #         t_wc = tvec.reshape(3,1)

    #         # Per-frame reprojection filter
    #         e = per_frame_reproj_err(R_wc, t_wc, W3, I2, K_cv, D_cv)
    #         if np.median(e) > pnp_reproj_clip:
    #             skip_higherr += 1
    #             continue

    #         # Add target->cam
    #         R_target2cam.append(R_wc.astype(np.float64))
    #         t_target2cam.append(t_wc.astype(np.float64))

    #         # Fetch world->system (base->gripper) for same frame, then invert to g->b
    #         frame_id = int(vicon_points['frame_ids'][idx])
    #         if not (0 <= frame_id < len(self.Ts_world_to_system)):
    #             skip_bounds += 1
    #             R_target2cam.pop(); t_target2cam.pop()
    #             continue

    #         T_w2s = self.Ts_world_to_system[frame_id]
    #         R_ws = T_w2s[:3, :3].astype(np.float64)
    #         t_ws = T_w2s[:3, 3].reshape(3,1).astype(np.float64)

    #         # system->world is inverse of world->system
    #         R_sw, t_sw = invert_Rt(R_ws, t_ws)          # g->b in OpenCV naming
    #         R_gripper2base.append(R_sw)
    #         t_gripper2base.append(t_sw)

    #         kept += 1

    #     print(f"Frames kept: {kept}  (few={skip_few}, pnp_fail={skip_pnp}, high_err={skip_higherr}, bounds={skip_bounds})")
    #     if kept < 3:
    #         raise RuntimeError("Not enough valid frames for hand–eye (need ≥3 with diverse motion).")

    #     # -------- 3) Hand–Eye calibration --------
    #     if handeye_method is None:
    #         handeye_method = cv2.CALIB_HAND_EYE_DANIILIDIS  # good default

    #     # OpenCV returns camera->gripper (cam->system). We need system->camera, so invert.
    #     R_c2s, t_c2s = cv2.calibrateHandEye(
    #         R_gripper2base, t_gripper2base,   # g->b  (system->world)
    #         R_target2cam,   t_target2cam,     # t->c  (world->camera)
    #         method=handeye_method
    #     )

    #     R_s2c, t_s2c = invert_Rt(R_c2s, t_c2s)          # system->camera
    #     T_sc = to_SE3(R_s2c, t_s2c)

    #     # -------- 4) Global scoring (reprojection over ALL valid correspondences) --------
    #     sys3, img2 = [], []
    #     for idx in range(n_frames):
    #         dvs_frame = labeled_points['points'][idx] or {}
    #         w_frame   = vicon_points['points'][idx] or {}
    #         if not dvs_frame or not w_frame:
    #             continue
    #         frame_id = int(vicon_points['frame_ids'][idx])
    #         if not (0 <= frame_id < len(self.Ts_world_to_system)):
    #             continue
    #         T_w2s = self.Ts_world_to_system[frame_id]
    #         for lab, pix in dvs_frame.items():
    #             if lab not in w_frame: continue
    #             W = np.asarray(w_frame[lab], dtype=np.float64)
    #             if W is None or np.any(~np.isfinite(W)): continue
    #             p_sys = (T_w2s @ np.append(W, 1.0))[:3]
    #             sys3.append(p_sys)
    #             img2.append([float(pix['x']), float(pix['y'])])

    #     P3 = np.ascontiguousarray(np.asarray(sys3, dtype=np.float64).reshape(-1,3))
    #     P2 = np.ascontiguousarray(np.asarray(img2, dtype=np.float64).reshape(-1,2))
    #     rvec_sc, _ = cv2.Rodrigues(T_sc[:3, :3])
    #     tvec_sc    = T_sc[:3, 3].reshape(3,1)
    #     proj, _ = cv2.projectPoints(P3, rvec_sc, tvec_sc, K_cv, D_cv)
    #     err = np.linalg.norm(proj.reshape(-1,2) - P2, axis=1)
    #     print(f"[Hand–Eye] global reproj -> mean={err.mean():.3f}px, median={np.median(err):.3f}px, 95%={np.percentile(err,95):.3f}px, n={len(err)}")

    #     self.T_syst_to_camera_opt = T_sc
    #     print("Estimated T_sys->cam (from calibrateHandEye):\n", T_sc)
    #     return T_sc

    # def reprojection_error(params, Ps, pc, K, dist):
    #     """
    #     params: 6 dimensions (rotation_vector [3], translation [3]) Optimization target
    #     Ps: (N, 3) 3D points (system coordinates)
    #     pc: (N, 2) 2D points (image coordinates)
    #     K: (3, 3) camera matrix
    #     dist: (5,) distortion parameters (OpenCV format)
    #     """
    #     rvec = params[:3]
    #     tvec = params[3:6]

    #     # Project 3D system points to 2D image points
    #     projected_points, _ = cv2.projectPoints(Ps, rvec, tvec, K, dist)
    #     projected_points = projected_points.reshape(-1, 2)

    #     return (projected_points - pc).ravel()  # Flatten and return


    # def estimate_Tstoc(Ps, pc, K, dist, init_params=None):
    #     """
    #     Ps: (N, 3) 3D points (system coordinates)
    #     pc: (N, 2) 2D points (image coordinates)
    #     K: (3, 3) camera matrix
    #     dist: (5,) distortion parameters
    #     return: (4, 4) system→camera coordinate transformation matrix Tstoc
    #     """

    #     # Initial values: zero rotation, zero translation
    #     if init_params is None:
    #         # If no initial parameters are provided, set them to zero
    #         init_params = np.zeros(6)

    #     # Minimize (choose LM method, etc.)
    #     res = least_squares(
    #         helpers.reprojection_error,
    #         init_params,
    #         args=(Ps, pc, K, dist),
    #         method='lm'  # Levenberg-Marquardt
    #     )

    #     rvec_opt = res.x[:3]
    #     tvec_opt = res.x[3:6]
    #     R_opt, _ = cv2.Rodrigues(rvec_opt)

    #     # Construct homogeneous transformation matrix
    #     T = np.eye(4)
    #     T[:3, :3] = R_opt
    #     T[:3, 3] = tvec_opt

    #     return T

    def optimize_calibration(self, labels_path: str,
                         min_points_per_frame: int = 4,
                         ransac_reproj_err: float = 3.0,
                         ransac_iters: int = 300,
                         pnp_reproj_clip: float = 5.0):
        """
        Enhanced calibration using hybrid approach:
        • Joint delay + transformation optimization (like the reference code)
        • Fallback to complex pipeline if simple approach fails
        """

        import numpy as np, cv2
        import helpers
        from helpers import ViconHelper
        from scipy.spatial.transform import Rotation
        from scipy.optimize import minimize_scalar

        print("🔄 HYBRID CALIBRATION APPROACH:")
        print("   1. Try simple joint delay+transform optimization (like reference code)")
        print("   2. Fallback to complex pipeline if needed")

        # Load correspondences
        labeled_points = helpers.read_points_labels(labels_path)

        def setup_simple_correspondences(delay_test):
            """Setup correspondences for a given delay (simplified approach)."""
            
            # Build simple world->camera correspondences (no complex transforms)
            world_points = []
            image_points = []
            
            for frame_idx, dvs_frame in enumerate(labeled_points['points']):
                if not dvs_frame:
                    continue
                    
                # Compute VICON time for this frame with test delay
                dvs_time = labeled_points['timestamps'][frame_idx]
                vicon_time = dvs_time + delay_test
                
                # Get VICON frame index (simple linear interpolation)
                vicon_frame_float = (vicon_time - self.marker_t[0]) * self.c3d_data.point_rate
                vicon_frame_idx = int(round(vicon_frame_float))
                
                # Check bounds
                if vicon_frame_idx < 0 or vicon_frame_idx >= len(self.marker_t):
                    continue
                    
                for lab, pix in dvs_frame.items():
                    # Get 3D VICON position directly from C3D data
                    if lab not in self.c3d_data.point_labels:
                        continue
                        
                    marker_idx = self.c3d_data.point_labels.index(lab)
                    W = self.points_3d[vicon_frame_idx, marker_idx, :]
                    
                    # Check for valid point
                    if np.any(~np.isfinite(W)) or np.allclose(W, 0):
                        continue
                        
                    world_points.append(W)
                    image_points.append([pix['x'], pix['y']])

            world_pts = np.array(world_points)
            image_pts = np.array(image_points)
            
            # Apply undistortion to image points for better calibration accuracy
            if hasattr(self, 'K') and hasattr(self, 'D') and self.D is not None and len(image_pts) > 0:
                try:
                    # Use the helper method from ViconProjector for consistency
                    temp_projector = helpers.ViconProjector(None, None, None, self.K, self.D)
                    image_pts = temp_projector.undistort_image_points(image_pts)
                except Exception as e:
                    print(f"⚠️ Could not undistort image points in simple setup: {e}")
            
            return world_pts, image_pts

        def simple_calibration_error(delay_test):
            """Error function for joint delay+transform optimization."""
            if delay_test < -1.0 or delay_test > 1.0:  # Reasonable delay bounds
                return 1e6
                
            try:
                world_pts, image_pts = setup_simple_correspondences(delay_test)
                
                if len(world_pts) < 6:  # Need minimum points for PnP
                    return 1e6

                K_cv = np.ascontiguousarray(self.K, dtype=np.float64)
                D_cv = None if self.D is None else np.ascontiguousarray(self.D, dtype=np.float64)

                # Direct OpenCV PnP (like reference code)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    world_pts, image_pts, K_cv, D_cv,
                    iterationsCount=300,
                    reprojectionError=5.0,
                    flags=cv2.SOLVEPNP_EPNP
                )
                
                if not success or len(inliers) < 4:
                    return 1e6

                # Refine with LM
                rvec, tvec = cv2.solvePnPRefineLM(
                    world_pts[inliers[:,0]], image_pts[inliers[:,0]],
                    K_cv, D_cv, rvec, tvec
                )

                # Measure reprojection error
                proj_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, K_cv, D_cv)
                proj_pts = proj_pts.reshape(-1, 2)
                errors = np.linalg.norm(proj_pts - image_pts, axis=1)
                mean_error = np.mean(errors)
                
                return mean_error
                
            except Exception as e:
                print(f"Error in simple calibration for delay {delay_test:.6f}: {e}")
                return 1e6

        print("\n🎯 PHASE 1: Simple Joint Optimization (like reference code)")
        
        # Optimize delay around current estimate
        delay_bounds = (self.delay - 0.2, self.delay + 0.2)
        print(f"   Optimizing delay in range: {delay_bounds}")
        
        try:
            result = minimize_scalar(
                simple_calibration_error,
                bounds=delay_bounds,
                method='bounded',
                options={'xatol': 1e-6}
            )
            
            if result.success and result.fun < 15.0:  # If we get good results
                optimal_delay = result.x
                final_error = result.fun
                
                print(f"✅ Simple approach succeeded!")
                print(f"   Optimal delay: {optimal_delay:.6f}s (was {self.delay:.6f}s)")
                print(f"   Final error: {final_error:.2f}px")
                
                # Get final transformation with optimal delay
                self.delay = optimal_delay
                world_pts, image_pts = setup_simple_correspondences(optimal_delay)
                
                K_cv = np.ascontiguousarray(self.K, dtype=np.float64)
                D_cv = None if self.D is None else np.ascontiguousarray(self.D, dtype=np.float64)
                
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    world_pts, image_pts, K_cv, D_cv,
                    iterationsCount=300, reprojectionError=5.0
                )
                rvec, tvec = cv2.solvePnPRefineLM(
                    world_pts[inliers[:,0]], image_pts[inliers[:,0]],
                    K_cv, D_cv, rvec, tvec
                )
                
                # Build transformation matrix
                R_opt, _ = cv2.Rodrigues(rvec)
                T_sc = np.eye(4, dtype=np.float64)
                T_sc[:3, :3] = R_opt.astype(np.float64)
                T_sc[:3, 3] = tvec.reshape(3).astype(np.float64)
                
                self.T_syst_to_camera_opt = T_sc
                print("📈 Final Simple Calibration Results:")
                print(f"   Delay: {self.delay:.6f}s")
                print(f"   Reprojection error: {final_error:.2f}px")
                print(f"   Inliers: {len(inliers)}/{len(world_pts)}")
                print("   Transformation matrix:")
                print(T_sc)
                
                return T_sc
            else:
                print(f"❌ Simple approach failed: error={result.fun:.2f}px")
                print("   Falling back to complex pipeline...")
                
        except Exception as e:
            print(f"❌ Simple approach crashed: {e}")
            print("   Falling back to complex pipeline...")

        print("\n🔄 PHASE 2: Complex Pipeline (Fallback)")
        # Continue with your original complex approach
        return self._optimize_calibration_complex(labels_path, min_points_per_frame, 
                                                 ransac_reproj_err, ransac_iters, pnp_reproj_clip)

    def _optimize_calibration_complex(self, labels_path: str,
                                    min_points_per_frame: int = 4,
                                    ransac_reproj_err: float = 3.0,
                                    ransac_iters: int = 300,
                                    pnp_reproj_clip: float = 5.0):
        """
        Original complex pipeline approach with world->system->camera transforms.
        """
        
        import numpy as np, cv2
        import helpers
        from helpers import ViconHelper
        from scipy.spatial.transform import Rotation

        print("Running complex pipeline calibration ...")

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
        # Following the pipeline: p(estimated) = camera_distortion * Tsystem_to_camera * Tvicon_to_system * 3d_marker_pose
        # We need to store: (vicon_points_3D, corresponding_frame_transforms, image_points_2D)
        vicon_3d_points = []     # 3D points in vicon/world coordinates  
        frame_transforms = []    # T_vicon_to_system for each point (Tvicon_to_system)
        image_points = []        # 2D labeled points (p(GT))

        for idx in range(n_frames):
            dvs_frame = labeled_points['points'][idx] or {}
            w_frame   = vicon_points['points'][idx] or {}
            if not dvs_frame or not w_frame:
                continue

            frame_id = int(vicon_points['frame_ids'][idx])
            if not (0 <= frame_id < len(self.Ts_world_to_system)):
                continue

            T_w2s = self.Ts_world_to_system[frame_id]  # This is Tvicon_to_system

            for lab, pix in dvs_frame.items():
                if lab not in w_frame:
                    continue
                W = np.asarray(w_frame[lab], dtype=np.float64)  # 3D vicon/world point
                if np.any(~np.isfinite(W)):
                    continue

                # Store the 3D vicon point, its transform, and corresponding 2D image point
                vicon_3d_points.append(W)
                frame_transforms.append(T_w2s)
                image_points.append([pix['x'], pix['y']])

        Pw = np.asarray(vicon_3d_points, dtype=np.float64)    # 3D vicon/world points  
        Tw2s_list = np.asarray(frame_transforms, dtype=np.float64)  # per-point transforms
        pc = np.asarray(image_points, dtype=np.float64)       # 2D labeled points

        print(f"Global correspondences: {len(Pw)}")
        
        # Apply undistortion to labeled image points for better calibration accuracy
        if hasattr(self, 'K') and hasattr(self, 'D') and self.D is not None:
            print(f"🔧 Applying undistortion to {len(pc)} labeled image points...")
            try:
                # Use the helper method from ViconProjector for consistency
                temp_projector = helpers.ViconProjector(None, None, None, self.K, self.D)
                pc_undistorted = temp_projector.undistort_image_points(pc)
                pc = pc_undistorted
                print(f"✅ Successfully undistorted {len(pc)} image points for calibration")
            except Exception as e:
                print(f"⚠️ Could not undistort image points: {e}, using original points")
        else:
            print("ℹ️ No distortion coefficients available, using original image points")
        
        # Add diagnostic information about the system
        print("\n🔍 DIAGNOSTIC INFORMATION:")
        print(f"   Camera setup: {getattr(self, 'camera_setup', 'Unknown')}")
        print(f"   Number of T_world_to_system transforms: {len(self.Ts_world_to_system)}")
        print(f"   T_world_to_system[0] sample:\n{self.Ts_world_to_system[0]}")
        print(f"   Is camera static or dynamic? {'STATIC' if len(set(tuple(T.flatten()) for T in self.Ts_world_to_system[:5])) == 1 else 'DYNAMIC'}")
        if hasattr(self, 'T_syst_to_camera_opt'):
            print(f"   Current T_syst_to_camera_opt:\n{self.T_syst_to_camera_opt}")
        
        # Deep coordinate system analysis
        if len(Pw) > 0:
            print("\n🔬 COORDINATE SYSTEM ANALYSIS:")
            # Analyze first few correspondences
            for i in range(min(3, len(Pw))):
                p_vicon = Pw[i]
                T_w2s = Tw2s_list[i] 
                p_img = pc[i]
                
                # Transform VICON -> system coordinates
                p_sys_homog = T_w2s @ np.append(p_vicon, 1.0)
                p_sys = p_sys_homog[:3]
                
                print(f"   Point {i}:")
                print(f"     VICON: {p_vicon} (magnitude: {np.linalg.norm(p_vicon):.1f}mm)")
                print(f"     System: {p_sys} (magnitude: {np.linalg.norm(p_sys):.1f}mm)")
                print(f"     Image GT: {p_img}")
                
                # Check if coordinates are reasonable for a typical indoor scene
                if np.linalg.norm(p_sys) > 10000:  # > 10 meters
                    print(f"     ⚠️  System coordinates seem very large (>{np.linalg.norm(p_sys)/1000:.1f}m)")
                if np.linalg.norm(p_vicon) > 10000:  # > 10 meters  
                    print(f"     ⚠️  VICON coordinates seem very large (>{np.linalg.norm(p_vicon)/1000:.1f}m)")
            
            # Check transformation matrix properties
            print(f"\n   Transform matrix analysis:")
            T_sample = Tw2s_list[0]
            print(f"     Translation component: {T_sample[:3, 3]} (magnitude: {np.linalg.norm(T_sample[:3, 3]):.1f}mm)")
            det_R = np.linalg.det(T_sample[:3, :3])
            print(f"     Rotation determinant: {det_R:.6f}")
            if abs(det_R - 1.0) > 0.01:
                print(f"     ⚠️  Rotation matrix determinant != 1, possible scaling/reflection!")
        
        # SCALE CORRECTION: Detect and fix coordinate system scale mismatch
        if len(Pw) > 0:
            print(f"\n🔧 SCALE CORRECTION ANALYSIS:")
            
            # Compute scale factor from a sample of correspondences
            scale_factors = []
            sample_size = min(20, len(Pw))
            
            for i in range(sample_size):
                p_vicon = Pw[i]
                T_w2s = Tw2s_list[i]
                p_sys = (T_w2s @ np.append(p_vicon, 1.0))[:3]
                
                vicon_mag = np.linalg.norm(p_vicon)
                sys_mag = np.linalg.norm(p_sys)
                
                if vicon_mag > 100:  # Only consider points far enough from origin
                    scale_factors.append(sys_mag / vicon_mag)
            
            if len(scale_factors) > 0:
                avg_scale_factor = np.mean(scale_factors)
                std_scale_factor = np.std(scale_factors)
                
                print(f"   Detected scale factor: {avg_scale_factor:.3f} ± {std_scale_factor:.3f}")
                print(f"   Scale factor range: [{np.min(scale_factors):.3f}, {np.max(scale_factors):.3f}]")
                
                # Apply scale correction if factor is significantly different from 1.0
                if abs(avg_scale_factor - 1.0) > 0.3:  # More than 30% scale difference
                    print(f"   🎯 APPLYING SCALE CORRECTION: {avg_scale_factor:.3f}")
                    
                    # Apply inverse scale to transforms to normalize coordinates
                    inv_scale = 1.0 / avg_scale_factor
                    
                    # Create scale correction matrix
                    scale_correction = np.eye(4)
                    scale_correction[:3, :3] *= inv_scale
                    
                    # Apply scale correction to all transforms
                    Tw2s_list_corrected = []
                    for T_w2s in Tw2s_list:
                        # Apply scale correction: T_corrected = scale_correction @ T_w2s
                        T_corrected = scale_correction @ T_w2s
                        Tw2s_list_corrected.append(T_corrected)
                    
                    # Update the transform list
                    Tw2s_list = np.array(Tw2s_list_corrected)
                    
                    # Verify the correction worked
                    print("   ✓ Scale correction applied. Verifying...")
                    p_vicon_test = Pw[0]
                    T_w2s_test = Tw2s_list[0]
                    p_sys_test = (T_w2s_test @ np.append(p_vicon_test, 1.0))[:3]
                    
                    new_scale_factor = np.linalg.norm(p_sys_test) / np.linalg.norm(p_vicon_test)
                    print(f"   New scale factor (sample): {new_scale_factor:.3f}")
                    
                    if abs(new_scale_factor - 1.0) < 0.2:
                        print("   ✅ Scale correction successful!")
                    else:
                        print("   ⚠️  Scale correction may not be fully effective")
                else:
                    print(f"   ✓ Scale factor close to 1.0, no correction needed")
        
        # Sample correspondence analysis
        if len(Pw) > 0:
            print(f"\n📊 CORRESPONDENCE SAMPLE ANALYSIS (after scale correction):")
            for i in range(min(3, len(Pw))):
                print(f"   Point {i}: 3D_world={Pw[i]} -> 2D_image={pc[i]}")
                T_sample = Tw2s_list[i]
                p_sys = (T_sample @ np.append(Pw[i], 1.0))[:3]
                print(f"             3D_world -> 3D_system: {p_sys}")
        print()

        # ------------------ initial guess ------------------
        # Use manual calibration estimate if available, otherwise average of per-frame estimates
        if hasattr(self, 'T_syst_to_camera_opt') and self.T_syst_to_camera_opt is not None:
            # Check if we have a meaningful transformation (not just identity matrix)
            if not np.allclose(self.T_syst_to_camera_opt, np.eye(4), atol=1e-6):
                R_init = self.T_syst_to_camera_opt[:3, :3]
                t_init = self.T_syst_to_camera_opt[:3, 3]
                print("Using manual calibration estimate as initial guess")
            else:
                # Fallback to per-frame average if T_syst_to_camera_opt is identity
                Rs = np.stack(R_wc_list, axis=0)
                ts = np.stack([t.reshape(3) for t in t_wc_list], axis=0)
                R_init = Rotation.from_matrix(Rs).mean().as_matrix()
                t_init = np.mean(ts, axis=0)
                print("T_syst_to_camera_opt is identity - using per-frame average as initial guess")
        else:
            # Fallback to average of per-frame estimates
            Rs = np.stack(R_wc_list, axis=0)
            ts = np.stack([t.reshape(3) for t in t_wc_list], axis=0)
            R_init = Rotation.from_matrix(Rs).mean().as_matrix()
            t_init = np.mean(ts, axis=0)
            print("No T_syst_to_camera_opt available - using per-frame average as initial guess")

        rvec_init = Rotation.from_matrix(R_init).as_rotvec()
        init_params = np.hstack([rvec_init, t_init])

        print(f"Initial rotation (degrees): {np.rad2deg(rvec_init)}")
        print(f"Initial translation: {t_init}")
        
        # Add timing offset optimization if errors are still high after scale correction
        optimize_timing = True  # Enable temporal offset optimization
        
        if optimize_timing:
            # Extend parameters to include a temporal offset
            init_params_with_timing = np.hstack([init_params, [0.0]])  # Add 0.0 second initial offset
            print(f"🕐 Including temporal offset optimization (initial offset: 0.0s)")
        else:
            init_params_with_timing = init_params

        # ------------------ Global optimization using correct projection pipeline ------------------
        print("Running global optimization following pipeline: camera_distortion * Tsystem_to_camera * Tvicon_to_system * 3d_marker_pose ...")

        # Build reprojection residual function following the full pipeline
        def reprojection_residual_pipeline(params, Pw_in, Tw2s_in, pc_in, K_in, D_in):
            """
            Enhanced implementation with optional temporal offset optimization:
            p(estimated) = camera_distortion * Tsystem_to_camera * Tvicon_to_system * 3d_marker_pose
            
            params: [rvec(3), tvec(3), temporal_offset(1)] - parameters for Tsystem_to_camera + optional timing
            Pw_in: (N,3) - 3D marker positions in vicon/world coordinates
            Tw2s_in: (N,4,4) - per-point Tvicon_to_system transforms  
            pc_in: (N,2) - 2D labeled image coordinates p(GT)
            """
            if len(params) == 7:  # With temporal offset
                rvec_s2c = params[:3].astype(np.float64)  
                tvec_s2c = params[3:6].astype(np.float64)  
                temporal_offset = params[6]  # Time offset in seconds
                use_temporal_offset = True
            else:  # Original 6-parameter version
                rvec_s2c = params[:3].astype(np.float64)  
                tvec_s2c = params[3:6].astype(np.float64)  
                temporal_offset = 0.0
                use_temporal_offset = False
            
            residuals = []
            points_behind_camera = 0
            large_errors = 0
            
            for i in range(len(Pw_in)):
                try:
                    # Step 1: 3d_marker_pose (input)
                    pw = Pw_in[i]  # 3D point in vicon coordinates
                    
                    # Step 2: Apply temporal offset if optimizing timing
                    if use_temporal_offset and abs(temporal_offset) > 0.001:  # Only if significant offset
                        # For temporal offset, we'd need to interpolate transforms
                        # For now, use a simple approximation by selecting different transform indices
                        frame_offset = int(temporal_offset * 100)  # Assuming 100Hz VICON
                        adjusted_idx = max(0, min(i + frame_offset, len(Tw2s_in) - 1))
                        T_w2s = Tw2s_in[adjusted_idx]
                    else:
                        T_w2s = Tw2s_in[i]
                    
                    # Step 3: Tvicon_to_system * 3d_marker_pose
                    ps_homog = T_w2s @ np.append(pw, 1.0)  # transform to system coordinates
                    ps = ps_homog[:3]  # 3D point in system coordinates
                    
                    # Step 4: Tsystem_to_camera * (result from step 3)  
                    # Use cv2.projectPoints which handles Tsystem_to_camera + camera_distortion
                    proj_pt, _ = cv2.projectPoints(
                        ps.reshape(1, 3), rvec_s2c, tvec_s2c.reshape(3, 1), K_in, D_in
                    )
                    p_estimated = proj_pt.reshape(2)
                    
                    # Step 5: Compute residual = p(GT) - p(estimated)
                    p_gt = pc_in[i]
                    res_x = p_gt[0] - p_estimated[0]  # x residual
                    res_y = p_gt[1] - p_estimated[1]  # y residual
                    
                    residuals.append(res_x)
                    residuals.append(res_y)
                    
                    # Debug: Check for common issues
                    error_magnitude = np.sqrt(res_x**2 + res_y**2)
                    if error_magnitude > 100:  # Very large error
                        large_errors += 1
                        
                    # Check if point is behind camera (in camera coordinates)
                    R_s2c, _ = cv2.Rodrigues(rvec_s2c)
                    p_camera = R_s2c @ ps + tvec_s2c
                    if p_camera[2] < 0:  # Behind camera
                        points_behind_camera += 1
                    
                except Exception as e:
                    # If any step fails, add large residuals to steer optimizer away
                    residuals.append(1000.0)  # large x residual
                    residuals.append(1000.0)  # large y residual
                    
            # Debug output every few iterations
            if hasattr(reprojection_residual_pipeline, 'call_count'):
                reprojection_residual_pipeline.call_count += 1
            else:
                reprojection_residual_pipeline.call_count = 1
                
            if reprojection_residual_pipeline.call_count % 5 == 0:  # Every 5th call
                mean_residual = np.mean(np.abs(residuals))
                if use_temporal_offset:
                    print(f"     [Iter {reprojection_residual_pipeline.call_count:3d}] Mean residual: {mean_residual:.2f}px, "
                          f"Behind camera: {points_behind_camera}/{len(Pw_in)}, Large errors: {large_errors}, "
                          f"Time offset: {temporal_offset:.3f}s")
                else:
                    print(f"     [Iter {reprojection_residual_pipeline.call_count:3d}] Mean residual: {mean_residual:.2f}px, "
                          f"Behind camera: {points_behind_camera}/{len(Pw_in)}, Large errors: {large_errors}")
                    
            return np.array(residuals, dtype=np.float64)

        # compute initial reprojection error using correct pipeline  
        try:
            initial_residuals = reprojection_residual_pipeline(init_params_with_timing, Pw, Tw2s_list, pc, K_cv, D_cv)
            initial_errors = initial_residuals.reshape(-1, 2)  # reshape to (N, 2) for per-point errors
            err0 = np.linalg.norm(initial_errors, axis=1)  # compute per-point error magnitudes
            print(f"Initial reprojection error (after scale correction): mean={err0.mean():.3f}px, median={np.median(err0):.3f}px, n={len(err0)}")
        except Exception as e:
            print(f"Initial projection failed ({e}) - continuing to optimization")

        # Use robust loss to reduce sensitivity to outliers
        try:
            from scipy.optimize import least_squares
            
            # Try multiple optimization strategies to escape local minima
            best_result = None
            best_cost = float('inf')
            
            optimization_strategies = [
                # Strategy 1: Standard approach with tighter tolerances
                {
                    'method': 'trf',
                    'loss': 'soft_l1',
                    'ftol': 1e-12,
                    'xtol': 1e-12, 
                    'gtol': 1e-12,
                    'max_nfev': 5000,
                    'name': 'Standard TRF'
                },
                # Strategy 2: More robust loss function for outliers
                {
                    'method': 'trf',
                    'loss': 'huber',
                    'ftol': 1e-10,
                    'xtol': 1e-10,
                    'gtol': 1e-10, 
                    'max_nfev': 3000,
                    'name': 'Huber Loss'
                },
                # Strategy 3: Different solver algorithm
                {
                    'method': 'lm',
                    'ftol': 1e-10,
                    'xtol': 1e-10,
                    'gtol': 1e-10,
                    'max_nfev': 2000,
                    'name': 'Levenberg-Marquardt'
                }
            ]
            
            print(f"🔧 Trying multiple optimization strategies to break through {12}px barrier...")
            
            for i, strategy in enumerate(optimization_strategies):
                try:
                    print(f"   Strategy {i+1}: {strategy['name']}")
                    
                    # Remove name from strategy dict for passing to least_squares
                    opt_params = {k: v for k, v in strategy.items() if k != 'name'}
                    
                    lsq = least_squares(
                        reprojection_residual_pipeline,
                        init_params_with_timing,
                        args=(Pw, Tw2s_list, pc, K_cv, D_cv),
                        verbose=1,
                        **opt_params
                    )
                    
                    print(f"   Result: cost={lsq.cost:.1f}, success={lsq.success}, optimality={lsq.optimality:.2e}")
                    
                    if lsq.success and lsq.cost < best_cost:
                        best_result = lsq
                        best_cost = lsq.cost
                        print(f"   ✅ New best result found!")
                        
                except Exception as e:
                    print(f"   ❌ Strategy failed: {e}")
                    continue
            
            if best_result is not None:
                lsq = best_result
                print(f"\n🎯 Best optimization result: cost={lsq.cost:.1f}, optimality={lsq.optimality:.2e}")
            else:
                print(f"\n⚠️ All strategies failed, using original approach")
                lsq = least_squares(
                    reprojection_residual_pipeline,
                    init_params_with_timing,
                    args=(Pw, Tw2s_list, pc, K_cv, D_cv),
                    method='trf',
                    loss='soft_l1',
                    ftol=1e-8, xtol=1e-8, gtol=1e-8,
                    max_nfev=2000,
                    verbose=1
                )

            rvec_opt = lsq.x[:3]
            tvec_opt = lsq.x[3:6]
            
            # Extract temporal offset if optimized
            if len(lsq.x) == 7:
                temporal_offset_opt = lsq.x[6]
                print(f"Optimized temporal offset: {temporal_offset_opt:.4f}s")
            else:
                temporal_offset_opt = 0.0
                
            R_opt, _ = cv2.Rodrigues(rvec_opt)
            T_sc = np.eye(4, dtype=np.float64)
            T_sc[:3, :3] = R_opt.astype(np.float64)
            T_sc[:3, 3] = tvec_opt.astype(np.float64)

            print(f"Optimization completed: success={lsq.success}, cost={lsq.cost:.6f}, optimality={lsq.optimality:.2e}")

        except Exception as e:
            print(f"Pipeline optimization failed ({e}), falling back to simplified approach")
            # Fallback to the old approach if the new one fails
            # Transform all points to system coordinates first, then optimize
            Ps_fallback = []
            for i in range(len(Pw)):
                pw = Pw[i]
                T_w2s = Tw2s_list[i]  
                ps = (T_w2s @ np.append(pw, 1.0))[:3]
                Ps_fallback.append(ps)
            Ps_fallback = np.asarray(Ps_fallback, dtype=np.float64)
            T_sc = helpers.estimate_Tstoc(Ps_fallback, pc, K_cv, D_cv, init_params)

        # ------------------ diagnostics using correct pipeline ------------------
        try:
            final_residuals = reprojection_residual_pipeline(
                np.hstack([Rotation.from_matrix(T_sc[:3, :3]).as_rotvec(), T_sc[:3, 3]]),
                Pw, Tw2s_list, pc, K_cv, D_cv
            )
            final_errors = final_residuals.reshape(-1, 2)
            err_final = np.linalg.norm(final_errors, axis=1)

            print(f"[PIPELINE OPT] Final reprojection error: mean={err_final.mean():.3f}px, "
                  f"median={np.median(err_final):.3f}px, 95%={np.percentile(err_final,95):.3f}px, n={len(err_final)}")
            
            # Additional diagnostics: show error improvement
            if 'err0' in locals():
                print(f"[PIPELINE OPT] Error reduction: {((err0.mean() - err_final.mean()) / err0.mean() * 100):.1f}%")
            
            # Debug: Check for outliers and show worst correspondences
            if len(err_final) > 0:
                outlier_threshold = np.percentile(err_final, 95)
                outliers = err_final > outlier_threshold
                print(f"[PIPELINE OPT] Found {np.sum(outliers)} outliers (>{outlier_threshold:.1f}px)")
                
                if np.sum(outliers) > 0:
                    worst_idx = np.argmax(err_final)
                    print(f"[PIPELINE OPT] Worst error: {err_final[worst_idx]:.1f}px at correspondence {worst_idx}")
                
                # Detailed outlier analysis
                if np.sum(outliers) > 10:  # If many outliers, analyze patterns
                    print(f"\n🔍 OUTLIER ANALYSIS:")
                    outlier_indices = np.where(outliers)[0]
                    
                    # Check if outliers are clustered in certain image regions
                    outlier_image_coords = pc[outliers]
                    print(f"   Outlier image coordinates:")
                    print(f"     X range: [{outlier_image_coords[:,0].min():.0f}, {outlier_image_coords[:,0].max():.0f}]")
                    print(f"     Y range: [{outlier_image_coords[:,1].min():.0f}, {outlier_image_coords[:,1].max():.0f}]")
                    
                    # Check if outliers have common VICON coordinate patterns
                    outlier_vicon_coords = Pw[outliers]
                    outlier_vicon_mags = np.linalg.norm(outlier_vicon_coords, axis=1)
                    print(f"   Outlier VICON coordinates:")
                    print(f"     Magnitude range: [{outlier_vicon_mags.min():.0f}, {outlier_vicon_mags.max():.0f}]mm")
                    print(f"     Mean magnitude: {outlier_vicon_mags.mean():.0f}mm vs all points: {np.linalg.norm(Pw, axis=1).mean():.0f}mm")
                    
                    # Check if outliers are concentrated in specific transforms/frames
                    outlier_transforms = Tw2s_list[outliers]
                    outlier_translations = np.array([T[:3, 3] for T in outlier_transforms])
                    outlier_trans_mags = np.linalg.norm(outlier_translations, axis=1)
                    print(f"   Outlier transform translations:")
                    print(f"     Magnitude range: [{outlier_trans_mags.min():.0f}, {outlier_trans_mags.max():.0f}]mm")
                    print(f"     Mean: {outlier_trans_mags.mean():.0f}mm vs all: {np.linalg.norm([T[:3, 3] for T in Tw2s_list], axis=1).mean():.0f}mm")
                    
                    # Suggest possible causes
                    print(f"\n💡 POSSIBLE ERROR SOURCES:")
                    if outlier_vicon_mags.mean() > np.linalg.norm(Pw, axis=1).mean() * 1.2:
                        print(f"   • Outliers tend to be far from VICON origin (possibly poor tracking)")
                    if outlier_trans_mags.mean() > np.linalg.norm([T[:3, 3] for T in Tw2s_list], axis=1).mean() * 1.2:
                        print(f"   • Outliers correspond to extreme camera positions")
                    if np.std(outlier_image_coords[:,0]) < 50 or np.std(outlier_image_coords[:,1]) < 50:
                        print(f"   • Outliers clustered in image → possible distortion model issues")
                    if np.sum(outliers) > len(err_final) * 0.1:
                        print(f"   • High outlier rate ({100*np.sum(outliers)/len(err_final):.1f}%) → possible systematic calibration issues")
                        
        except Exception as e:
            print(f"Final diagnostics failed: {e}")
            
        # ------------------ transformation validation ------------------
        print(f"[PIPELINE OPT] Final transformation matrix T_system_to_camera:")
        print(T_sc)
        
        # Validate transformation properties
        det_R = np.linalg.det(T_sc[:3, :3])
        print(f"[PIPELINE OPT] Rotation matrix determinant: {det_R:.6f} (should be ~1.0)")
        if abs(det_R - 1.0) > 0.01:
            print("[WARNING] Rotation matrix determinant significantly differs from 1.0 - possible optimization issue")

        # store and return
        self.T_syst_to_camera_opt = T_sc
        print("Estimated T_sys→cam (optimized):\n", T_sc)
        return T_sc

    ###           
                
    def save_calibration(self, output_file: str):
        """Save transformation matrix and delay."""
        # Handle case where output_file is a directory
        if os.path.isdir(output_file):
            # Create a default filename in the directory
            filename = "calibration_result.txt"
            output_file = os.path.join(output_file, filename)
            print(f"Directory provided for calibration save, saving to: {output_file}")
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        def format_matrix_block(T: np.ndarray) -> str:
            rows = []
            for i, row in enumerate(T):
                row_str = " ".join(f"{val: .5f}" for val in row).lstrip()
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
        
    # def plot_3d_marker_trajectory(self, marker_name: str = None, save_path: str = None):
    #     """Plot 3D trajectory of a specific marker over time."""
        
    #     if marker_name is None:
    #         # Ask user to select a marker
    #         available_markers = [label.strip() for label in self.c3d_data.point_labels]
    #         print(f"\nAvailable markers: {available_markers[:10]}{'...' if len(available_markers) > 10 else ''}")
    #         while True:
    #             marker_name = input("Enter marker name for 3D trajectory plot: ").strip()
    #             if marker_name in available_markers:
    #                 break
    #             print(f"Marker '{marker_name}' not found. Available markers: {available_markers}")
        
    #     # Get 3D marker positions over time
    #     marker_positions = []
    #     valid_times = []
        
    #     for i, frame_data in enumerate(self.points_3d.values()):
    #         if marker_name in [label.strip() for label in self.c3d_data.point_labels]:
    #             marker_idx = [label.strip() for label in self.c3d_data.point_labels].index(marker_name)
    #             if marker_idx < len(frame_data):
    #                 pos = frame_data[marker_idx][:3]  # x, y, z
    #                 if np.all(np.isfinite(pos)) and not np.allclose(pos, 0):
    #                     marker_positions.append(pos)
    #                     valid_times.append(self.marker_t[i] if i < len(self.marker_t) else i * (1/100))
        
    #     if not marker_positions:
    #         print(f"No valid positions found for marker '{marker_name}'")
    #         return
            
    #     positions = np.array(marker_positions)
    #     times = np.array(valid_times)
        
    #     fig = plt.figure(figsize=(15, 5), num=f"3D Marker Trajectory: {marker_name}")
        
    #     # 3D trajectory
    #     ax1 = fig.add_subplot(131, projection='3d')
    #     ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7, linewidth=2)
    #     ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    #     ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
    #     ax1.set_xlabel('X (mm)')
    #     ax1.set_ylabel('Y (mm)')
    #     ax1.set_zlabel('Z (mm)')
    #     ax1.set_title(f'3D Trajectory: {marker_name}')
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
        
    #     # X, Y, Z vs time
    #     ax2 = fig.add_subplot(132)
    #     ax2.plot(times, positions[:, 0], 'r-', label='X', alpha=0.8)
    #     ax2.plot(times, positions[:, 1], 'g-', label='Y', alpha=0.8)
    #     ax2.plot(times, positions[:, 2], 'b-', label='Z', alpha=0.8)
    #     ax2.set_xlabel('Time (s)')
    #     ax2.set_ylabel('Position (mm)')
    #     ax2.set_title(f'Position vs Time: {marker_name}')
    #     ax2.legend()
    #     ax2.grid(True, alpha=0.3)
        
    #     # Velocity magnitude
    #     ax3 = fig.add_subplot(133)
    #     if len(positions) > 1:
    #         dt = np.diff(times)
    #         velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    #         ax3.plot(times[1:], velocity, 'purple', linewidth=2)
    #         ax3.set_xlabel('Time (s)')
    #         ax3.set_ylabel('Velocity (mm/s)')
    #         ax3.set_title(f'Velocity Magnitude: {marker_name}')
    #         ax3.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"3D marker plot saved: {save_path}")
        
    #     plt.show(block=False)  # Non-blocking show to allow multiple plots

    # def compute_camera_motion_and_plot(self, camera_markers: list = None, save_path: str = None, show_plots: bool = False, open_after: bool = True):
    #     """Compute camera origins and forward vectors from three camera markers per Vicon frame
    #     and plot the camera motion in 3D and 2D (top-down).

    #     Parameters
    #     - camera_markers: list of 3 marker names used to define camera frame. If None, uses
    #       self.camera_markers if available, else attempts to pick first 3 markers from the c3d file.
    #     - save_path: path to save the combined plot PNG. If None, saved to output directory.
    #     - show_plots: if True, call plt.show() to display interactively.
    #     """
    #     import matplotlib
    #     # If the caller doesn't want interactive plots, force a non-interactive backend
    #     if not show_plots:
    #         matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    #     # Determine camera markers
    #     if camera_markers is None:
    #         camera_markers = getattr(self, 'camera_markers', None)
    #     if camera_markers is None or len(camera_markers) < 3:
    #         # pick first three markers from c3d if available
    #         camera_markers = [name.strip() for name in self.c3d_data.point_labels][:3]

    #     if len(camera_markers) < 3:
    #         print("Need 3 camera markers to compute camera frame. Aborting.")
    #         return None

    #     # Map marker names to indices
    #     labels = [name.strip() for name in self.c3d_data.point_labels]
    #     try:
    #         idxs = [labels.index(name) for name in camera_markers]
    #     except ValueError as e:
    #         print(f"Camera marker not found in c3d labels: {e}")
    #         return None

    #     # Re-read c3d frames to extract positions for chosen markers
    #     reader = c3d.Reader(open(self.vicon_path, 'rb'))
    #     origins = []
    #     forwards = []
    #     rotations = []
    #     times = []
    #     frame_rate = getattr(reader, 'point_rate', None) or getattr(self, 'c3d_data', None) and getattr(self.c3d_data, 'point_rate', None) or 100.0
    #     for i, points, analog in reader.read_frames():
    #         # points is a list/array of marker coords (N x 3)
    #         try:
    #             p0 = np.asarray(points[idxs[0]][:3], dtype=np.float64)
    #             p1 = np.asarray(points[idxs[1]][:3], dtype=np.float64)
    #             p2 = np.asarray(points[idxs[2]][:3], dtype=np.float64)
    #         except Exception:
    #             origins.append(None)
    #             forwards.append(None)
    #             times.append(i / frame_rate)
    #             continue

    #         # define camera triad: x along p1-p0, temp = p2-p0
    #         v_x = p1 - p0
    #         v_temp = p2 - p0
    #         # normalize
    #         def nrm(v):
    #             nv = np.linalg.norm(v)
    #             return v / nv if nv > 1e-8 else v

    #         x_axis = nrm(v_x)
    #         z_axis = nrm(np.cross(x_axis, v_temp))
    #         y_axis = np.cross(z_axis, x_axis)

    #         # camera origin: use centroid of three markers
    #         origin = (p0 + p1 + p2) / 3.0
    #         R_world_cam = np.column_stack((x_axis, y_axis, z_axis))
    #         # forward vector in world coords (camera optical axis)
    #         forward = z_axis

    #         origins.append(origin)
    #         forwards.append(forward)
    #         rotations.append(R_world_cam)
    #         times.append(i / frame_rate)

    #     # Filter None frames
    #     valid_idx = [i for i, o in enumerate(origins) if o is not None]
    #     if not valid_idx:
    #         print("No valid camera marker frames found.")
    #         return None

    #     origins_arr = np.array([origins[i] for i in valid_idx])
    #     forwards_arr = np.array([forwards[i] for i in valid_idx])
    #     rotations_arr = np.array([rotations[i] for i in valid_idx])
    #     times_arr = np.array([times[i] for i in valid_idx])

    #     # Save into self for later use in frame rendering (and rotation matrices)
    #     self.camera_origins = origins_arr
    #     self.camera_forwards = forwards_arr
    #     self.camera_rotations = rotations_arr
    #     self.camera_motion_times = times_arr

    #     # Create combined plot: left 3D trajectory, right 2D top-down (X,Y)
    #     fig = plt.figure(figsize=(14, 6), num='Camera Motion (3D & 2D)')
    #     ax3d = fig.add_subplot(121, projection='3d')
    #     ax2d = fig.add_subplot(122)

    #     ax3d.plot(origins_arr[:, 0], origins_arr[:, 1], origins_arr[:, 2], '-k', alpha=0.7)
    #     ax3d.scatter(origins_arr[:, 0], origins_arr[:, 1], origins_arr[:, 2], c=times_arr, cmap='viridis', s=6)
    #     # draw sampled camera triads (X/Y/Z axes) to show full orientation
    #     sample_step = max(1, len(origins_arr)//40)
    #     span = np.linalg.norm(origins_arr[-1] - origins_arr[0]) if len(origins_arr) > 1 else 1.0
    #     axis_len = span * 0.03
    #     for i in range(0, len(origins_arr), sample_step):
    #         o = origins_arr[i]
    #         R = rotations_arr[i]
    #         # R columns are x_axis, y_axis, z_axis in world coords
    #         x = R[:, 0]
    #         y = R[:, 1]
    #         z = R[:, 2]
    #         # draw axes: X=red, Y=green, Z=blue
    #         ax3d.quiver(o[0], o[1], o[2], x[0], x[1], x[2], length=axis_len, color='r', normalize=True)
    #         ax3d.quiver(o[0], o[1], o[2], y[0], y[1], y[2], length=axis_len, color='g', normalize=True)
    #         ax3d.quiver(o[0], o[1], o[2], z[0], z[1], z[2], length=axis_len, color='b', normalize=True)

    #     ax3d.set_xlabel('X (mm)')
    #     ax3d.set_ylabel('Y (mm)')
    #     ax3d.set_zlabel('Z (mm)')
    #     ax3d.set_title('Camera Origin Trajectory (3D)')

    #     ax2d.plot(origins_arr[:, 0], origins_arr[:, 1], '-k', alpha=0.7)
    #     ax2d.scatter(origins_arr[:, 0], origins_arr[:, 1], c=times_arr, cmap='viridis', s=6)
    #     # In the top-down view draw x/y axes projections for sampled frames
    #     scale2d = span * 0.02 if 'span' in locals() else 10.0
    #     head_w = max(1.0, scale2d * 0.12)
    #     head_l = max(1.0, scale2d * 0.18)
    #     for i in range(0, len(origins_arr), sample_step):
    #         o = origins_arr[i]
    #         R = rotations_arr[i]
    #         x = R[:, 0]
    #         y = R[:, 1]
    #         # project X and Y axes onto XY plane
    #         ax2d.arrow(o[0], o[1], x[0]*scale2d, x[1]*scale2d, head_width=head_w, head_length=head_l, fc='r', ec='r')
    #         ax2d.arrow(o[0], o[1], y[0]*scale2d, y[1]*scale2d, head_width=head_w, head_length=head_l, fc='g', ec='g')

    #     ax2d.set_xlabel('X (mm)')
    #     ax2d.set_ylabel('Y (mm)')
    #     ax2d.set_title('Camera Trajectory (Top-down XY)')
    #     ax2d.axis('equal')

    #     plt.tight_layout()
    #     if save_path is None:
    #         out_dir = self._get_output_directory()
    #         save_path = os.path.join(out_dir, 'camera_motion.png')
    #     plt.savefig(save_path, dpi=200, bbox_inches='tight')
    #     print(f"Saved camera motion plots: {save_path}")

    #     # If caller wants an interactive display, show and block until closed
    #     if show_plots:
    #         try:
    #             plt.show(block=True)
    #         except Exception:
    #             # Fallback: print location and close
    #             print("Unable to open interactive plot window; saved to disk instead.")
    #             plt.close(fig)
    #     else:
    #         plt.close(fig)

    #     # Optionally open the saved PNG with the system image viewer (Linux: xdg-open)
    #     if open_after and save_path and os.name == 'posix':
    #         try:
    #             import shutil, subprocess
    #             opener = shutil.which('xdg-open')
    #             if opener:
    #                 # Launch non-blocking so script can finish and user sees the image
    #                 subprocess.Popen([opener, save_path])
    #             else:
    #                 print(f"No xdg-open found; open the image manually: {save_path}")
    #         except Exception as e:
    #             print(f"Failed to open image with system viewer: {e}")

    #     return {'origins': origins_arr, 'forwards': forwards_arr, 'rotations': rotations_arr, 'times': times_arr, 'plot_path': save_path}

    # def animate_camera_reference_frames(self, camera_marker_groups: dict, save_path: str = None,
    #                                     fps: int = 20, step: int = 1, max_frames: int = None,
    #                                     show: bool = False, open_after: bool = True):
    #     """Create an animation (MP4 or GIF) showing camera reference frames moving over time.

    #     Args:
    #         camera_marker_groups: dict mapping camera name -> list of 3 marker names, e.g.
    #             {'cam_right': ['CAM:RY', 'CAM:RX', 'CAM:RZ'], 'cam_left': [...]}
    #         save_path: output file path. If None, saved to output directory as camera_motion_anim.mp4
    #         fps: frames per second for the output video
    #         step: sample step over C3D frames (1 = all frames)
    #         max_frames: maximum number of frames to include (after stepping)
    #         show: if True, display interactive matplotlib window after creation (blocks)
    #         open_after: if True, attempt to open saved file with system viewer (xdg-open)

    #     Returns:
    #         dict with per-camera arrays and 'anim_path'
    #     """
    #     import matplotlib
    #     # prefer a non-interactive backend for file writing unless show requested
    #     if not show:
    #         matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    #     import matplotlib.animation as animation
    #     import shutil, subprocess

    #     # Validate input
    #     if not isinstance(camera_marker_groups, dict) or not camera_marker_groups:
    #         print("camera_marker_groups must be a non-empty dict of name -> [marker1,marker2,marker3]")
    #         return None

    #     # Read C3D and compute per-camera origins and rotations (per-frame)
    #     reader = c3d.Reader(open(self.vicon_path, 'rb'))
    #     frame_rate = getattr(reader, 'point_rate', None) or getattr(self, 'c3d_data', None) and getattr(self.c3d_data, 'point_rate', None) or 100.0

    #     cams_data = {}
    #     # Initialize per-camera storage as lists of length frame_count
    #     # We'll iterate frames and populate
    #     for cam_name, markers in camera_marker_groups.items():
    #         if not markers or len(markers) < 3:
    #             print(f"Camera '{cam_name}' needs 3 marker names. Skipping.")
    #             continue
    #         cams_data[cam_name] = {'markers': markers, 'origins': [], 'rots': [], 'times': []}

    #     if not cams_data:
    #         print("No valid camera groups to animate.")
    #         return None

    #     # Build label->idx map from c3d header
    #     labels = [name.strip() for name in reader.point_labels]
    #     label_to_idx = {lab: i for i, lab in enumerate(labels)}

    #     # Precompute indices for each camera
    #     cam_idxs = {}
    #     for cam_name, d in cams_data.items():
    #         try:
    #             cam_idxs[cam_name] = [label_to_idx[name] for name in d['markers']]
    #         except KeyError as e:
    #             print(f"Marker not found for camera {cam_name}: {e}. Skipping this camera.")
    #             cam_idxs[cam_name] = None

    #     # Iterate frames
    #     frame_i = 0
    #     for i, points, analog in reader.read_frames():
    #         for cam_name, idxs in cam_idxs.items():
    #             if idxs is None:
    #                 continue
    #             try:
    #                 p0 = np.asarray(points[idxs[0]][:3], dtype=np.float64)
    #                 p1 = np.asarray(points[idxs[1]][:3], dtype=np.float64)
    #                 p2 = np.asarray(points[idxs[2]][:3], dtype=np.float64)
    #             except Exception:
    #                 cams_data[cam_name]['origins'].append(None)
    #                 cams_data[cam_name]['rots'].append(None)
    #                 cams_data[cam_name]['times'].append(i / frame_rate)
    #                 continue

    #             def nrm(v):
    #                 nv = np.linalg.norm(v)
    #                 return v / nv if nv > 1e-8 else v

    #             x_axis = nrm(p1 - p0)
    #             z_axis = nrm(np.cross(x_axis, p2 - p0))
    #             y_axis = np.cross(z_axis, x_axis)
    #             origin = (p0 + p1 + p2) / 3.0
    #             R = np.column_stack((x_axis, y_axis, z_axis))

    #             cams_data[cam_name]['origins'].append(origin)
    #             cams_data[cam_name]['rots'].append(R)
    #             cams_data[cam_name]['times'].append(i / frame_rate)

    #         frame_i += 1

    #     # Convert lists to arrays and determine animation length
    #     max_len = 0
    #     for cam_name, d in cams_data.items():
    #         origins = np.array([o for o in d['origins'] if o is not None])
    #         rots = np.array([r for r in d['rots'] if r is not None])
    #         times = np.array([t for t, o in zip(d['times'], d['origins']) if o is not None])
    #         cams_data[cam_name]['origins_arr'] = origins
    #         cams_data[cam_name]['rots_arr'] = rots
    #         cams_data[cam_name]['times_arr'] = times
    #         max_len = max(max_len, len(times))

    #     if max_frames is not None:
    #         max_len = min(max_len, max_frames)

    #     if max_len == 0:
    #         print("No valid frames found for any camera groups.")
    #         return None

    #     # Colors for cameras (fallback cycle)
    #     default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    #     # Setup figure
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Plot static trajectories and create line objects for triad axes per camera
    #     cam_artists = {}
    #     all_origins_concat = []
    #     for ci, (cam_name, d) in enumerate(cams_data.items()):
    #         col = default_colors[ci % len(default_colors)]
    #         origins = d['origins_arr']
    #         if origins.size:
    #             ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-', color=col, alpha=0.5)
    #             ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=[col], s=4)
    #             all_origins_concat.append(origins)

    #         # initialize three lines for X/Y/Z axes (empty)
    #         lx, = ax.plot([], [], [], color='r', linewidth=2)
    #         ly, = ax.plot([], [], [], color='g', linewidth=2)
    #         lz, = ax.plot([], [], [], color='b', linewidth=2)
    #         cam_artists[cam_name] = {'lx': lx, 'ly': ly, 'lz': lz, 'color': col}

    #     # Set axes limits from data
    #     if all_origins_concat:
    #         all_pts = np.vstack(all_origins_concat)
    #         mins = all_pts.min(axis=0)
    #         maxs = all_pts.max(axis=0)
    #         span = maxs - mins
    #         margin = span.max() * 0.1 if span.max() > 0 else 10.0
    #         ax.set_xlim(mins[0] - margin, maxs[0] + margin)
    #         ax.set_ylim(mins[1] - margin, maxs[1] + margin)
    #         ax.set_zlim(mins[2] - margin, maxs[2] + margin)

    #     ax.set_xlabel('X (mm)')
    #     ax.set_ylabel('Y (mm)')
    #     ax.set_zlabel('Z (mm)')
    #     ax.set_title('Camera Reference Frames Animation')

    #     # Determine axis length for drawing triads
    #     diag = np.linalg.norm(np.array(ax.get_xlim())[[1,0]]) if all_origins_concat else 1.0
    #     axis_len = (np.max([np.linalg.norm(pts.max(axis=0)-pts.min(axis=0)) for pts in all_origins_concat]) * 0.05) if all_origins_concat else 10.0

    #     # Animation update function
    #     def update(frame_idx):
    #         k = frame_idx * step
    #         for cam_name, d in cams_data.items():
    #             times = d['times_arr']
    #             if k >= len(times):
    #                 # clear lines
    #                 cam_artists[cam_name]['lx'].set_data([], [])
    #                 cam_artists[cam_name]['lx'].set_3d_properties([])
    #                 cam_artists[cam_name]['ly'].set_data([], [])
    #                 cam_artists[cam_name]['ly'].set_3d_properties([])
    #                 cam_artists[cam_name]['lz'].set_data([], [])
    #                 cam_artists[cam_name]['lz'].set_3d_properties([])
    #                 continue

    #             o = d['origins_arr'][k]
    #             R = d['rots_arr'][k]
    #             x = R[:, 0] * axis_len
    #             y = R[:, 1] * axis_len
    #             z = R[:, 2] * axis_len

    #             # update X axis line
    #             cam_artists[cam_name]['lx'].set_data([o[0], o[0]+x[0]], [o[1], o[1]+x[1]])
    #             cam_artists[cam_name]['lx'].set_3d_properties([o[2], o[2]+x[2]])
    #             # update Y axis line
    #             cam_artists[cam_name]['ly'].set_data([o[0], o[0]+y[0]], [o[1], o[1]+y[1]])
    #             cam_artists[cam_name]['ly'].set_3d_properties([o[2], o[2]+y[2]])
    #             # update Z axis line
    #             cam_artists[cam_name]['lz'].set_data([o[0], o[0]+z[0]], [o[1], o[1]+z[1]])
    #             cam_artists[cam_name]['lz'].set_3d_properties([o[2], o[2]+z[2]])

    #         return [a for cam in cam_artists.values() for a in (cam['lx'], cam['ly'], cam['lz'])]

    #     n_frames = int(np.ceil(max_len / max(1, step)))

    #     anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0/fps, blit=False)

    #     # Choose writer
    #     out_dir = self._get_output_directory()
    #     if save_path is None:
    #         save_path = os.path.join(out_dir, 'camera_motion_anim.mp4')

    #     ffmpeg = shutil.which('ffmpeg')
    #     try:
    #         if ffmpeg:
    #             Writer = animation.FFMpegWriter
    #             writer = Writer(fps=fps)
    #             anim.save(save_path, writer=writer, dpi=200)
    #         else:
    #             # fallback to GIF via Pillow
    #             from matplotlib.animation import PillowWriter
    #             writer = PillowWriter(fps=fps)
    #             gif_path = os.path.splitext(save_path)[0] + '.gif'
    #             anim.save(gif_path, writer=writer)
    #             save_path = gif_path
    #     except Exception as e:
    #         print(f"Failed to save animation: {e}")
    #         plt.close(fig)
    #         return None

    #     print(f"Saved animation: {save_path}")

    #     if show:
    #         try:
    #             plt.show(block=True)
    #         except Exception:
    #             print("Unable to open interactive animation window; saved to disk instead.")
    #     else:
    #         plt.close(fig)

    #     # Open with system viewer if requested (non-blocking)
    #     if open_after and os.name == 'posix':
    #         opener = shutil.which('xdg-open')
    #         if opener:
    #             try:
    #                 subprocess.Popen([opener, save_path])
    #             except Exception as e:
    #                 print(f"Failed to open animation: {e}")
    #         else:
    #             print(f"Animation saved to {save_path}; open it manually (no xdg-open found)")

    #     return {'cams_data': cams_data, 'anim_path': save_path}
    
    # def plot_2d_marker_projections(self, marker_name: str = None, save_path: str = None):
    #     """Plot 2D projections of a specific marker over time."""
        
    #     if marker_name is None:
    #         # Ask user to select a marker
    #         available_markers = [label.strip() for label in self.c3d_data.point_labels]
    #         print(f"\nAvailable markers: {available_markers[:10]}{'...' if len(available_markers) > 10 else ''}")
    #         while True:
    #             marker_name = input("Enter marker name for 2D projection plot: ").strip()
    #             if marker_name in available_markers:
    #                 break
    #             print(f"Marker '{marker_name}' not found. Available markers: {available_markers}")
        
    #     try:
    #         # Create projector to get 2D projections
    #         if not hasattr(self, 'T_syst_to_camera_opt') or self.T_syst_to_camera_opt is None:
    #             print("No optimized calibration found. Cannot compute 2D projections.")
    #             return
                
    #         projector = helpers.ViconProjector(
    #             [marker_name], self.c3d_data, self.points_3d,
    #             self.T_syst_to_camera_opt, self.Ts_world_to_system,
    #             self.K, self.cam_res, D=self.D, subject=self.subject
    #         )
            
    #         # Get 2D projections
    #         if marker_name not in projector.image_points:
    #             print(f"Could not compute projections for marker '{marker_name}'")
    #             return
                
    #         projections = projector.image_points[marker_name]
    #         times = self.marker_t[:len(projections)]
            
    #         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), num=f"2D Marker Projections: {marker_name}")
            
    #         # 2D trajectory on image plane
    #         valid_mask = (projections[:, 0] >= 0) & (projections[:, 0] < self.cam_res[1]) & \
    #                     (projections[:, 1] >= 0) & (projections[:, 1] < self.cam_res[0])
    #         valid_projs = projections[valid_mask]
    #         valid_times = times[valid_mask]
            
    #         if len(valid_projs) > 0:
    #             ax1.plot(valid_projs[:, 0], valid_projs[:, 1], 'b-', alpha=0.7, linewidth=2)
    #             ax1.scatter(valid_projs[0, 0], valid_projs[0, 1], color='green', s=100, label='Start')
    #             ax1.scatter(valid_projs[-1, 0], valid_projs[-1, 1], color='red', s=100, label='End')
    #             ax1.set_xlim(0, self.cam_res[1])
    #             ax1.set_ylim(self.cam_res[0], 0)  # Invert Y axis for image coordinates
    #             ax1.set_xlabel('X (pixels)')
    #             ax1.set_ylabel('Y (pixels)')
    #             ax1.set_title(f'2D Trajectory on Image Plane: {marker_name}')
    #             ax1.legend()
    #             ax1.grid(True, alpha=0.3)
    #             ax1.set_aspect('equal')
            
    #         # X pixel position vs time
    #         ax2.plot(valid_times, valid_projs[:, 0], 'r-', linewidth=2)
    #         ax2.set_xlabel('Time (s)')
    #         ax2.set_ylabel('X Position (pixels)')
    #         ax2.set_title(f'X Projection vs Time: {marker_name}')
    #         ax2.grid(True, alpha=0.3)
            
    #         # Y pixel position vs time
    #         ax3.plot(valid_times, valid_projs[:, 1], 'g-', linewidth=2)
    #         ax3.set_xlabel('Time (s)')
    #         ax3.set_ylabel('Y Position (pixels)')
    #         ax3.set_title(f'Y Projection vs Time: {marker_name}')
    #         ax3.grid(True, alpha=0.3)
            
    #         # 2D velocity
    #         if len(valid_projs) > 1:
    #             dt = np.diff(valid_times)
    #             velocity_2d = np.linalg.norm(np.diff(valid_projs, axis=0), axis=1) / dt
    #             ax4.plot(valid_times[1:], velocity_2d, 'purple', linewidth=2)
    #             ax4.set_xlabel('Time (s)')
    #             ax4.set_ylabel('2D Velocity (pixels/s)')
    #             ax4.set_title(f'2D Velocity: {marker_name}')
    #             ax4.grid(True, alpha=0.3)
            
    #         plt.tight_layout()
            
    #         if save_path:
    #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #             print(f"2D marker plot saved: {save_path}")
            
    #         plt.show(block=False)  # Non-blocking show to allow multiple plots
            
    #     except Exception as e:
    #         print(f"Error plotting 2D marker projections: {e}")
    
    # def plot_camera_motion(self, save_path: str = None):
    #     """Plot camera motion trajectory and orientation over time."""
        
    #     try:
    #         if not hasattr(self, 'T_syst_to_camera_opt') or self.T_syst_to_camera_opt is None:
    #             print("No optimized calibration found. Cannot compute camera motion.")
    #             return
                
    #         if not hasattr(self, 'Ts_world_to_system') or self.Ts_world_to_system is None:
    #             print("No world-to-system transforms found. Cannot compute camera motion.")
    #             return
            
    #         # Compute camera poses in world frame
    #         camera_positions = []
    #         camera_orientations = []
    #         times = self.marker_t[:len(self.Ts_world_to_system)]
            
    #         T_sc = self.T_syst_to_camera_opt  # System to camera (fixed)
            
    #         for i, T_ws in enumerate(self.Ts_world_to_system):
    #             # World to camera transform
    #             T_wc = T_sc @ T_ws
                
    #             # Camera position in world frame (inverse transform)
    #             T_cw = np.linalg.inv(T_wc)
    #             camera_pos = T_cw[:3, 3]
    #             camera_positions.append(camera_pos)
                
    #             # Camera orientation (rotation matrix to Euler angles)
    #             R_cw = T_cw[:3, :3]
    #             rot = Rotation.from_matrix(R_cw)
    #             euler_angles = rot.as_euler('xyz', degrees=True)
    #             camera_orientations.append(euler_angles)
            
    #         positions = np.array(camera_positions)
    #         orientations = np.array(camera_orientations)
            
    #         fig = plt.figure(figsize=(18, 10), num="Camera Motion Analysis")
            
    #         # Camera trajectory in 3D
    #         ax1 = fig.add_subplot(231, projection='3d')
    #         ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.8, linewidth=2)
    #         ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    #         ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
    #         ax1.set_xlabel('X (mm)')
    #         ax1.set_ylabel('Y (mm)')
    #         ax1.set_zlabel('Z (mm)')
    #         ax1.set_title('Camera 3D Trajectory')
    #         ax1.legend()
    #         ax1.grid(True, alpha=0.3)
            
    #         # Camera position vs time
    #         ax2 = fig.add_subplot(232)
    #         ax2.plot(times, positions[:, 0], 'r-', label='X', alpha=0.8)
    #         ax2.plot(times, positions[:, 1], 'g-', label='Y', alpha=0.8)
    #         ax2.plot(times, positions[:, 2], 'b-', label='Z', alpha=0.8)
    #         ax2.set_xlabel('Time (s)')
    #         ax2.set_ylabel('Position (mm)')
    #         ax2.set_title('Camera Position vs Time')
    #         ax2.legend()
    #         ax2.grid(True, alpha=0.3)
            
    #         # Camera orientation vs time
    #         ax3 = fig.add_subplot(233)
    #         ax3.plot(times, orientations[:, 0], 'r-', label='Roll (X)', alpha=0.8)
    #         ax3.plot(times, orientations[:, 1], 'g-', label='Pitch (Y)', alpha=0.8)
    #         ax3.plot(times, orientations[:, 2], 'b-', label='Yaw (Z)', alpha=0.8)
    #         ax3.set_xlabel('Time (s)')
    #         ax3.set_ylabel('Orientation (degrees)')
    #         ax3.set_title('Camera Orientation vs Time')
    #         ax3.legend()
    #         ax3.grid(True, alpha=0.3)
            
    #         # Camera velocity
    #         ax4 = fig.add_subplot(234)
    #         if len(positions) > 1:
    #             dt = np.diff(times)
    #             velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    #             ax4.plot(times[1:], velocity, 'purple', linewidth=2)
    #             ax4.set_xlabel('Time (s)')
    #             ax4.set_ylabel('Velocity (mm/s)')
    #             ax4.set_title('Camera Linear Velocity')
    #             ax4.grid(True, alpha=0.3)
            
    #         # Camera angular velocity
    #         ax5 = fig.add_subplot(235)
    #         if len(orientations) > 1:
    #             dt = np.diff(times)
    #             angular_velocity = np.linalg.norm(np.diff(orientations, axis=0), axis=1) / dt
    #             ax5.plot(times[1:], angular_velocity, 'orange', linewidth=2)
    #             ax5.set_xlabel('Time (s)')
    #             ax5.set_ylabel('Angular Velocity (deg/s)')
    #             ax5.set_title('Camera Angular Velocity')
    #             ax5.grid(True, alpha=0.3)
            
    #         # Top-down view of trajectory
    #         ax6 = fig.add_subplot(236)
    #         ax6.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.8, linewidth=2)
    #         ax6.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
    #         ax6.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
    #         ax6.set_xlabel('X (mm)')
    #         ax6.set_ylabel('Y (mm)')
    #         ax6.set_title('Camera Trajectory (Top View)')
    #         ax6.legend()
    #         ax6.grid(True, alpha=0.3)
    #         ax6.set_aspect('equal')
            
    #         plt.tight_layout()
            
    #         if save_path:
    #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #             print(f"Camera motion plot saved: {save_path}")
            
    #         plt.show(block=False)  # Non-blocking show to allow multiple plots
            
    #         # Print camera motion statistics
    #         print(f"\n=== CAMERA MOTION SUMMARY ===")
    #         print(f"Total recording time: {times[-1] - times[0]:.2f} seconds")
    #         print(f"Camera position range:")
    #         print(f"  X: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f} mm (range: {positions[:, 0].max() - positions[:, 0].min():.1f} mm)")
    #         print(f"  Y: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f} mm (range: {positions[:, 1].max() - positions[:, 1].min():.1f} mm)")
    #         print(f"  Z: {positions[:, 2].min():.1f} to {positions[:, 2].max():.1f} mm (range: {positions[:, 2].max() - positions[:, 2].min():.1f} mm)")
            
    #         if len(positions) > 1:
    #             total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    #             avg_velocity = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times))
    #             print(f"Total distance traveled: {total_distance:.1f} mm")
    #             print(f"Average velocity: {avg_velocity:.1f} mm/s")
            
    #     except Exception as e:
    #         print(f"Error plotting camera motion: {e}")
        
    def run_full_pipeline(self, use_projections: bool = False, create_video: bool = True,
        init_file_path: str = None, chosen_marker: str = None, perform_error_analysis: bool = False,
        save_frames: bool = False, frames_video: bool = False):
        
        """Run the complete pipeline."""
        print("Starting VICON-DVS pipeline...")
        
        # 1. Load all data
        self.load_event_data()
        self.load_vicon_data()
        self.load_calibration_data()
        
        # 2. Try to load existing calibration
        if init_file_path is None:
            init_file = os.path.join(self._get_output_directory(), "init_file.txt")
        else:
            init_file = init_file_path
            
        calibration_exists = self.load_existing_calibration(init_file)
        
        # 3. Compute world to system transforms
        self.compute_world_to_system_transforms()
        
        if not calibration_exists:
            print("No existing calibration found. Starting manual calibration...")
            
            # 4. Manual rotation estimation with marker filter
            tvec_init = np.array([0.0, 0.0, 0.0])       # TODO: make it user-input, 5.5, 5, 0.5
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
            # self.optimize_calibration(
            #     labels_path=self.output_path,
            #     min_points_per_frame=4,      # minimum points per frame
            #     ransac_reproj_err=3.0,       # RANSAC reprojection error threshold
            #     ransac_iters=300,            # RANSAC iterations
            #     pnp_reproj_clip=5.0,         # reprojection error clipping threshold
            #     handeye_method=cv2.CALIB_HAND_EYE_DANIILIDIS  # or None for default
            # )

            # 8. Save calibration
            self.save_calibration(init_file)
            
        # 9. Save transformation matrices
        # transforms_file = os.path.join(os.path.dirname(self.vicon_path), "transformation_matrices.txt")
        # self.save_transforms_txt(transforms_file)
        
        # 10. Create projection video
        if create_video:
            # Extract sequence name for unique video naming
            sequence_name = self._extract_sequence_name()
            base_video_file = os.path.join(self._get_output_directory(), f"{sequence_name}_projection_video.mp4")
            video_file = self._generate_unique_video_path(base_video_file)

            # Run the projection session but DO NOT save yet — just collect buffers
            result = self.create_projection_video(video_file)  # now returns dict with segments & points
            collected_video_segments = result["segments"]
            all_projected_points = result["points"]

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

                    # Optional: error analysis after saving
                    if perform_error_analysis:
                        labels_path = self.output_path
                        try:
                            self.calculate_projection_error(labels_path, visualize=True)
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
                            
                            frames_dir = self.save_frames_with_gt_and_csv_projections(
                                self.output_path, 
                                create_video=frames_video
                            )
                            print(f"Frame visualization completed! Check: {frames_dir}")
                            
                        except Exception as e:
                            print(f"Error during frame-by-frame visualization: {e}")
                            print("Pipeline completed but frame visualization failed.")

                    # Run joint extraction automation automatically
                    print("\n" + "="*60)
                    print("STARTING JOINT EXTRACTION AUTOMATION")
                    print("="*60)
                    # Skip init file creation since we're using an existing calibration
                    success = self.run_joint_extraction_automation(skip_init_file_creation=True)
                    if success:
                        print("Joint extraction automation completed successfully!")
                    else:
                        print("Joint extraction automation encountered errors.")
            
                        # Attempt to create camera reference-frame animation if camera markers are available
                        try:
                            # Look for any markers that mention 'cam' or 'camera'
                            cam_candidates = [m for m in getattr(self, 'marker_names', []) if ('cam' in m.lower() or 'camera' in m.lower())]
                            groups = {}
                            if cam_candidates:
                                # try to classify by side keywords
                                right = [m for m in cam_candidates if 'right' in m.lower()]
                                left = [m for m in cam_candidates if 'left' in m.lower()]
                                back = [m for m in cam_candidates if ('back' in m.lower() or 'rear' in m.lower())]
                                if right:
                                    groups['cam_right'] = right[:3]
                                if left:
                                    groups['cam_left'] = left[:3]
                                if back:
                                    groups['cam_back'] = back[:3]
                                # If no specific sides found but there are at least 3 cam markers, make a default group
                                if not groups and len(cam_candidates) >= 3:
                                    groups['camera'] = cam_candidates[:3]
                            else:
                                print("No camera markers (containing 'cam' or 'camera') found — skipping camera animation.")

                            if groups:
                                print(f"Found camera marker groups for animation: {list(groups.keys())}")
                                res_anim = self.animate_camera_reference_frames(camera_marker_groups=groups, save_path=None, fps=20, step=1, show=False, open_after=True)
                                if res_anim is None:
                                    print("Camera animation was not created (see earlier messages).")
                                else:
                                    print(f"Camera animation saved: {res_anim.get('anim_path')}")
                            else:
                                print("No valid camera marker groups detected; animation skipped.")
                        except Exception as e:
                            print(f"Error while attempting to create camera animation: {e}")

                        print("Pipeline completed successfully!")
                        return

                    # After handling user confirmation, break out of review loop
                    break

                elif response in ['n', 'no']:
                    print("Okay — discarding this run (no video, no CSV saved).")
                    # You can loop back or exit depending on your pipeline design
                    break

                else:
                    print("Please answer 'y' or 'n'.")

        if perform_error_analysis:
            labels_path = self.output_path
            try:
                self.calculate_projection_error(labels_path, visualize=True)
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
                
                frames_dir = self.save_frames_with_gt_and_csv_projections(
                    self.output_path, 
                    create_video=frames_video
                )
                print(f"Frame visualization completed! Check: {frames_dir}")
                
            except Exception as e:
                print(f"Error during frame-by-frame visualization: {e}")
                print("Pipeline completed but frame visualization failed.")
        
            # Run joint extraction automation automatically
            print("\n" + "="*60)
            print("STARTING JOINT EXTRACTION AUTOMATION")
            print("="*60)
            # Determine if we should skip init file creation based on whether we loaded an existing calibration
            skip_init = calibration_exists
            success = self.run_joint_extraction_automation(skip_init_file_creation=skip_init)
            if success:
                print("Joint extraction automation completed successfully!")
            else:
                print("Joint extraction automation encountered errors.")
            
            print("Pipeline completed successfully!")
            return
            
        else:
            # If no video creation, just complete the pipeline
            # Step 11: Perform error analysis if requested
            if perform_error_analysis:
                labels_path = self.output_path
                try:
                    self.calculate_projection_error(labels_path, visualize=True)
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
                    
                    frames_dir = self.save_frames_with_gt_and_csv_projections(
                        self.output_path, 
                        create_video=frames_video
                    )
                    print(f"Frame visualization completed! Check: {frames_dir}")
                    
                except Exception as e:
                    print(f"Error during frame-by-frame visualization: {e}")
                    print("Pipeline completed but frame visualization failed.")
            
            # Run joint extraction automation automatically
            print("\n" + "="*60)
            print("STARTING JOINT EXTRACTION AUTOMATION")
            print("="*60)
            # Determine if we should skip init file creation based on whether we loaded an existing calibration
            skip_init = calibration_exists
            success = self.run_joint_extraction_automation(skip_init_file_creation=skip_init)
            if success:
                print("Joint extraction automation completed successfully!")
            else:
                print("Joint extraction automation encountered errors.")
            
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
    parser.add_argument('--save_frames', action='store_true',
                       help='Create frame-by-frame GT vs projection visualization images')
    parser.add_argument('--frames_video', action='store_true',
                       help='Create video from frame-by-frame comparison (use with --save_frames)')
    
    # TODO: add argument to let the user choose the size of the time windows????
    
    args = parser.parse_args()
    
    # Require --marker_list_path when --list is used
    if args.list and args.marker_list_path is None:
        parser.error("--marker_list_path is required when using --list")
    
    # Validate frame visualization arguments
    if args.frames_video and not args.save_frames:
        parser.error("--frames_video requires --save_frames to be enabled")
    
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
    
    # Run full pipeline
    pipeline.run_full_pipeline(
        use_projections=args.projections,
        create_video=not args.no_video,
        init_file_path=args.init_file,
        chosen_marker=args.chosen_marker,
        perform_error_analysis=args.error,
        save_frames=args.save_frames,
        frames_video=args.frames_video
    )
    

if __name__ == "__main__":
    main()