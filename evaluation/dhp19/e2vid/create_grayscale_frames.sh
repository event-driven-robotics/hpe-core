#!/bin/bash

# read cmd line arguments
while getopts ":i:o:" opt; do
  case $opt in
    i) input_events="$OPTARG"
    ;;
    o) output_folder="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&3
    ;;
  esac
done

python run_e2vid.py --events_file "$input_events" --cam_id 0 --output_folder "$output_folder" --auto_hdr
python run_e2vid.py --events_file "$input_events" --cam_id 1 --output_folder "$output_folder" --auto_hdr
python run_e2vid.py --events_file "$input_events" --cam_id 2 --output_folder "$output_folder" --auto_hdr
python run_e2vid.py --events_file "$input_events" --cam_id 3 --output_folder "$output_folder" --auto_hdr
