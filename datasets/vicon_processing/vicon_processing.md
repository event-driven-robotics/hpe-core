# Vicon processing

This module allows to take the joints output from the output of the gait model for a Vicon system.
The methods requires to label some of the points manually in order to find the right transformation.

## Vicon Nexus processing

The action to perform on the data are defined in the hpe pipeline. They are:

- Cut the recording only to the relevant portions
- Reconstruct
- Label (Autolabel static is recomended but not working well)
- (Optional) Fill gaps. Alternatively the gap filling can be done manually (see later)
- Scale subject VSK
- Static Skeleton Calibration - Markers only
- Process Static Plug-in Gait model

The pipeline is defined in `Tools/Pipeline` (upper right, gear icon).

Note: make sure when running the pipeline that only the body subject is selected in the left panel in `Subjects`. The subject `Pxxx (PluginGait Full Body Ai)` should be selected.

### Cut the recording

We can the sequences only between the flashing of the calibration wand. This may help with the other steps as at the beginning and end there might be more markers that are not visible.
`I would only do this if the cut is small. Otherwise it could effect the time synchronization. The optimal time is searched only in a fixed range of values`

### Action: Reconstruct

The reconstruction is an automotic process. In the parameters it is recomended to set `Minimum Cameras to Start Trajectory` to at least `3`. If there are a lot of probelms with the labeling because not all the markers are visible, reducing this value to 2 might help. But it is not recomended.

### Action: Label

The labeling should be done automatically in the `Label` action of the pipeline. For various reasons this may not work. The `log` panel might give some information about the error. Alternatively one can do manual labeling.

Note: I find the manual labeling to be more reliable. Sometimes the automatic labeling works fine, but sometime it doesn't. When it fails it is usually more time consuming to fix. I tend to always manually label the first frame.

### Gap Filling

After the labeling there will be some gaps in the trajecories. There are two option for this:

- manual gap filling
- automatic gap filling

**Manual FIlling**: For the manual filling go into the `Label/Edit` tab. The section `Gap Filling` shows the gaps in the trajectories.
For each gap:

- double click on label
- There are two possible cases now:
  - **a label is missing**:
    - a merker is visible, but not labeld (Gray)
    - select the corresponding label from the list above and manually assign the marker to it
  - **a merker is missing** for a portion of the trajectory
    - no marker is present
    - select filling strategy. Suggested `pattern fill` and `rigid body fill`.
    - then select the points (or chose `auto`)
    - For small gaps (<10) other methods of filling should also be fine

### Action: Process Static Plug-in Gait Model

The only option that needs to be changed is the: `Output Joint centers`. It should be turned on.

The other actions of the piepline should be allautomatic and should not give too many problems. They are:

- `scale vsk`
- `fit gait model` (for this deselect the camera from the left panel)
- `Export .c3d` (remeber to reselect the camera before performing this step)

## Calibration, projection and export

There are two main requirements for the projection and the generation of the data:

- an extrinsic projection (from marker frame to camera frame)
- a delay to be applied to the vicon data such that the dvs and the vicon datas are aligned.

The extrinsic calibration is obtained from the calibration process and doesn't need to be done very often. (There are 2 different calibrations, one for the static and one for the dyanamic camera)

The delay has to be computed for every sequence as there is always a small variation and a single delay can't be used for every sequence.

Note: The calibration needs to be re-done only if the projection looks wrong. Over time the position of the markers might shift a little bit resulting in errors.

The static camera should not need to be recalibrated, only in case it was accidentaly moved.

General pipeline:

- generate frames
- label
- calibrate
- find delay
- generate video / data file / etc ...

### Folder structure

Some of the bash scripts to automate all the processing expect a certain folder structure. Bold elements are required, other are generated later:

- subject_folder1:
  - extrinsic_d.npy (extrinsic calibration, see later)
  - extrinsic_s.npy (extrinsic calibration static, see later)
  - **calib-d.txt** (intrinsic calibration dynamic camera, should already be available)
  - **calib-s.txt** (intrinsic static)
  - **sequence_name1/** (e.g. jump_s1)
    - **atis-d** (contains the events for the moving camera in yarp format)
    - **atis-s** (static camera)
    - atis_d_frames (generated frames, see next section)
      - times.yml (timing information for the generated frames)
      - labeled_points.yml (output of the labeling process, see later sections)
    - atis_s_frames (same for static)
  - **sequence_name1.c3d** (generated by the vicon processing)
  - **sequence_name2/**
  - **sequence_name2.c3d**
  - ...
- subject_folder2:
  - ...

### Generating frames

In order to calibrate and find the optimal delay, we need to manually label some points.

To do that, we first generate and save some frames from the events. Then we go over the frames and do the labeling.
It is more convinient to first generate all the frames and then label. Loading the events might take a while and doing it on the fly when labeling takes more time.
`Some of file for the events are very large. It is best to do the generation of the frames on a machine with a lot of memory.`

For generating the frames for a single sequence there is a script: `scripts/save_dvs_frames.py`. The help messages in the scritp should help with arguments required.

The parameters allow to select the range and the frequency of the frame generation. It is not important to match exactly the times of the vicon frames as the points are interpolated anyway.

Example: `python3 save_dvs_frames.py --dvs_path /home/schiavazza/data/hpe/vicon_dataset/processed/gaurvi/box_s
1/atis-d/ --output_path /home/schiavazza/data/hpe/vicon_dataset/processed/gaurvi/box_s1/atis_d_frames/ --n_frames 200 --time_window 0.04`

It is convient to generate all frames for every sequence of a subject. Thre is a bash script for this: `scripts/bash/save_frames_all.sh`. The path is **hardcoded** in the script. Change it to change the subject.

### Labeling

The labeling for the calibration needs to be more accurate and in general is better to have as many points as possible. For finding the delay it is ok to use less points. (2/3 points, from 2/3 different points in the sequence)

In general we aim to label the markers that are on the body of the person. They are sometime visible in the events (but not always).

The script for labeling is: `scripts/label_dvs.py`. Again, look a the help messages for the arguments. The sames for labeling all the sequences one after another is: `scripts/bash/label_all.sh`

Note: one of the arguments is: `--manual`. Which controls how the labeling is done (see next sections).

In both cases, a window will pop up showing the frames generated from events. Click on the points to label them, press `Space` to skip a frame.

**IMPORTANT: Each subject has a code. e.g. P11, P10 etc... The labeling script requires this code in `--subject`. The `label_all.sh` script also defines an hardcoded value for this. Remember to change it for every subject. The vicon nexus software saves all the subjects with this code**

#### Manual labels (for calibration)

Each time a point is clicked in the frame, allows to chose what label should be used. This method is more time consuming, but allows to have a different set of points for each frame.

The label needs to be inserted from terminal from a list of option. Just write the number corresponding to the desired label and press `Enter`. Press `K` to finish labeling a frame. `Space` to skip (it also discard any label for the current frame if any, good in case of a mistake).

Use this for the extrinsic calibration. (as many points as possible [>5 frames at least])

#### Fixed labels (for delay)

Label from a predefined list of labels (see script args). This method is faster, but less flexible. The name of the current marker to label will be shown on the frame.

Note: is is better to use marker that move fast, i.g. hands, arms.
The `label_all.sh` script uses the fixed labels method.
The default markers used by the `label_all.sh` script should already work for all sequences.
**Careful that the marker used when the camera is static of moving slowly are different from when the camera is moving fast.**

Labeling all the sequences can take a while. The `label_all.sh` script can be stopped (ctrl+c) at any time. Runnig it again will ignore all the sequences that have already been labeled.

### Calibration

If the labeling is done it should be strightforward. Just run the script: `scripts/extrinsic_calibrate.py` with the proper arguments. A projetion matrix will be saved.

Example: `python3 extrinsic_calibrate.py --dvs_path ~/data/hpe/vicon_dataset/processed/gaurvi/box_s1/atis-d/ --annotated_points ~/data/hpe/vicon_dataset/processed/gaurvi/jump_s1/atis_d_frames/labeled_points.yml --vicon_path ~/data/hpe/vicon_dataset/processed/gaurvi/jump_s1.c3d --intrinsic ~/data/hpe/vicon_dataset/processed/gaurvi/calib-d.txt --output_path ~/data/hpe/vicon_dataset/processed/gaurvi/extrinsic_d.npy`

An important option for this script is the: `--no_camera_markers` option. This tells the script if it should use the markers on the camera for the projection.
**Use this option only when calibrating the static camera.** (sequences named with `_s`).
For moving camera calibration omit it.

Again, this process should not be done for every sequence. Maybe once per subject or once per day (of recording). Just pick a sequence and run the calibration on it (after properly labeling it).
A sequence like `jump_s1` has fast movements, but it still easy to identify the markers.

### Delay optimization

After the extrinsic calibration is found we can find the optimal delay for each sequence. We need at least a couple of points for each sequence from different points in time. The more the better, but there are a lot of sequences...

Run the script: `scripts/bash/delay_optimise_all.sh`. Again change the path at the beginnig of the script to the desired one. The script will save the delay in a file inside the sequence folder.

To run delay optimisation on a single sequence use: `scritps/delay_optimise.py`

### Degug

The `scripts/bash/debug_all.sh` script allows to verify the resutls. It projects the points on all the previously generated frames.

### Video

`generate_video_all.sh` creates a video combining the events and the projections. It can take a long time to run for every sequence. (could be improved by running each sequence in a different process, just add & at the end of the scritp call. The prints won't be reliable however).

The script automatically reads the delay found in the step before from the saved file.

### Export data

There are 2 basic scripts for generating the data:

- `export_cdf.py`
- `export_yarp.py` (NOT TESTED YET)

Both scripts output the projected 2d points in the dvs camera frame. Both also have a corresponding bash script to porcess all the sequences of a subject with a single command:

- `bash/export_cdf_all.sh`
- `bash/export_yarp_all.sh`

The `export_cdf.py` file has a bit more comments to understand the general process for reading and generating the projected points.
