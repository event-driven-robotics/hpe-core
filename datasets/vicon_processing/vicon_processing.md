# Vicon processing

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

## Cut the recording

We can the sequences only between the flashing of the calibration wand. This may help with the other steps as at the beginning and end there might be more markers that are not visible.
`I would only do this if the cut is small. Otherwise it could effect the time synchronization. The optimal time is searched only in a fixed range of values`

## Action: Reconstruct

The reconstruction is an automotic process. In the parameters it is recomended to set `Minimum Cameras to Start Trajectory` to at least `3`. If there are a lot of probelms with the labeling because not all the markers are visible, reducing this value to 2 might help. But it is not recomended.

Likewiese, an higher number of cameras required for continuing the trajectory might imporve the results, but it may create more gaps in the reconsrtuction. If the some of the camera markers are "jumping" a little bit. Increasnig this value might help.

## Action: Label

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

## Action: Process Static Plug-in Gait Model
The only option that needs to be changed is the: `Output Joint centers`. It should be turned on.

The other actions of the piepline should be allautomatic and should not give too many problems. They are:
  - `scale vsk`
  - `fit gait model` (for this deselect the camera from the left panel)
  - `Export .c3d` (remeber to reselect the camera before performing this step)

## Calibration, projection and export

There are two main requirements for the projection and the generation of the data:
  - an extrinsic projection (from marker frame to camera frame)
  - a delay to be applied to the vicon data such that the dvs and the vicon datas are aligned.

The extrinsic calibration is obtained from the calibration process and need to be done very often. (There are 2 different calibrations, one for the static and one for the dyanamic camera)

The delay has to be computed for every sequence as there is always a small variation and a single delay can't be used for every sequence.

Note: The calibration needs to be done only if the projection looks wrong. Over time the position of the markers might shift a little bit resulting in errors. 

The static camera should not need to be recalibrated, only in case it was accidentaly moved.

### Generating frames

In order to calibrate and find the optimal delay, we need some manuaally lable some points. 

In order to do that we first generate and save some frames from the events. Then we go over the frames and do the labeling. 
It is more convinient to first generate all the frames and then label. Loading the events might take a while and doing it on the fly when labeling takes more time. 

For generating the frames for a single sequence there is a script: `scripts/save_dvs_frames.py`. The help messages in the scritp should help with arguments required.

It is convient to generate all frames for every sequence of a subject. Thre is a bash script for this: `scripts/bash/save_frames_all.sh`. The path is hardcoded in the script. Change it to change the path.

### Labeling

The labeling for the calibration needs to be more accurate and in general is better to have as many points as possible. For finding the delay it is ok to use less points. (2/3 points, from 2/3 different points in the sequence)

In general we aim to label the markers that are on the body of the person. They are sometime visible in the events (but not always).

The script for labeling is: `scripts/label_dvs.py`. Again, look a the help messages for the arguments. The sames for labeling all the sequences one after another is: `scripts/bash/label_all.sh`

Note: one of the arguments is: `--manual`. Which controls how the labeling is done.

In both cases, a window will pop up showing the frames generated from events. Click on the points to label them, press `Space` to skip a frame.

#### Manual labels (for calibration)

Each time a point is clicked in the frame, allows to chose what label should be used. This method is more time consumin, but allows to have a different set of points for each frame.

The label needs to be inserted from terminal from a list of option. Just write the number corresponding to the desired label and press `Enter`. Press `K` to finish labeling a frame. `Space` to skip (it also discard any label for the current frame if any, good in case of a mistake).

Use this for the extrinsic calibration. (as many points as possible [>5 frames at least])

#### Fixed labels (for delay)

Label from a predefined list of labels (see script args). This method is faster, but less flexible. The name of the current marker to label will be shown on the frame.

Note: is is better to use marker that move fast, i.g. hands, arms.
The default markers used by the `label_all.sh` script should already work for all sequences. Careful that the marker used when the camera is static of moving slowly are different from when the camera is moving fast.

### Calibration

If the labeling is done it should be strightforward. Just run the script: `scripts/extrinsic_calibrate.py` with the proper arguments. A projetion matrix will be saved.

### Delay optimization

After the extrinsic calibration is found we can find the optimal delay for each sequence. We need at least a couple of points for each sequence from different points in time. 

Run the script: `scripts/bash/delay_optimise_all.sh`. Again change the path to the desired one.

### Degug

The `debug_all.sh` script allows to verify the resutls. It projects the points on all the previously generated frames.

### Video

`generate_video_all.sh` creates a video combining the events and the projections. It can take a long time to run for every sequence. (could be improved by running each sequence in a different process, just add & at the end of the method call. The prints won't be reliable).

### Export data

TODO

export the projections in file format to be used by other methods.