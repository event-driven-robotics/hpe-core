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

## Action: Reconstruct

The reconstruction is an automotic process. In the parameters it is recomended to set `Minimum Cameras to Start Trajectory` to at least `3`. If there are a lot of probelms with the labeling because not all the markers are visible, reducing this value to 2 might help. But it is not recomended.

## Action: Label

The labeling should be done automatically in the `Label` action of the pipeline. For various reasons this may not work. The `log` panel might give some information about the error. Alternatively one can do manual labeling. 

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
    - the select the points

## Action: Process Static Plug-in Gait Model
The only option that needs to be changed is the: `Output Joint centers`. It should be turned on.

The other actions of the piepline should be allautomatic and should not give too many problems.