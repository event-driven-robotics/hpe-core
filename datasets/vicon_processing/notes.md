# vicon recordings

The c3d file containes all the 3d points recorded with labels. The data is divided in frames containing the measurement at each timestep.

**note**: not all the data seems to be labelled. Only `giovanna1` containes labelled joints, but only for the lower body. Labelling of marker and joint position estimation should probably be done in vicon nexus and the then exports in `c3d` data. Still, we can perform all the projections on the marker position as a test and then repeat the same process for the joint center positions.

## image projection

We have the markers position in 3D in the world coorinates. We also have the marker postions for the camera. 
What is missing for the projection:
- camera intrinsic (we have this from calibration)
- camera markers frame (can be computed from vicon data)
- transformation between camera marker frame and actual camera frame (needs to be found)

If we assume the camera is calibrated, and the we have a frame defined by the camera markers. We can estimate the projection / transformation between the camera marker frame and the actual camera frame with least square (or similar) (?). OpencV should also have a calibrateCamera that should do this.

### test without camera markers

We can check if the projection estimation works by ptojecting 3D points in world coordinates into camera image points. 

The method seems to be working both with and without giving the intrinsic camera matrix. The notebooks `test_projection.ipynb` and `test_projection_with_k.ipynb` show the results without and with the intrinsic K respectively.

When K is used we also take into account the camera distortion since it is given by the calibration procedure. 
The solution is found by using `scipy.optimize.minimize`. The goal is to minimize the geometric distance between the projection and the labeled data. 
$$ p_p = K(R|t)P_w $$
$$ p_{ul} = undistort(p_l)$$
$$min(||p_{ul} - p_p||_2)$$

where $K$ and the distortion coeffiecients are given by the calibration. $P_w$ are the 3D points from the vicon in world coordinates. $p_l$ is the labelled data from the camera imaeg and $P_{ul}$ are the same points undistorted. Only $R|t$ need to be estimated. 

**multiple solutions**: is the transformation unique? Or are there multiple solutions? When trying to plot the estimated pose in 3D world coordinates, the position of the camera appears to be behind the person (which is wrong). It could be a mistake in the estimation or the solution found is only one of the possible solutions.

## image labeling

The class `DvsLabeler` in `projection.py` allows to manually label the points from DVS frames. 