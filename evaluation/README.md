# Evaluating HPE and Joints Tracking Algorithms

This folder contains scripts for evaluating HPE and joints tracking algorithms. In particular, each algorithm to be compared should process each dataset to produce a `.csv` file and stored in a particular folder structure as outlined below. The scripts can be run to evaluate joint position comparisons, or joint velocity comparisons.

## Usage

The pose evaluation script can be run with
```shell
$ python3 evaluate_hpe.py
$   -d path/to/datasets/folder
$   -p path/to/predictions/folder
$   -o path/to/output/folder
$   -pck <list of floats in (0.0, 1.0]>
$   -rmse
$   -latex
```

- `-d path/to/datasets/folder` must point to a folder containing all datasets in yarp format the algorithms must be evaluated
    ```
    path/to/datasets/folder
    └───ds_a
         └─── ch<n>skeleton
                   └─── data.log
                   └─── info.log
    └───ds_b
         └─── ch<n>skeleton
                   └─── data.log
                   └─── info.log
    ...
    ```
- `-p path/to/predictions/folder` must point to a folder containing the outputs of all algorithms that must be evaluated
    ```
    path/to/predictions/folder
    └───ds_a
         └─── ch0skeleton
               └─── algo_a.txt
               └─── algo_b.txt
               ...
         └─── ch<n>skeleton
               └─── algo_a.txt
               └─── algo_b.txt
               ...
    └───ds_b
         └─── ch0skeleton
               └─── algo_a.txt
               └─── algo_b.txt
               ...
         └─── ch<m>skeleton
               └─── algo_a.txt
               └─── algo_b.txt
               ...
    ...
    ```
- `-o path/to/output/folder` is the path to the output folder that will be created and will contain all plots and tables for
  the specified metrics
- `-pck <list of floats in (0.0, 1.0]>` specifies a list of thresholds over which the PCK will be computed; if not specified, 
  PCK will not be computed
- `-rmse` flag specifying that RMSE must be computed
- `-latex` flag specifying that tables must be saved to latex format.

The output folder will contain files with metric tables and prediction plots.

## Metrics

The following metrics are computed for every algorithm

### Percentage of Correct Keypoints (PCK)
PCK is computed
* for every single body joint
* as an average on all body joints

and

* for each single input dataset
* for all datasets
* for every specified classification threshold.

The classification uses the body's head or torso size, depending on the available ground truth.

Metric output
* tables with all computed PCK values for every input algorithm's predictions
* plots showing average PCK for every algorithm over the selected classification thresholds.

### Root Mean Square Error (RMSE)
RMSE is computed
* for every body joint coordinate
* as an average on all joints for each coordinate

and

* for each single input dataset
* for all datasets.

Metric output
* tables with all computed RMSE values for every input algorithm's predictions
* plots for each joint showing ground truth and predicted x and y coordinates over time.
