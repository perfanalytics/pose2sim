[![Continuous integration](https://github.com/perfanalytics/pose2sim/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/perfanalytics/pose2sim/actions/workflows/continuous-integration.yml)
[![PyPI version](https://badge.fury.io/py/Pose2Sim.svg)](https://badge.fury.io/py/Pose2Sim) \
[![Downloads](https://pepy.tech/badge/pose2sim)](https://pepy.tech/project/pose2sim)
[![Stars](https://badgen.net/github/stars/perfanalytics/pose2sim)](https://github.com/perfanalytics/pose2sim/stargazers)
[![GitHub forks](https://badgen.net/github/forks/perfanalytics/pose2sim)](https://GitHub.com/perfanalytics/pose2sim/forks)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub issues](https://img.shields.io/github/issues/perfanalytics/pose2sim)](https://github.com/perfanalytics/pose2sim/issues)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/perfanalytics/pose2sim)](https://GitHub.com/perfanalytics/pose2sim/issues?q=is%3Aissue+is%3Aclosed)
\
[![status](https://joss.theoj.org/papers/a31cb207a180f7ac9838d049e3a0de26/status.svg)](https://joss.theoj.org/papers/a31cb207a180f7ac9838d049e3a0de26)
[![DOI](https://zenodo.org/badge/501642916.svg)](https://zenodo.org/badge/latestdoi/501642916)


# Pose2Sim
`Pose2Sim` provides a workflow for 3D markerless kinematics, as an alternative to the more usual marker-based motion capture methods.\
Pose2Sim stands for "OpenPose to OpenSim", as it uses OpenPose inputs (2D keypoints coordinates obtained from multiple videos) and leads to an OpenSim result (full-body 3D joint angles). Other 2D solutions can alternatively be used as inputs.

If you can only use a single camera and don't mind losing some accuracy, please consider using [Sports2D](https://github.com/davidpagnon/Sports2D).

<img src="Content/Pose2Sim_workflow.jpg" width="760">

<img src='Content/Activities_verylow.gif' title='Other more or less challenging tasks and conditions.' width="760">

## Contents
1. [Installation and Demonstration](#installation-and-demonstration)
   1. [Installation](#installation)
   2. [Demonstration Part-1: Build 3D TRC file on Python](#demonstration-part-1-build-3d-trc-file-on-python)
   3. [Demonstration Part-2: Obtain 3D joint angles with OpenSim](#demonstration-part-2-obtain-3d-joint-angles-with-opensim)
2. [Use on your own data](#use-on-your-own-data)
   1. [Prepare for running on your own data](#prepare-for-running-on-your-own-data)
   2. [2D pose estimation](#2d-pose-estimation)
      1. [With OpenPose](#with-openpose)
      2. [With BlazePose (MediaPipe)](#with-blazepose-mediapipe)
      3. [With DeepLabCut](#with-deeplabcut)
      4. [With AlphaPose](#with-alphapose)
   3. [Camera calibration](#camera-calibration)
   4. [2D Tracking of person](#2d-tracking-of-person)
   5. [3D triangulation](#3d-triangulation)
   6. [3D filtering](#3d-filtering)
   7. [OpenSim kinematics](#opensim-kinematics)
3. [Utilities](#utilities)
4. [How to cite and how to contribute](#how-to-cite-and-how-to-contribute)
   1. [How to cite](#how-to-cite)
   2. [How to contribute](#how-to-contribute)

## Installation and Demonstration

### Installation
1. **Install OpenPose** (instructions [there](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md)). \
*Windows portable demo is enough.*
2. **Install OpenSim 4.x** ([there](https://simtk.org/frs/index.php?group_id=91)). \
*Tested up to v4.4-beta on Windows. Has to be compiled from source on Linux (see [there](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Linux+Support)).*
3. ***Optional.*** *Install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). \
   Open an Anaconda terminal and create a virtual environment with typing:*
   <pre><i>conda create -n Pose2Sim python=3.7 
   conda activate Pose2Sim</i></pre>
   
3. **Install Pose2Sim**: \
If you don't use Anaconda, type `python -V` in terminal to make sure python>=3.6 is installed.
   - OPTION 1: **Quick install:** Open a terminal. 
       ```
       pip install pose2sim
       ```
     
   - OPTION 2: **Build from source and test the last changes:**
     Open a terminal in the directory of your choice and Clone the Pose2Sim repository.
       ```
       git clone https://github.com/perfanalytics/pose2sim.git
       cd pose2sim
       pip install .
       ```
          
### Demonstration Part-1: Build 3D TRC file on Python  
> _**This demonstration provides an example experiment of a person balancing on a beam, filmed with 4 calibrated cameras processed with OpenPose.**_ 

Open a terminal, enter `pip show pose2sim`, report package location. \
Copy this path and go to the Demo folder with `cd <path>\pose2sim\Demo`. \
Type `ipython`, and test the following code:
```
from Pose2Sim import Pose2Sim
Pose2Sim.calibrateCams()
Pose2Sim.track2D()
Pose2Sim.triangulate3D()
Pose2Sim.filter3D()
```
You should obtain a plot of all the 3D coordinates trajectories. You can check the logs in`Demo\Users\logs.txt`.\
Results are stored as .trc files in the `Demo/pose-3d` directory.

*N.B.:* Default parameters have been provided in `Demo\Users\Config.toml` but can be edited.
<br/>

### Demonstration Part-2: Obtain 3D joint angles with OpenSim  
> _**In the same vein as you would do with marker-based kinematics, start with scaling your model, and then perform inverse kinematics.**_ 

#### Scaling
1. Open OpenSim.
2. Open the provided `Model_Pose2Sim_Body25b.osim` model from `pose2sim/Demo/opensim`. *(File -> Open Model)*
3. Load the provided `Scaling_Setup_Pose2Sim_Body25b.xml` scaling file from `pose2sim/Demo/opensim`. *(Tools -> Scale model -> Load)*
4. Run. You should see your skeletal model take the static pose.

#### Inverse kinematics
1. Load the provided `IK_Setup_Pose2Sim_Body25b.xml` scaling file from `pose2sim/Demo/opensim`. *(Tools -> Inverse kinematics -> Load)*
2. Run. You should see your skeletal model move in the Vizualizer window.
<br/>

## Use on your own data

> _**Deeper explanations and instructions are given below.**_

### Prepare for running on your own data
  > _**Get ready.**_
  
  1. Find your `Pose2Sim\Empty_project`, copy-paste it where you like and give it the name of your choice.
  2. Edit the `User\Config.toml` file as needed, **especially regarding the path to your project**. 
  3. Populate the `raw-2d`folder with your videos.
  
       <pre>
       Project
       │
       ├──opensim
       │    ├──Geometry
       │    ├──Model_Pose2Sim_Body25b.osim
       │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
       │    └──IK_Setup_Pose2Sim_Body25b.xml
       │        
       ├── <b>raw-2d
       │    ├──vid_cam1.mp4 (or other extension)
       │    ├──...
       │    └──vid_camN.mp4</b>
       │
       └──User
           └──Config.toml
       </b>
    
### 2D pose estimation
> _**Estimate 2D pose from images with Openpose or an other pose estimation solution.**_ \
N.B.: Note that the names of your camera folders must follow the same order as in the calibration file, and end with '_json'.


  #### With OpenPose:
The accuracy and robustness of Pose2Sim have been thoroughly assessed only with OpenPose, and especially with the BODY_25B model. Consequently, we recommend using this 2D pose estimation solution. See [OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for installation and running.
* Open a command prompt in your **OpenPose** directory. \
  Launch OpenPose for each raw image folder: 
  ```
  bin\OpenPoseDemo.exe --model_pose BODY_25B --video <PATH_TO_PROJECT_DIR>\raw-2d\vid_cam1.mp4 --write_json <PATH_TO_PROJECT_DIR>\pose-2d\pose_cam1_json
  ```
* The [BODY_25B model](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/experimental_models) has more accurate results than the standard BODY_25 one and has been extensively tested for Pose2Sim. \
You can also use the [BODY_135 model](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/experimental_models), which allows for the evaluation of pronation/supination, wrist flexion, and wrist deviation.\
All other OpenPose models (BODY_25, COCO, MPII) are also supported.\
Make sure you modify the `User\Config.toml` file accordingly.
* Use one of the `json_display_with_img.py` or `json_display_with_img.py` scripts (see [Utilities](#utilities)) if you want to display 2D pose detections.

**N.B.:** *OpenPose BODY_25B is the default 2D pose estimation model used in Pose2Sim. However, other skeleton models from other 2D pose estimation solutions can be used alternatively.* \
    - You will first need to convert your 2D detection files to the OpenPose format (see [Utilities](#utilities)). \
    - Then, change the `pose_model` in the `User\Config.toml` file. You may also need to choose a different `tracked_keypoint` if the Neck is not detected by the chosen model. \
    - Finally, use the corresponding OpenSim model and setup files, which are provided in the `Empty_project\opensim` folder. 
  
 Available models are:    
    - OpenPose BODY_25B, BODY_25, BODY_135, COCO, MPII \
    - Mediapipe BLAZEPOSE \
    - DEEPLABCUT \
    - AlphaPose HALPE_26, HALPE_68, HALPE_136, COCO_133, COCO, MPII

#### With BlazePose (MediaPipe):
[BlazePose](https://google.github.io/mediapipe/solutions/pose.html) is very fast, fully runs under Python, handles upside-down postures and wrist movements (but no subtalar ankle angles). \
However, it is less robust and accurate than OpenPose, and can only detect a single person.
* Use the script `Blazepose_runsave.py` (see [Utilities](#utilities)) to run BlazePose under Python, and store the detected coordinates in OpenPose (json) or DeepLabCut (h5 or csv) format: 
  ```
  python -m Blazepose_runsave -i r"<input_file>" -dJs
  ```
  Type in `python -m Blazepose_runsave -h` for explanation on parameters and for additional ones.
* Make sure you change the `pose_model` and the `tracked_keypoint` in the `User\Config.toml` file.

#### With DeepLabCut:
If you need to detect specific points on a human being, an animal, or an object, you can also train your own model with [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).
1. Train your DeepLabCut model and run it on your images or videos (more intruction on their repository)
2. Translate the h5 2D coordinates to json files (with `DLC_to_OpenPose.py` script, see [Utilities](#utilities)): 
   ```
   python -m DLC_to_OpenPose -i r"<input_h5_file>"
   ```
3. Report the model keypoints in the 'skeleton.py' file, and make sure you change the `pose_model` and the `tracked_keypoint` in the `User\Config.toml` file.
4. Create an OpenSim model if you need 3D joint angles.

#### With AlphaPose:
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) is one of the main competitors of OpenPose, and its accuracy is comparable. As a top-down approach (unlike OpenPose which is bottom-up), it is faster on single-person detection, but slower on multi-person detection.
* Install and run AlphaPose on your videos (more intruction on their repository)
* Translate the AlphaPose single json file to OpenPose frame-by-frame files (with `AlphaPose_to_OpenPose.py` script, see [Utilities](#utilities)): 
   ```
   python -m AlphaPose_to_OpenPose -i r"<input_alphapose_json_file>"
   ```
* Make sure you change the `pose_model` and the `tracked_keypoint` in the `User\Config.toml` file.




<img src="Content/Pose2D.png" width="760">

N.B.: Markers are not needed in Pose2Sim and were used here for validation


<details>
  <summary>The project hierarchy becomes: (CLICK TO SHOW)</summary>
    <pre>
   Project
   │
   ├──opensim
   │    ├──Geometry
   │    ├──Model_Pose2Sim_Body25b.osim
   │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
   │    └──IK_Setup_Pose2Sim_Body25b.xml
   │
   <i><b>├──pose-2d
   │    ├──pose_cam1_json
   │    ├──...
   │    └──pose_camN_json</i></b>
   │        
   ├── raw-2d
   │    ├──vid_cam1.mp4
   │    ├──...
   │    └──vid_camN.mp4
   │
   └──User
       └──Config.toml
   </pre>
</details>

### Camera calibration
> _**Calibrate your cameras.**_

> - Note 1: Calibration is tricky. I plan to make it more user-friendly, but in the meantime, you can refer to [this issue](https://github.com/perfanalytics/pose2sim/issues/18) if you struggle.
> - Note 2: Cameras need to be synchronized. If they are not natively, you can use [this script](https://github.com/perfanalytics/pose2sim/blob/draft/Pose2Sim/Utilities/synchronize_cams.py).


1. If you already have a calibration file (.qca.txt from Qualisys or .xcp from Vicon for example):
- copy it in the `calib-2d` folder
- set [calibration] type to 'qca' or to 'vicon' in your `Config.toml` file. Change `binning_factor` to 2 if you film in 540p

or

2. If you have taken pictures or videos of a checkerboard with your cameras:
- create a folder for each camera in your `calib-2d` folder,
- copy there the images or videos of the checkerboard
- set [calibration] type to 'checkerboard' in your `Config.toml` file, and adjust other parameters.

Open an Anaconda prompt or a terminal, type `ipython`.\
By default, `calibrateCams()` will look for `Config.toml` in the `User` folder of your current directory. If you want to store it somewhere else (e.g. in your data directory), specify this path as an argument: `Pose2Sim.calibrateCams(r'path_to_config.toml')`.

```
from Pose2Sim import Pose2Sim
Pose2Sim.calibrateCams()
```

Output:\
<img src="Content/Calib2D.png" width="760">

<img src="Content/CalibFile.png" width="760">


<details>
  <summary>The project hierarchy becomes: (CLICK TO SHOW)</summary>
    <pre>
   Project
   │
   ├──<i><b>calib-2d
   │   ├──calib_cam1_img
   │   ├──...
   │   ├──calib_camN_img
   │   └──Calib.toml</i></b>
   │
   ├──opensim
   │    ├──Geometry
   │    ├──Model_Pose2Sim_Body25b.osim
   │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
   │    └──IK_Setup_Pose2Sim_Body25b.xml
   │
   ├──pose-2d
   │    ├──pose_cam1_json
   │    ├──...
   │    └──pose_camN_json
   │        
   ├── raw-2d
   │    ├──vid_cam1.mp4
   │    ├──...
   │    └──vid_camN.mp4
   │
   └──User
       └──Config.toml
   </pre>
</details>

### 2D tracking of person
> _**Track the person viewed by the most cameras, in case of several detections by OpenPose.**_ \
*N.B.: Skip this step if only one person is in the field of view.*

Open an Anaconda prompt or a terminal, type `ipython`.\
By default, `track2D()` will look for `Config.toml` in the `User` folder of your current directory. If you want to store it somewhere else (e.g. in your data directory), specify this path as an argument: `Pose2Sim.track2D(r'path_to_config.toml')`.
```
from Pose2Sim import Pose2Sim
Pose2Sim.track2D()
```

Check printed output. If results are not satisfying, try and release the constraints in the `Config.toml` file.

Output:\
<img src="Content/Track2D.png" width="760">


<details>
  <summary>The project hierarchy becomes: (CLICK TO SHOW)</summary>
    <pre>
   Project
   │
   ├──calib-2d
   │   ├──calib_cam1_img
   │   ├──...
   │   ├──calib_camN_img
   │   └──Calib.toml
   │
   ├──opensim
   │    ├──Geometry
   │    ├──Model_Pose2Sim_Body25b.osim
   │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
   │    └──IK_Setup_Pose2Sim_Body25b.xml
   │
   ├──pose-2d
   │   ├──pose_cam1_json
   │   ├──...
   │   └──pose_camN_json
   │
   <i><b>├──pose-2d-tracked
   │   ├──tracked_cam1_json
   │   ├──...
   │   └──tracked_camN_json</i></b>
   │        
   ├── raw-2d
   │    ├──vid_cam1.mp4
   │    ├──...
   │    └──vid_camN.mp4
   │
   └──User
       └──Config.toml
   </pre>
</details>
   
### 3D triangulation
> _**Triangulate your 2D coordinates in a robust way.**_

Open an Anaconda prompt or a terminal, type `ipython`.\
By default, `triangulate3D()` will look for `Config.toml` in the `User` folder of your current directory. If you want to store it somewhere else (e.g. in your data directory), specify this path as an argument: `Pose2Sim.triangulate3D(r'path_to_config.toml')`.

```
from Pose2Sim import Pose2Sim
Pose2Sim.triangulate3D()
```

Check printed output, and vizualise your trc in OpenSim.\
If your triangulation is not satisfying, try and release the constraints in the `Config.toml` file.

Output:\
<img src="Content/Triangulate3D.png" width="760">


<details>
  <summary>The project hierarchy becomes: (CLICK TO SHOW)</summary>
    <pre>
   Project
   │
   ├──calib-2d
   │   ├──calib_cam1_img
   │   ├──...
   │   ├──calib_camN_img
   │   └──Calib.toml
   │
   ├──opensim
   │    ├──Geometry
   │    ├──Model_Pose2Sim_Body25b.osim
   │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
   │    └──IK_Setup_Pose2Sim_Body25b.xml
   │
   ├──pose-2d
   │   ├──pose_cam1_json
   │   ├──...
   │   └──pose_camN_json
   │
   ├──pose-2d-tracked
   │   ├──tracked_cam1_json
   │   ├──...
   │   └──tracked_camN_json
   │
   <i><b>├──pose-3d
       └──Pose-3d.trc</i></b>>
   │        
   ├── raw-2d
   │    ├──vid_cam1.mp4
   │    ├──...
   │    └──vid_camN.mp4
   │
   └──User
       └──Config.toml
   </pre>
</details>


### 3D Filtering
> _**Filter your 3D coordinates.**_

Open an Anaconda prompt or a terminal, type `ipython`.\
By default, `filter3D()` will look for `Config.toml` in the `User` folder of your current directory. If you want to store it somewhere else (e.g. in your data directory), specify this path as an argument: `Pose2Sim.filter3D(r'path_to_config.toml')`.

```
from Pose2Sim import Pose2Sim
Pose2Sim.filter3D()
```

Check your filtration with the displayed figures, and vizualise your trc in OpenSim. If your filtering is not satisfying, try and change the parameters in the `Config.toml` file.

Output:\
<img src="Content/FilterPlot.png" width="760">

<img src="Content/Filter3D.png" width="760">


<details>
  <summary>The project hierarchy becomes: (CLICK TO SHOW)</summary>
    <pre>
   Project
   │
   ├──calib-2d
   │   ├──calib_cam1_img
   │   ├──...
   │   ├──calib_camN_img
   │   └──Calib.toml
   │
   ├──opensim
   │    ├──Geometry
   │    ├──Model_Pose2Sim_Body25b.osim
   │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
   │    └──IK_Setup_Pose2Sim_Body25b.xml
   │
   ├──pose-2d
   │   ├──pose_cam1_json
   │   ├──...
   │   └──pose_camN_json
   │
   ├──pose-2d-tracked
   │   ├──tracked_cam1_json
   │   ├──...
   │   └──tracked_camN_json
   │
   <i><b>├──pose-3d
   │   ├──Pose-3d.trc
   │   └──Pose-3d-filtered.trc</i></b>
   │        
   ├── raw-2d
   │    ├──vid_cam1.mp4
   │    ├──...
   │    └──vid_camN.mp4
   │
   └──User
       └──Config.toml
   </pre>
</details>


### OpenSim kinematics
> _**Obtain 3D joint angles.**_

#### Scaling
1. Use the previous steps to capture a static pose, typically an A-pose or a T-pose.
2. Open OpenSim.
3. Open the provided `Model_Pose2Sim_Body25b.osim` model from `pose2sim/Empty_project/opensim`. *(File -> Open Model)*
4. Load the provided `Scaling_Setup_Pose2Sim_Body25b.xml` scaling file from `pose2sim/Empty_project/opensim`. *(Tools -> Scale model -> Load)*
5. Replace the example static .trc file with your own data.
6. Run
7. Save the new scaled OpenSim model.

#### Inverse kinematics
1. Use Pose2Sim to generate 3D trajectories.
2. Open OpenSim.
3. Load the provided `IK_Setup_Pose2Sim_Body25b.xml` scaling file from `pose2sim/Empty_project/opensim`. *(Tools -> Inverse kinematics -> Load)*
4. Replace the example .trc file with your own data, and specify the path to your angle kinematics output file.
5. Run
6. Motion results will appear as .mot file in the `pose2sim/Empty_project/opensim` directory (automatically saved).

<img src="Content/OpenSim.JPG" width="380">


#### Command line
Alternatively, you can use command-line tools:

- Open an Anaconda terminal in your OpenSim/bin directory, typically `C:\OpenSim <Version>\bin`.\
  You'll need to adjust the `time_range`, `output_motion_file`, and enter the full paths to the input and output `.osim`, `.trc`, and `.mot` files in your setup file.
  ```
  opensim-cmd run-tool <PATH TO YOUR SCALING OR IK SETUP FILE>.xml
  ```

- You can also run OpenSim directly in Python:
  ```
  import subprocess
  subprocess.call(["opensim-cmd", "run-tool", r"<PATH TO YOUR SCALING OR IK SETUP FILE>.xml"])
  ```

- Or take advantage of the full the OpenSim Python API. See [there](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python) for installation instructions. \
Note that it is easier to install on Python 3.7 and with OpenSim 4.2. 

<details>
  <summary>The project hierarchy becomes: (CLICK TO SHOW)</summary>
    <pre>
   Project
   │
   ├──calib-2d
   │   ├──calib_cam1_img
   │   ├──...
   │   ├──calib_camN_img
   │   └──Calib.toml
   │
   ├──<i><b>opensim</i></b>  
   │    ├──Geometry
   │    ├──Model_Pose2Sim_Body25b.osim
   │    ├──Scaling_Setup_Pose2Sim_Body25b.xml
   │    ├──<i><b>Model_Pose2Sim_Body25b_Scaled.osim</i></b>  
   │    ├──IK_Setup_Pose2Sim_Body25b.xml
   │    └──<i><b>IK_result.mot</i></b>   
   │
   ├──pose-2d
   │   ├──pose_cam1_json
   │   ├──...
   │   └──pose_camN_json
   │
   ├──pose-2d-tracked
   │   ├──tracked_cam1_json
   │   ├──...
   │   └──tracked_camN_json
   │
   ├──pose-3d
   │   ├──Pose-3d.trc
   │   └──Pose-3d-filtered.trc
   │        
   ├── raw-2d
   │    ├──vid_cam1.mp4
   │    ├──...
   │    └──vid_camN.mp4
   │
   └──User
       └──Config.toml
   </pre>
</details>

## Utilities
A list of standalone tools (see [Utilities](https://github.com/perfanalytics/pose2sim/tree/main/Pose2Sim/Utilities)), which can be either run as scripts, or imported as functions. Check usage in the docstrings of each Python file. The figure below shows how some of these toolscan be used to further extend Pose2Sim usage.


<details>
  <summary><b>Converting files and Calibrating</b> (CLICK TO SHOW)</summary>
    <pre>
`Blazepose_runsave.py`
Runs BlazePose on a video, and saves coordinates in OpenPose (json) or DeepLabCut (h5 or csv) format.

`DLC_to_OpenPose.py`
Converts a DeepLabCut (h5) 2D pose estimation file into OpenPose (json) files.

`c3d_to_trc.py`
Converts 3D point data of a .c3d file to a .trc file compatible with OpenSim. No analog data (force plates, emg) nor computed data (angles, powers, etc) are retrieved.

`calib_from_checkerboard.py`
Calibrates cameras with images or a video of a checkerboard, saves calibration in a Pose2Sim .toml calibration file.

`calib_qca_to_toml.py`
Converts a Qualisys .qca.txt calibration file to the Pose2Sim .toml calibration file (similar to what is used in [AniPose](https://anipose.readthedocs.io/en/latest/)).

`calib_toml_to_qca.py`
Converts a Pose2Sim .toml calibration file (e.g., from a checkerboard) to a Qualisys .qca.txt calibration file.

`calib_yml_to_toml.py`
Converts OpenCV intrinsic and extrinsic .yml calibration files to an OpenCV .toml calibration file.

`calib_toml_to_yml.py`
Converts an OpenCV .toml calibration file to OpenCV intrinsic and extrinsic .yml calibration files.
   </pre>
</details>

<details>
  <summary><b>Plotting tools</b> (CLICK TO SHOW)</summary>
    <pre>

`json_display_with_img.py` 
Overlays 2D detected json coordinates on original raw images. High confidence keypoints are green, low confidence ones are red.

`json_display_without_img.py`
Plots an animation of 2D detected json coordinates. 

`trc_plot.py`
Displays X, Y, Z coordinates of each 3D keypoint of a TRC file in a different matplotlib tab.
   </pre>
</details>

<details>
  <summary><b>Other trc tools</b> (CLICK TO SHOW)</summary>
    <pre>
    
`trc_desample.py`
Undersamples a trc file.

`trc_Zup_to_Yup.py`
Changes Z-up system coordinates to Y-up system coordinates.

`trc_filter.py`
Filters trc files. Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median.

`trc_gaitevents.py`
Detects gait events from point coordinates according to [Zeni et al. (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0966636207001804?via%3Dihub).

`trc_combine.py`
Combine two trc files, for example a triangulated DeepLabCut trc file and a triangulated OpenPose trc file.
   </pre>
</details>

<img src="Content/Pose2Sim_workflow_utilities.jpg" width="760">

## How to cite and how to contribute
#### How to cite
If you use this code or data, please cite [Pagnon et al., 2022b](https://doi.org/10.21105/joss.04362), [Pagnon et al., 2022a](https://www.mdpi.com/1424-8220/22/7/2712), or [Pagnon et al., 2021](https://www.mdpi.com/1424-8220/21/19/6530).
    
    @Article{Pagnon_2022_JOSS, 
      AUTHOR = {Pagnon, David and Domalain, Mathieu and Reveret, Lionel}, 
      TITLE = {Pose2Sim: An open-source Python package for multiview markerless kinematics}, 
      JOURNAL = {Journal of Open Source Software}, 
      YEAR = {2022},
      DOI = {10.21105/joss.04362}, 
      URL = {https://joss.theoj.org/papers/10.21105/joss.04362}
     }

    @Article{Pagnon_2022_Accuracy,
      AUTHOR = {Pagnon, David and Domalain, Mathieu and Reveret, Lionel},
      TITLE = {Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2: Accuracy},
      JOURNAL = {Sensors},
      YEAR = {2022},
      DOI = {10.3390/s22072712},
      URL = {https://www.mdpi.com/1424-8220/22/7/2712}
    }

    @Article{Pagnon_2021_Robustness,
      AUTHOR = {Pagnon, David and Domalain, Mathieu and Reveret, Lionel},
      TITLE = {Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 1: Robustness},
      JOURNAL = {Sensors},
      YEAR = {2021},
      DOI = {10.3390/s21196530},
      URL = {https://www.mdpi.com/1424-8220/21/19/6530}
    }

#### How to contribute

I would happily welcome any proposal for new features, code improvement, and more!\
If you want to contribute to Pose2Sim, please follow [this guide](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) on how to fork, modify and push code, and submit a pull request. I would appreciate it if you provided as much useful information as possible about how you modified the code, and a rationale for why you're making this pull request. Please also specify on which operating system and on which python version you have tested the code.

*Here is a to-do list, for general guidance purposes only:*
> <li> <b>calibrateCams:</b> (1) Intrinsic with checkerboard, extrinsic with object or ChArUco board, or (2) SBA calibration with wand (cf <a href='https://argus.web.unc.edu/'>Argus</a>, see converter <a href='https://github.com/backyardbiomech/DLCconverterDLT/blob/master/DLTcameraPosition.py'>here</a>), or (3) autocalibration <a href='https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12130'>based on a person's dimensions</a>. Also see <a href='https://github.com/lambdaloop/aniposelib'>aniposelib</a> for calibration with ChArUco.</li>
> <li> <b>synchronizeCams:</b> Synchronize cameras on 2D keypoint speeds.</li>   
> <li> <b>track2D:</b> Multiple persons association (rename to peopleAssociation and ensure backcompatibility). With a neural network instead of brute force?</li>
> <li> <b>triangulate3D:</b> Multiple persons kinematics (output multiple .trc coordinates files). See <a href="https://arxiv.org/pdf/1901.04111.pdf">Dong 2021</a>.</li>
> <li> <b>GUI:</b> Blender add-on, or webapp (e.g., with <a href="https://napari.org/stable/">Napari</a>). See <a href="https://github.com/davidpagnon/Maya-Mocap">Maya-Mocap</a> and <a href="https://github.com/JonathanCamargo/BlendOsim">BlendOSim</a>.</li>
> <li> <b>Doc:</b> Use <a href="https://www.sphinx-doc.org/en/master/">Sphinx</a> or <a href="https://www.mkdocs.org/">MkDocs</a> for clearer documentation.</li>
> </br>
> <li> Catch errors</li>
> <li> Conda package and Docker image</li>
> <li> Copy-paste muscles from OpenSim <a href="https://simtk.org/projects/lfbmodel">lifting full-body model</a> for inverse dynamics and more</li>
> <li> Implement optimal fixed-interval Kalman smoothing for inverse kinematics (<a href='https://github.com/pyomeca/biorbd/blob/f776fe02e1472aebe94a5c89f0309360b52e2cbc/src/RigidBody/KalmanReconsMarkers.cpp'>Biorbd</a> or <a href='https://github.com/antoinefalisse/opensim-core/blob/kalman_smoother/OpenSim/Tools/InverseKinematicsKSTool.cpp'>OpenSim fork</a>)</li>
> </br>
> <li> <a href="https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga887960ea1bde84784e7f1710a922b93c">Undistort</a> 2D points before triangulating (and <a href="https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L301">distort</a> them before computing reprojection error).</li>
> <li> Offer the possibility of triangulating with Sparse Bundle Adjustment (SBA), Extended Kalman Filter (EKF), Full Trajectory Estimation (FTE) (see <a href="https://github.com/African-Robotics-Unit/AcinoSet">AcinoSet</a>). </li>
> <li> Implement SLEAP as an other 2D pose estimation solution (converter, skeleton.py, OpenSim model and setup files).</li>
> <li> Outlier rejection (sliding z-score?) Also solve limb swapping</li>
> <li> Implement normalized DLT and RANSAC triangulation, as well as a triangulation refinement step (cf DOI:10.1109/TMM.2022.3171102)</li>
> <li> Utilities: convert Vicon xcp calibration file to toml</li>
> <li> Run from command line via click or typer</li>


