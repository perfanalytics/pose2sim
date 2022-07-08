---
title: 'Pose2Sim: An Open Source Python Package for multiview markerless kinematics'
tags:
  - python
  - markerless kinematics
  - motion capture
  - sports performance analysis
  - openpose
  - opensim
authors:
  - name: David Pagnon^[corresponding author] 
    orcid: 0000-0002-6891-8331
    affiliation: 1, 2
  - name: Mathieu Domalain
    orcid: 0000-0002-4518-8479
    affiliation: 2
  - name: Lionel Reveret
    orcid: 0000-0002-0810-5187
    affiliation: 1, 3
affiliations:
 - name: Laboratoire Jean Kuntzmann, Université Grenoble Alpes, 700 avenue Centrale, 38400 Saint Martin d’Hères, France
   index: 1
 - name: Institut Pprime, 2 Bd des Frères Lumière, 86360 Chasseneuil-du-Poitou, France
   index: 2
 - name: Inria Grenoble Rhône-Alpes, 38330 Montbonnot-Saint-Martin, France
   index: 3
date: January 24 2021
bibliography: paper.bib
---

# Summary

`Pose2Sim` provides a workflow for 3D markerless kinematics, as an alternative to the more usual marker-based motion capture methods.\
`Pose2Sim` stands for "OpenPose to OpenSim", as it uses OpenPose inputs (2D coordinates obtained from multiple videos) and leads to an OpenSim result (full-body 3D joint angles). 

The repository presents a framework for:\
• Detecting 2D joint coordinates from videos, e.g. via OpenPose [@Cao_2019], \
• Calibrating cameras, \
• Tracking the main person on the scene, \
• Triangulating 2D joint coordinates and storing them as 3D positions in a .trc file, \
• Filtering these calculated 3D positions, \
• Scaling and running inverse kinematics via OpenSim [@Delp_2007; @Seth_2018], in order to obtain full-body 3D joint angles.

Each task is easily customizable, and requires only moderate Python skills. Pose2Sim is accessible at [https://github.com/perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim). 

# Statement of need

For the last few decades, marker-based kinematics has been considered as the best choice for the analysis of human movement, when regarding the trade-off between ease-of-use and accuracy. However, a marker-based system is hard to set outdoors or in "ecological" conditions, and it requires placing markers on the body, which can hinder natural movement. 

The emergence of markerless kinematics opens up new possibilities. Indeed, the interest in deep learning pose estimation neural networks has been growing fast since 2015 [@Zheng_2022], which makes it now possible to collect accurate and reliable kinematic data without the use of physical markers. OpenPose, for example, is a widespread open-source software which provides 2D joint coordinate estimations from videos. These coordinates can then be triangulated in order to produce 3D positions. Yet, when it comes to biomechanic analysis of human motion, it is often more useful to obtain joint angles than absolute positions. Indeed, joint angles allow for better comparison among trials and individuals, and they represent the first step for other analysis such as inverse dynamics. 

OpenSim is an other widespread open-source software which helps compute 3D joint angles, usually from marker coordinates. It lets scientists define a detailed muskuloskeletal model, scale it to individual subjects, and perform inverse kinematics with customizable biomechanical constraints. It provides other features such as net joint moments calculation or individual muscle forces resolution, although this is out of the scope of our contribution.

The goal of `Pose2Sim` is to build a bridge between the communities of computer vision and biomechanics, by providing a simple and open-source pipeline connecting the two aforementioned state-of-the-art tools: OpenPose and OpenSim. 
`Pose2Sim` has already been used and tested in a number of situations (walking, running, cycling, balancing, swimming, boxing), and published in peer-review scientific publications [@Pagnon_2021; @Pagnon_2022] assessing its robustness and accuracy. The combination of its ease of use, customizable characteristics, and robustness and accuracy makes it promising, especially for "in-the-wild" sports movement analysis.

So far, little work has been done towards obtaining 3D angles from multiple views [@Zheng_2022]. However, two softwares are worth being mentionend. Anipose [@Karashchuk_2021] proposes a Python open-source framework which allows for joint angle estimation with spatio-temporal constraints, but it is primarily designed for animal motion analysis. Theia3D [@Kanko_2021] provides a software for human gait kinematics from videos. Although the GUI is more user friendly, it is not open-source nor easily customizable. Our results on inverse kinematics are similar, or slightly better [@Pagnon_2022]. 

# Features
## Pose2Sim workflow

`Pose2Sim` connects two of the most widely recognized (and open source) pieces of software of their respective fields:\
• OpenPose [@Cao_2019], a 2D human pose estimation neural network\
• OpenSim [@Delp_2007], a 3D biomechanics analysis software

![Pose2Sim full pipeline: (1) OpenPose 2D joint detection; (2i) Camera calibration; (2ii–iv) Tracking the person of interest, Triangulating his coordinates, and Filtering them; (3) Constraining the 3D coordinates to a physically consistent OpenSim skeletal model.\label{fig:pipeline}](Pipeline.png)

The workflow is organized as follows \autoref{fig:pipeline}:\
1. Preliminary OpenPose [@Cao_2019] 2D keypoint detection.\
2. Pose2Sim core includes 4 customizable steps:\
&nbsp;&nbsp;&nbsp;&nbsp;2.i. Camera calibration\
&nbsp;&nbsp;&nbsp;&nbsp;2.ii. Tracking of the person viewed by the most cameras\
&nbsp;&nbsp;&nbsp;&nbsp;2.iii. 2D keypoint triangulation\
&nbsp;&nbsp;&nbsp;&nbsp;2.iv. 3D coordinates filtering\
3. A full-body OpenSim [@Delp_2007] skeletal model with OpenPose keypoints is provided, as well as scaling and inverse kinematics setup files. As the position of triangulated keypoints are not dependent on the operator nor on the subject, these setup files can be taken as is.

OpenPose, OpenSim, and the whole `Pose2Sim` workflow run from any video cameras, on any computer, equipped with any operating system. 


## Pose2Sim core
Each step of the Pose2Sim core is easily customizable through the 'User/Config.toml' file. Among other things, users can edit:\
• The project hierarchy, the video framerate, the range of analyzed frames,\
• The OpenPose model they wish to use. They can also use AlphaPose [@Fang_2017], or even create their own model (e.g. with DeepLabCut [@Mathis_2018]),\
• Whether they are going to calibrate their cameras with a checkerboard, or to simply convert a calibration file provided by a Qualisys system,\
• Which keypoint they want to track in order to automatically single out the person of interest,\
• The thresholds in confidence and in reprojection error for using or not a camera while triangulating a keypoint,\
• The minimum number of cameras below which the keypoint won't be triangulated at this frame,\
• The interpolation and filter types and parameters.

## Pose2Sim utilities
Some standalone Python tools are also provided.

**Conversion to and from Pose2Sim** 

• `DLC_to_OpenPose.py`
Converts a DeepLabCut [@Mathis_2018] (h5) 2D pose estimation file into OpenPose [@Cao_2019] (json) files.\
• `calib_qca_to_toml.py`
Converts a Qualisys .qca.txt calibration file to the Pose2Sim .toml calibration file.\
• `calib_toml_to_qca.py`
Converts a Pose2Sim .toml calibration file (e.g., from a checkerboard) to a Qualisys .qca.txt calibration file.\
• `calib_from_checkerboard.py`
Calibrates cameras with images or a video of a checkerboard, saves calibration in a Pose2Sim .toml calibration file.\
• `c3d_to_trc.py`
Converts 3D point data of a .c3d file to a .trc file compatible with OpenSim. No analog data (force plates, emg) nor computed data (angles, powers, etc) are retrieved.


**Plotting tools**

• `json_display_with_img.py` 
Overlays 2D detected json coordinates on original raw images. High confidence keypoints are green, low confidence ones are red.\
• `json_display_without_img.py`
Plots an animation of 2D detected json coordinates.\
• `trc_plot.py`
Displays X, Y, Z coordinates of each 3D keypoint of a TRC file in a different matplotlib tab.\


**Other trc tools**

• `trc_desample.py`
Undersamples a trc file.
• `trc_Zup_to_Yup.py`
Changes Z-up system coordinates to Y-up system coordinates.\
• `trc_filter.py`
Filters trc files. Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median.\
• `trc_gaitevents.py`
Detects gait events from point coordinates according to [@Zeni_2008].\

# Acknowledgements

We acknowledge the dedicated people involved in major software programs and packages used by Pose2Sim, such as Python, OpenPose, OpenSim, OpenCV [@Bradski_2000], and many others. 

# References
