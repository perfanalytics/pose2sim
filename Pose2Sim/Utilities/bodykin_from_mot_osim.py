#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Build csv from mot and osim files            ##
    ##################################################
    
    Build a csv file which stores locations and orientations of all bodies
    calculated from a .mot motion file and a .osim model file.
    
    Equivalent to OpenSim Analysis -> BodyKinematics but without the bugs in 
    orientations due to their use of Euler angle instead of homography matrices
    Angles are unwrapped (np.unwrap) with an assumed period of 2*pi/360.

    Transforms from OpenSim's yup to Blender's zup unless you set direction = 'yup'
    An optional arugment may be passed by the user to maintain the opensim convention ('yup') 

    Beware, it can be quite slow depending on the complexity 
    of the model and on the number of frames.
    
    Usage: 
    from Pose2Sim.Utilities import bodykin_from_mot_osim; bodykin_from_mot_osim.bodykin_from_mot_osim_func(r'<input_mot_file>', r'<output_osim_file>', r'<output_csv_file>')
    bodykin_from_mot_osim -m input_mot_file -o input_osim_file
    bodykin_from_mot_osim -m input_mot_file -o input_osim_file -c output_csv_file -d direction
'''

## INIT
import os
import numpy as np
import opensim as osim
import argparse

#direction = 'zup' # 'zup' or 'yup'

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, Jonathan Camargo"
__copyright__ = "Copyright 2023, BlendOSim & Sim2Blend"
__credits__ = ["David Pagnon", "Jonathan Camargo"]
__license__ = "MIT License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--input_mot_file', required = True, help='input .mot file')
    parser.add_argument('-o', '--input_osim_file', required = True, help='input .osim file')
    parser.add_argument('-c', '--csv_output_file', required = False, help='output csv file')
    parser.add_argument('-d', '--direction', required=False, help='vertical axis direction')
    args = vars(parser.parse_args())
    
    bodykin_from_mot_osim_func(args)

def bodykin_from_mot_osim_func(*args):
    '''
    Build a csv file which stores locations and orientations of all bodies
    calculated from a .mot motion file and a .osim model file.
    
    Equivalent to OpenSim Analysis -> BodyKinematics but without the bugs in 
    orientations due to their use of Euler angle instead of homography matrices
    
    Usage: 
    from Pose2Sim.Utilities import bodykin_from_mot_osim; bodykin_from_mot_osim.bodykin_from_mot_osim_func(r'<input_mot_file>', r'<output_osim_file>', r'<output_csv_file>','<direction>')
    bodykin_from_mot_osim -m input_mot_file -o input_osim_file
    bodykin_from_mot_osim -m input_mot_file -o input_osim_file -t output_csv_file -d direction
    '''
    
    try:
        motion_path = args[0]['input_mot_file'] # invoked with argparse
        osim_path = args[0]['input_osim_file']
        if args[0]['csv_output_file'] == None:
            output_csv_file = motion_path.replace('.mot', '.csv')
        else:
            output_csv_file = args[0]['csv_output_file']
        if args[0]['direction'] == None:
            direction = 'zup'
        else:
            direction = args[0]['direction']
    except:
        motion_path = args[0] # invoked as a function
        osim_path = args[1]
        try:
            output_csv_file = args[2]
        except:
            output_csv_file = motion_path.replace('.mot', '.csv')
        try:
            direction = args[3]
        except:
            direction = 'zup'
    
    
    # Read model and motion files
    model = osim.Model(osim_path)
    motion_data = osim.TimeSeriesTable(motion_path)
    
    # # Degrees or radians
    # with open(motion_path) as m_p:
        # while True:
            # line =  m_p.readline()
            # if 'inDegrees' in line:
                # break
    # if 'yes' in line:
        # in_degrees = True
    # else:
        # in_degrees = False

    # Model: get model coordinates and bodies
    model_coordSet = model.getCoordinateSet()
    # coordinates = [model_coordSet.get(i) for i in range(model_coordSet.getSize())]
    # coordinates = [c for c in coordinates if '_beta' not in c.getName()]
    # coordinateNames = [c.getName() for c in coordinates]
    coordinateNames = motion_data.getColumnLabels()
    model_bodySet = model.getBodySet()
    bodies = [model_bodySet.get(i) for i in range(model_bodySet.getSize())]
    bodyNames = [b.getName() for b in bodies]

    # Motion: read coordinates and convert to radians
    times = motion_data.getIndependentColumn()
    motion_data_np = motion_data.getMatrix().to_numpy()
    for i, c in enumerate(coordinateNames):
        if model_coordSet.get(c).getMotionType() == 1: # 1: rotation, 2: translation, 3: coupled
            if  motion_data.getTableMetaDataAsString('inDegrees') == 'yes':
    # if in_degrees:
        # for i in range(len(coordinates)):
            # if coordinates[i].getMotionType() == 1: # 1: rotation, 2: translation, 3: coupled
                motion_data_np[:,i] = motion_data_np[:,i] * np.pi/180 # if rotation, convert to radians

    # Animate model
    state = model.initSystem()
    loc_rot_frame_all = []
    H_zup = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]])
    print('Time frame:')
    for n in range(motion_data.getNumRows()):
        print(times[n], 's')
        # Set model struct in each time state
        for c, coord in enumerate(coordinateNames): ## PROBLEME QUAND HEADERS DE MOTION_DATA_NP ET COORDINATENAMES SONT PAS DANS LE MEME ORDRE
            try:
                model.getCoordinateSet().get(coord).setValue(state, motion_data_np[n,c], enforceContraints=False)
            except:
                pass
        # model.assemble(state)
        model.realizePosition(state) # much faster (IK already done, no need to compute it again)
           
        # Use state of model to get body coordinates in ground
        loc_rot_frame = []
        for b in bodies:
            H_swig = b.getTransformInGround(state)
            T = H_swig.T().to_numpy()
            R_swig = H_swig.R()
            R = np.array([[R_swig.get(0,0), R_swig.get(0,1), R_swig.get(0,2)],
                [R_swig.get(1,0), R_swig.get(1,1), R_swig.get(1,2)],
                [R_swig.get(2,0), R_swig.get(2,1), R_swig.get(2,2)]])
            H = np.block([ [R,T.reshape(3,1)], [np.zeros(3), 1] ])
            
            # y-up to z-up
            if direction=='zup':
                H = H_zup @ H
            
            # Convert matrix to loc and rot, and export to csv
            loc_x, loc_y, loc_z = H[0:3,3]
            R_mat = H[0:3,0:3]
            sy = np.sqrt(R_mat[1,0]**2 +  R_mat[0,0]**2) # singularity when y angle is +/- pi/2
            if sy>1e-6:
                rot_x = np.arctan2(R_mat[2,1], R_mat[2,2])
                rot_y = np.arctan2(-R_mat[2,0], sy)
                rot_z = np.arctan2(R_mat[1,0], R_mat[0,0])
            else: # to be verified
                rot_x = np.arctan2(-R_mat[1,2], R_mat[1,1])
                rot_y = np.arctan2(-R[2,0], sy)
                rot_z = 0
            loc_rot_frame.extend([loc_x, loc_y, loc_z, rot_x, rot_y, rot_z])
        
        loc_rot_frame_all.append(loc_rot_frame)

    # Create arrays and headers
    loc_rot_frame_all_np = np.array(loc_rot_frame_all)
    loc_rot_frame_all_np = np.insert(loc_rot_frame_all_np, 0, times, axis=1) # insert time column
    bodyHeader = 'times, ' + ''.join([f'{b}_x, {b}_y, {b}_z, {b}_rotx, {b}_roty, {b}_rotz, ' for b in bodyNames])[:-2]

    # Unwrap angles
    for n,col in enumerate(bodyHeader.split(', ')):
        if '_rot' in col:
            loc_rot_frame_all_np[:,n] = np.unwrap(loc_rot_frame_all_np[:,n],period=2*np.pi)

    # Export to csv
    np.savetxt(os.path.splitext(output_csv_file)[0]+'.csv', loc_rot_frame_all_np, delimiter=',', header=bodyHeader)
    
    print(f'CSV file generated at {os.path.splitext(output_csv_file)[0]+".csv"}.\n')
    
if __name__ == '__main__':
    main()
