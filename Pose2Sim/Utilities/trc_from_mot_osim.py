#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Build trc from mot and osim files            ##
    ##################################################
    
    Build a trc file which stores all real and virtual markers 
    calculated from a .mot motion file and a .osim model file.
    
    Beware, it can be quite slow depending on the ccomplexity 
    of the model and on the number of frames.
    
    Also, make sure that OpenSim is installed (e.g. via conda)
    
    Usage: 
    from Pose2Sim.Utilities import trc_from_mot_osim; trc_from_mot_osim.trc_from_mot_osim_func(r'<input_mot_file>', r'<output_osim_file>', r'<output_trc_file>')
    python -m trc_from_mot_osim -m input_mot_file -o input_osim_file
    python -m trc_from_mot_osim -m input_mot_file -o input_osim_file -t output_trc_file
'''


## INIT
import os
import pandas as pd
import numpy as np
import opensim as osim
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def get_marker_positions(motion_data, model, in_degrees=True):
    '''
    Get dataframe of marker positions
    
    INPUTS: 
    - motion_data: .mot file opened with osim.TimeSeriesTable
    - model: .osim file opened with osim.Model 
    
    OUTPUT:
    - marker_positions_pd: DataFrame of marker positions 
    '''
    
    # Markerset
    marker_set = model.getMarkerSet()
    marker_set_names = [mk.getName() for mk in list(marker_set)]
    marker_set_names_xyz = np.array([[m+'_x', m+'_y', m+'_z'] for m in marker_set_names]).flatten()

    # Data
    times = motion_data.getIndependentColumn()
    joint_angle_set_names = motion_data.getColumnLabels() # or [c.getName() for c in model.getCoordinateSet()]
    joint_angle_set_names = [j for j in joint_angle_set_names if not j.endswith('activation')]
    motion_data_pd = pd.DataFrame(motion_data.getMatrix().to_numpy()[:,:len(joint_angle_set_names)], columns=joint_angle_set_names)

    # Get marker positions at each state
    state = model.initSystem()
    marker_positions = []
    print('Time frame:')
    for n,t in enumerate(times):
        print('t = ', t, 's')
        # put the model in the right position
        for coord in joint_angle_set_names:
            if in_degrees and not coord.endswith('_tx') and not coord.endswith('_ty') and not coord.endswith('_tz'):
                value = motion_data_pd.loc[n,coord]*np.pi/180
            else:
                value = motion_data_pd.loc[n,coord]
            model.getCoordinateSet().get(coord).setValue(state,value)
        # get marker positions
        marker_positions += [np.array([marker_set.get(mk_name).findLocationInFrame(state, model.getGround()).to_numpy() for mk_name in marker_set_names]).flatten()]
    marker_positions_pd = pd.DataFrame(marker_positions, columns=marker_set_names_xyz)
    marker_positions_pd.insert(0, 'time', times)
    marker_positions_pd.insert(0, 'frame', np.arange(len(times)))
    
    return marker_positions_pd
    
    
def trc_from_mot_osim_func(*args):
    '''
    Build a trc file which stores all real and virtual markers 
    calculated from a .mot motion file and a .osim model file.
    
    Usage: 
    from Pose2Sim.Utilities import trc_from_mot_osim; trc_from_mot_osim.trc_from_mot_osim_func(r'<input_mot_file>', r'<output_osim_file>', r'<trc_output_file>')
    python -m trc_from_mot_osim -m input_mot_file -o input_osim_file
    python -m trc_from_mot_osim -m input_mot_file -o input_osim_file -t trc_output_file
    '''

    try:
        motion_path = args[0]['input_mot_file'] # invoked with argparse
        osim_path = args[0]['input_osim_file']
        if args[0]['trc_output_file'] == None:
            trc_path = motion_path.replace('.mot', '.trc')
        else:
            trc_path = args[0]['trc_output_file']
    except:
        motion_path = args[0] # invoked as a function
        osim_path = args[1]
        try:
            trc_path = args[2]
        except:
            trc_path = motion_path.replace('.mot', '.trc')

    # Create dataframe with marker positions
    model = osim.Model(osim_path)
    motion_data = osim.TimeSeriesTable(motion_path)
    
    # In degrees or radians
    with open(motion_path) as m_p:
        while True:
            line =  m_p.readline()
            if 'inDegrees' in line:
                break
    if 'yes' in line:
        in_degrees = True
    else:
        in_degrees = False
    
    marker_positions_pd = get_marker_positions(motion_data, model, in_degrees=in_degrees)
    
    # Trc header
    times = motion_data.getIndependentColumn()
    marker_set = model.getMarkerSet()
    marker_set_names = [mk.getName() for mk in list(marker_set)]
    
    fps = str( int(1/ ((times[-1]-times[0]) / (len(times)-1))))
    nb_frames = str(len(times))
    nb_markers = str(len(marker_set_names))
    header0_str = 'PathFileType\t4\t(X/Y/Z)\t' + os.path.basename(trc_path)

    header1 = {}
    header1['DataRate'] = fps
    header1['CameraRate'] = fps
    header1['NumFrames'] = nb_frames
    header1['NumMarkers'] = nb_markers
    header1['Units'] = 'm'
    header1['OrigDataRate'] = fps
    header1['OrigDataStartFrame'] = '0'
    header1['OrigNumFrames'] = nb_frames
    header1_str1 = '\t'.join(header1.keys())
    header1_str2 = '\t'.join(header1.values())

    header2_str1 = 'Frame#\tTime\t' + '\t\t\t'.join([mk.strip() for mk in marker_set_names]) + '\t\t'
    header2_str2 = '\t\t'+'\t'.join(['X{i}\tY{i}\tZ{i}'.format(i=i+1) for i in range(int(header1['NumMarkers']))])

    header_trc = '\n'.join([header0_str, header1_str1, header1_str2, header2_str1, header2_str2])
    
    # write data
    with open(trc_path, 'w') as trc_o:
        trc_o.write(header_trc+'\n')
    marker_positions_pd.to_csv(trc_path, header=False, sep = '\t', mode='a', index=False)
    print(f'trc file successfully saved as {trc_path}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--input_mot_file', required = True, help='input mot file')
    parser.add_argument('-o', '--input_osim_file', required = True, help='input osim file')
    parser.add_argument('-t', '--trc_output_file', required=False, help='trc output file')
    args = vars(parser.parse_args())
    
    trc_from_mot_osim_func(args)