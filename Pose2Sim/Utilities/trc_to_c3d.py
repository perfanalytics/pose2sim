#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Convert trc files to c3d                     ##
    ##################################################
    
    Converts trc files to c3d files.
    
    Usage: 
    from Pose2Sim.Utilities import trc_to_c3d; trc_to_c3d.trc_to_c3d_func(r'<input_trc_file>')
    python -m trc_to_c3d -t <path_to_trc_path>
    python -m trc_to_c3d --trc_path <path_to_trc_path> --c3d_path <output_c3d_file>
'''


## INIT
import argparse
import numpy as np
import c3d


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HuMin Kim, David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.8'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def extract_marker_data(trc_path):
    '''
    Extract marker names and coordinates from a trc file.

    INPUTS:
    - trc_path: Path to the trc file

    OUTPUTS:
    - marker_names: List of marker names
    - marker_coords: Array of marker coordinates (n_frames, t+3*n_markers)
    '''

    # marker names
    with open(trc_path, 'r') as file:
        lines = file.readlines()
        marker_names_line = lines[3]
        marker_names = marker_names_line.strip().split('\t')[2::3]

    # time and marker coordinates
    trc_data_np = np.genfromtxt(trc_path, skip_header=5, delimiter = '\t')[:,1:] 

    return marker_names, trc_data_np


def create_c3d_file(c3d_path, marker_names, trc_data_np):
    '''
    Create a c3d file from the data extracted from a trc file.

    INPUTS:
    - c3d_path: Path to the c3d file
    - marker_names: List of marker names
    - trc_data_np: Array of marker coordinates (n_frames, t+3*n_markers)

    OUTPUTS:
    - c3d file
    '''

    # retrieve frame rate
    times = trc_data_np[:,0]
    frame_rate = round((len(times)-1) / (times[-1] - times[0]))

    # write c3d file
    writer = c3d.Writer(point_rate=frame_rate, analog_rate=0, point_scale=1.0, point_units='mm', gen_scale=-1.0)
    writer.set_point_labels(marker_names)
    
    for frame in trc_data_np:
        residuals = np.full((len(marker_names), 1), 0.0)
        cameras = np.zeros((len(marker_names), 1))
        coords = frame[1:].reshape(-1,3)*1000
        points = np.hstack((coords, residuals, cameras))
        writer.add_frames([(points, np.array([]))])

    writer.set_start_frame(0)
    writer._set_last_frame(len(trc_data_np)-1)

    with open(c3d_path, 'wb') as handle:
        writer.write(handle)


def trc_to_c3d_func(*args):
    '''
    Converts trc files to c3d files.
    
    Usage: 
    from Pose2Sim.Utilities import trc_to_c3d; trc_to_c3d.trc_to_c3d_func(r'<input_trc_file>')
    python -m trc_to_c3d -t <path_to_trc_path>
    python -m trc_to_c3d --trc_path <path_to_trc_path> --c3d_path <output_c3d_file>    
    '''

    try:
        trc_path = args[0]['trc_path'] # invoked with argparse
        if args[0]['c3d_path'] == None:
            c3d_path = trc_path.replace('.trc', '.c3d')
        else:
            c3d_path = args[0]['c3d_path']
    except:
        trc_path = args[0] # invoked as a function
        c3d_path = trc_path.replace('.trc', '.c3d')

    marker_names, trc_data_np = extract_marker_data(trc_path)
    create_c3d_file(c3d_path, marker_names, trc_data_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TRC files to C3D files.')
    parser.add_argument('-t', '--trc_path', type=str, required=True, help='trc input file path')
    parser.add_argument('-c', '--c3d_path', type=str, required=False, help='c3d output file path')
    args = vars(parser.parse_args())

    trc_to_c3d_func(args)
