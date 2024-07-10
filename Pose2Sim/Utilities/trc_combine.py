#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Combine two trc files                        ##
    ##################################################
    
    Combine two trc files.
    Example: you have run Pose2Sim with OpenPose AND with a DeepLabCut model 
    (or any other marker-based or markerless pose estimation algorithm), 
    and you want to assemble both detections before running OpenSim.
    
    Usage:
    from Pose2Sim.Utilities import trc_combine; trc_combine.trc_combine_func(r'<first_path>', r'<second_path>', r'<output_path>')
    OR python -m trc_combine -i first_path -j second_path -o output_path
    OR python -m trc_combine -i first_path -j second_path
'''


## INIT
import os
import pandas as pd
import numpy as np
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2022, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def df_from_trc(trc_path):
    '''
    Retrieve header and data from trc path.

    INPUT:
    trc_path: path to trc file

    OUTPUT:
    header: dictionary of header data
    data: pandas dataframe of data
    '''

    # DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
    df_header = pd.read_csv(trc_path, sep="\t", skiprows=1, header=None, nrows=2, encoding="ISO-8859-1")
    header = dict(zip(df_header.iloc[0].tolist(), df_header.iloc[1].tolist()))
    
    # Label1_X  Label1_Y    Label1_Z    Label2_X    Label2_Y
    df_lab = pd.read_csv(trc_path, sep="\t", skiprows=3, nrows=1)
    labels = df_lab.columns.tolist()[2:-1:3]
    labels_XYZ = np.array([[labels[i]+'_X', labels[i]+'_Y', labels[i]+'_Z'] for i in range(len(labels))], dtype='object').flatten()
    labels_FTXYZ = np.concatenate((['Frame#','Time'], labels_XYZ))
    
    data = pd.read_csv(trc_path, sep="\t", skiprows=5, index_col=False, header=None, names=labels_FTXYZ)
    
    return header, data


def combine_trc_headerdata (first_path, second_path):
    '''
    Combine headers and data from two different trc files.

    INPUT:
    first_path: path to first trc file
    second_path: path to second trc file

    OUTPUT:
    Header: dictionary of combined headers
    Data: dataframe of combined trc data
    '''

    first = df_from_trc(first_path)
    second = df_from_trc(second_path)

    frames_first = int(first[0].get('NumFrames'))
    frames_second = int(second[0].get('NumFrames'))
    NumFrames = min(frames_first, frames_second)
    OrigNumFrames = NumFrames
    NumMarkers = int(first[0].get('NumMarkers')) + int(second[0].get('NumMarkers'))

    Header = first[0]
    Header.update({'NumFrames': str(NumFrames), 'OrigNumFrames':str(OrigNumFrames), 'NumMarkers':str(NumMarkers)})

    Data = pd.concat([first[1].iloc[:NumFrames,:], second[1].iloc[:NumFrames, 2:]], axis=1)

    return Header, Data


def trc_from_header_data(Header, Data, combined_path):
    '''
    Opposite of df_from_trc: builds trc from header and data.

    INPUT:
    Header: Header dictionary
    Data: Dataframe of trc data
    combined_path: output path of combined trc files

    OUTPUT:
    writes combined trc file
    '''

    header0_str = 'PathFileType\t4\t(X/Y/Z)\t' + combined_path

    header1_str1 = '\t'.join(Header.keys())
    header1_str2 = '\t'.join(Header.values())

    labels_markers = [s.split('_')[0] for s in Data.columns][2::3]
    header2_str1 = 'Frame#\tTime\t' + '\t\t\t'.join([item.strip() for item in labels_markers]) + '\t\t'
    header2_str2 = '\t\t'+'\t'.join(['X{i}\tY{i}\tZ{i}'.format(i=i+1) for i in range(int(Header['NumMarkers']))])

    header_trc = '\n'.join([header0_str, header1_str1, header1_str2, header2_str1, header2_str2])

    with open(combined_path, 'w') as trc_o:
        trc_o.write(header_trc+'\n')
        Data.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')
        

def trc_combine_func(*args):
    '''
    Combine two trc files.
    Example: you have run Pose2Sim with OpenPose AND with a DeepLabCut model 
    (or any other marker-based or markerless pose estimation algorithm), 
    and you want to assemble both detections before running OpenSim.

    Usage:
    from Pose2Sim.Utilities import trc_combine; trc_combine.trc_combine_func(r'<first_path>', r'<second_path>', r'<output_path>')
    OR python -m trc_combine -i first_path -j second_path -o output_path
    OR python -m trc_combine -i first_path -j second_path
    '''

    try:
        first_path = os.path.realpath(args[0].get('first_path')) # invoked with argparse
        second_path = os.path.realpath(args[0].get('second_path'))
        output_path = args[0].get('output_path')
        if output_path == None: 
            output_path = os.path.join(os.path.dirname(first_path), 'combined.trc')
        else: 
            output_path = os.path.realpath(output_path)
    except:
        first_path = os.path.realpath(args[0]) # invoked as a function
        second_path = os.path.realpath(args[1])
        try:
            output_path = os.path.realpath(args[2])
        except:
            output_path = os.path.join(os.path.dirname(first_path), 'combined.trc')
    
    Header, Data = combine_trc_headerdata (first_path, second_path)
    trc_from_header_data(Header, Data, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--first_path', required = True, help='first trc file path')
    parser.add_argument('-j', '--second_path', required = True, help='second trc file path')
    parser.add_argument('-o', '--output_path', required = False, help='path of combined trc files')
    args = vars(parser.parse_args())
    
    trc_combine_func(args)
    
