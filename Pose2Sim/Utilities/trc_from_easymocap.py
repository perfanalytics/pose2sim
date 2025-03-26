#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #####################################################
    ## Convert EasyMocap results to TRC                ##
    #####################################################
    
    Converts EasyMocap keypoints3D results to TRC files.
    If several people are detected, writes one TRC file per person.
    
    N.B.: If you run EasyMocap with a different model, edit KEYPOINT_NAMES 
    accordingly (names and order of keypoints).

    N.B.: EasyMocap currently attributes even IDs to the detected people, 
    so the trc files with odd numbers may be empty. Left this way for it
    to still work the day they fix it.

    N.B.: Trc framerate is set to 1 by default.

    Usage: 
    trc_from_easymocap -i input_keypoint_dir
    trc_from_easymocap -i input_keypoint_dir -o output_trc_dir
    import trc_from_easymocap; trc_from_easymocap.trc_from_easymocap_func(input_keypoint_dir=r'<input_keypoint_dir>', output_trc_dir=r'<output_trc_dir>')
'''


## CONSTANTS
KEYPOINT_NAMES = ['Nose', 'Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist','CHip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye','REar','LEar','LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']


## INIT
import argparse
import os
import glob
import json
import numpy as np
import pandas as pd


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_keypoint_dir', required = True, help='directory on input keypoints3D folder with EasyMocap json files')
    parser.add_argument('-o', '--output_trc_dir', required = False, help='direction of the gait. If negative, you need to include an equal sign in the argument, eg -d=-Z')
    
    kwargs = vars(parser.parse_args())
    trc_from_easymocap_func(**kwargs)


def zup2yup(Q):
    '''
    X->Y, Y->Z, Z->X
    '''
    
    cols = list(Q.columns)
    cols = np.array([[cols[i*3+1],cols[i*3+2],cols[i*3]] for i in range(int(len(cols)/3))]).flatten()
    Q = Q[cols]
    return Q


def max_persons(keypoint_files):
    '''
    Max number of persons in easymocap results
    '''
    
    max_id = 0
    for f in keypoint_files:
        with open(f, 'r') as json_f:
            js = json.load(json_f)
            for p in js:
                if p['id'] > max_id:
                    max_id = p['id']
    max_id += 1
    return max_id


def df_from_easymocap(keypoint_files, max_id):
    '''
    Stores keypoint_files data in a list of dataframes with the IDs given by EasyMocap.
    '''
    
    Q = [[] for n in range(max_id)]
    for f in keypoint_files:
        with open(f, 'r') as json_f:
            js = json.load(json_f)
            idx_persons_in_frame = [p['id'] for p in js]
            for idx in range(max_id):
                if idx in idx_persons_in_frame:
                    x = [p['id'] for p in js].index(idx)
                    keypoints3d_frame = np.array(js[x]['keypoints3d'])[:, :3].flatten().tolist()
                else:
                    keypoints3d_frame = [np.nan]*len(KEYPOINT_NAMES)*3
                Q[idx].append(keypoints3d_frame)
    Q_df = [pd.DataFrame(Q[idx]) for idx in range(max_id)]
    return Q_df


def write_trc(Q_df, output_trc_dir, trc_root_name):
    '''
    Writes a list of dataframes in a directory with a given root name.
    '''
    
    for idx, Q in enumerate(Q_df):
        DataRate = CameraRate = OrigDataRate = 1
        NumFrames = len(Q)
        NumMarkers = len(KEYPOINT_NAMES)
        header_trc = ['PathFileType\t4\t(X/Y/Z)\t'+trc_root_name+str(idx), 
                'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
                '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, 0, NumFrames])),
                'Frame#\tTime\t' + '\t\t\t'.join(KEYPOINT_NAMES) + '\t\t',
                '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(KEYPOINT_NAMES))])]
                
        Q = zup2yup(Q_df[idx])
        Q.index = np.array(range(NumFrames)) + 1
        Q.insert(0, 't', Q.index / DataRate)
        
        trc_path = os.path.realpath(os.path.join(output_trc_dir, trc_root_name+str(idx)+'.trc'))
        with open(trc_path, 'w') as trc_o:
            [trc_o.write(line+'\n') for line in header_trc]
            Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')


def trc_from_easymocap_func(**kwargs):
    '''
    Converts EasyMocap keypoints3D results to TRC files.
    If several people are detected, writes one TRC file per person.
    
    N.B.: If you run EasyMocap with a different model, edit KEYPOINT_NAMES 
    accordingly (names and order of keypoints).

    N.B.: EasyMocap currently attributes even IDs to the detected people, 
    so the trc files with odd numbers may be empty. Left this way for it
    to still work the day they fix it.

    N.B.: Trc framerate is set to 1 by default.

    Usage: 
    trc_from_easymocap -i input_keypoint_dir
    trc_from_easymocap -i input_keypoint_dir -o output_trc_dir
    import trc_from_easymocap; trc_from_easymocap.trc_from_easymocap_func(input_keypoint_dir=r'<input_keypoint_dir>', output_trc_dir=r'<output_trc_dir>')
    '''

    input_keypoint_dir = kwargs.get('input_keypoint_dir')
    output_trc_dir = kwargs.get('output_trc_dir')

    input_keypoint_dir = os.path.abspath(input_keypoint_dir)
    output_trc_dir = os.path.abspath(output_trc_dir) if output_trc_dir else os.path.dirname(input_keypoint_dir)
    if not os.path.exists(output_trc_dir): os.makedirs(output_trc_dir)
    trc_root_name = os.path.basename(os.path.dirname(os.path.dirname(input_keypoint_dir)))

    keypoint_files = sorted(glob.glob(input_keypoint_dir+'/*.json'))
    max_id = max_persons(keypoint_files)
    Q_df = df_from_easymocap(keypoint_files, max_id)
    write_trc(Q_df, output_trc_dir=output_trc_dir, trc_root_name=trc_root_name)

    print(f"TRC files written in {output_trc_dir}")


if __name__ == '__main__':
    main()
