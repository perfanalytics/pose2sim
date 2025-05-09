#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Convert c3d files to trc                     ##
    ##################################################
    
    Converts c3d files to trc files.
    Beware that it only allows you to retrieve 3D points, you won't get analog data nor computed data such as angles or powers with this code. 
    
    Usage: 
    from Pose2Sim.Utilities import c3d_to_trc; c3d_to_trc.c3d_to_trc_func(r'<input_c3d_file>')
    c3d_to_trc -i input_c3d_file
    c3d_to_trc -i input_c3d_file -o output_c3d_file
'''


## INIT
import c3d
import numpy as np
import argparse


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
    parser.add_argument('-i', '--input', required = True, help='c3d input file')
    parser.add_argument('-o', '--output', required=False, help='trc output file')
    args = vars(parser.parse_args())
    
    c3d_to_trc_func(args)


def c3d_to_trc_func(*args):
    '''
    Convert c3d to trc
    /!\ Only point data are retrieved. Analog data (force plates, emg) and 
    computed data (angles, powers, etc) will be lost.
    
    Usage: 
    import c3d_to_trc; c3d_to_trc.c3d_to_trc_func(r'<input_c3d_file>')
    c3d_to_trc -i input_c3d_file
    c3d_to_trc -i input_c3d_file -o output_c3d_file
    '''
    
    try:
        c3d_path = args[0]['input'] # invoked with argparse
        if args[0]['output'] == None:
            trc_path = c3d_path.replace('.c3d', '.trc')
        else:
            trc_path = args[0]['output']
    except:
        c3d_path = args[0] # invoked as a function
        try:
            trc_path = args[1]
        except:
            trc_path = c3d_path.replace('.c3d', '.trc')
        
    # c3d header
    reader = c3d.Reader(open(c3d_path, 'rb'))
    items_header = str(reader.header).split('\n')
    items_header_list = [item.strip().split(': ') for item in items_header]
    label_item = [item[0] for item in items_header_list]
    value_item = [item[1] for item in items_header_list]
    header_c3d = dict(zip(label_item, value_item))
    
    # unit
    for k1 in reader.group_items():
        if k1[0]=='POINT':
            for k2 in k1[1].param_items():
                if k2[0]=='UNITS':
                    if 'mm' in k2[1].bytes[:].decode('utf-8'):
                        unit = 'mm'
                        unit_scale= 0.001
                    else:
                        unit = 'm'
                        unit_scale= 1 # mm
    
    # c3d data: reads 3D points (no analog data) and takes off computed data
    labels = reader.point_labels
    index_labels_markers = [i for i, s in enumerate(labels) if 'Angle' not in s and 'Power' not in s and 'Force' not in s and 'Moment' not in s and 'GRF' not in s]
    labels_markers = [labels[ind] for ind in index_labels_markers]
    
    # trc header
    header0_str = 'PathFileType\t4\t(X/Y/Z)\t' + trc_path

    header1 = {}
    header1['DataRate'] = str(int(float(header_c3d['frame_rate'])))
    header1['CameraRate'] = header1['DataRate']
    header1['NumFrames'] = str(int(header_c3d['last_frame']) - int(header_c3d['first_frame']) + 1)
    header1['NumMarkers'] = str(len(labels_markers))
    header1['Units'] = unit
    header1['OrigDataRate'] = header1['DataRate']
    header1['OrigDataStartFrame'] = header_c3d['first_frame']
    header1['OrigNumFrames'] = header1['NumFrames']
    header1_str1 = '\t'.join(header1.keys())
    header1_str2 = '\t'.join(header1.values())

    header2_str1 = 'Frame#\tTime\t' + '\t\t\t'.join([item.strip() for item in labels_markers]) + '\t\t'
    header2_str2 = '\t\t'+'\t'.join(['X{i}\tY{i}\tZ{i}'.format(i=i+1) for i in range(int(header1['NumMarkers']))])
    
    header_trc = '\n'.join([header0_str, header1_str1, header1_str2, header2_str1, header2_str2])
    
    with open(trc_path, 'w') as trc_o:
        trc_o.write(header_trc+'\n')
    
    # trc data
        index_data_markers = np.sort(np.concatenate([np.array(index_labels_markers)*3, np.array(index_labels_markers)*3+1, np.array(index_labels_markers)*3+2]))
        t0 = int(float(header_c3d['first_frame'])) / int(float(header_c3d['frame_rate']))
        tf = int(float(header_c3d['last_frame'])) / int(float(header_c3d['frame_rate']))
        trc_time = np.linspace(t0, tf, num=(int(header_c3d['last_frame']) - int(header_c3d['first_frame']) + 1))
        for n, (i, points, _) in enumerate(list(reader.read_frames())):
            c3d_line = np.concatenate([item[:3] for item in points])*unit_scale
            c3d_line_markers = c3d_line[index_data_markers]
            trc_line = '{i}\t{t}\t'.format(i=i, t=trc_time[n]) + '\t'.join(map(str,c3d_line_markers))
            trc_o.write(trc_line+'\n')

    print(f'Converted c3d file to {trc_path}')
    
    
if __name__ == '__main__':
    main()
