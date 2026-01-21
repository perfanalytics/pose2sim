#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Rotate trc coordinates by 90°                ##
    ##################################################
    
    Rotate trc coordinates by 90° around an axis
    You can either choose an axis to rotate around,
    or use one of the predefined conversions from and axis-up to another one.

    90° rotation around:
    - "X"  corresponds to:  yup_to_zup     or zup_to_ydown
    - "-X" corresponds to:  yup_to_zdown   or zup_to_yup
    - "Y"  corresponds to:  zup_to_xup     or xup_to_zdown
    - "-Y" corresponds to:  zup_to_xdown   or xup_to_zup
    - "Z"  corresponds to:  yup_to_xdown
    - "-Z" corresponds to:  yup_to_xup

    The output file argument is optional. If not specified, 
    '_X', '_-X', '_Y', '_-Y', '_Z' or '_-Z' is appended to the input filename.

    Usage: 
    from Pose2Sim.Utilities import trc_rotate; trc_rotate.trc_rotate_func(input=r'<input_trc_file>', output=r'<output_trc_file>', rotate90='-X')
    
    trc_rotate -i input_trc_file                    # Will rotate around -X by default (Y-up -> Z-up)
    trc_rotate -i input_trc_file -o output_trc_file

    trc_rotate -i input_trc_file --zup_to_yup
    trc_rotate -i input_trc_file --rotate90=-X      # Equivalently
'''


## INIT
import pandas as pd
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
def trc_rotate_func(**args):
    '''
    Rotate trc coordinates by 90° around an axis
    You can either choose an axis to rotate around,
    or use one of the predefined conversions from and axis-up to another one.

    90° rotation around:
    - "X"  corresponds to:  yup_to_zup     or zup_to_ydown
    - "-X" corresponds to:  yup_to_zdown   or zup_to_yup
    - "Y"  corresponds to:  zup_to_xup     or xup_to_zdown
    - "-Y" corresponds to:  zup_to_xdown   or xup_to_zup
    - "Z"  corresponds to:  yup_to_xdown
    - "-Z" corresponds to:  yup_to_xup

    The output file argument is optional. If not specified, 
    '_X', '_-X', '_Y', '_-Y', '_Z' or '_-Z' is appended to the input filename.

    Usage: 
    from Pose2Sim.Utilities import trc_rotate; trc_rotate.trc_rotate_func(input=r'<input_trc_file>', output=r'<output_trc_file>', rotate90='-X')
    
    trc_rotate -i input_trc_file                    # Will rotate around -X by default (Y-up -> Z-up)
    trc_rotate -i input_trc_file -o output_trc_file

    trc_rotate -i input_trc_file --zup_to_yup
    trc_rotate -i input_trc_file --rotate90=-X      # Equivalently
    '''

    trc_path = args.get('input')
    output_trc_path = args.get('output')
    rotate90 = args.get('rotate90')
    if rotate90 is None:
        rotate90 = "-X"
    if output_trc_path is None:
        output_trc_path = trc_path.replace('.trc', f'_{rotate90}.trc')

    # header
    with open(trc_path, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]

    # data
    trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
    frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
    Q_coord = Q_coord.loc[:, ~Q_coord.columns.str.contains('^Unnamed')]

    # rotate coordinates
    cols = Q_coord.values.reshape(-1,3)
    if rotate90 == "X": # X->X, Y->-Z, Z->Y
        cols = np.stack([cols[:,0],-cols[:,2],cols[:,1]], axis=-1)
    elif rotate90 == "-X": # X->X, Y->Z, Z->-Y
        cols = np.stack([cols[:,0],cols[:,2],-cols[:,1]], axis=-1)
    elif rotate90 == "Y": # X->Z, Y->Y, Z->-X
        cols = np.stack([cols[:,2],cols[:,1],-cols[:,0]], axis=-1)
    elif rotate90 == "-Y": # X->-Z, Y->Y, Z->X
        cols = np.stack([-cols[:,2],cols[:,1],cols[:,0]], axis=-1)
    elif rotate90 == "Z": # X->-Y, Y->X, Z->Z
        cols = np.stack([-cols[:,1],cols[:,0],cols[:,2]], axis=-1)
    elif rotate90 == "-Z": # X->Y, Y->-X, Z->Z
        cols = np.stack([cols[:,1],-cols[:,0],cols[:,2]], axis=-1)
    Q_coord = pd.DataFrame(cols.reshape(Q_coord.values.shape[0],-1), columns=Q_coord.columns, index=Q_coord.index)
  
    # write file
    with open(output_trc_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q_coord.insert(0, 'Frame#', frames_col)
        Q_coord.insert(1, 'Time', time_col)
        Q_coord.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    print(f"trc file rotated by 90° around {rotate90}. Saved to {output_trc_path}")


def main():
    parser = argparse.ArgumentParser(description="Rotate trc coordinates by 90°")

    parser.add_argument('-i', '--input', required=True, help='trc input file')
    parser.add_argument('-o', '--output', required=False, help='trc output file')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--rotate90",
                    choices=["X","-X","Y","-Y","Z","-Z"], default="-X",
                    help="Axis and direction for a 90-degree rotation")
    group.add_argument("--yup_to_zup",   action="store_const", const="X",  dest="rotate90",
                    help="Corresponds to a 90-degree rotation around +X")
    group.add_argument("--zup_to_ydown", action="store_const", const="X",  dest="rotate90",
                    help="Corresponds to a 90-degree rotation around +X")
    group.add_argument("--yup_to_zdown", action="store_const", const="-X", dest="rotate90",
                    help="Corresponds to a 90-degree rotation around -X")
    group.add_argument("--zup_to_yup",   action="store_const", const="-X", dest="rotate90",
                    help="Corresponds to a 90-degree rotation around -X")
    group.add_argument("--zup_to_xup",   action="store_const", const="Y",  dest="rotate90",
                    help="Corresponds to a 90-degree rotation around +Y")
    group.add_argument("--xup_to_zdown", action="store_const", const="Y",  dest="rotate90",
                    help="Corresponds to a 90-degree rotation around +Y")
    group.add_argument("--zup_to_xdown", action="store_const", const="-Y", dest="rotate90",
                    help="Corresponds to a 90-degree rotation around -Y")
    group.add_argument("--xup_to_zup",   action="store_const", const="-Y", dest="rotate90",
                    help="Corresponds to a 90-degree rotation around -Y")
    group.add_argument("--yup_to_xdown", action="store_const", const="Z",  dest="rotate90",
                    help="Corresponds to a 90-degree rotation around +Z")
    group.add_argument("--yup_to_xup",   action="store_const", const="-Z", dest="rotate90",
                    help="Corresponds to a 90-degree rotation around -Z")

    args = vars(parser.parse_args())
    
    trc_rotate_func(**args)


if __name__ == '__main__':
    main()
