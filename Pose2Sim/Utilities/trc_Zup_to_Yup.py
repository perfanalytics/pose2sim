#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Convert trc Z-up files to Y-up files         ##
    ##################################################
    
    Convert trc files with Z-up system coordinates to Y-up files.
    
    Usage: 
    from Pose2Sim.Utilities import trc_Zup_to_Yup; trc_Zup_to_Yup.trc_Zup_to_Yup_func(r'<input_trc_file>', r'<output_trc_file>')
    trc_Zup_to_Yup -i input_trc_file
    trc_Zup_to_Yup -i input_trc_file -o output_trc_file
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required = True, help='trc Zup input file')
    parser.add_argument('-o', '--output', required=False, help='trc Yup output file')
    args = vars(parser.parse_args())
    
    trc_Zup_to_Yup_func(args)


def trc_Zup_to_Yup_func(*args):
    '''
    Turns trc files with Z-up system coordinates into Y-up files.

    Usage: 
    import trc_Zup_to_Yup; trc_Zup_to_Yup.trc_Zup_to_Yup_func(r'<input_trc_file>', r'<output_trc_file>')
    trcZup_to_Yup -i input_trc_file
    trcZup_to_Yup -i input_trc_file -o output_trc_file
    '''

    try:
        trc_path = args[0]['input'] # invoked with argparse
        if args[0]['output'] == None:
            trc_yup_path = trc_path.replace('.trc', '_Yup.trc')
        else:
            trc_yup_path = args[0]['output']
    except:
        trc_path = args[0] # invoked as a function
        trc_yup_path = trc_path.replace('.trc', '_Yup.trc')

    # header
    with open(trc_path, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]

    # data
    trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4)
    frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    # Y->Z, Z->Y
    cols = list(Q_coord.columns)
    # cols = np.array([[cols[i*3+1],cols[i*3+2],cols[i*3]] for i in range(int(len(cols)/3))]).flatten() # X->Y, Y->Z, Z->X
    cols = np.array([[cols[i*3],cols[i*3+2],cols[i*3+1]] for i in range(int(len(cols)/3))]).flatten() # Y->Z, Z->-Y
    Q_Yup = Q_coord[cols]
    # Q_Yup.iloc[:,2::3] = - Q_Yup.iloc[:,2::3]

    # write file
    with open(trc_yup_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q_Yup.insert(0, 'Frame#', frames_col)
        Q_Yup.insert(1, 'Time', time_col)
        Q_Yup.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    print(f"trc file converted from Z-up to Y-up: {trc_yup_path}")
    
if __name__ == '__main__':
    main()
