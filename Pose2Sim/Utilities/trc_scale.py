#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Scale trc coordinates                        ##
    ##################################################
    
    Scale trc coordinates by a desired factor.
    
    Usage: 
    from Pose2Sim.Utilities import trc_scale; trc_scale.trc_scaled_func(r'<input_trc_file>', 0.001, r'<output_trc_file>')
    trc_scale -i input_trc_file -s 0.001
    trc_scale -i input_trc_file -s 0.001 -o output_trc_file
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
    parser.add_argument('-s', '--scale_factor', required=True, type=float, help='scaling factor to apply to the trc coordinates. mm to m would be 0.001')
    args = vars(parser.parse_args())
    
    trc_scale_func(args)


def trc_scale_func(*args):
    '''
    Scale trc coordinates by a desired factor.
    
    Usage: 
    from Pose2Sim.Utilities import trc_scale; trc_scale.trc_scaled_func(r'<input_trc_file>', 0.001, r'<output_trc_file>')
    trc_scale -i input_trc_file -s 0.001
    trc_scale -i input_trc_file -s 0.001 -o output_trc_file
    '''

    try:
        trc_path = args[0]['input'] # invoked with argparse
        scale_factor = args[0]['scale_factor']
        if args[0]['output'] == None:
            trc_scaled_path = trc_path.replace('.trc', '_scaled.trc')
        else:
            trc_scaled_path = args[0]['output']
    except:
        trc_path = args[0] # invoked as a function
        scale_factor = args[1]
        try:
            trc_scaled_path = args[2]
        except:
            trc_scaled_path = trc_path.replace('.trc', '_scaled.trc')

    # header
    with open(trc_path, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]

    # data
    trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4)
    frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    # scaling
    Q_scaled = Q_coord * scale_factor

    # write file
    with open(trc_scaled_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q_scaled.insert(0, 'Frame#', frames_col)
        Q_scaled.insert(1, 'Time', time_col)
        Q_scaled.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    print(f"trc file scaled with a {scale_factor} factor. Saved to {trc_scaled_path}")
    
if __name__ == '__main__':
    main()
