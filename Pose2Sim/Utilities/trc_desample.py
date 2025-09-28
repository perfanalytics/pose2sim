#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Undersample a trc file                       ##
    ##################################################
    
    Undersample a trc file
    
    Usage: 
    trc_desample -i input_trc_file -f <output_frequency>
    trc_desample -i input_trc_file -f <output_frequency> -o output_trc_file
    from Pose2Sim.Utilities import trc_desample; trc_desample.trc_desample_func(r'input_trc_file', output_frequency, r'output_trc_file')
'''


## INIT
import pandas as pd
import re
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
    parser.add_argument('-i', '--input_file', required = True, help='trc input file')
    parser.add_argument('-f', '--output_frequency', required = True, help='required output frequency')
    parser.add_argument('-o', '--output_file', required=False, help='trc desampled output file')
    args = vars(parser.parse_args())
    
    trc_desample_func(args)


def trc_desample_func(*args):
    '''
    Undersample a trc file

    Usage: 
    trc_desample -i input_trc_file -f <output_frequency>
    trc_desample -i input_trc_file -f <output_frequency> -o output_trc_file
    import trc_desample; trc_desample.trc_desample_func(r'input_trc_file', output_frequency, r'output_trc_file')
    '''
    
    try:
        trc_path = args[0]['input_file'] # invoked with argparse
        f_out = int(args[0]['output_frequency'])
        if args[0]['output_file'] == None:
            trc_desampled_path = trc_path.replace('.trc', f'_{f_out}fps.trc')
        else:
            trc_desampled_path = args[0]['output']
    except:
        trc_path = args[0] # invoked as a function
        f_out = int(args[1])
        trc_desampled_path = trc_path.replace('.trc', f'_{f_out}fps.trc')
    
    # header
    with open(trc_path, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]
    params_in = re.split('\t|\n', header[2])[:-1]
    params_in = [int(p) if i not in [3,4] else p for i,p in enumerate(params_in)]
    f_in = params_in[0]
    params_out = [int(p*f_out/f_in) if i not in [3,4] else p for i,p in enumerate(params_in)]
    params_out = [str(p) for i,p in enumerate(params_out)]
    header[2] = '\t'.join(params_out) + '\n'
    
    # data
    Q = pd.read_csv(trc_path, sep="\t", skiprows=4)
    Q = Q.iloc[::int(f_in/f_out),:]
    Q.iloc[:,0] = range(len(Q))
    
    # write trc
    with open(trc_desampled_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    print(f"trc file desampled at {f_out} fps: {trc_desampled_path}")
    

if __name__ == '__main__':
    main()