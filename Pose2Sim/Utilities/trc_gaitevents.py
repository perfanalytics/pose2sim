#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## GAIT EVENTS DETECTION                        ##
    ##################################################
    
    Determine gait on and off from a TRC file of point coordinates.

    Events are cleaned to make sure that each contact phases has a start and an end.
    Optionally and in case of noisy data, you can constrain the right and left steps to alternate (gait), 
    or the right and left steps to alternate with a flight phase inbetween (sprint), 
    or not constrain at all in case of hops on one foot (eg triple jump).

    Contacts and curves can be plotted to check the results.
    

    N.B.: Could implement the methods listed there in the future:
          https://www.tandfonline.com/doi/full/10.1080/19424280903133938 
          Please feel free to make a pull-request or keep me informed if you do so!

    Three available methods, each of them with their own pros and cons: 

    - "forward_coordinates": 
        on =  max(XHeel - Xsacrum)
        off = min(XToe - XSacrum)
        ++: Works well for walking (Zeni et al., 2008)
        ++: No argument nor tuning necessary
        --: Not suitable for running
        --: does not work if the person is not going on a straight line

    - "height_coordinates": 
        on =  YToe < height_threshold
        off = YToe > height_threshold
        ++: Best results for running and walking
        ++: Works if the person is not going on a straight line
        --: Does not work is the person is grazing the ground
        --: Does not work if the field is not flat
        --: height_threshold might need to be tuned
        --: Heel point might be more accurate than toe point for walking

    - "forward_velocity": 
        on =  VToe < forward_velocity_threshold
        off = VToe > forward_velocity_threshold
        ++: Works for running
        --: More sensitive to noise
        --: Tends to anticipate off if the marker is not at the tip of the toe
        --: forward_velocity_threshold might need to be tuned
        --: Does not work if the person is not going on a straight line
    
    Usage: 
    List of available arguments:
        trc_gaitevents -h

        import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>')
        OR import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>', method='forward_coordinates', gait_direction='-X', motion_type='gait', plot=True, save_output=True, output_file='gaitevents.txt')
        OR import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>', method='height_coordinates', up_direction='Y', height_threshold=6, motion_type='gait', right_toe_marker='RBigToe', left_toe_marker='LBigToe')
        OR import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>', method='forward_velocity', gait_direction='-Z', forward_velocity_threshold=1, motion_type='gait', right_toe_marker='RBigToe', left_toe_marker='LBigToe')
        
        trc_gaitevents -i input_trc_file
        OR trc_gaitevents -i input_trc_file --method forward_coordinates --gait_direction=-X --motion_type gait --plot True --save_output True --output_file gaitevents.txt
        OR trc_gaitevents -i input_trc_file --method height_coordinates --up_direction=Y --height_threshold 6 --motion_type gait --right_toe_marker RBigToe --left_toe_marker LBigToe
        OR trc_gaitevents -i input_trc_file --method forward_velocity --gait_direction=-Z --forward_velocity_threshold 1 --motion_type gait --right_toe_marker RBigToe --left_toe_marker LBigToe
'''


## INIT
import os
import argparse
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


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
    parser = argparse.ArgumentParser(description='Determine gait on and off with "forward_coordinates", "height_coordinates", or "forward_velocity" method. More details in the file description.')
    parser.add_argument('-i', '--trc_path', required = True, help='Trc input file')
    parser.add_argument('-g', '--gait_direction', default = 'X', required = False, help='Direction of the gait. "X", "Y", "Z", "-X", "-Y", or "-Z". Default: "X". Requires an equal sign if negative, eg -g=-X')
    parser.add_argument('-u', '--up_direction', default = 'Y', required = False, help='Up direction. "X", "Y", or "Z", "-X", "-Y", or "-Z". Default: "Y"')
    parser.add_argument('-m', '--method', default = 'height_coordinates', required = False, help='Method to determine gait events. "forward_coordinates", "height_coordinates", or "forward_velocity". Default:"height_coordinates"')
    parser.add_argument('-V', '--forward_velocity_threshold', default = 1, type=float, required = False, help='Forward velocity below which the foot is considered to have touched the ground (m/s). Used if method is forward_velocity. Default: 1.5')
    parser.add_argument('-H', '--height_threshold', default = 6, type=float, required = False, help='Height below which the foot is considered to have touched the ground (cm). Used if method is height_coordinates. Default: 6')
    parser.add_argument('-t', '--motion_type', default = 'gait', required = False, help='Motion type. "gait" for walking, "sprint" for sprinting, "" if there is no specific alternation. Default: "gait"')
    parser.add_argument('--sacrum_marker', default = 'Hip', required = False, help='Hip marker name. Default: "Hip"')
    parser.add_argument('--right_heel_marker', default = 'RHeel', required = False, help='Right heel marker name. Default: "RHeel"')
    parser.add_argument('--right_toe_marker', default = 'RBigToe', required = False, help='Right toe marker name. Default: "RBigToe"')
    parser.add_argument('--left_heel_marker', default = 'LHeel', required = False, help='Left heel marker name. Default: "LHeel"')
    parser.add_argument('--left_toe_marker', default = 'LBigToe', required = False, help='Left toe marker name. Default: "LBigToe"')
    parser.add_argument('-f', '--cut_off_frequency', default = 10, type=float, required = False, help='Butterworth filter cutoff frequency (Hz). Default: 10')
    parser.add_argument('-p', '--plot', default = True, required = False, help='Plot results. Default: True')
    parser.add_argument('-s', '--save_output', default = True, required = False, help='Save output in csv file. Default: True')
    parser.add_argument('-o', '--output_file', default = 'gaitevents.txt', required = False, help='Output file name. Default: "gaitevents.txt"')

    args = vars(parser.parse_args())
    
    trc_gaitevents_func(**args)


def start_end_true_seq(series):
    '''
    Find starts and ends of sequences of True values in a pandas Series

    INPUTS:
    - series: pandas Series of boolean values

    OUTPUTS:
    - start_indices: list of start indices
    - end_indices: list of end indices
    '''

    diff = series.ne(series.shift())
    start_indices = series.index[diff & series].tolist()
    end_indices = (series.index[diff & ~series]-1).tolist()
    if end_indices[0] == -1: end_indices.pop(0)

    return start_indices, end_indices


def read_trc(trc_path):
    '''
    Read trc file

    INPUTS:
    - trc_path: path to the trc file

    OUTPUTS:
    - Q_coords: dataframe of coordinates
    - frames_col: series of frames
    - time_col: series of time
    - markers: list of marker names
    - header: list of header lines
    '''
    
    with open(trc_path, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]
    markers = header[3].split('\t')[2::3]
    
    trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4)
    frames_col, time_col = pd.Series(trc_df.iloc[:,0], name='frames'), pd.Series(trc_df.iloc[:,1], name='time')
    Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    return Q_coords, frames_col, time_col, markers, header


def first_step_side(Ron, Lon):
    '''
    Get first step side

    INPUTS:
    - Ron: list of right on times or frames
    - Lon: list of left on times or frames

    OUTPUTS:
    - first_side: 'R' if right first, 'L' if left first
    '''

    if Ron[0]<Lon[0]: 
        first_side = 'R'
    else:
        first_side = 'L'

    return first_side


def alternate_lists(*lists, strategy='last'):
    '''
    Alternates values from multiple sorted lists while preserving their order.
    Two strategies are available: 'first' to take the first non-alternating value, 'last' to take the last.

    For example, with 
    lists = [[1, 4, 7, 10], 
             [2, 3, 5, 6, 8, 9]]
    and strategy 'first', the output would be:
            [[1, 4, 7, 10], 
             [2, 5, 8]]
    while with strategy 'last', it would be:
            [[1, 4, 7, 10], 
             [3, 6, 9]]
    
    INPUTS:
    - lists: list of lists of values
    - strategy: 'first' to take the first non-alternating value, 'last' to take the last

    OUTPUTS:
    - lists: list of lists of values
    '''

    # Combine all lists into a list of tuples with list index and value, and sort them by value
    combined = [(i, value) for i, lst in enumerate(lists) for value in lst]
    combined.sort(key=lambda x: x[1])

    # Make sure we start with an item from the first list
    while combined and combined[0][0] != 0: 
        combined.pop(0)

    # Alternate values between lists
    result = {i: [] for i in range(len(lists))}
    new_index, last_value = 0, 0
    while combined:
        # print(f'{combined}')
        index, value = combined[0]
        if index == new_index and value >= last_value:
            result[index].append(value)
            last_index = new_index
            new_index = new_index+1
        # print(index, value)
        # print(new_index, last_value, last_index)
        if strategy == 'last' and index == last_index and value > last_value:
            result[last_index][-1] = value
        last_value = value
        if new_index>=len(lists): new_index=0
        combined.pop(0)
        # print(f'{result}')
        # input()

    lists = [result[i] for i in range(len(lists))]

    # Extract final lists in the original order
    return lists


def clean_gait_events(gait_events, motion_type='gait'):
    '''

    Events are cleaned to make sure that each contact phases has a start and an end.
    
    Optionally and in case of noisy data, you can constrain the right and left steps to alternate (gait), 
    or the right and left steps to alternate with a flight phase inbetween (sprint), 
    or not constrain at all in case of hops on one foot (eg triple jump).

    INPUTS:
    - gait_events: tuple of lists (Ron, Lon, Roff, Loff)
    - motion_type: 'gait' for walking (right-left alternation),
                   'sprint' for sprinting (right-left alternation with flight phase inbetween), 
                   '' if there is no specific alternation

    OUTPUTS:
    - Ron, Lon, Roff, Loff = cleaned gait events
    '''

    Ron, Lon, Roff, Loff = gait_events

    # Make right and left contacts alternate. Take the first strike and the last take off detection (some outliers may be detected in the middle)
    if motion_type=='gait':
        first_side = first_step_side(Ron, Lon)
        if first_side == 'R':
            Ron, Lon = alternate_lists(Ron, Lon, strategy='first')
            Roff, Loff = alternate_lists(Roff, Loff, strategy='last')
        else:
            Lon, Ron = alternate_lists(Lon, Ron, strategy='first')
            Loff, Roff = alternate_lists(Loff, Roff, strategy='last')

    # Make sure that left on, left off, right on, right off events alternate (or the other way around). 
    if motion_type=='sprint':
        first_side = first_step_side(Ron, Lon)
        if first_side == 'R':
            Ron, Roff, Lon, Loff = alternate_lists(Ron, Roff, Lon, Loff, strategy='last')
        else:
            Lon, Loff, Ron, Roff = alternate_lists(Lon, Loff, Ron, Roff, strategy='last')

    # Don't fix it if hopping on one foot (eg triple jump). Beware that noisy data may lead to wrong results
    else: 
        pass
    
    # Remove incomplete pairs at the start and end
    if Ron[0]>Roff[0]: Roff.pop(0)
    if Lon[0]>Loff[0]: Loff.pop(0)

    if Ron[-1]>Roff[-1]: Ron.pop(-1)
    if Lon[-1]>Loff[-1]: Lon.pop(-1)

    return Ron, Lon, Roff, Loff


def gait_events_fwd_coords(trc_path, gait_direction, motion_type='gait', markers=['RHeel', 'RBigToe', 'LHeel', 'LBigToe', 'Hip'], plot=True):
    '''
    Determine gait on and off with "forward_coordinates" method
    
    on =  max(XHeel - Xsacrum)
    off = min(XToe - XSacrum)
    ++: Works well for walking (Zeni et al., 2008)
    ++: No argument nor tuning necessary
    --: Not suitable for running
    --: does not work if the person is not going on a straight line

    INPUTS:
    - trc_path: path to the trc file
    - gait_direction: tuple (sign, direction) with sign in {-1, 1} and direction in {'X', 'Y', 'Z'}
    - markers: list of marker names in the following order: [right_heel_marker, right_toe_marker, left_heel_marker, left_toe_marker, sacrum_marker]
    - plot: plot results or not (boolean)

    OUTPUTS:
    - t_Ron: list of right on times
    - t_Lon: list of left on times
    - t_Roff: list of right off times
    - t_Loff: list of left off times
    '''

    # Retrieve gait direction
    sign, direction = gait_direction
    axis = ['X', 'Y', 'Z'].index(direction)

    # Read trc file
    Q_coords, _, time_col, trc_markers, header = read_trc(trc_path)

    unit = header[2].split('\t')[4]
    peak_prominence = .1 if unit=='m' else 1 if unit=='dm' else 10 if unit=='cm' else 100 if unit=='mm' else np.inf

    RHeel_idx, RBigToe_idx, LHeel_idx, LBigToe_idx, Hip_idx = [trc_markers.index(m) for m in markers]
    RHeel_df, RBigToe_df, LHeel_df, LBigToe_df, Hip_df = (Q_coords.iloc[:,axis+idx*3] for idx in [RHeel_idx, RBigToe_idx, LHeel_idx, LBigToe_idx, Hip_idx])

    # Find gait events
    max_r_heel_hip_proj = sign*(RHeel_df-Hip_df)
    frame_Ron = signal.find_peaks(max_r_heel_hip_proj, prominence=peak_prominence)[0].tolist()
    t_Ron = time_col[frame_Ron].tolist()

    max_l_heel_hip_proj = sign*(LHeel_df-Hip_df)
    frame_Lon = signal.find_peaks(max_l_heel_hip_proj, prominence=peak_prominence)[0].tolist()
    t_Lon = time_col[frame_Lon].tolist()

    max_r_hip_toe_proj = sign*(Hip_df-RBigToe_df)
    frame_Roff = signal.find_peaks(max_r_hip_toe_proj, prominence=peak_prominence)[0].tolist()
    t_Roff = time_col[frame_Roff].tolist()

    max_l_hip_toe_proj = sign*(Hip_df-LBigToe_df)
    frame_Loff = signal.find_peaks(max_l_hip_toe_proj, prominence=peak_prominence)[0].tolist()
    t_Loff = time_col[frame_Loff].tolist()

    # Clean gait events
    frame_Ron, frame_Lon, frame_Roff, frame_Loff = clean_gait_events((frame_Ron, frame_Lon, frame_Roff, frame_Loff), motion_type=motion_type)
    t_Ron, t_Lon, t_Roff, t_Loff = clean_gait_events((t_Ron, t_Lon, t_Roff, t_Loff), motion_type=motion_type)

    # Plot
    if plot:
        plt.plot(time_col, max_r_heel_hip_proj, 'C0', label='Right on')
        plt.plot(time_col[frame_Ron], max_r_heel_hip_proj[frame_Ron], 'g+')

        plt.plot(time_col, max_l_heel_hip_proj, 'C1', label='Left on')
        plt.plot(time_col[frame_Lon], max_l_heel_hip_proj[frame_Lon], 'g+')

        plt.plot(time_col, max_r_hip_toe_proj, 'C0', label='Right off')
        plt.plot(time_col[frame_Roff], max_r_hip_toe_proj[frame_Roff], 'r+')

        plt.plot(time_col, max_l_hip_toe_proj, 'C1', label='Left off')
        plt.plot(time_col[frame_Loff], max_l_hip_toe_proj[frame_Loff], 'r+')

        plt.title('Gait events')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (cm)')
        plt.legend()
        plt.show()

    print('Times:')
    print('Right on:', t_Ron)
    print('Right off:', t_Roff)
    print('Left on:', t_Lon)
    print('Left off:', t_Loff)
    print('\nFrames:')
    print('Right on:', frame_Ron)
    print('Right off:', frame_Roff)
    print('Left on:', frame_Lon)
    print('Left off:', frame_Loff)

    return (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff)


def gait_events_height_coords(trc_path, up_direction, height_threshold=6, motion_type='gait', cut_off_frequency=10, markers=['RBigToe', 'LBigToe'], plot=True):
    '''
    Determine gait on and off with "height_coordinates" method
    
    on =  YToe < height_threshold
    off = YToe > height_threshold
    ++: Best results for running and walking
    ++: Works if the person is not going on a straight line
    --: Does not work is the person is grazing the ground
    --: Does not work if the field is not flat
    --: height_threshold might need to be tuned
    --: Heel point might be more accurate than toe point for walking

    INPUTS:
    - trc_path: path to the trc file
    - up_direction: tuple (sign, direction) with sign in {-1, 1} and direction in {'X', 'Y', 'Z'}
    - height_threshold: height below which the foot is considered to have touched the ground (cm)
    - cut_off_frequency: butterworth filter cutoff frequency (Hz)
    - markers: list of marker names in the following order: [right_toe_marker, left_toe_marker]
    - plot: plot results or not (boolean)

    OUTPUTS:
    - t_Ron: list of right on times
    - t_Lon: list of left on times
    - t_Roff: list of right off times
    - t_Loff: list of left off times
    '''

    # Retrieve gait direction
    sign, direction = up_direction
    axis = ['X', 'Y', 'Z'].index(direction)

    # Read trc file
    Q_coords, _, time_col, trc_markers, header = read_trc(trc_path)
    unit = header[2].split('\t')[4]
    unit_factor = 100 if unit=='m' else 10 if unit=='dm' else 1 if unit=='cm' else .1 if unit=='mm' else np.inf
    Q_coords *= unit_factor

    # Calculate height
    Y_height = Q_coords.iloc[:,axis::3]
    Y_height.columns = trc_markers
    Rfoot_height, Lfoot_height = (Y_height[m] for m in markers)

    dt = time_col.diff().mean()
    b, a = signal.butter(4/2, cut_off_frequency*dt*2, 'low', analog=False)
    Rfoot_height_filtered = pd.Series(signal.filtfilt(b, a, Rfoot_height[1:]), name=Rfoot_height.name)
    Lfoot_height_filtered = pd.Series(signal.filtfilt(b, a, Lfoot_height[1:]), name=Lfoot_height.name)

    # Find gait events
    low_Rfoot_height = Rfoot_height_filtered<height_threshold
    frame_Ron, frame_Roff = start_end_true_seq(low_Rfoot_height)
    if 0 in frame_Ron: frame_Ron.remove(0)
    t_Ron, t_Roff = time_col[frame_Ron].tolist(), time_col[frame_Roff].tolist()

    low_Lfoot_height = Lfoot_height_filtered<height_threshold
    frame_Lon, frame_Loff = start_end_true_seq(low_Lfoot_height)
    if 0 in frame_Lon: frame_Lon.remove(0)
    t_Lon, t_Loff = time_col[frame_Lon].tolist(), time_col[frame_Loff].tolist()

    # Clean gait events
    frame_Ron, frame_Lon, frame_Roff, frame_Loff = clean_gait_events((frame_Ron, frame_Lon, frame_Roff, frame_Loff), motion_type=motion_type)
    t_Ron, t_Lon, t_Roff, t_Loff = clean_gait_events((t_Ron, t_Lon, t_Roff, t_Loff), motion_type=motion_type)

    # Plot
    if plot:
        plt.plot(time_col[1:], Rfoot_height_filtered, 'C0', label='Right foot height filtered')
        plt.plot(time_col[1:][frame_Ron], Rfoot_height_filtered[frame_Ron], 'g+')
        plt.plot(time_col[1:][frame_Roff], Rfoot_height_filtered[frame_Roff], 'r+')

        plt.plot(time_col[1:], Lfoot_height_filtered, 'C1', label='Left foot height filtered')
        plt.plot(time_col[1:][frame_Lon], Lfoot_height_filtered[frame_Lon], 'g+')
        plt.plot(time_col[1:][frame_Loff], Lfoot_height_filtered[frame_Loff], 'r+')

        plt.title('Gait events')
        plt.xlabel('Time (s)')
        plt.ylabel('Height (cm)')
        plt.legend()
        plt.show()

    print('Times:')
    print('Right on:', t_Ron)
    print('Right off:', t_Roff)
    print('Left on:', t_Lon)
    print('Left off:', t_Loff)
    print('\nFrames:')
    print('Right on:', frame_Ron)
    print('Right off:', frame_Roff)
    print('Left on:', frame_Lon)
    print('Left off:', frame_Loff)

    return (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff)


def gait_events_fwd_vel(trc_path, gait_direction, forward_velocity_threshold=1, motion_type='gait', cut_off_frequency=10, markers=['RBigToe', 'LBigToe'], plot=True):
    '''
    Determine gait on and off with "forward_velocity" method
    
    on =  VToe < forward_velocity_threshold
    off = VToe > forward_velocity_threshold
    ++: Works for running and walking
    --: Tends to anticipate off if marker is not at the tip of the toe
    --: forward_velocity_threshold might need to be tuned
    --: does not work if the person is not going on a straight line
    
    INPUTS:
    - trc_path: path to the trc file
    - gait_direction: tuple (sign, direction) with sign in {-1, 1} and direction in {'X', 'Y', 'Z'}
    - forward_velocity_threshold: forward velocity below which the foot is considered to have touched the ground (m/s)
    - cut_off_frequency: butterworth filter cutoff frequency (Hz)
    - markers: list of marker names in the following order: [right_toe_marker, left_toe_marker]
    - plot: plot results or not (boolean)

    OUTPUTS:
    - t_Ron: list of right on times
    - t_Lon: list of left on times
    - t_Roff: list of right off times
    - t_Loff: list of left off times
    '''
    
    # Retrieve gait direction
    sign, direction = gait_direction
    axis = ['X', 'Y', 'Z'].index(direction)

    # Read trc file
    Q_coords, _, time_col, trc_markers, header = read_trc(trc_path)
    unit = header[2].split('\t')[4]
    unit_factor = 1 if unit=='m' else 10 if unit=='dm' else 100 if unit=='cm' else 1000 if unit=='mm' else np.inf
    forward_velocity_threshold *= unit_factor
    Q_coords *= unit_factor

    # Calculate speed
    dt = time_col.diff().mean()
    b, a = signal.butter(4/2, cut_off_frequency*dt*2, 'low', analog=False)
    X_speed = Q_coords.iloc[:,axis::3].diff()/dt
    X_speed.columns = trc_markers
    Rfoot_speed, Lfoot_speed = (X_speed[m] for m in markers)

    Rfoot_speed = Rfoot_speed.where(Rfoot_speed<0, other=0) if sign==-1 else Rfoot_speed.where(Rfoot_speed>0, other=0)
    Rfoot_speed = Rfoot_speed.abs()
    # Rfoot_speed_filtered = pd.Series(signal.filtfilt(b, a, Rfoot_speed[1:]), name=Rfoot_speed.name)
    Rfoot_speed_filtered = pd.Series(gaussian_filter1d(Rfoot_speed[1:], 5), name=Rfoot_speed.name)
        
    Lfoot_speed = Lfoot_speed.where(Lfoot_speed<0, other=0) if sign==-1 else Lfoot_speed.where(Lfoot_speed>0, other=0)
    Lfoot_speed = Lfoot_speed.abs()
    # Lfoot_speed_filtered = pd.Series(signal.filtfilt(b, a, Lfoot_speed[1:]), name=Lfoot_speed.name)
    Lfoot_speed_filtered = pd.Series(gaussian_filter1d(Lfoot_speed[1:], 5), name=Lfoot_speed.name)

    # Find gait events
    low_Rfoot_speed = Rfoot_speed_filtered<forward_velocity_threshold
    frame_Ron, frame_Roff = start_end_true_seq(low_Rfoot_speed)
    if 0 in frame_Ron: frame_Ron.remove(0)
    t_Ron, t_Roff = time_col[frame_Ron].tolist(), time_col[frame_Roff].tolist()

    low_Lfoot_speed = Lfoot_speed_filtered<forward_velocity_threshold
    frame_Lon, frame_Loff = start_end_true_seq(low_Lfoot_speed)
    if 0 in frame_Lon: frame_Lon.remove(0)
    t_Lon, t_Loff = time_col[frame_Lon].tolist(), time_col[frame_Loff].tolist()

    # Clean gait events
    frame_Ron, frame_Lon, frame_Roff, frame_Loff = clean_gait_events((frame_Ron, frame_Lon, frame_Roff, frame_Loff), motion_type=motion_type)
    t_Ron, t_Lon, t_Roff, t_Loff = clean_gait_events((t_Ron, t_Lon, t_Roff, t_Loff), motion_type=motion_type)

    # Plot
    if plot:
        plt.plot(time_col[1:], Rfoot_speed_filtered, 'C0', label='Right foot speed filtered')
        plt.plot(time_col[1:][frame_Ron], Rfoot_speed_filtered[frame_Ron], 'g+')
        plt.plot(time_col[1:][frame_Roff], Rfoot_speed_filtered[frame_Roff], 'r+')

        plt.plot(time_col[1:], Lfoot_speed_filtered, 'C1', label='Left foot speed filtered')
        plt.plot(time_col[1:][frame_Lon], Lfoot_speed_filtered[frame_Lon], 'g+')
        plt.plot(time_col[1:][frame_Loff], Lfoot_speed_filtered[frame_Loff], 'r+')

        plt.title('Gait events')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.show()

    print('Times:')
    print('Right on:', t_Ron)
    print('Right off:', t_Roff)
    print('Left on:', t_Lon)
    print('Left off:', t_Loff)
    print('\nFrames:')
    print('Right on:', frame_Ron)
    print('Right off:', frame_Roff)
    print('Left on:', frame_Lon)
    print('Left off:', frame_Loff)

    return (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff)


def trc_gaitevents_func(**args):
    '''
    Determine gait on and off from a TRC file of point coordinates.

    Events are cleaned to make sure that each contact phases has a start and an end.
    Optionally and in case of noisy data, you can constrain the right and left steps to alternate (gait), 
    or the right and left steps to alternate with a flight phase inbetween (sprint), 
    or not constrain at all in case of hops on one foot (eg triple jump).

    Contacts and curves can be plotted to check the results.
    

    N.B.: Could implement the methods listed there in the future:
          https://www.tandfonline.com/doi/full/10.1080/19424280903133938 
          Please feel free to make a pull-request or keep me informed if you do so!

    Three available methods, each of them with their own pros and cons: 

    - "forward_coordinates": 
        on =  max(XHeel - Xsacrum)
        off = min(XToe - XSacrum)
        ++: Works well for walking (Zeni et al., 2008)
        ++: No argument nor tuning necessary
        --: Not suitable for running
        --: does not work if the person is not going on a straight line

    - "height_coordinates": 
        on =  YToe < height_threshold
        off = YToe > height_threshold
        ++: Best results for running and walking
        ++: Works if the person is not going on a straight line
        --: Does not work is the person is grazing the ground
        --: Does not work if the field is not flat
        --: height_threshold might need to be tuned
        --: Heel point might be more accurate than toe point for walking

    - "forward_velocity": 
        on =  VToe < forward_velocity_threshold
        off = VToe > forward_velocity_threshold
        ++: Works for running
        --: More sensitive to noise
        --: Tends to anticipate off if the marker is not at the tip of the toe
        --: forward_velocity_threshold might need to be tuned
        --: Does not work if the person is not going on a straight line
    
    Usage: 
    List of available arguments:
        trc_gaitevents -h

        import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>')
        OR import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>', method='forward_coordinates', gait_direction='-X', motion_type='gait', plot=True, save_output=True, output_file='gaitevents.txt')
        OR import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>', method='height_coordinates', up_direction='Y', height_threshold=6, motion_type='gait', right_toe_marker='RBigToe', left_toe_marker='LBigToe')
        OR import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(trc_path=r'<input_trc_file>', method='forward_velocity', gait_direction='-Z', forward_velocity_threshold=1, motion_type='gait', right_toe_marker='RBigToe', left_toe_marker='LBigToe')
        
        trc_gaitevents -i input_trc_file
        OR trc_gaitevents -i input_trc_file --method forward_coordinates --gait_direction=-X --motion_type gait --plot True --save_output True --output_file gaitevents.txt
        OR trc_gaitevents -i input_trc_file --method height_coordinates --up_direction=Y --height_threshold 6 --motion_type gait --right_toe_marker RBigToe --left_toe_marker LBigToe
        OR trc_gaitevents -i input_trc_file --method forward_velocity --gait_direction=-Z --forward_velocity_threshold 1 --motion_type gait --right_toe_marker RBigToe --left_toe_marker LBigToe
    '''

    # Retrieve arguments
    trc_path = args.get('trc_path')
    method = args.get('method')
    gait_direction = args.get('gait_direction')
    up_direction = args.get('up_direction')
    forward_velocity_threshold = args.get('forward_velocity_threshold')
    height_threshold = args.get('height_threshold')
    motion_type = args.get('motion_type')
    sacrum_marker = args.get('sacrum_marker')
    right_heel_marker = args.get('right_heel_marker')
    right_toe_marker = args.get('right_toe_marker')
    left_heel_marker = args.get('left_heel_marker')
    left_toe_marker = args.get('left_toe_marker')
    cut_off_frequency = args.get('cut_off_frequency')
    plot = args.get('plot')
    save_output = args.get('save_output')
    output_file = args.get('output_file')

    # If invoked via a function
    if method == None: method = 'height_coordinates'
    if gait_direction == None: gait_direction = '+X'
    if up_direction == None: up_direction = '+Y'
    if forward_velocity_threshold == None: forward_velocity_threshold = 1
    if height_threshold == None: height_threshold = 6
    if motion_type == None: motion_type = 'gait'
    if sacrum_marker == None: sacrum_marker = 'Hip'
    if right_heel_marker == None: right_heel_marker = 'RHeel'
    if right_toe_marker == None: right_toe_marker = 'RBigToe'
    if left_heel_marker == None: left_heel_marker = 'LHeel'
    if left_toe_marker == None: left_toe_marker = 'LBigToe'
    if cut_off_frequency == None: cut_off_frequency = 10
    if plot == None: plot = True
    if save_output == None: save_output = True
    if output_file == None: output_file = 'gaitevents.txt'

    # In case of a sign in direction (eg -X)
    if len(gait_direction)==1:
        gait_direction = +1, gait_direction
    elif len(gait_direction)==2:
        gait_direction = int(gait_direction[0]+'1'), gait_direction[1]
    if len(up_direction)==1:
        up_direction = +1, up_direction
    elif len(up_direction)==2:
        up_direction = int(up_direction[0]+'1'), up_direction

    if method not in ['forward_coordinates', 'height_coordinates', 'forward_velocity']:
        raise ValueError('Method must be "forward_coordinates", "height_coordinates", or "forward_velocity"')

    # Retrieve gait events
    if method == 'forward_coordinates':
        print('Method: forward_coordinates')
        print(f'Motion type: {motion_type}')
        markers = [right_heel_marker, right_toe_marker, left_heel_marker, left_toe_marker, sacrum_marker]
        (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff) \
                                = gait_events_fwd_coords(trc_path, gait_direction, motion_type=motion_type, markers=markers, plot=plot)
    elif method == 'height_coordinates':
        print(f'Method: height_coordinates. Height threshold: {height_threshold} cm')
        print(f'Motion type: {motion_type}')
        markers = [right_toe_marker, left_toe_marker]
        (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff) \
                                = gait_events_height_coords(trc_path, up_direction, height_threshold=height_threshold, motion_type=motion_type, cut_off_frequency=cut_off_frequency, markers=markers, plot=plot)
    elif method == 'forward_velocity':
        print(f'Method: forward_velocity. Forward velocity threshold: {forward_velocity_threshold} m/s')
        print(f'Motion type: {motion_type}')
        markers = [right_toe_marker, left_toe_marker]
        (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff) \
                                = gait_events_fwd_vel(trc_path, gait_direction, forward_velocity_threshold=forward_velocity_threshold, motion_type=motion_type, cut_off_frequency=cut_off_frequency, markers=markers, plot=plot)

    if save_output or save_output==None:
        trc_dir = os.path.dirname(trc_path)
        trc_name = os.path.basename(trc_path)
        if output_file == None: output_file = 'gaitevents.txt'
        with open(os.path.join(trc_dir, output_file), 'a') as gaitevents:
            L = trc_name + '\n'
            L += f'Method: {method}. '
            L += f'Height threshold: {height_threshold}\n' if method=='height_coordinates' else f'Forward velovity threshold: {forward_velocity_threshold}\n' if method == 'forward_velocity' else '\n'
            L += f'Motion type: {motion_type}\n'
            L += 'Times:\n'
            L += '\tRight on: ' + str(t_Ron) + '\n'
            L += '\tLeft on: ' + str(t_Lon) + '\n'
            L += '\tRight off: ' + str(t_Roff) + '\n'
            L += '\tLeft off: ' + str(t_Loff) + '\n'
            L += 'Frames:\n'
            L += '\tRight on: ' + str(frame_Ron) + '\n'
            L += '\tLeft on: ' + str(frame_Lon) + '\n'
            L += '\tRight off: ' + str(frame_Roff) + '\n'
            L += '\tLeft off: ' + str(frame_Loff) + '\n\n'

            gaitevents.write(L)
    
    return (t_Ron, t_Lon, t_Roff, t_Loff), (frame_Ron, frame_Lon, frame_Roff, frame_Loff)


if __name__ == '__main__':
    main()