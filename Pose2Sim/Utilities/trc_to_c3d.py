"""
Extracts marker data from a TRC file and creates a corresponding C3D file.

Usage:
    python trc_to_c3d.py --trc_path <path_to_trc_file> --f <frame_rate>

--trc_path: Path to the TRC file.
--f: Frame rate of the 3D data.
c3d file will be saved in the same directory as the TRC file with the same name.

"""

import os
import argparse
import numpy as np
import pandas as pd
import c3d

def extract_marker_data(trc_file):
    with open(trc_file, 'r') as file:
        lines = file.readlines()
        marker_names_line = lines[3]
        marker_names = marker_names_line.strip().split('\t')[2::3]

    trc_data = pd.read_csv(trc_file, sep='\t', skiprows=5)
    marker_coords = trc_data.iloc[:, 2:].to_numpy().reshape(-1, len(marker_names), 3)
    # marker_coords = np.nan_to_num(marker_coords, nan=0.0)
    marker_coords *= 1000  # Convert from meters to millimeters

    return marker_names, marker_coords

def create_c3d_file(trc_file, marker_names, marker_coords, frame_rate):
    writer = c3d.Writer(point_rate=frame_rate, analog_rate=0, point_scale=1.0, point_units='mm', gen_scale=-1.0)
    writer.set_point_labels(marker_names)
    
    markers_group = writer.point_group

    for frame in marker_coords:
        residuals = np.full((frame.shape[0], 1), 0.0)
        cameras = np.zeros((frame.shape[0], 1))
        points = np.hstack((frame, residuals, cameras))
        writer.add_frames([(points, np.array([]))])

    writer.set_start_frame(1)
    writer._set_last_frame(len(marker_coords))

    c3d_file_path = trc_file.replace('.trc', '.c3d')
    with open(c3d_file_path, 'wb') as handle:
        writer.write(handle)
    print(f"Successfully created c3d file.")

def trc_to_c3d(trc_file, frame_rate):
    marker_names, marker_coords = extract_marker_data(trc_file)
    create_c3d_file(trc_file, marker_names, marker_coords, frame_rate)

def main():
    parser = argparse.ArgumentParser(description='Convert TRC files to C3D files.')
    parser.add_argument('--trc_path', type=str, required=True, help='Path to the TRC file')
    parser.add_argument('--f', type=int, required=True, help='Frame rate')

    args = parser.parse_args()

    trc_file = args.trc_path
    frame_rate = args.f

    if not os.path.isfile(trc_file):
        print(f"Error: {trc_file} does not exist.")
        return

    trc_to_c3d(trc_file, frame_rate)

if __name__ == '__main__':
    main()
