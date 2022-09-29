import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from calib_qca_to_toml import toml_write

def convert_calib(*args):
    '''
    Read a mc_dev calibration file
    Returns 5 lists of size N (N=number of cameras):
    - C (camera name),
    - S (image size),
    - D (distorsion),
    - K (intrinsic parameters),
    - R (extrinsic rotation),
    - T (extrinsic translation)
    '''

    root_path = args[0].get('input_dir') # invoked with argparse
    if args[0].get('output_file') is not None:
        toml_path = args[0].get('output_file')
    else:
        toml_path = os.path.join(root_path,"Calib.toml")


    C, S, D, K, R, T = [], [], [], [], [], []
    cam_num = 0
    for item in os.listdir(root_path):

        if os.path.splitext(item)[-1] == ".calib":
            cam_num +=1
            calib_path = os.path.join(root_path,item)
            with open(calib_path,"r") as ifile:
                lines = ifile.readlines()
                C.append("cam" + str(cam_num))
                S.append([int(lines[0].rstrip("\n")),int(lines[1].rstrip("\n"))])
                k_line = []
                for i in range(2,5):
                    # read the line, remove the newline character (.rstrip(\n)), remove trailing white space (.rstrip()) , split
                    # values on whitespace and convert to float
                    k_line.extend([float(x) for x in lines[i].rstrip("\n").rstrip().split(" ")])

                K.append(np.array(k_line).reshape(3,3))
                L = []
                for i in range(6,10):
                    # same as for k matrix
                    l_line = [float(x) for x in lines[i].rstrip("\n").rstrip().split(" ")]
                    L.extend(l_line)
                L = np.array(L).reshape(4,4)
                # now we have the L matrix, we can invert it and get the rotations and translations using scipy
                L_inv = np.linalg.inv(L)
                R.append(Rot.from_matrix(L_inv[:3,:3]).as_mrp())
                # camera origin is the rightmost column of L_inv
                T.append(list(L_inv[:,-1][:3]))
                D.append([float(x) for x in lines[11].rstrip("\n").rstrip().split(" ")])
        toml_write(toml_path, C, S, D, K, R, T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required = True, help='Directory where .calib files are located')
    parser.add_argument('-o', '--output_file', required=False, help='OpenCV .toml output calibration file in')
    args = vars(parser.parse_args())
    convert_calib(args)

