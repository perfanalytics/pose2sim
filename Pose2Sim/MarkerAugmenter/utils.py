from . import utilsDataman
import numpy as np
import os

def TRC2numpy(pathFile, markers,rotation=None):
    # rotation is a dict, eg. {'y':90} with axis, angle for rotation
    
    trc_file = utilsDataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    
    if rotation != None:
        for axis,angle in rotation.items():
            trc_file.rotate(axis,angle)
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out

def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame+1,(frame)/fc+t_start)) # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")

def getOpenPoseMarkerNames():
    
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    
    return markerNames

def getOpenPoseFaceMarkers():
    
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]
    
    return faceMarkerNames, idxFaceMarkers

def getOpenPoseMarkers_fullBody():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

def getOpenPoseMarkers_lowerExtremity():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

# Different order of markers compared to getOpenPoseMarkers_lowerExtremity 
def getOpenPoseMarkers_lowerExtremity2():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = [
        'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
        'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 
        'r_ankle_study', 'r_mankle_study', 'r_toe_study', 
        'r_5meta_study', 'r_calc_study', 'L_knee_study', 
        'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
        'L_toe_study', 'L_calc_study', 'L_5meta_study', 
        'r_shoulder_study', 'L_shoulder_study', 'C7_study', 
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
        'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']

    return feature_markers, response_markers


def getMarkers_upperExtremity_pelvis():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
        "RWrist", "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

def getMarkers_upperExtremity_noPelvis():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

# Different order of markers compared to getMarkers_upperExtremity_noPelvis.
def getMarkers_upperExtremity_noPelvis2():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers
