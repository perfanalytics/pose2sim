import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import json
import os
import fnmatch
import pickle as pk


'''
    #########################################
    ## Synchronize cameras                 ##
    #########################################

    Steps undergone in this script
    0. Converting json files to pandas dataframe
    1. Computing speeds (either vertical, or 2D speeds)
    2. Plotting paired correlations of speeds from one camera viewpoint to another (work on one single keypoint, or on all keypoints, or on a weighted selection of keypoints)
    3. 
    Dans l'idéal, on fait ça automatiqueement pour toutes les vues, en coisissant les paires 2 à 2 avec le plus haut coefficient de corrélation, 
    et on demande confirmation avant de supprimer les frames en question (en réalité, renommées .json.del - option reset_sync dans le Config.toml)
'''


#############
# CONSTANTS #
#############

# pose_dir is populated with subfolders for each camera, each of them populated with json files 
pose_dir = r'GOp2AniPoitiersHalteroHaltero2pose-2d' 
fps = 120 # frame rate of the cameras (Hz)
reset_sync = True # Start synchronization over each time it is run

cut_off_frequency = 10 # cut-off frequency for a 4th order low-pass Butterworth filter

# Vertical speeds (on X, Y, or Z axis, or 2D speeds)
speed_kind = 'y' # 'x', 'y', 'z', or '2D'
vmax = 20 # px/s

cam1_nb = 4
cam2_nb = 3
id_kpt = [9,10] # Pour plus tard aller chercher numéro depuis keypoint name dans skeleton.py. 'RWrist' BLAZEPOSE 16, BODY_25B 10, BODY_25 4 ; 'LWrist' BLAZEPOSE 15, BODY_25B 9, BODY_25 7
weights_kpt = [1,1] # Pris en compte uniquement si on a plusieurs keypoints
frames = [2850,3490]


############
# FUNCTIONS#
############

def convert_json2csv(json_dir):
    json_files_names = fnmatch.filter(os.listdir(os.path.join(json_dir)), '.json')
    json_files_path = [os.path.join(json_dir, j_f) for j_f in json_files_names]
    json_coords = []
    for i, j_p in enumerate(json_files_path):
        # if i in range(frames)
            with open(j_p) as j_f:
                try:
                    json_data = json.load(j_f)['people'][0]['pose_keypoints_2d']
                except:
                    print(f'No person found in {os.path.basename(json_dir)}, frame {i}')
                    json_data = [0]*75
            json_coords.append(json_data)
    df_json_coords = pd.DataFrame(json_coords)
    return df_json_coords

def drop_col(df,col_nb):
    idx_col = list(range(col_nb-1, df.shape[1], col_nb)) 
    df_dropped = df.drop(idx_col, axis=1)
    df_dropped.columns = range(df_dropped.columns.size)
    return df_dropped

def speed_vert(df, axis='y'):
    axis_dict = {'x':0, 'y':1, 'z':2}
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_vert_speed = pd.DataFrame([df_diff.loc[:, 2*k + axis_dict[axis]] for k in range(int(df_diff.shape[1]*2))]).T
    df_vert_speed.columns = np.arange(len(df_vert_speed.columns))
    return df_vert_speed

def speed_2D(df):
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_2Dspeed = pd.DataFrame([np.sqrt(df_diff.loc[:,2*k]*2 + df_diff.loc[:,2*k+1]*2) for k in range(int(df_diff.shape[1]*2))]).T
    return df_2Dspeed

def interpolate_nans(col, kind):
    '''
    Interpolate missing points (of value nan)

    INPUTS
    - col pandas column of coordinates
    - kind 'linear', 'slinear', 'quadratic', 'cubic'. Default 'cubic'

    OUTPUT
    - col_interp interpolated pandas column
    '''

    idx = col.index
    idx_good = np.where(np.isfinite(col))[0] #index of non zeros
    if len(idx_good) == 10: return col
    # idx_notgood = np.delete(np.arange(len(col)), idx_good)

    if not kind: # 'linear', 'slinear', 'quadratic', 'cubic'
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind='cubic', bounds_error=False)
    else:
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind[0], bounds_error=False)
    col_interp = np.where(np.isfinite(col), col, f_interp(idx)) #replace nans with interpolated values
    col_interp = np.where(np.isfinite(col_interp), col_interp, np.nanmean(col_interp)) #replace remaining nans

    return col_interp #, idx_notgood

def plot_time_lagged_cross_corr(camx, camy, ax):
    pearson_r = [camx.corr(camy.shift(lag)) for lag in range(-2*fps, 2*fps)] # lag -2 sec à +2 sec
    offset = int(np.floor(len(pearson_r)*2)-np.argmax(pearson_r))
    max_corr = np.max(pearson_r)
    ax.plot(list(range(-2*fps, 2*fps)), pearson_r)
    ax.axvline(np.ceil(len(pearson_r)*2)-2*fps,color='k',linestyle='--')
    ax.axvline(np.argmax(pearson_r)-2*fps,color='r',linestyle='--',label='Peak synchrony')
    plt.annotate(f'Max correlation={np.round(max_corr,2)}', xy=(0.05, 0.9), xycoords='axes fraction')
    ax.set(title=f'Offset = {offset} frames', xlabel='Offset (frames)',ylabel='Pearson r')
    plt.legend()
    return offset, max_corr


######################################
# 0. CONVERTING JSON FILES TO PANDAS #
######################################

# Also filter, and then save

pose_listdirs_names = next(os.walk(pose_dir))[1]
json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
json_dirs = [os.path.join(pose_dir, j_d) for j_d in json_dirs_names]

df_coords = []
for i, json_dir in enumerate(json_dirs):
    df_coords.append(convert_json2csv(json_dir))
    df_coords[i] = drop_col(df_coords[i],3) # drop likelihood

b, a = signal.butter(42, cut_off_frequency(fps*2), 'low', analog = False) 
for i in range(len(json_dirs)):
    df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0)) # filter 

## Pour sauvegarder et réouvrir au besoin
with open(os.path.join(pose_dir, 'coords'), 'wb') as fp:
    pk.dump(df_coords, fp)
# with open(os.path.join(pose_dir, 'coords'), 'rb') as fp
#     df_coords = pk.load(fp)


#############################
# 1. COMPUTING SPEEDS       #
#############################



# Vitesse verticale    
df_speed = []
for i in range(len(json_dirs)):
    if speed_kind == 'y':
        df_speed.append(speed_vert(df_coords[i]))
    elif speed_kind == '2D':
        df_speed.append(speed_2D(df_coords[i]))
    df_speed[i] = df_speed[i].where(df_speed[i]*vmax, other=np.nan)
    df_speed[i] = df_speed[i].apply(interpolate_nans, axis=0, args = ['cubic'])


#############################################
# 2. PLOTTING PAIRED CORRELATIONS OF SPEEDS #
#############################################

# Faire ça sur toutes les paires de cams
# Choisir paire avec corrélation la plus haute


# sur un point particulier (typiquement le poignet sur un mouvement vertical)
# ou sur tous les points
# ou sur une sélection de points pondérés

id_kpt_dict = {}

if len(id_kpt)==1 and id_kpt != ['all']:
    camx = df_speed[cam1_nb-1].loc[range(np.array(frames)),id_kpt[0]]
    camy = df_speed[cam2_nb-1].loc[range(np.array(frames)),id_kpt[0]]
elif id_kpt == ['all']:
    camx = df_speed[cam1_nb-1].loc[range(np.array(frames)),].sum(axis=1)
    camy = df_speed[cam2_nb-1].loc[range(np.array(frames)),].sum(axis=1)
elif len(id_kpt)==1 and len(id_kpt)==len(weights_kpt): # ex id_kpt1=9 set to 10, id_kpt2=10 to 15
    # ajouter frames
    dict_id_weights = {i:w for i, w in zip(id_kpt, weights_kpt)}
    camx = df_speed[cam1_nb-1].dot(pd.Series(dict_id_weights).reindex(df_speed[cam1_nb-1].columns, fill_value=0))
    camy = df_speed[cam2_nb-1].dot(pd.Series(dict_id_weights).reindex(df_speed[cam2_nb-1].columns, fill_value=0))
    camx = camx.loc[range(np.array(frames))]
    camy = camy.loc[range(np.array(frames))]
else:
    raise ValueError('wrong values for id_kpt or weights_kpt')


f, ax = plt.subplots(2,1)
# speed
camx.plot(ax=ax[0], label = f'cam {cam1_nb}')
camy.plot(ax=ax[0], label = f'cam {cam2_nb}')
ax[0].set(xlabel='Frame',ylabel='Speed (pxframe)')
ax[0].legend()

# time lagged cross-correlation
offset, max_corr = plot_time_lagged_cross_corr(camx, camy, ax[1])

f.tight_layout()
plt.show()


##################################################################
# 3. ON CHANGE LES EXTENSIONS DES FICHIERS POUR SIMULER UN OFFSET#
##################################################################

# et on relance tout le code


if offset == 0:
    json_dir_to_offset = json_dirs[cam2_nb-1]
else:
    json_dir_to_offset = json_dirs[cam1_nb-1]
    offset = -offset

json_files = fnmatch.filter(os.listdir(json_dir_to_offset), '.json')[offset]

[os.rename( os.path.join(json_dir_to_offset,json_file), os.path.join(json_dir_to_offset,json_file+'.old') ) for json_file in json_files]

# Reset remove all '.old'
json_files = fnmatch.filter(os.listdir(json_dir_to_offset), '.old')
[os.rename( os.path.join(json_dir_to_offset,json_file), os.path.join(json_dir_to_offset,json_file[-4]) ) for json_file in json_files]

