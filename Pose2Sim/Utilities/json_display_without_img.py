#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #####################################################
    ## Display json 2d detections on an animated graph ##
    #####################################################
    
    This tool lets you display 2D coordinates json files on an animated graph.
    High confidence keypoints are green, low confidence ones are red.

    Note: See 'json_display_without_img.py' if you want to overlay the json 
    coordinates on the original images.
    
    Usage: 
    json_display_without_img -j json_folder -W 1920 -H 1080
    json_display_without_img -j json_folder -o output_img_folder -d True -s True -W 1920 -H 1080 --id_persons 1 2
    import json_display_without_img; json_display_without_img.json_display_without_img_func(json_folder=r'<json_folder>', image_width=1920, image_height = 1080, id_persons=(1,2))
'''


## INIT
import os
import numpy as np
import json
import re
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, FileMovieWriter 
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
    parser.add_argument('-j', '--json_folder', required = True, help='folder of json 2D coordinate files')
    parser.add_argument('-i', '--id_persons', required = False, nargs="+", type=int, help='ids of the persons you want to display')
    parser.add_argument('-W', '--image_width', required = False, type=int, help='image width')
    parser.add_argument('-H', '--image_height', required = False, type=int, help='image height')
    parser.add_argument('-f', '--frame_rate', required = False, type=float, help='frame rate')
    parser.add_argument('-o', '--output_img_folder', required=False, help='custom folder name for coordinates overlayed on images')
    parser.add_argument('-d', '--display', default=True, required = False, help='display images with overlayed coordinates')
    parser.add_argument('-s', '--save', default=False, required = False, help='save images with overlayed 2D coordinates')
    
    args = vars(parser.parse_args())
    json_display_without_img_func(**args)


def sort_stringlist_by_last_number(string_list):
    '''
    Sort a list of strings based on the last number in the string.
    Works if other numbers in the string, if strings after number. Ignores alphabetical order.

    Example: ['json1', 'js4on2.b', 'eypoints_0000003.json', 'ajson0', 'json10']
    gives: ['ajson0', 'json1', 'js4on2.b', 'eypoints_0000003.json', 'json10']
    '''
    
    def sort_by_last_number(s):
        return int(re.findall(r'\d+', s)[-1])
    
    return sorted(string_list, key=sort_by_last_number)


def save_inp_as_output(_img, c_name, dpi=100):
    h, w, _ = _img.shape
    fig, axes = plt.subplots(figsize=(h/dpi, w/dpi))
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
    axes.imshow(_img)
    axes.axis('off')
    plt.savefig(c_name, dpi=dpi, format='jpeg')


def json_display_without_img_func(**args):
    '''
    This function lets you display 2D coordinates json files on an animated graph.
    High confidence keypoints are green, low confidence ones are red.

    Note: See 'json_display_without_img.py' if you want to overlay the json 
    coordinates on the original images.
    
    Usage: 
    json_display_without_img -j json_folder -W 1920 -H 1080
    json_display_without_img -j json_folder -o output_img_folder -d True -s True -W 1920 -H 1080 --id_persons 1 2
    import json_display_without_img; json_display_without_img.json_display_without_img_func(json_folder=r'<json_folder>', image_width=1920, image_height = 1080, id_persons=(1,2))
    '''

    json_folder = os.path.realpath(args.get('json_folder'))
    json_fnames = [f for f in os.listdir(json_folder) if os.path.isfile(os.path.join(json_folder, f))]
    json_fnames = sort_stringlist_by_last_number(json_fnames)
    
    output_img_folder =  args.get('output_img_folder')
    if output_img_folder==None: 
        output_img_folder = os.path.join(json_folder+'_img')
    else:
        output_img_folder =  os.path.realpath(output_img_folder)
    image_width =  args.get('image_width')
    if image_width==None: 
        image_width = 2000
    image_height =  args.get('image_height')
    if image_height==None: 
        image_height = 2000
    id_persons =  args.get('id_persons')
    if id_persons == None:
        id_persons = 'all'
    frame_rate = args.get('frame_rate')
    if frame_rate==None:
        frame_rate = 30
    display = args.get('display')
    if display==None: 
        display = True
    save = args.get('save')

    # Save
    if save == True or save == 'True':
        if not os.path.exists(output_img_folder):
            os.mkdir(output_img_folder)

    # Données json
    X,Y,CONF = [], [], []
    for json_fname in json_fnames:    
        xfrm, yfrm, conffrm = np.array([]), np.array([]), np.array([])    # Coordinates of all people in frame
        with open(os.path.join(json_folder,json_fname)) as json_f:
            json_file = json.load(json_f)
            if id_persons == 'all':
                for ppl in range(len(json_file['people'])):  
                    keypt = np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1,3)
                    xfrm = np.concatenate((xfrm,keypt[:,0]))
                    yfrm = np.concatenate((yfrm,keypt[:,1]))
                    conffrm = np.concatenate((conffrm,keypt[:,2]))
            elif isinstance(id_persons, list):
                for ppl in id_persons:  
                    try:
                        keypt = np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1,3)
                        xfrm = np.concatenate((xfrm,keypt[:,0]))
                        yfrm = np.concatenate((yfrm,keypt[:,1]))
                        conffrm = np.concatenate((conffrm,keypt[:,2]))
                    except:
                        xfrm = np.concatenate((xfrm,[]))
                        yfrm = np.concatenate((yfrm,[]))
                        conffrm = np.concatenate((conffrm,[]))
        X += [xfrm]
        Y += [yfrm]
        CONF += [conffrm]

    # Scatterplot
    def update(frame):
        if frame==len(json_fnames)-1:
            plt.close(fig)
        else:
            scat.set_offsets(np.c_[X[frame], image_height-Y[frame]])
            scat.set_array(CONF[frame])
            if save == True or save=='True' or save == '1':
                output_name = os.path.join(output_img_folder, f'{os.path.basename(output_img_folder)}_{str(frame).zfill(5)}.png')
                plt.savefig(output_name)
        return scat,
    
    fig = plt.figure()
    ax = plt.axes(xlim = (0,image_width), ylim = (0,image_height))
    ax.set_aspect('equal', adjustable='box')
    scat = ax.scatter(X[0],image_height-Y[0], marker='+', cmap='RdYlGn', c=CONF[0])
    
    interval_img = int(1000/frame_rate)
    anim = FuncAnimation(fig, update, interval=interval_img, frames=np.arange(len(json_fnames)), repeat=False) #, blit=True

    # Display
    if display == True or display == 'True' or display =='1':    
        plt.show()

    plt.close('all')


if __name__ == '__main__':
    main()
