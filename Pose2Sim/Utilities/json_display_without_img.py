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
    python -m json_display_without_img -j json_folder -W 1920 -H 1080
    python -m json_display_without_img -j json_folder -o output_img_folder -d True -s True -W 1920 -H 1080 - 30
    import json_display_without_img; json_display_without_img.json_display_without_img_func(json_folder=r'<json_folder>', image_width=1920, image_height = 1080)
'''


## INIT
import os
import numpy as np
import json
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, FileMovieWriter 
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.6'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
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
    json_display_without_img -j json_folder -o output_img_folder -d True -s True -W 1920 -H 1080
    import json_display_without_img; json_display_without_img.json_display_without_img_func(json_folder=r'<json_folder>', image_width=1920, image_height = 1080)
    '''

    json_folder = os.path.realpath(args.get('json_folder'))
    json_fnames = [f for f in os.listdir(json_folder) if os.path.isfile(os.path.join(json_folder, f))]
    json_fnames.sort(key=lambda f: int(f.split('_')[0])) # sort by frame number
    
    output_img_folder =  args.get('output_img_folder')
    if output_img_folder==None: 
        output_img_folder = os.path.join(json_folder+'_img')
    else:
        output_img_folder =  os.path.realpath(output_img_folder)
    image_width =  args.get('image_width')
    if image_width==None: 
        image_width = 2000
    else:
        image_width = int(image_width)
    image_height =  args.get('image_height')
    if image_height==None: 
        image_height = 2000
    else:
        image_height = int(image_height)

    frame_rate = int(args.get('frame_rate'))
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
            for ppl in range(len(json_file['people'])):  
                keypt = np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1,3)                
                xfrm = np.concatenate((xfrm,keypt[:,0]))
                yfrm = np.concatenate((yfrm,keypt[:,1]))
                conffrm = np.concatenate((conffrm,keypt[:,2]))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_folder', required = True, help='folder of json 2D coordinate files')
    parser.add_argument('-W', '--image_width', required = False, help='image width')
    parser.add_argument('-H', '--image_height', required = False, help='image height')
    parser.add_argument('-f', '--frame_rate', required = False, help='frame rate')
    parser.add_argument('-o', '--output_img_folder', required=False, help='custom folder name for coordinates overlayed on images')
    parser.add_argument('-d', '--display', default=True, required = False, help='display images with overlayed coordinates')
    parser.add_argument('-s', '--save', default=False, required = False, help='save images with overlayed 2D coordinates')
    
    args = vars(parser.parse_args())
    json_display_without_img_func(**args)
