#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #############################################################
    ## Display json 2d detections overlayed on original images ##
    #############################################################
    
    If you didn't save images when running OpenPose (parameter --write_images 
    not used), this tool lets you display the pose from json outputs, overlayed
    on the original raw images.
    High confidence keypoints are green, low confidence ones are red.

    Note: See 'json_display_without_img.py' if you only want to display the
    json coordinates on an animated graph or if don't have the original raw
    images.
    
    Usage: 
    python -m json_display_with_img -j json_folder -i raw_img_folder
    python -m json_display_with_img -j json_folder -i raw_img_folder -o output_img_folder -d True -s True
    from Pose2Sim.Utilities import json_display_with_img; json_display_with_img.json_display_with_img_func(json_folder=r'<json_folder>', raw_img_folder=r'<raw_img_folder>')
'''


## INIT
import os
import numpy as np
import json
import cv2
import cmapy
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def json_display_with_img_func(**args):
    '''
    Displays json 2d detections overlayed on original raw images.
    High confidence keypoints are green, low confidence ones are red.
     
    Note: See 'json_display_without_img.py' if you only want to display the
    json coordinates on an animated graph or if don't have the original raw
    images.
    
    Usage: 
    json_display_with_img -j json_folder -i raw_img_folder
    json_display_with_img -j json_folder -i raw_img_folder -o output_img_folder -d True -s True
    import json_display_with_img; json_display_with_img.json_display_with_img_func(json_folder=r'<json_folder>', raw_img_folder=r'<raw_img_folder>')
    '''

    json_folder = os.path.realpath(args.get('json_folder'))
    json_fnames = os.listdir(json_folder)
    raw_img_folder = os.path.realpath(args.get('raw_img_folder'))
    img_fnames = os.listdir(raw_img_folder)
    img_fnames = [e for e in img_fnames if not e.endswith('.db')]
    output_img_folder =  args.get('output_img_folder')
    if output_img_folder==None: 
        output_img_folder = os.path.join(json_folder+'_img')
    else:
        output_img_folder =  os.path.realpath(output_img_folder)
    save = args.get('save')
    display = args.get('display')
    
    for img_fname, json_fname in zip(img_fnames, json_fnames):
        # Données json
        xfrm, yfrm, conffrm = np.array([]), np.array([]), np.array([])    # Les coordonnées de toutes les personnes dans la frame
        with open(os.path.join(json_folder,json_fname)) as json_f:
            json_file = json.load(json_f)
            for ppl in range(len(json_file['people'])):  
                keypt = np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1,3)                
                xfrm = np.concatenate((xfrm,keypt[:,0]))
                yfrm = np.concatenate((yfrm,keypt[:,1]))
                conffrm = np.concatenate((conffrm,keypt[:,2]))
        
        # Read images and overlay json 2D coords
        img = cv2.imread(os.path.join(raw_img_folder,img_fname))
        for pt in range(len(xfrm)):
            cv2.circle(img, (int(xfrm[pt]), int(yfrm[pt])), 5, tuple(cmapy.color('RdYlGn', conffrm[pt])), -1) #thickness -1 pour que le cercle soit rempli
        
        # Display
        if display == True or display == 'True' or display =='1':
            cv2.imshow('', img)
            cv2.waitKey(0)
        
        # Save
        if save == True or save == 'True' or save == '1':
            if not os.path.exists(output_img_folder):
                os.mkdir(output_img_folder)
            cv2.imwrite(os.path.join(output_img_folder, img_fname), img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_folder', required = True, help='folder of json 2D coordinate files')
    parser.add_argument('-i', '--raw_img_folder', required = True, help='folder of original images')
    parser.add_argument('-o', '--output_img_folder', required=False, help='custom folder name for coordinates overlayed on images')
    parser.add_argument('-d', '--display', default=True, required = False, help='display images with overlayed coordinates')
    parser.add_argument('-s', '--save', default=False, required = False, help='save images with overlayed 2D coordinates')
    
    args = vars(parser.parse_args())
    json_display_with_img_func(**args)
