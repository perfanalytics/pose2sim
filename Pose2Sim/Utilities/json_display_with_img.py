#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #############################################################
    ## Display json 2d detections overlayed on original images ##
    #############################################################
    
    Overlay json pose estimation results on a video or series of images.
    High confidence keypoints are green, low confidence ones are red.

    Note: See 'json_display_without_img.py' if you only want to display the
    json coordinates.
    
    Usage: 
    json_display_with_img -j json_folder -i img_folder
    json_display_with_img -j json_folder -i video_file.mp4
    json_display_with_img -j json_folder -i img_folder -o output_img_folder -d True -s True
    json_display_with_img -j json_folder -i video_file.mp4 -t images
    from Pose2Sim.Utilities import json_display_with_img; json_display_with_img.json_display_with_img_func(json_folder=r'<json_folder>', input=r'<vid_or_img_folder>')
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
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_folder', required = True, help='folder of json 2D coordinate files')
    parser.add_argument('-i', '--input', required = True, help='video file or image folder')
    parser.add_argument('-o', '--output_img_folder', required=False, help='custom output path for overlayed results')
    parser.add_argument('-t', '--output_type', default='video', required=False, choices=['video', 'images'], help='output type: "video" (default) or "images"')
    parser.add_argument('-d', '--display', default=True, required = False, help='display images with overlayed coordinates')
    parser.add_argument('-s', '--save', default=True, required = False, help='save images with overlayed 2D coordinates')
    
    args = vars(parser.parse_args())
    json_display_with_img_func(**args)


def _is_video_file(path):
    '''Check if the given path is a video file based on its extension.'''
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in video_extensions


def _parse_json(json_path):
    '''Parse a JSON keypoint file and return x, y, confidence arrays.'''
    xfrm, yfrm, conffrm = np.array([]), np.array([]), np.array([])
    with open(json_path) as json_f:
        json_file = json.load(json_f)
        for ppl in range(len(json_file['people'])):
            try:
                keypt = np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1, 3)
                xfrm = np.concatenate((xfrm, keypt[:, 0]))
                yfrm = np.concatenate((yfrm, keypt[:, 1]))
                conffrm = np.concatenate((conffrm, keypt[:, 2]))
            except:
                xfrm = np.concatenate((xfrm, np.full((25,), np.nan)))
                yfrm = np.concatenate((yfrm, np.full((25,), np.nan)))
                conffrm = np.concatenate((conffrm, np.full((25,), np.nan)))
    return xfrm, yfrm, conffrm


def _overlay_keypoints(img, xfrm, yfrm, conffrm):
    '''Draw keypoint circles on the image, skipping NaN coordinates.'''
    for pt in range(len(xfrm)):
        if np.isnan(xfrm[pt]) or np.isnan(yfrm[pt]) or np.isnan(conffrm[pt]):
            continue
        color = tuple(cmapy.color('RdYlGn', conffrm[pt]))
        cv2.circle(img, (int(xfrm[pt]), int(yfrm[pt])), 5, color, -1)
    return img


def _to_bool(value):
    '''Convert various truthy representations to bool.'''
    return value in (True, 'True', 'true', '1', 1)


def _frame_iterator(input_path, json_fnames):
    '''
    Yield (frame_image, output_filename) pairs from either a video or image folder,
    matched with the sorted json filenames.
    '''
    if _is_video_file(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f'Could not open video: {input_path}')
        for idx, json_fname in enumerate(json_fnames):
            ret, img = cap.read()
            if not ret:
                break
            out_fname = f'{idx:06d}.png'
            yield img, out_fname
        cap.release()
    else:
        img_fnames = sorted(os.listdir(input_path))
        img_fnames = [e for e in img_fnames if not e.endswith('.db')]
        for img_fname, json_fname in zip(img_fnames, json_fnames):
            img = cv2.imread(os.path.join(input_path, img_fname))
            yield img, img_fname

    
def json_display_with_img_func(**args):
    '''
    Overlay json pose estimation results on a video or series of images.
    High confidence keypoints are green, low confidence ones are red.
    Output can be saved as a video or as an image folder.

    Note: See 'json_display_without_img.py' if you only want to display the
    json coordinates.
    
    Usage: 
    json_display_with_img -j json_folder -i img_folder
    json_display_with_img -j json_folder -i video_file.mp4
    json_display_with_img -j json_folder -i img_folder -o output_img_folder -d True -s True
    json_display_with_img -j json_folder -i video_file.mp4 -t images
    from Pose2Sim.Utilities import json_display_with_img; json_display_with_img.json_display_with_img_func(json_folder=r'<json_folder>', input=r'<vid_or_img_folder>')
    '''

    json_folder = os.path.realpath(args.get('json_folder'))
    json_fnames = sorted(os.listdir(json_folder))
    input_path = os.path.realpath(args.get('input'))
    output_path = args.get('output_img_folder')
    output_type = args.get('output_type', 'video')
    save = _to_bool(args.get('save'))
    display = _to_bool(args.get('display'))
    is_video = _is_video_file(input_path)
    
    # Default output path
    if output_path is None:
        base = json_folder.replace('json', 'overlay') if 'json' in json_folder else json_folder + '_overlay'
        if output_type == 'video':
            output_path = base + '.mp4'
        else:
            output_path = base
    else:
        output_path = os.path.realpath(output_path)
    
    # Set up video writer or output directory
    out_writer = None
    if save:
        if output_type == 'video':
            # Need first frame to get dimensions; peek via iterator
            frame_iter = _frame_iterator(input_path, json_fnames)
            first_img, first_out_fname = next(frame_iter)
            
            # Determine fps: use source video fps if available, else 30
            if is_video:
                cap_tmp = cv2.VideoCapture(input_path)
                fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30
                cap_tmp.release()
            else:
                fps = 30
            
            h, w = first_img.shape[:2]
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Process the first frame we already read
            xfrm, yfrm, conffrm = _parse_json(os.path.join(json_folder, json_fnames[0]))
            _overlay_keypoints(first_img, xfrm, yfrm, conffrm)
            if display:
                cv2.imshow('', first_img)
                key = cv2.waitKey(1) if is_video else cv2.waitKey(0)
            out_writer.write(first_img)
            
            # Process remaining frames
            for (img, out_fname), json_fname in zip(frame_iter, json_fnames[1:]):
                xfrm, yfrm, conffrm = _parse_json(os.path.join(json_folder, json_fname))
                _overlay_keypoints(img, xfrm, yfrm, conffrm)
                if display:
                    cv2.imshow('', img)
                    key = cv2.waitKey(1) if is_video else cv2.waitKey(0)
                    if key & 0xFF == ord('q'):
                        break
                out_writer.write(img)
            
            out_writer.release()
            print(f'Saved overlayed video to: {output_path}')
            cv2.destroyAllWindows()
            return
        
        else:
            os.makedirs(output_path, exist_ok=True)
    
    # Process frames (output_type == 'images' or not saving)
    for (img, out_fname), json_fname in zip(_frame_iterator(input_path, json_fnames), json_fnames):
        xfrm, yfrm, conffrm = _parse_json(os.path.join(json_folder, json_fname))
        _overlay_keypoints(img, xfrm, yfrm, conffrm)
        
        if display:
            cv2.imshow('', img)
            key = cv2.waitKey(1) if is_video else cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break
        if save:
            cv2.imwrite(os.path.join(output_path, out_fname), img)
    
    if save:
        print(f'Saved overlayed images to: {output_path}')
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()