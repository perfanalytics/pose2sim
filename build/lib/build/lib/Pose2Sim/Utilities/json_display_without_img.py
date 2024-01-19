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
    python -m json_display_without_img -j json_folder
    python -m json_display_without_img -j json_folder -o output_img_folder -d True -s True
    from Pose2Sim.Utilities import json_display_without_img; json_display_without_img.json_display_without_img_func(json_folder=r'<json_folder>')
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
__version__ = '0.4'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# ## CLASSES
class BunchOFiles(FileMovieWriter):
    '''
    Borrowed from https://stackoverflow.com/a/41273312/12196632
    '''

    supported_formats = ['png', 'jpeg', 'bmp', 'svg', 'pdf']

    def __init__(self, *args, extra_args=None, **kwargs):
        # extra_args aren't used but we need to stop None from being passed
        super().__init__(*args, extra_args=(), **kwargs)

    def setup(self, fig, dpi, frame_prefix):
        super().setup(fig, dpi, frame_prefix) #, clear_temp=False)
        self.fname_format_str = '%s%%d.%s'
        self.temp_prefix, self.frame_format = self.outfile.split('.')

    def grab_frame(self, **savefig_kwargs):
        '''
        Grab the image information from the figure and save as a movie frame.
        All keyword arguments in savefig_kwargs are passed on to the 'savefig' command that saves the figure.
        '''
        # Tell the figure to save its data to the sink, using the frame format and dpi.
        with self._frame_sink() as myframesink:
            self.fig.savefig(myframesink, format=self.frame_format, dpi=self.dpi, **savefig_kwargs)

    def finish(self):
        self._frame_sink().close()


## FUNCTIONS
def json_display_without_img_func(**args):
    '''
    This function lets you display 2D coordinates json files on an animated graph.
    High confidence keypoints are green, low confidence ones are red.

    Note: See 'json_display_without_img.py' if you want to overlay the json 
    coordinates on the original images.
    
    Usage: 
    json_display_without_img -j json_folder
    json_display_without_img -j json_folder -o output_img_folder -d True -s True
    import json_display_without_img; json_display_without_img.json_display_without_img_func(json_folder=r'<json_folder>')
    '''

    json_folder = os.path.realpath(args.get('json_folder'))
    json_fnames = [f for f in os.listdir(json_folder) if os.path.isfile(os.path.join(json_folder, f))]
    output_img_folder =  args.get('output_img_folder')
    if output_img_folder==None: 
        output_img_folder = os.path.join(json_folder+'_img')
    else:
        output_img_folder =  os.path.realpath(output_img_folder)
    display = args.get('display')
    if display==None: 
        display = True
    save = args.get('save')

    # Save
    if save == True or save == 'True':
        if not os.path.exists(output_img_folder):
            os.mkdir(output_img_folder)

    # Données json
    width, height = 2000,2000
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
            scat.set_offsets(np.c_[X[frame], height-Y[frame]])
            scat.set_array(CONF[frame])
        return scat,
    
    fig = plt.figure()
    ax = plt.axes(xlim = (0,width), ylim = (0,height))
    scat = ax.scatter(X[0],height-Y[0], marker='+', cmap='RdYlGn', c=CONF[0])
    anim = FuncAnimation(fig, update, interval=33, frames=np.arange(len(json_fnames)), repeat=False) #, blit=True

    # Display
    if display == True or display == 'True' or display =='1':    
        plt.show()

    # Save
    if save == True or save=='True' or save == '1':
        anim.save(os.path.join(output_img_folder, 'image.png'), writer=BunchOFiles())

    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_folder', required = True, help='folder of json 2D coordinate files')
    parser.add_argument('-o', '--output_img_folder', required=False, help='custom folder name for coordinates overlayed on images')
    parser.add_argument('-d', '--display', default=True, required = False, help='display images with overlayed coordinates')
    parser.add_argument('-s', '--save', default=False, required = False, help='save images with overlayed 2D coordinates')
    
    args = vars(parser.parse_args())
    json_display_without_img_func(**args)
