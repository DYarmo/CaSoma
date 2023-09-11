#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:43:06 2022

@author: ch184656
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo


def normCOR(obj):
    
    filename =  obj.DBpath
    var_name_hdf5 = obj.DB['Animals'][obj.curAnimal][obj.curFOV][obj.dataFocus].name
    
    max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
    max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    
    #%% start the cluster 
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False, ignore_preexisting=True)
    
    # create a motion correction object
    mc = MotionCorrect(filename, dview=dview, max_shifts=max_shifts,
                      strides=strides, overlaps=overlaps,
                      max_deviation_rigid=max_deviation_rigid, 
                      shifts_opencv=shifts_opencv, nonneg_movie=True,
                      border_nan=border_nan)
    
    mc.motion_correct(save_movie=True)
    
    
    # load motion corrected movie
    m_rig = cm.load(mc.mmap_file)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
    #%% visualize templates
    plt.figure(figsize = (20,10))
    plt.imshow(mc.total_template_rig, cmap = 'gray')
    
    #%% inspect movie
    m_rig.resize(1, 1, 0.2).play(
        q_max=99.5, fr=30, magnification=2, bord_px = 0*bord_px_rig) # press q to exit
    
    
    
    #%% plot rigid shifts
    plt.close()
    plt.figure(figsize = (20,10))
    plt.plot(mc.shifts_rig)
    plt.legend(['x shifts','y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')



