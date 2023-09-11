#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:23:28 2022

@author: ch184656
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage import data, util, transform, feature, measure, filters, metrics
from pystackreg import StackReg
import copy

def getTransform(template, target, mode = StackReg.RIGID_BODY):
    sr = StackReg(mode)
    return(sr.register(template, target))
    
def getTransforms(stacks, mode = StackReg.RIGID_BODY, filt_vasc = None):
    Images = []
    for stack in stacks:
         
        medImage = np.median(stack[0:10,...], axis=0)
       
        fillValue = np.median(medImage)
        medImage[medImage==0] = fillValue
        Images.append(medImage)
    template = Images[0]
    tfs = []
    for image in Images:
        tfs.append(getTransform(template, image, mode=mode))
    return(tfs, Images)



def filter_to_vasc(stack, q = 0.1):
    output = copy.copy(stack)
    thresh = np.quantile(stack, q)
    max_v = np.amax(stack)
    output[np.where(stack>thresh)] = max_v
    plt.figure(f'Vasc filt, q = {q}')
    plt.imshow(np.median(output, axis=0))
    return(output)

def padStacksForAlignment(stacks):
    widths = []
    heights = []
    for stack in stacks:
        heights.append(stack.shape[1])
        widths.append(stack.shape[2])
    padheight = np.amax(heights)
    padwidth = np.amax(widths)
    output = []
    for stack in stacks:
        paddedStack = np.zeros([stack.shape[0], padheight, padwidth], np.float32)
        Hoffset = int((padheight-stack.shape[1])/2)
        Woffset = int((padwidth-stack.shape[2])/2)
        paddedStack[:,Hoffset:Hoffset+stack.shape[1], Woffset:Woffset+stack.shape[2]] = stack
        output.append(paddedStack)
    return(output)


    

# def Stitch_deprecated(stacks,  mode=StackReg.RIGID_BODY):
#     stacks = padStacksForAlignment(stacks)
#     numFrames = 0
#     for stack in stacks:
#         numFrames = numFrames + stack.shape[0]
#     tfs, Images = getTransforms(stacks, mode=mode) ## get local transforms between stack avg and temeplate
#     Margin = 50
#     height = Images[0].shape[0]
#     width = Images[0].shape[1]
#     out_shape = (height + 2*Margin, width + 2*Margin)
#     glob_transform = np.eye(3)
#     glob_transform[:2,2] = -Margin, -Margin
#     glob_im_list = []
#     #for im, trfm, in zip(Images, tfs):
#         #glob_im_list.append(transform.warp(im, trfm.dot(glob_transform), output_shape=out_shape, mode='constant', cval=np.nan))
#     for stack, trfm, in zip(stacks, tfs):
#         for frame in stack:
#             glob_im_list.append(transform.warp(frame, trfm.dot(glob_transform), output_shape=out_shape, mode='constant', cval=np.nan))
    
#     return(np.array(glob_im_list), tfs, glob_transform)

def Stitch(stacks,  mode=StackReg.RIGID_BODY, transform_ROIs = False, ROIstacks = None, traceArrays = None, split_output = False, registration_input = 'rois'):
    stacks = padStacksForAlignment(stacks)
    
    ## Pad ROI stacks:
   
    if transform_ROIs:

        rolledROIs = []  ## put roi # axis at 0 position to make compatible with padding func
        for ROIstack in ROIstacks:
            rolled = np.moveaxis(ROIstack,[0,1,2],[1,2,0])
            rolledROIs.append(rolled)
            #print(f'rolled shape: {rolled.shape}')
        
        paddedROIstacks = padStacksForAlignment(rolledROIs)
        #print
       
    
    # numFrames = 0
    # for stack in stacks:
    #     numFrames = numFrames + stack.shape[0]
        
    print(f'{registration_input=}')
    if registration_input == 'raw':
        tfs, Images = getTransforms(stacks, mode=mode) ## get local transforms between stack avg and temeplate
        
    elif registration_input == 'rois':
        ROIinput = []
        for ROIstack in paddedROIstacks:
            maxim = np.amax(ROIstack, axis=0)
            exim = np.expand_dims(maxim, axis=0)
            repim = exim.repeat(10, axis=0)
            ROIinput.append(repim)
        print(f'{len(ROIinput)=}')
        print(f'{ROIinput[0].shape=}')
        tfs, Images = getTransforms(ROIinput, mode=mode)
    elif registration_input == 'last_3_rois':
        ROIinput = []
        for ROIstack in paddedROIstacks:
            maxim = np.amax(ROIstack[:,:,-3:], axis=0)
            exim = np.expand_dims(maxim, axis=0)
            repim = exim.repeat(10, axis=0)
            ROIinput.append(repim)
        print(f'{len(ROIinput)=}')
        print(f'{ROIinput[0].shape=}')
        tfs, Images = getTransforms(ROIinput, mode=mode)
    elif registration_input == 'vasc':
        ROIinput = []
        for stack in stacks:
            ROIinput.append(filter_to_vasc(stack, q = 0.1))
        tfs, Images = getTransforms(ROIinput, mode=mode)    
        
    Margin = 50
    height = Images[0].shape[0]
    width = Images[0].shape[1]
    out_shape = (height + 2*Margin, width + 2*Margin)
    glob_transform = np.eye(3)
    glob_transform[:2,2] = -Margin, -Margin
    glob_im_list = []
    transformedROIstacks = []
    if split_output:
        transformedStacks = []
    #for im, trfm, in zip(Images, tfs):
        #glob_im_list.append(transform.warp(im, trfm.dot(glob_transform), output_shape=out_shape, mode='constant', cval=np.nan))
    
    if transform_ROIs:
        iterator = zip(stacks, tfs, paddedROIstacks)
        
    else:
        iterator = zip(stacks, tfs, stacks)
        
    for stack, trfm, ROIstack in iterator:
        if transform_ROIs:
            transformedROIstack = np.zeros([ROIstack.shape[0], out_shape[0], out_shape[1]])
            print(f'ROIstack shape: {ROIstack.shape}')
            for count, ROI in enumerate(ROIstack):
                #print(f'ROI shape: {ROI.shape}')
                warpedROI = transform.warp(ROI, trfm.dot(glob_transform), output_shape=out_shape, mode='constant', cval=np.nan)
                transformedROIstack[count,:,:] = warpedROI
            
            plt.figure()
            transformedROIstack = np.moveaxis(transformedROIstack,[0,1,2],[2,0,1])
            transformedROIstack = np.nan_to_num(transformedROIstack)
            print(f'transformed shape: {transformedROIstack.shape}')
            plt.imshow(np.sum(transformedROIstack, axis= 2))
            transformedROIstacks.append(transformedROIstack)
        
        if split_output:
            transformedStack = np.zeros([stack.shape[0], out_shape[0], out_shape[1]], np.float32)
            for count, frame in enumerate(stack):
                transformedStack[count,:,:] = transform.warp(frame, trfm.dot(glob_transform), output_shape=out_shape, mode='constant', cval=np.nan)
            transformedStacks.append(np.nan_to_num(transformedStack))
        else:
            for frame in stack:
                glob_im_list.append(transform.warp(frame, trfm.dot(glob_transform), output_shape=out_shape, mode='constant', cval=np.nan))
    

    if split_output:
        return(transformedStacks, tfs, glob_transform, transformedROIstacks)
    else:
        return(np.nan_to_num(np.array(glob_im_list)), tfs, glob_transform, transformedROIstacks)