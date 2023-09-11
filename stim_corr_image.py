#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:23:38 2023

@author: ch184656
"""
import numpy as np
from matplotlib import pyplot as plt

def corrImSig_original(im,sig,imTime,sigTime):
    # Convert time base of signal for correlation to correspond to image series: (THIS METHOD IS FLAWED! )
    regSig = np.zeros(imTime.shape[0])
    for ii in range(0,imTime.shape[0]):
        sampleIndex = np.searchsorted(sigTime,imTime[ii])
        while sampleIndex >= sig.shape[0]:
            sampleIndex=sampleIndex-1
        regSig[ii] = sig[sampleIndex]
   

    output = np.zeros([1,im.shape[1],im.shape[2]])
    for ii in range(0,im.shape[1]):
        for jj in range(0,im.shape[2]):
            output[0,ii,jj] = np.corrcoef(im[:,ii,jj],regSig)[0,1]
 

    output = np.nan_to_num(output)
    output = np.squeeze(output)
    return(output)  

def corrImSig(im, sig, imTime, sigTime, tolerance = 1):
    regSig = np.zeros(imTime.shape[0], dtype = np.float64)
    for i, t in enumerate(imTime):
        min_distance = abs(t-sigTime) 
        sig_exists = len(np.where(abs(t-sigTime)<=tolerance)[0])
        if sig_exists:
            IX = np.where(abs(t-sigTime)<=tolerance)[0][0]
            value = sig[IX]
            regSig[i] = value
        else:
            regSig[i] = np.nan
    missing = np.isnan(regSig)
    regSig = np.delete(regSig, missing)
    im = np.delete(im, missing, axis = 0)
    
    output = np.zeros([1,im.shape[1],im.shape[2]])
    for ii in range(0,im.shape[1]):
        for jj in range(0,im.shape[2]):
            output[0,ii,jj] = np.corrcoef(im[:,ii,jj],regSig)[0,1]
 

    output = np.nan_to_num(output)
    output = np.squeeze(output)
    return(output) 
        
        