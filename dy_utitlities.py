#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:22:41 2022

@author: ch184656
"""

from matplotlib import pyplot as plt
import numpy as np

def colorbar(cmap = None, F = None, A = None, pos = None, bounds = None, fontsize=12):
    if A is None:
        F = plt.figure(cmap, figsize = [2,0.2])
        A = F.add_axes([0,0,1,1])
        
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient,gradient))
    
    A.imshow(gradient, cmap=cmap, aspect = 'auto')
    A.xaxis.set_visible(False)
    A.yaxis.set_visible(False)
    if not bounds is None:
        A.text(0,2,str(bounds[0]), horizontalalignment='left', verticalalignment='top', fontsize=fontsize)
        A.text(255,2,str(bounds[1]), horizontalalignment='right', verticalalignment='top', fontsize=fontsize)
    return(F)
    