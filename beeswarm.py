#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:08:46 2023

@author: ch184656
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def beeswarm(y, xpos, nbins=None, width=1., A = None, **kwargs):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        # nbins = len(y) // 6
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get upper bounds of bins
    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()

    #Divide indices into bins
    ibs = []#np.nonzero((y>=ybins[0])*(y<=ybins[1]))[0]]
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y>ymin)*(y<=ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx
    x = x+xpos
    if A is None:
        F = plt.figure()
        A = F.add_subplot(1,1,1)
    A.scatter(x,y, **kwargs)
  
    
    return x

def beeswarms(ys, xposs, nbins=None, A=None, width = 0.5, **kwarg):
    for y, xpos in zip(ys, xposs):
        beeswarm(y, xpos, nbins, width = width, A=A)
    

