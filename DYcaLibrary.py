#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:32:15 2021

@author: ch184656
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mat

class ROI():
    
    def __init__(self, name = ''):
        self.name = name
        self.parentDB = ''
        self.parentAnimal = ''
        self.parentFOV = ''
        self.parentData = ''
        self.cellType = ''
        self.neuronID = ''
        self.extractionMethod = ''
        self.thermStim = None
        self.mechStim = None
        self.otherStim = None
        self.FOVimage = np.array([])
        self.mask = np.array([])
        self.binMask = np.array([])
        self.rawMeanIntensity = np.array([])
        self.dff          = np.array([])
        self.timeData     = np.array([])
        self.displayColor = np.array([1,1,1])
        
    def save(self, path):
        path.attrs['name'] = self.name
        path.attrs['parentDB'] = self.parentDB
        path.attrs['parentAnimal'] = self.parentAnimal
        path.attrs['parentFOV'] = self.parentFOV
        path.attrs['parentData'] = self.parentData
        path.attrs['cellType'] = self.cellType
        path.attrs['neuronID'] = self.neuronID
        path.attrs['extractionMethod'] = self.extractionMethod
        
        path.create_dataset('mask', data = self.mask)
        path.create_dataset('binMask', data = self.binMask)
        path.create_dataset('rawMeanIntensity', data = self.rawMeanIntensity)
        path.create_dataset('dff', data = self.dff)
        path.create_dataset('timeData', data = self.timeData)
        path.create_dataset('displayColor', data = self.displayColor)
        
        

def loadROI(path):
    R = ROI()
    for attr in path.attrs:
        setattr(R, attr, path.attrs[attr][:])
    for key in path.keys():
        setattr(R, key, path[key][:])
    return(R)
        
        
    