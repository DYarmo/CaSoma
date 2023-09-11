#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:51:36 2022

@author: ch184656
"""

#import thermalRingLib as trg
import numpy as np
#import libAnalysis as la
#import h5py 
from summaryTable import rasterize, dbContainer
from matplotlib import pyplot as plt
import pdb
from libAnalysis import box_off, jitter
#from DYpreProcessingLibraryV2 import jitter

def format_cold_plate_data(obj, animal=None, fov=None, cam_data = None, therm_data = None, precision=5):
    #####
    ##### First ROI as ipsilateral, second ROI is contralateral
    #####
    ##### FLIR data has one ROI with temp
    ##### 
    ##### Get average event(lift/guarding rate) at 35 deg and at 5 deg bins down to 5 deg c
    
    
    
    cam_time  = obj.DB['Animals'][animal][fov]['T'][cam_data][...]
    ipsi_trace = obj.DB['Animals'][animal][fov]['R'][cam_data]['traceArray'][0,:]
    cont_trace = obj.DB['Animals'][animal][fov]['R'][cam_data]['traceArray'][1,:]
    
    therm_trace = obj.DB['Animals'][animal][fov]['R'][therm_data]['traceArray'][:]
    therm_time  = obj.DB['Animals'][animal][fov]['T'][therm_data][...]
    
    time = [therm_time, cam_time, cam_time]
    data = [therm_trace, ipsi_trace, cont_trace]
    labels = ['temp', 'ipsi', 'contra']
    
    raster = rasterize(time, data, labels, plot=False, precision = precision)
    return(raster)
    
def analyze_cold_plate_data(obj, DBpath = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Cold plate /cold plate deposit 11-7-2022.h5'):
    
    F={}
    
    bin_lower_bounds = np.arange(0, 36, 5)
    bin_upper_bounds = np.arange(5,41, 5)
    
    if obj is None:
        obj = dbContainer(DBpath)
        for a in obj.DB['Animals'].keys():
            animal = a
        FOVs = obj.DB['Animals'][animal].keys()
   
    else:
        animal = obj.curAnimal
        
        FOVs=[]
        for item in obj.FOVlist.selectedItems():
            FOVs.append(item.text())
            
   #FOVs = obj.DB['Animals'][animal].keys()
    
    sham_count = 0
    sni_count = 0
    for fov in FOVs:
        if 'sham' in fov:
            sham_count = sham_count+1
        elif 'sni' in fov:
            sni_count = sni_count+1
        else:
            print('what is this?')
            continue
        
    sham_ips_freq = np.zeros([sham_count, bin_lower_bounds.shape[0]])
    sni_ips_freq = np.zeros([sni_count, bin_lower_bounds.shape[0]])
    sham_con_freq = np.zeros([sham_count, bin_lower_bounds.shape[0]])
    sni_con_freq = np.zeros([sni_count, bin_lower_bounds.shape[0]])
    
    sham_count = -1
    sni_count = -1
    
    print(FOVs)
    F['Cold plate trace'] = plt.figure()
    A = F['Cold plate trace'].add_axes([0,0,1,1])
    for cc, fov in enumerate(FOVs):
        if 'sham' in fov:
            sham = True
            sni = False
            sham_count = sham_count+1
        elif 'sni' in fov:
            sham = False
            sni = True
            sni_count = sni_count+1
        else:
            print('what is this?')
            continue
        
        for data in obj.DB['Animals'][animal][fov].keys():
            if 'FLIR' in data:
                therm_data = data
            elif 'camera' in data:
                cam_data = data
        analysis = format_cold_plate_data(obj, animal = animal, fov=fov, cam_data = cam_data, therm_data = therm_data)
        raster = analysis['raster']
        raster[np.where(raster==-2**15)]=np.nan
        drange = [0,14800]
        
        therm = raster[0,:]
        ipsi = raster[1,:]
        cont = raster[2,:]
        
        
        if sham:
            color = ['k', [0.5,0.5,0.5], 'b']
        elif sni:
            color = ['r', 'm', 'c']
          
        
        A.plot(therm[drange[0]:drange[1]], color=color[2])
        A.plot(3*ipsi[drange[0]:drange[1]], color = color[0])
        A.plot(3*cont[drange[0]:drange[1]], color = color[1])
        
        #pdb.set_trace()
        ### get total counts, total poiint at each temp biini, and calc freq per bin
        for c, (l, u) in enumerate(zip(bin_lower_bounds, bin_upper_bounds)):
            
            
            IX = np.where((therm>l) & (therm<u))[0]
            num_samples = len(IX)
            ipsi_count = np.sum(ipsi[IX])
            cont_count = np.sum(cont[IX])
            i_freq = ipsi_count/num_samples
            c_freq = cont_count/num_samples
            if sham:
                sham_con_freq[sham_count, c] = c_freq
                sham_ips_freq[sham_count, c] = i_freq
            if sni:
                sni_con_freq[sni_count, c] = c_freq
                sni_ips_freq[sni_count, c] = i_freq
            
    box_off(A)  
    xticks = np.arange(0, 15001, 6000)
    xticklabels = []
    for tick in xticks:
        xticklabels.append(str(int(tick/600)))
    A.set_xticks(xticks)
    A.set_xticklabels(xticklabels, fontsize=32)
    A.set_yticks([0,15,30])
    A.set_yticklabels(['0','15','30'], fontsize=32)
    
    # F['Bin rasters'] = plt.figure() 
    # B = F['Bin rasters'].add_axes([0,0,1,1])    
    # B.subplot(2,2,1)
    # B.imshow(sham_con_freq)
    # B.subplot(2,2,2)
    # B.imshow(sham_ips_freq)
    # B.subplot(2,2,3)
    # B.imshow(sni_con_freq)
    # B.subplot(2,2,4)
    # B.imshow(sni_ips_freq)
    
    F['Temperature vs guarding, SNI vs sham'] = plt.figure()
    C= F['Temperature vs guarding, SNI vs sham'].add_axes([0,0,1,1])    
    datas = [sham_con_freq, sham_ips_freq, sni_con_freq, sni_ips_freq]
    colors = [[0.5,0.5,0.5], 'k', 'm', 'r']
    for cc, data in enumerate(datas):
        X= np.zeros([data.shape[1]])
        Y = np.zeros(X.shape)
        Yerr = np.zeros(X.shape)
        for c, (l, u) in enumerate(zip(bin_lower_bounds, bin_upper_bounds)):
            X[c] = (l+u)/2
            Y[c] = np.average(data[:, c])
            Yerr[c] = np.std(data[:,c])/np.sqrt(data.shape[0])
        C.errorbar(X,Y, yerr = Yerr, color= colors[cc])
        #plt.errorbar(X,Yerr, color=colors[cc])
    box_off(C)
    xticks = ([10,15,20,25,30,35])
    C.set_xticks(xticks)
    C.set_xticklabels(['10','15','20','25','30','35'], fontsize=32)
   
    ytick = C.get_yticks()*600
    yticklabels = []
    yticks=[]
    for c, tick in enumerate(ytick):
        if c%2: 
            yticklabels.append(str(tick))
            yticks.append(tick/600)
    C.set_yticks(yticks)
    C.set_yticklabels(yticklabels, fontsize=32)
    
       

    return(F)
    