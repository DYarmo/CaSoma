# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:50:40 2020

@author: USER
"""
import cv2
import math
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from matplotlib import cm
import h5py
from pystackreg import StackReg
from skimage.transform import downscale_local_mean
from tkinter import filedialog
from tkinter import *
import glob
from sklearn.cluster import KMeans
#import scipy
import time
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, \
    QLineEdit, QMainWindow, QSizePolicy, QLabel, QSlider, QMenu, QAction, \
    QComboBox, QListWidget, QGridLayout, QPlainTextEdit, QDateTimeEdit, QTextEdit, QInputDialog, QColorDialog

import DYroiLibrary as DYroi
import json
import natsort

from skimage.data import colorwheel
from skimage.draw import disk

from scipy import signal
import LIR #least inscribed rectangle
import libAnalysis as LA

##Export plotting functions



def exportPlotsOriginal(X,Y,saveName):

    #Slice up data to elide long intervals with no readings:
    F=plt.figure()
    allTimes= np.array([], dtype = float)
    for d in X:
        allTimes =  np.append(allTimes, X[d], 0)
    allTimeUnique = set(allTimes)
    allTimeList   = list(allTimeUnique)
    allTimeSorted = sorted(allTimeList)
    allTime =np.squeeze(np.array([allTimeSorted], dtype = float))
    gaps = np.diff(allTime)
    medGap = np.median(gaps)
    splitGap = 100*medGap
    
    gapLocations = np.argwhere(gaps>splitGap)

    
    
    Tstarts = allTime[gapLocations+1]
    Tstarts = np.insert(Tstarts,0,allTime[0])
    Tends   = allTime[gapLocations]
    Tends   = np.insert(Tends,len(Tends),allTime[-1])
    Times = np.stack((Tstarts,Tends,Tends-Tstarts), axis = 1)

    
    totalTime = Times.sum(axis = 0)[2]  ## Calculate total time being plotted
    
    XaxisWidths = Times[:,2]/totalTime
    YaxisHeight = 1/len(X)
    bottom = 1-YaxisHeight
    for d in X:
        data = X[d]
        left = 0
        for section in range(0,len(XaxisWidths),1):
            A = F.add_axes((left,bottom, XaxisWidths[section], YaxisHeight))
            startTime = Times[section,0]
            endTime = Times[section,1]   
            A.plot(X[d],Y[d],'k')
            plt.xlim(startTime, endTime)
            plt.axis('off')
            left = left + XaxisWidths[section]
        bottom = bottom - YaxisHeight
    plt.savefig(saveName)
    
def watershed(floatMask):
    pass

def sortByKmean(traces, nClusters):
    normTraces = np.zeros(traces.shape)
    if nClusters>traces.shape[0]:
        nClusters = traces.shape[0]
        
    for cell, trace in enumerate(traces):
        trace = trace-np.quantile(trace, 0.1)
        if np.max(trace) == 0:
            normTraces[cell,:] = trace
        else:
            normTraces[cell,:] = trace/np.max(trace)
    kmeans = KMeans(n_clusters = nClusters).fit(normTraces)
    
    IX = np.linspace(0, traces.shape[0]-1, traces.shape[0])
    newIX = np.array([], dtype = np.uint32)
    
    for label in range(nClusters):
        newIX = np.concatenate((newIX, IX[kmeans.labels_ == label]), axis = 0)
        
    return(newIX.astype(np.uint32))


        
        
def exportPlots(X,Y,saveName):
    np.savetxt(saveName, X)
    return
    #Slice up data to elide long intervals with no readings:
    F=plt.figure()
    allTimes= np.array([], dtype = float)
    for d in X:
        allTimes =  np.append(allTimes, X[d], 0)
    allTimeUnique = set(allTimes)
    allTimeList   = list(allTimeUnique)
    allTimeSorted = sorted(allTimeList)
    allTime =np.squeeze(np.array([allTimeSorted], dtype = float))
    gaps = np.diff(allTime)
    medGap = np.median(gaps)
    splitGap = 100*medGap
    
    gapLocations = np.argwhere(gaps>splitGap)
    
    Tstarts = allTime[gapLocations+1]
    Tstarts = np.insert(Tstarts,0,allTime[0])
    Tends   = allTime[gapLocations]
    Tends   = np.insert(Tends,len(Tends),allTime[-1])
    Times = np.stack((Tstarts,Tends,Tends-Tstarts), axis = 1)

    
    totalTime = Times.sum(axis = 0)[2]  ## Calculate total time being plotted
    
    ##Need scale bars (X and Y)
    ##Plot raster
    
    XaxisWidths = Times[:,2]/totalTime
    YaxisHeight = 0.75/len(X)
    bottom = 0.75-YaxisHeight
    for d in X:
        data = X[d]
        left = 0
        for section in range(0,len(XaxisWidths),1):
            A = F.add_axes((left,bottom, XaxisWidths[section], YaxisHeight))
            startTime = Times[section,0]
            endTime = Times[section,1]   
            A.plot(X[d],Y[d],'k')
            plt.xlim(startTime, endTime)
            plt.axis('off')
            left = left + XaxisWidths[section]
        bottom = bottom - YaxisHeight
        
    ## Plot rasters
    YaxisHeight = 0.25/len(X)
    bottom = 1-YaxisHeight
    left = 0
    for d in X:
        data = X[d]
        left = 0
        for section in range(0,len(XaxisWidths),1):
            A = F.add_axes((left,bottom, XaxisWidths[section], YaxisHeight))
            startTime = Times[section,0]
            endTime = Times[section,1]   
            rasterData = np.array(X[d])
            rasterData  = np.expand_dims(rasterData, axis = 0)
            print(rasterData.shape)
            A.imshow(rasterData, cmap = 'hot')
            plt.xlim(startTime, endTime)
            plt.axis('off')
            left = left + XaxisWidths[section]
        bottom = bottom - YaxisHeight
    
        
    plt.savefig(saveName)            
            
def dummy(something):
    print(something)
    
    
    
#These functions are for spatial filtering/background correction
def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def bgCorrectImage(img, LPfilter):
    fft       = np.fft.fft2(img)
    center    = np.fft.fftshift(fft)
    LPcenter  = center * LPfilter
    LP        = np.fft.ifftshift(LPcenter)
    invLP     = np.fft.ifft2(LP)
    blurred   = np.abs(invLP)
    corrected = np.float32(np.divide(img,blurred))
    return(corrected)

def findFilesFromFOV(FOVfolder, fileNameString):
    output=[]
    trialFolders = os.listdir(FOVfolder)
    for trialFolder in trialFolders:
        output.extend(glob.glob(os.path.join(FOVfolder,trialFolder,fileNameString)))
            
    return(output)

def findFilesFromTrial(TrialFolder, fileNameString):
    output = []
    if type(fileNameString) == tuple:
        for string in fileNameString:
            result = glob.glob(os.path.join(TrialFolder,string))
            if result:
                output = glob.glob(os.path.join(TrialFolder,string))
    else:
        output = glob.glob(os.path.join(TrialFolder,fileNameString))
    return(output)

def batchCorrect(FOVfolder):
    CERNAfiles = findFilesFromFov(FOVfolder, '*CERNA.tif*')
    for Cfile in CERNAfiles:
        imdata = imageio.volread(Cfile)
        output = np.zeros(imdata.shape, dtype = np.float32)
        image = imdata[0,:,:]
        LPfilter = gaussianLP(5, image.shape)

        for ii in range(imdata.shape[0]):
            output[ii,:,:] = bgCorrectImage(np.float32(imdata[ii,:,:]), LPfilter)
            print(ii)

        savename = os.path.join(os.path.dirname(Cfile),'CERNAbgcorrected.tif')

        imageio.volwrite(savename, output, bigtiff=True)
        print('Corrected file saved'+ savename)

def FFTcorrectImageStack(IMG, obj):
    #default filter spec should be 5 for 10x objective 2x2 binned data
    output = np.zeros(IMG.shape, dtype = np.float32)
    LPfilter = gaussianLP(5, IMG[0,:,:].shape)
    for ii in range(IMG.shape[0]):
        output[ii,:,:] = bgCorrectImage(np.float32(IMG[ii,:,:]), LPfilter)
        print(f'{ii+1} of {IMG.shape[0]+1} corrected')
    #output = output-np.min(np.min(np.min(output)))
    #output = output*(((2**16)-1)/np.max(np.max(np.max(output))))
    #output = np.uint16(output)
    return(output)

def medFilt2(IMG, obj):
    print('Median filtering...')
    output = np.zeros(IMG.shape, dtype = IMG.dtype)#np.uint16)
    for ii in range(IMG.shape[0]):
        output[ii,:,:] = cv2.medianBlur(IMG[ii,:,:], 3)
        #pctDone = ii/IMG.shape[0]*100
        #print(f'\r{pctDone}', end="")
    return(output)

def simpleThreshold(img, obj):
    levels = obj.dLevels[obj.dataFocus]
    minV = np.min(np.min(np.min(img)))
    maxV = np.max(np.max(np.max(img)))
    img[img<levels[0]] = minV
    img[img>levels[1]] = maxV
    return(img)

    

def prep(IMG, obj, templateIM='X'):
    return(pReg(convertUint16(pFFThighPass(medFilt2(IMG, obj),obj),obj), obj, template=templateIM))

def recolor(stack, obj):
    im = np.median(stack, axis=0)
    im = im-np.min(np.min(np.min(im)))
    monochrome = im/np.max(np.max(np.max(im)))
  
    
    colorDialog = QColorDialog(obj)
    color = colorDialog.getColor()
    hexColor = color.name()
    h = hexColor.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0,2,4))
    r=monochrome*rgb[0]
    g=monochrome*rgb[1]
    b=monochrome*rgb[2]
    
    r=np.expand_dims(r, axis=2)
    b=np.expand_dims(b, axis=2)
    g=np.expand_dims(g, axis=2)
    
    RGB = np.concatenate((r,g,b), axis = 2)
    RGB= np.expand_dims(RGB, 0)
    output = RGB.repeat(stack.shape[0], 0)
    return(output)
    
    
def stackMax(stack, obj):  
    maxIm = np.amax(stack, axis = 0)
    maxIm = np.expand_dims(maxIm,0)
    output = maxIm.repeat(stack.shape[0], 0)
    return(output)

def stackDev(stack, obj):  
    maxIm = np.std(stack, axis = 0)
    maxIm = np.expand_dims(maxIm,0)
    output = maxIm.repeat(stack.shape[0], 0)
    return(output)

def decimateTime(allTimes,Tstep,Tbegin,Tend):
        
        Tnum = np.array(allTimes)
        Thig = Tnum[Tnum>=Tbegin]
        Tlow = Thig[Thig<=Tend]
        Tbig = Tlow/Tstep
        Trou = np.round(Tbig)
        Tcor = Trou*Tstep
        Tset = set(Tcor)
        Tlist = list(Tset)
        Tsort = sorted(Tlist)
        output = np.array(Tsort)
        return(output)

def testCorr():
    db = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Inactive windows/Mouse 66/66.h5'
    DB = h5py.File(db,'a')
    im = DB['Animals']['66']['1']['CERNAraw_medFilt2_343_1243_FFT_HP5_3_900'][:]
    im = im[:,100:500,100:400]
    imageio.volwrite('corInput.tif',im)
    #return
    
    imTime = DB['Animals']['66']['1']['T']['CERNAraw_medFilt2_343_1243_FFT_HP5_3_900']
    
    
    sig = DB['Animals']['66']['1']['R']['FLIRraw_copy_2_600']['Traces']['1']
    sigTime = DB['Animals']['66']['1']['T']['FLIRraw_copy_2_600']
    #plt.plot(imTime)
    #plt.Hold(True)
    #plt.plot(sigTime)
    #return
    #plt.plot(sigTime,sig)
    #return
   
    output = corrImSig(im,sig,imTime,sigTime)
    output = output - np.min(np.min(output))
    output = output/np.max(np.max(output))
    output = output*2**16
    toWrite = output[0,:,:]
    toWrite = np.uint16(np.squeeze(toWrite))
    imageio.imwrite('cortest.tif',toWrite)
    
    #output = output.repeat(im.shape[0], 0)
    return(output)
    

def regColor(stack, obj):
    Ro = np.squeeze(stack[:,:,:,0])
    Go = np.squeeze(stack[:,:,:,1])
    Bo = np.squeeze(stack[:,:,:,2])
    
    Rt = pGetTransform(Ro, None)
    output = np.zeros(stack.shape)
    sr = StackReg(StackReg.RIGID_BODY)
    
    for color in range(3):
        source = np.squeeze(stack[:,:,:,color])
        for c, frame in enumerate(source):
            output[c,:,:,color] = sr.transform(frame, tmat=Rt[c])
            
    return(output)
          
def createPawMap(stack, obj):
    im=imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
    im = cv2.resize(im, (534,534))
    im  = np.fliplr(im)
    im = np.rot90(im)
    im = np.expand_dims(im,0)
    output = im.repeat(stack.shape[0], 0)
    return(output)
    
def generatePawColorCode():
    paws =imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
    paws = cv2.resize(paws, (534,534))
    paws  = np.fliplr(paws)
    paws = np.rot90(paws)
    #plt.imshow(paws)
    overlay  = np.zeros([paws.shape[0], paws.shape[1],4], dtype = np.uint8)
    basewheel  = colorwheel()
    
    alpha = np.zeros([paws.shape[0],paws.shape[1]],  dtype = np.uint8)
    
    xx, yy = disk((int(basewheel.shape[0]/2), int(basewheel.shape[1]/2)), 110, shape = basewheel.shape)
    wheel = np.zeros([220,220,3], dtype = np.uint8)
    x,y = disk((110,110), 110, shape = [220,220])
    wheel[x,y,...] = basewheel[xx,yy,...]
    alpha = np.expand_dims(np.zeros([220,220], dtype = np.uint8), 2)
    alpha[x,y] = 128
    RGBAwheel = np.concatenate((wheel,alpha), axis = 2)
    plt.imshow(RGBAwheel)

    Xscale = 0.95
    Yscale = 0.58
    
    newWheel = cv2.resize(RGBAwheel, (int(Xscale*220), int(Yscale*220)))
    
    pawCorners = [(140,42),  (300,42), (300,253),  (138,255)]
    w = newWheel.shape[0]
    h = newWheel.shape[1]

    for c, corner in enumerate(pawCorners):
        #aa,bb = disk(center, 110, shape = overlay.shape)
        if c%2:
            overlay[corner[0]:corner[0]+w,corner[1]:corner[1]+h,...] = np.flipud(newWheel)
        else:
            overlay[corner[0]:corner[0]+w,corner[1]:corner[1]+h,...] = newWheel
        
   
   
    #alpha = np.expand_dims(alpha, 2)
    #RGBA = np.concatenate((overlay, alpha), axis = 2)
    plt.imshow(paws)
    plt.imshow(overlay)
    plt.show
    return(overlay)

def mapRGB(value, colormap=plt.cm.cool, start=0, stop=255):
    clipped = np.clip(value,start,stop)-start
    clipped = clipped/(stop-start)
    
    return(colormap(clipped))
                 
    
def mechThreshold(obj, cells=None, cellSource=None, stimDataSource = None, plot = False):
    if cells == None:
        cells = obj.selectedROI
    
    if cellSource == None:
        cellSource = obj.dataFocus
        
    if stimDataSource == None:
        stimDataSource = obj.getDataDialog(prompt = "Select mech stim:")
    
    mechTrace, mechTime = getSubStack(obj, datakey=stimDataSource)
    
    
  
    
    rProm = 10
    sProm = 10
    rInt = 10
    sInt = 10
    
    Istart,Iend = getTimeBounds(obj,  datakey = cellSource)
    responseTime = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][cellSource][Istart:Iend]
    mechTrace = np.squeeze(alignTimeScales(responseTime, mechTrace, mechTime))
    
    traceArray = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][cellSource]['traceArray'][:,Istart:Iend]
    
    
    minStim = 10 ## Find stims over 20 mN
    maxStim = np.amax(mechTrace)
    #maxStim = 500
    nStim = mechTrace/maxStim
    
    threshold_mat = np.zeros([cells.shape[0]])
    sPeaks = signal.find_peaks(mechTrace, distance = sInt, prominence = minStim)[0]
    for cCount, cell in enumerate(cells):
        response = traceArray[cell,:]
        
        nResponse = response/np.amax(response)
        
        rPeaks = signal.find_peaks(response, distance = rInt, prominence = np.amax(response)/rProm)[0]
        
        if plot:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(nResponse, color='gray');# plt.scatter(rPeaks, np.ones(rPeaks.shape)*-0.5)
            plt.plot(nStim+1.1, color = 'gray'); #plt.scatter(sPeaks, np.ones(sPeaks.shape)*-1, c='r')
            
        
        tMat = np.zeros([rPeaks.shape[0], 2])
        invalid = []
        for count, rPeak in enumerate(rPeaks):
            tMat[count, 0] = nResponse[rPeak]
            if plot:
                plt.scatter(rPeak,nResponse[rPeak], c='green')
            prevPeaks = sPeaks<=rPeak

            if prevPeaks.sum()>0:
                IX = sPeaks[prevPeaks][-1]
                stimIntensity = mechTrace[IX]
                if plot:
                    plt.scatter(IX,nStim[IX]+1.1, c='cyan')
            else:
                IX = 0
                stimIntensity = np.nan
                invalid.append(count)
                
            tMat[count, 1] = stimIntensity
        if plot:
            plt.subplot(1,2,2)
            plt.scatter(tMat[:,1],tMat[:,0])
            plt.xlim(0, maxStim)
            plt.ylim(0,1)
            plt.show()
        
        stim_response = np.delete(tMat, invalid, axis =0)
        min_stim = np.amin(stim_response[:,1])
        threshold_mat[cCount] = min_stim
        
    
    ## Color code ROI map by mechanical threshold
    ROIs = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][cellSource]['floatMask'][:,:,cells]
    im = obj.DB['Animals'][obj.curAnimal][obj.curFOV][cellSource][0,...]
    
    R = np.zeros([im.shape[0], im.shape[1], 1])
    G = np.zeros([im.shape[0], im.shape[1], 1])
    B = np.zeros([im.shape[0], im.shape[1], 1])
    
    threshMin = 1
    threshMax = 100
    cmap = mapRGB(threshold_mat, start = threshMin, stop = threshMax)
    
    for r in range(ROIs.shape[2]):
        R[ROIs[:,:,r]>0] = int(cmap[r,0]*255)
        G[ROIs[:,:,r]>0] = int(cmap[r,1]*255)
        B[ROIs[:,:,r]>0] = int(cmap[r,2]*255)
      
    
    A = np.max(ROIs, axis=2)
    A = A.astype(bool)
    A = A.astype(np.uint8)*200
    A = np.expand_dims(A, axis = 2)
    RGBA = np.concatenate((R,G,B,A), axis = 2)/255
    
  
    F = plt.figure('Mech threshold coded ' + obj.curAnimal + ' ' +obj.curFOV)
    A = F.add_axes([0, 0, 1, 1])
    dlevels = obj.dLevels[obj.dataFocus]
    A.imshow(im.T, aspect='auto', interpolation='none', cmap='gist_gray', vmin = dlevels[0], vmax = dlevels[1])
    A.xaxis.set_visible(False)
    A.yaxis.set_visible(False)
    A.set_frame_on(False)

    if RGBA != np.array([]):
        R = np.swapaxes(RGBA,0,1)
        A.imshow(R)
    B = F.add_axes([0,0,0.25,0.05])

    plt.imshow(np.expand_dims(np.linspace(threshMin,threshMax,num=100),axis=0), cmap=cm.cool)
    plt.text(0,1, str(threshMin))
    
    plt.text(100,1, str(threshMax))
    B.xaxis.set_visible(False)
    B.yaxis.set_visible(False)
    B.set_frame_on(False)
    plt.show()
    print(threshold_mat)
    plt.figure()
    
        
        
def mechanoHeatMap(obj, cells=None, cellSource=None, stimDataSource=None): ##cell = ROI # of cell, cellSource = key of calcium imaging data
    #Plot stim/ response relationship for a cell
    #
    if cells == None:
        cells = obj.selectedROI
    
    if cellSource == None:
        cellSource = obj.dataFocus
        
    if stimDataSource == None:
        stimDataSource = obj.getDataDialog(prompt = "Select paw map data:")

    stimTraces = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][stimDataSource]['traceArray'][...]
    stimMasks = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][stimDataSource]['floatMask']
    ## get cell and stim traces on ssame timebase:
    calciumTime = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][cellSource]
    stimTime = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][stimDataSource]
    astims = alignTimeScales(calciumTime, stimTraces, stimTime)
    
    
    plt.figure()
    plt.imshow(astims, aspect = 100)
    plt.show()
    
    ## eliminate duplicates :
    unique_aligned_stim, indices = np.unique(astims, axis = 0, return_index = True)
    
    maxStim = np.amax(unique_aligned_stim)
    maxSStim = 500 ## make 500mM maximum stim intensity
                      
    print(f'Indices: {indices}')
    indices.sort()
    print(f'Sorted indices: {indices}')
    unique_masks = stimMasks[:,:,indices] # select masks corresponding to unique stims
    plt.figure()
    plt.imshow(np.amax(unique_masks, axis = -1))
    plt.show()
    
    
    
    ##sort stims by start times
    start_times = np.zeros(indices.shape)
    for c, stim in enumerate(unique_aligned_stim): #loop through to collect start times
        print(f'Finding start time of stim #{c}')
        if np.size(np.where(stim>0)[0]) > 0:
            start_times[c] = np.where(stim>0)[0][0] #index of first non-zero item
        else:
            print(f'No non-zero values found in stim # {c}')
        
    ## plot to confirm stimuli are processed well enough
    ix = np.argsort(start_times)
    sorted_unique_aligned_stim = unique_aligned_stim[ix]
    sorted_unique_masks = unique_masks[:,:,ix]
    
    
    ## create stim intensity map
  #  stimMap = np.zeros(sorted_unique_masks.shape)
  #  for ii in range(sorted_unique_masks.shape[-1]):
  #      omask = sorted_unique_masks[:,:,ii]
  #      nmask = np.zeros(omask.shape)
  #      location = np.where(omask>0)
  #      intensity = np.amax(unique_aligned_stim[ii]) #peak of stimulus
  #      nmask[location] = intensity
  #      stimMap[:,:,ii] = nmask
    
    stimDiam =  10
    pawMap = createPawMap(np.zeros([1,1,1]), None)[0,...]
    fig, ax = plt.subplots()
    plt.imshow(pawMap)
    stimCenters = []
    # show all stim:
    for ii in range(sorted_unique_masks.shape[-1]):
        omask = sorted_unique_masks[:,:,ii].astype(np.uint8)
        M = cv2.moments(omask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])  ## getting centroids of mask for stim position
        stimCenters.append((cx,cy))
        
        #ax.add_patch(stimpatch)
        stim = sorted_unique_aligned_stim[ii,...]
        normStim = 255*np.amax(stim)/maxStim
        
        
        patch_color = plt.cm.cool(normStim, alpha = 1)
        map_patch = mat.patches.Ellipse((cx,cy), stimDiam,stimDiam, color = patch_color)
        ax.add_patch(map_patch)
    
    
    
    
    
       
        
    for cCount, cell in enumerate(cells):
        cellTrace = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][cellSource]['traceArray'][cell,:]
        
 
        maxstim = np.amax(sorted_unique_aligned_stim)
        maxResponse = np.amax(cellTrace)
        pawMap = np.squeeze(createPawMap(np.zeros([1,1,1]), None))
        mechMap = np.zeros(pawMap.shape)
    
        stimMags = np.zeros(sorted_unique_aligned_stim.shape[0])
        responseMags = np.zeros(sorted_unique_aligned_stim.shape[0])


        fig, ax = plt.subplots()
        plt.imshow(pawMap)
        
        maxResponse = np.amax(cellTrace)
        for c, stim in enumerate(sorted_unique_aligned_stim):
            response = np.zeros(stim.shape)
            #print(f'stim shape: {stim.shape}')
            stimstart = np.where(stim>0)[0][0]
            stimendOriginal = np.where(stim>0)[0][-1]  ## actual end of stim
            if c < sorted_unique_aligned_stim.shape[0]-1:
                stimend = np.where(sorted_unique_aligned_stim[c+1]>0)[0][0]-1 # extend stim period to start of next stim
            else:
                stimend = sorted_unique_aligned_stim.shape[1]-1           
            response[stimstart:stimend] = cellTrace[stimstart:stimend]
            if response.size == 0:
                print('zero size response')
                continue
            stimMags[c] = np.amax(stim)
            normStim = 255*np.amax(stim)/maxStim
            responseMags[c] = np.amax(response)
            normResponse = np.amax(response)/maxResponse
            patch_color = plt.cm.cool(255-normStim, alpha = normResponse)
            map_patch = mat.patches.Ellipse(stimCenters[c], stimDiam,stimDiam, color = patch_color)
            ax.add_patch(map_patch)
        
        
            #plt.figure(
            #plt.plot(stim, 'r')
            #plt.plot(cellTrace*1000, 'k')
            #plt.plot(response*1000, 'b')

        #plt.show()
        #plt.figure()
        #plt.scatter(stimMags,responseMags)
        
        #show ROIs color coded - alpha for responsse, Hue for stimulus itensity
        

    
    
    
def alignTimeScales(responseTime, stimY, stimTime):     # Aligns stimulus timescale to calcium response time scale 
    # Convert time base of signal for correlation to correspond to image series:
    if len(stimY.shape)<2:
        stimY = np.expand_dims(stimY, 0)
        
    regStim = np.zeros([stimY.shape[0], responseTime.shape[0]])
    for ii in range(0,responseTime.shape[0]):
        sampleIndex = np.searchsorted(stimTime,responseTime[ii])
        while sampleIndex >= stimY.shape[1]:
            sampleIndex=sampleIndex-1
        regStim[:,ii] = stimY[:,sampleIndex]

    return(regStim)

def decimateTimeScales(Ydata, Xdata, timeStep = 0.1):  #Data is formatted as dictionaries with Y for intensity  values and X for time values
    
        for key in Ydata.keys():
            if len(Ydata[key].shape) ==1:
                trace = Ydata[key][...]   
                Ydata[key] = np.expand_dims(trace, axis = 0)
                
                
        ## get list of all time points and create new regular time base spanning experiment:   
        allTimes = []
        for Time in Xdata:  
            allTimes = list(allTimes) + list(Xdata[Time][:])
        
        allTimes.sort()
        
        start = allTimes[0] - 1
        end = allTimes[-1] + 1
        
        length = end-start
        print(f'Length: {length}')
        timeBase = np.linspace(start, end, int(length/timeStep), endpoint=False)
        
        
        rowCounter = 0
        for count, datastream in enumerate(Ydata):
            
            
            raster = Ydata[datastream]
            replast = raster[:,-1]
            replast = np.expand_dims(replast, axis = 1)
            
            raster = np.append(raster, replast, axis = 1)
            dataArray = np.zeros([raster.shape[0], timeBase.shape[0]], dtype = np.float64)
            exTime = Xdata[datastream]
            IX = np.searchsorted(exTime, timeBase)
            
       
            exTimeExtended = np.append(exTime, exTime[-1])
            originalTimes = exTimeExtended[IX]
            #adjusted = timeBase-timeBase[0]
            error = np.absolute(timeBase-originalTimes)
            

            dataArray[:,:] = raster[:,IX]  #spacing preserved
            #realTimeArray = dataArray
            
            nullValue = -2**15
            #nullValue = np.nan
            dataArray[:,error>0.5] = nullValue    #set to 0 any reading that is too far from time point
           
            
            if count == 0:
                output = dataArray
            else:
                output = np.concatenate((output, dataArray), axis = 0)
            rowCounter = output.shape[0]
                


        
        maxArray = np.max(output, axis = 0)
        compressedArray = np.delete(output, maxArray == nullValue, axis = 1)  
        return(output)
    
    
def makeGrayscale(stack, obj):
    return(np.mean(stack, axis = 3))

def dynamicOverlay(stack, obj):
    pass

def corMap(stack, obj):
        dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
        selectedData, okPressed = QInputDialog.getItem(obj,"Select signal source:", "Data:", dataStreams, 0, False)
        if okPressed != True:
            return(stack)
        print(f'Data selected: {selectedData}')
        if len(obj.DB['Animals'][obj.curAnimal][obj.curFOV][selectedData].shape)==1: ##If 1D data selected, use directly for correlation
            sig = obj.DB['Animals'][obj.curAnimal][obj.curFOV][selectedData][...]
        else:          ## Otherwise select ROI with signal to correlate against
            traceSelector=[]
            traces = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][selectedData]['traceArray'][...]
            for t, void in enumerate(traces):
                traceSelector.append(str(t))
        #traceSelector = np.linspace(0,traces.shape[0],traces.shape[0], endpoint=False, dtype = np.uint16)
            selectedTraceNum, okPressed = QInputDialog.getItem(obj,"Select trace #:", "Trace:", traceSelector, 0, False)
            if okPressed != True:
                return(stack)
        
            s = int(selectedTraceNum)
            sig = traces[s,:]

        sigTime = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][selectedData][:]
        
        
        timescale =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus]
        Istart = np.searchsorted(timescale,obj.timeLUT[0])
        Iend = np.searchsorted(timescale,obj.timeLUT[-1])
    
        imTime = timescale[Istart:Iend]
        
        cMap = corrImSig(stack,sig,imTime,sigTime)
        return(cMap)
        
def segmentMechTrace(forceData, thresh=None, interval=None, prom=None, base=None):
    if thresh == None:
        thresh = 20
    if interval == None:
        interval = 10
    if prom == None:
        prom = 10
    if base == None:
        base = 1
    # find stimuli peaks:
    peaks = signal.find_peaks(forceData, distance = interval, prominence = np.amax(forceData)/prom)[0]
    plt.plot(forceData, color='gray'); plt.scatter(peaks, np.ones(peaks.shape)*-1)
    stimStarts = np.zeros(peaks.shape)
    stimStops = np.zeros(peaks.shape)
        #find starting point of each stimulus:
    for c, peak in enumerate(peaks):
        if c==0:
            start = 0
        else:
            start = peaks[c-1]
        if c == peaks.shape[0]-1:
            stop = forceData.shape[0]-1
        else:
            stop = peaks[c+1]
        preStim = forceData[start:peak-1]
        postStim = forceData[peak:stop]
     #   plt.plot(np.linspace(start,peak,(peak-start)-1),preStim)
     #   plt.plot(np.linspace(peak,stop,(stop-peak)),postStim)
        
        
        if np.amin(preStim)<base:
            stimStarts[c] = start+np.where(preStim<base)[0][-1]+1 ## If force value falls below baseline during interval,  stim starts 1 point after last reading below baseline
        else:
            stimStarts[c] = start+np.where(preStim==np.amin(preStim))[0][-1]+1 ## Otherwise start at inter-stim local min value
        if np.amin(postStim)<base:
            stimStops[c] = peak+np.where(postStim<base)[0][0] ## If force value falls below baseline during interval,  stim stops at first point below baseline
        else:
            stimStops[c] = peak+np.where(postStim==np.amin(postStim))[0][-1] ## Otherwise stim stops inter-stim local min value
    
    stimStarts = stimStarts.astype(np.uint16)
    stimStops = stimStops.astype(np.uint16)
    
    #return(stimStarts, stimStops, peaks)
    output = np.zeros([peaks.shape[0], forceData.shape[0]])   
    
    for c, (begin, end) in enumerate(zip(stimStarts,stimStops)):
      
        output[c,begin:end] = forceData[begin:end]
        plt.plot(output[c])
            
    #plt.imshow(output)
    return(output)
        



      
def stimResponseGradient(stim, response):
    pass       
        
def corrImSig(im,sig,imTime,sigTime):
    # Convert time base of signal for correlation to correspond to image series:
    regSig = np.zeros(imTime.shape[0])
    for ii in range(0,imTime.shape[0]):
        sampleIndex = np.searchsorted(sigTime,imTime[ii])
        while sampleIndex >= sig.shape[0]:
            sampleIndex=sampleIndex-1
        print(sampleIndex)
        regSig[ii] = sig[sampleIndex]
    plt.plot(sigTime)
    #plt.plot(imTime)

   
    print(regSig.shape)
    
    
    output = np.zeros([1,im.shape[1],im.shape[2]])
    for ii in range(0,im.shape[1]):
        for jj in range(0,im.shape[2]):
            output[0,ii,jj] = np.corrcoef(im[:,ii,jj],regSig)[0,1]
        print(ii)
    #output = output - np.min(np.min(output))
    #output = output/np.max(np.max(output))
    output = np.nan_to_num(output)
    #output = np.uint16(output*2**16)
    output = output.repeat(im.shape[0], 0)
    return(output)  
    


def copyData(data, obj):
    return(data)

def convertTempToC(stack, obj):
    return(stack/100)

def dff(stack, obj):
    
    Fnought  =  np.median(stack, axis = 0)
    DeltaF   =  np.float32(stack) - np.float32(Fnought)
    return (np.float32(DeltaF/Fnought))



#Utility functions for selecting files and folders:#ef selectFolder():
  #  root=Tk()
   # initDirectory =os.path.normpath('Y:/CERNA')
    #selectedFolder = filedialog.askdirectory(initialdir=initDirectory)
  #  root.destroy()
   # return(os.path.normpath(selectedFolder))

def selectFolder(msg = "Choose directory..."):
        result = QFileDialog.getExistingDirectory(None, msg, "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        result = os.path.normpath(result)
        print(result)
        return(result)
    
def selectFile():
        result = QFileDialog.getOpenFileName(None, "Choose file...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        return(os.path.normpath(result[0]))

#def selectFile():
#    root=Tk()
#    initDirectory =os.path.normpath('Y:/CERNA')
#    selectedFile = filedialog.askopenfilename(initialdir=initDirectory)
#    root.destroy()
#    return(selectedFile)

#Functions for registration:
def regStackFirstFrame(stack):
    sr=StackReg(StackReg.RIGID_BODY)
    registeredStack = sr.register_transform_stack(stack, reference='first')
    return(registeredStack)

def RBregisterStack(stack, obj):
    sr=StackReg(StackReg.RIGID_BODY)
    refImage = np.median(stack, axis = 0)
    output = np.zeros(stack.shape)
    for ii in range(stack.shape[0]):
        output[ii,:,:] = sr.register_transform(refImage,stack[ii,:,:])
        print(str(ii) + ' of ' + str(stack.shape[0]) + ' frames registered')
    #output = np.uint16(output)
    return(output)

#def cropStack(stack,obj):
#    obj.viewsList[obj.dataFocus]
#    if hasattr(obj, 'newROI'):
#        obj.viewsList[obj.dataFocus].removeItem(obj.newROI)
#        obj.newROI = pg.CircleROI([50,50], [20,20], removable=True)
        #self.newROI.sigRegionChanged.connect(self.spotlightROI)
#        self.newROI.sigRemoveRequested.connect(self.addROImanually)
 #       self.viewsList[self.dataFocus].addItem(self.newROI)
 #   return(output)



def removeSelected(stack, obj): #returns selected data to be removed, also creates a trimmed version of data without selected
    fullStack  = obj.DB['Animals'][obj.curAnimal][obj.curFOV][obj.dataFocus][...]
    timescale =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus]
    Istart = np.searchsorted(timescale,obj.timeLUT[0])
    Iend = np.searchsorted(timescale,obj.timeLUT[-1])
    mask = np.linspace(Istart,Iend, (Iend-Istart)+1, dtype = int)
    newStack = np.delete(fullStack,mask, axis = 0)
    newTime  = np.delete(timescale,mask, axis = 0)
    
    newDataKey = obj.dataFocus + '_trimmed' + '_' + str(Istart) + '_' + str(Iend)
    
    
    newShape = list(newStack.shape)
    if len(newShape) > 1:
        newShape[0] = 1
    chunkShape = tuple(newShape)
        
    obj.DB['Animals'][obj.curAnimal][obj.curFOV].create_dataset(newDataKey, data = newStack, track_order = True, chunks=chunkShape) 
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'].create_dataset(newDataKey,  data = newTime, dtype = newTime.dtype, shape = newTime.shape, maxshape = (None,) )
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'].require_group(newDataKey)
    
    
    return(stack)
    

def RBregisterStackfor66 (stack):
    sr=StackReg(StackReg.RIGID_BODY)
    avgOne = np.median(stack[0:1799,:,:], axis = 0)
    avgTwo = np.median(stack[1800:,:,:], axis = 0)
    
    tra_one = sr.register(avgTwo,avgOne)
    for ii in range(1799):
        stack[ii,:,:] = sr.transform(stack[ii,:,:],tra_one)
    print('Finished pre-registration')
    
    
    refImage = np.median(stack, axis = 0)
    output = np.zeros(stack.shape)
    for ii in range(stack.shape[0]):
        tra_two = sr.register(refImage[50:-50,50:-50],stack[ii,50:-50,50:-50])
        output[ii,:,:] = sr.transform(stack[ii,:,:],tra_two)
        print(str(ii) + ' of ' + str(stack.shape[0]) + ' frames registered')
    #output = np.uint16(output)
    return(output)
    
def batchRegister(FOVfolder):
    CERNAfiles = makeFileList(FOVfolder, '*CERNAbgcorrected.tif*')
    firstStack = imageio.volread(CERNAfiles[1])
    first10 = firstStack[0:9,:,:]
    first10reg = regStackFirstFrame(first10)
    refImage = np.mean(first10reg, axis=0)
    
    sr=StackReg(StackReg.RIGID_BODY)
    for Cfile in CERNAfiles:
        imdata = imageio.volread(Cfile)
        output = np.zeros(imdata.shape, dtype = np.float32)
        for ii in range(imdata.shape[0]):
            output[ii,:,:] = sr.register_transform(refImage,imdata[ii,:,:])
            print(ii)
            
        savename = os.path.join(os.path.dirname(Cfile),'CERNAbgcorrected_registered.tif')

        imageio.volwrite(savename, output, bigtiff=True)
        print('Corrected file saved'+ savename)    

def depositMiniscopeSession(obj, sessionFolder=None):
    A = obj.curAnimal
    
    if sessionFolder is None:
        sessionFolder = selectFolder()
    print(sessionFolder)
    sessionName = os.path.basename(sessionFolder)
    obj.DB['Animals'][A].require_group(sessionName)
    obj.curFov = sessionName
    F = sessionName
    
    trialFolders = []
    items = os.listdir(sessionFolder)
    camString  = 'My_WebCam'
    miniString = 'UCLA_miniscope'
    miniData = np.array([])
    miniTime = np.array([])
    camData = np.array([])
    camTime = np.array([])
    
    for item in items:
        if os.path.isdir(os.path.join(sessionFolder,item)):
            trialFolders.append(os.path.join(sessionFolder, item))
    if trialFolders == []:
        print('No trials found')
        return
    for trial in trialFolders:
        ##Find start time:
        mDataFile = os.path.join(trial,'metaData.json')
        f = open(mDataFile)
        j = json.load(f)
        startTime = j['recordingStartTime']['msecSinceEpoch']/1000
        print(f'Start time: {startTime}')
        
        for datadir in os.listdir(trial):
            if datadir == camString:
                datalist = natsort.natsorted(os.listdir(os.path.join(trial,datadir)))
                
                print(datalist)
                for datafile in datalist:
                    #datafile = os.path.join(trial,datadir,datafile)
                    if datafile.endswith('.avi'):
                        print(f'Adding movie {datafile}')
                        camdata = np.array(imageio.mimread(os.path.join(trial, datadir, datafile), memtest = False))
                        print(f'Cam data shape: {camdata.shape}')
                        camData = joinMini(camData, camdata)
                    elif datafile.endswith('timeStamps.csv'):
                        camtime = np.genfromtxt(os.path.join(trial, datadir, datafile), skip_header = 1, dtype = None, delimiter = ',')[:,1]/1000
                        #print(camtime)
                        #print(camtime.shape)
                        camTime = np.concatenate((camTime, camtime), axis = 0)+startTime
            elif datadir == miniString:
                datalist = natsort.natsorted(os.listdir(os.path.join(trial,datadir)))
                
                for datafile in datalist:
                    #datafile = os.path.join(trial,datadir,datafile)
                    if datafile.endswith('avi'):
                        print(f'Adding movie {datafile} from {trial} {datadir}')
                       
                        minidata = np.array(imageio.mimread(os.path.join(trial, datadir, datafile), memtest = False))
                        print(f'Miniscope data {minidata.shape}')
                        miniData = joinMini(miniData, minidata)
                    elif datafile.endswith('timeStamps.csv'):
                        minitime = np.genfromtxt(os.path.join(trial, datadir, datafile), skip_header = 1, dtype = None, delimiter = ',')[:,1]/1000
                        #print(minitime.shape)
                        #print(minitime)
                        miniTime = np.concatenate((miniTime, minitime), axis = 0)+startTime
    
    print(f'Cam data shape: {camData.shape}')
    print(f'Miniscope data {miniData.shape}')
    #camData  =  np.transpose(camData, [1,2,3,0])
    #miniData = np.transpose(miniData, [1,2,3,0])
    miniData = np.squeeze(miniData[:,:,:,0])
    
    obj.DB['Animals'][A][F].require_group('T')   ### Time data
    obj.DB['Animals'][A][F].require_group('R')  
            
    obj.DB['Animals'][A][F].require_dataset('Miniscope', shape =  miniData.shape,maxshape = (None, miniData.shape[1], miniData.shape[2] ),data = miniData, dtype = miniData.dtype, track_order = True, chunks=(1,miniData.shape[1],miniData.shape[2]))
    obj.DB['Animals'][A][F].require_dataset('BehaviorCam', shape =  camData.shape,maxshape = (None, camData.shape[1], camData.shape[2], camData.shape[3]),data = camData, dtype = camData.dtype, track_order = True, chunks=(1,camData.shape[1],camData.shape[2], camData.shape[3]))
        
    obj.DB['Animals'][A][F]['T'].require_dataset('Miniscope', maxshape = (None,), shape = miniTime.shape, data = miniTime, dtype =miniTime.dtype, track_order = True)
    obj.DB['Animals'][A][F]['T'].require_dataset('BehaviorCam', maxshape = (None,), shape = camTime.shape, data = camTime, dtype = miniTime.dtype, track_order = True)
    
    
    obj.DB['Animals'][A][F]['R'].require_group('Miniscope')
    obj.DB['Animals'][A][F]['R']['Miniscope'].require_group('Masks')
    obj.DB['Animals'][A][F]['R']['Miniscope'].require_group('Traces')
    
    obj.DB['Animals'][A][F]['R'].require_group('BehaviorCam')
    obj.DB['Animals'][A][F]['R']['BehaviorCam'].require_group('Masks')
    obj.DB['Animals'][A][F]['R']['BehaviorCam'].require_group('Traces')
        
    obj.updateFOVlist()
    print('Done')



def depositTrial2(obj, trialFolder, data_filter=None):  ## TODO
    A=obj.curAnimal
    F=obj.curFOV
    
    ## Input dictionary defines what file names to look for and
    ## function to read and process for deposit
    
    inputDict = {}
    
    if trialFolder == None:
        trialFolder = selectFolder(msg = 'Select trial to add from...')
    
    if data_filter != None:
        inputDict[data_filter] = obj.inputDict[data_filter]
    else:
        inputDict = obj.inputDict
   
    
   
    trialStartTime = float(os.path.basename(os.path.normpath(trialFolder)))
    
    for input_type in inputDict:
        dataPath  = findFilesFromTrial(trialFolder, inputDict[input_type]['data_string'])
        timePath =  findFilesFromTrial(trialFolder, inputDict[input_type]['time_string'])
        
        dName = inputDict[input_type]['data_name']
        
        if dataPath and timePath:
            dataPath = dataPath[0]
            timePath = timePath[0]
            print(f'Data path: {dataPath}')
            DATA, timedata, error = inputDict[input_type]['read_method'](dataPath, timePath)
            TIME = timedata + trialStartTime            
        else:
            continue
        if error:
            continue
        
        if obj.DB['Animals'][A][F].__contains__(dName): #Check if data stream already exists for FOV
            print('Appending to dataset...')
            #resize time scale 
            obj.DB['Animals'][A][F]['T'][dName].resize((obj.DB['Animals'][A][F]['T'][dName].shape[0]+TIME.shape[0]), axis = 0)
            obj.DB['Animals'][A][F]['T'][dName][-TIME.shape[0]:]=TIME
            #Resize dataset and add new data:
            obj.DB['Animals'][A][F][dName].resize((obj.DB['Animals'][A][F][dName].shape[0]+DATA.shape[0]),axis = 0)
            obj.DB['Animals'][A][F][dName][-DATA.shape[0]:]=DATA
        else: ## If data stream doesn't exist, create new:
            maxShape = (None,) + DATA.shape[1:]
            chunkShape = (1,) + DATA.shape[1:]
            obj.DB['Animals'][A][F].create_dataset(dName, shape =  DATA.shape, maxshape = maxShape, data = DATA, track_order = True, chunks=chunkShape)
            obj.DB['Animals'][A][F]['T'].create_dataset(dName, maxshape = (None,), shape = TIME.shape, data = TIME, track_order = True)
            #Create ROI mask and trace groups:
            obj.DB['Animals'][A][F]['R'].create_group(dName)
            obj.DB['Animals'][A][F]['R'][dName].create_group('Masks')
            obj.DB['Animals'][A][F]['R'][dName].create_group('Traces')
        print(f'Data {dName} deposited to FOV {F}')
        if inputDict[input_type]['transform'] == None:
            pass
        else:
            transform = inputDict[input_type]['transform']
            print(f'Transforming with {transform}')
            DATA, TIME, dName = transform(DATA, TIME)
            genericDepositTrial(obj, DATA, TIME, dName)
            
            
    print('Deposit finished')

def genericDepositTrial(obj, DATA, TIME, dName):

    A=obj.curAnimal 
    F=obj.curFOV
    if obj.DB['Animals'][A][F].__contains__(dName): #Check if data stream already exists for FOV
            print('Appending to dataset...')
            #resize time scale 
            obj.DB['Animals'][A][F]['T'][dName].resize((obj.DB['Animals'][A][F]['T'][dName].shape[0]+TIME.shape[0]), axis = 0)
            obj.DB['Animals'][A][F]['T'][dName][-TIME.shape[0]:]=TIME
            #Resize dataset and add new data:
            obj.DB['Animals'][A][F][dName].resize((obj.DB['Animals'][A][F][dName].shape[0]+DATA.shape[0]),axis = 0)
            obj.DB['Animals'][A][F][dName][-DATA.shape[0]:]=DATA
    else: ## If data stream doesn't exist, create new:
            maxShape = (None,) + DATA.shape[1:]
            chunkShape = (1,) + DATA.shape[1:]
            obj.DB['Animals'][A][F].create_dataset(dName, shape =  DATA.shape, maxshape = maxShape, data = DATA, track_order = True, chunks=chunkShape)
            obj.DB['Animals'][A][F]['T'].create_dataset(dName, maxshape = (None,), shape = TIME.shape, data = TIME, track_order = True)
            #Create ROI mask and trace groups:
            obj.DB['Animals'][A][F]['R'].create_group(dName)
            obj.DB['Animals'][A][F]['R'][dName].create_group('Masks')
            obj.DB['Animals'][A][F]['R'][dName].create_group('Traces')
    print(f'Data {dName} deposited to FOV {F}')
    
def makeTempWorkingDB(obj):
    pass
    
    
def returnTempDBToParent(obj):
    pass
  

def readCERNA(datapath, timepath):
    timedata = np.genfromtxt(timepath)
    data = imageio.volread(datapath)
    data = np.transpose(data, (0,2,1))
    #data = data[::2,:,:]   ## Just for 7924
    #timedata = timedata[::2] ##Just for 7924
    return(data, timedata, False)

def transformCERNA(data, time):
    data = medFilt2(data, None)
    data = pFFThighPass(data, None, poolNum = 23)
    if data.shape[1] > 500:
        data = downSampleSpatial(data, None)
    return(data, time, 'CERNAfiltered')

def readNIR(datapath, timepath):
    timedata = np.genfromtxt(timepath)
    data = np.array(imageio.mimread(datapath, memtest = False))[0:-1,...]
    return(data, timedata, False)
    


def readFLIR(datapath, timepath):
    data = imageio.volread(datapath)
    data = np.transpose(data, (0,2,1))
    timedata = np.genfromtxt(timepath)
    if timedata.shape[0] != data.shape[0]: ## Sometimes FLIR movies are missing frames for first n timestamps, this deletes extra timestamps from beginning of movie
                    print('Mismatch between time and data files!')
                    print(f'Offending file: {timepath[0]}')
                    tdif = timedata.shape[0] - data.shape[0]
                    if tdif >0:
                        timedata = np.delete(timedata, range(tdif))
                        print('Removed extra time stamps from FLIR movie!')
    data = data/100
    return(data, timedata, False)

def readVF(datapath, timepath):
    data = np.genfromtxt(datapath)*9.80665
    timedata = np.genfromtxt(timepath)
    return(data, timedata, False)

def readAurora(datapath, timepath):
    data = np.genfromtxt(datapath)
    timedata = np.genfromtxt(timepath)
    return(data, timedata, False)
    
def depositTrial(obj, trialFolder, chunkOn = True, processCERNA = False):
        A=obj.curAnimal
        F=obj.curFOV
        #trialFolderList = selectFolder()
        print('Depositing...')
        #trialFolder = os.path.normpath(trialFolder)
        print(trialFolder)
        trialStartTime = float(os.path.basename(os.path.normpath(trialFolder)))
        
        
        ##Data file name, timestamp file name, key for saving data, key for saving time, number of dimensions, Transpose image Data
        DataStreamTags   = ([('CERNA.tif','CERNAtime.txt','CERNAraw','CERNAraw',3, True),
                               ('FLIRmovie.tif','FLIRtime.txt','FLIRraw','FLIRraw',3, True),
                               ('NIRcamMovie.tif','NIRcamTime.txt','NIRraw','NIRraw',3, False),   
                               ('VFdata.txt', 'VFtime.txt', 'VFraw','VFraw',1, False),
                               ('NIRcamMovie.avi','NIRcamTime.txt','NIRavi','NIRavi',4,False),
                               ('AuroraData.txt', 'AuroraTime.txt', 'AuroraForce', 'AuroraForce', 1, False),
                               ('XstageData.txt', 'AuroraTime.txt', 'XstageData', 'XstageData', 1, False),
                               ('YstageData.txt', 'AuroraTime.txt', 'YstageData', 'YstageData', 1, False)
                               ])
        
        for stream in DataStreamTags: 
            print()
            dataPath  = findFilesFromTrial(trialFolder, stream[0])
            timePath =  findFilesFromTrial(trialFolder,stream[1])
        
            if dataPath and timePath:
                print('Depositing'+stream[2])
                if stream[4] == 3:
                    print(dataPath)
                    DATA   = imageio.volread(dataPath[0])
                    if stream[0] == 'NIRcamMovie.tif':
                        #DATA = DATA[0:-1,:,:]
                        DATA = DATA[0:-1,...] # 
                        print('Trimming 1 frame from NIR data')
                    if stream[5]:
                        DATA = np.transpose(DATA, (0,2,1))
                elif stream[4] == 1:
                    DATA = np.genfromtxt(dataPath[0])
                elif stream[4] == 4:
                    DATA = np.array(imageio.mimread(dataPath[0], memtest = False))
                    if stream[0] == 'NIRcamMovie.avi':
                        #DATA = DATA[0:-1,:,:]
                        DATA = DATA[0:-1,...] # 
                        print('Trimming 1 frame from NIR data')
                TIME  = np.genfromtxt(timePath[0])+trialStartTime
                if TIME.shape[0] != DATA.shape[0]:
                    print(TIME.shape[0])
                    print(DATA.shape[0])
                    print('Mismatch between time and data files!')
                    print(f'Offending file: {timePath[0]}')
                    tdif = TIME.shape[0] - DATA.shape[0]
                    if stream[0] == 'FLIRmovie.tif' and tdif >0:
                        TIME = np.delete(TIME, range(tdif))
                        print('Removed extra time stamps from FLIR movie!')
                    else:
                        continue
             
                    
                    
                    
                if obj.DB['Animals'][A][F].__contains__(stream[2]): #Check if data stream already exists for FOV
                    print('Appending to dataset...')
                    #Detach and resize time scale (no longer attaching, now just resizing)
                    #obj.DB['Animals'][A][F][stream[2]].dims[0].detach_scale(obj.DB['Animals'][A][F]['T'][stream[3]])
                    obj.DB['Animals'][A][F]['T'][stream[3]].resize((obj.DB['Animals'][A][F]['T'][stream[3]].shape[0]+TIME.shape[0]), axis = 0)
                    obj.DB['Animals'][A][F]['T'][stream[3]][-TIME.shape[0]:]=TIME
                    
                    #Resize dataset and add new data:
                    obj.DB['Animals'][A][F][stream[2]].resize((obj.DB['Animals'][A][F][stream[2]].shape[0]+DATA.shape[0]),axis = 0)
                    
                    obj.DB['Animals'][A][F][stream[2]][-DATA.shape[0]:]=DATA
                    
             #       if stream[4] == 3:
             #           obj.DB['Animals'][A][F][stream[2]][-DATA.shape[0]:]=DATA
             #       elif stream[4] == 1:
             #           obj.DB['Animals'][A][F][stream[2]][-DATA.shape[0]:]=DATA
                    
                        
                    #obj.DB['Animals'][A][F][stream[2]].dims[0].attach_scale(obj.DB['Animals'][A][F]['T'][stream[3]])
                    
                else: #Create new data stream if not present
                    print('Creating new dataset...')
                    if stream[4] == 3:
                        if chunkOn:
                            obj.DB['Animals'][A][F].create_dataset(stream[2], shape =  DATA.shape,maxshape = (None, DATA.shape[1], DATA.shape[2] ),data = DATA, track_order = True, chunks=(1,DATA.shape[1],DATA.shape[2]))
                        else:
                            obj.DB['Animals'][A][F].create_dataset(stream[2], shape =  DATA.shape,maxshape = (None, DATA.shape[1], DATA.shape[2] ),data = DATA, track_order = True)
                        #Create ROI mask and trace groups:
                        obj.DB['Animals'][A][F]['R'].create_group(stream[2])
                        obj.DB['Animals'][A][F]['R'][stream[2]].create_group('Masks')
                        obj.DB['Animals'][A][F]['R'][stream[2]].create_group('Traces')
                    elif stream[4] == 4: # 4 dimensional data (e.g. RGB movies)
                        if chunkOn:
                            obj.DB['Animals'][A][F].create_dataset(stream[2], shape =  DATA.shape, maxshape = (None, DATA.shape[1], DATA.shape[2] , DATA.shape[3]), data = DATA, track_order = True, chunks=(1, DATA.shape[1],DATA.shape[2],DATA.shape[3]))
                        else:
                            obj.DB['Animals'][A][F].create_dataset(stream[2], shape =  DATA.shape, maxshape = (None, DATA.shape[1], DATA.shape[2] , DATA.shape[3]), data = DATA, track_order = True)
                        
                        #Create ROI mask and trace groups:
                        obj.DB['Animals'][A][F]['R'].create_group(stream[2])
                        obj.DB['Animals'][A][F]['R'][stream[2]].create_group('Masks')
                        obj.DB['Animals'][A][F]['R'][stream[2]].create_group('Traces')
                    elif stream[4] == 1:
                        obj.DB['Animals'][A][F].create_dataset(stream[2], shape =  DATA.shape, maxshape = (None, ), data = DATA, track_order = True)
                    obj.DB['Animals'][A][F]['T'].create_dataset(stream[3], maxshape = (None,), shape = TIME.shape, data = TIME, track_order = True)
                    #obj.DB['Animals'][A][F]['T'][stream[3]].make_scale(stream[3])
                    #obj.DB['Animals'][A][F][stream[2]].dims[0].attach_scale(obj.DB['Animals'][A][F]['T'][stream[3]])
                
                print(stream[3] + ' is ' + str(obj.DB['Animals'][A][F]['T'][stream[3]].shape[0]) + ' samples long')
            
                
            
        obj.updateDataList()
        print('Done')
        

def depositSessionToFOV(obj, sessionFolder, transform=None, select_data = False):
    if select_data:
        input_types = obj.inputDict.keys()
        input_filter, okPressed = QInputDialog.getItem(obj,"Select data type:", "Data:", input_types, 0, False)
    else:
        input_filter = None
    
    directory = os.listdir(sessionFolder)
    directory.sort()
    print(directory)
    for c, item in enumerate(directory):
        if item.isdecimal() and len(item)==10:
            print(f'Depositing trial {c} of {len(directory)}')
            depositTrial2(obj, os.path.join(sessionFolder,os.path.normpath(item)), data_filter = input_filter)
    
    
 
def blockRegister(stack, obj):
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    templateKey, okPressed = QInputDialog.getItem(obj,"Select Template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        return(stack)
    templateStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][templateKey][:]
    template = np.median(templateStack, axis = 0)
    source = np.median(stack, axis = 0)
    output = pReg(stack, obj, template)
    sr = StackReg(StackReg.RIGID_BODY)
    transform = sr.register(template, source)
    output = np.zeros(stack.shape)
    for count, frame in enumerate(stack):
        output[count, :,:] = sr.transform(frame, tmat = transform)
    return(output)
    

def registerTemplate(stack, obj):
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    templateKey, okPressed = QInputDialog.getItem(obj,"Select Template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        return(stack)
    templateStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][templateKey][:]
    template = np.median(templateStack, axis = 0)
    output = pReg(stack, obj, template=template)
    return(output)
    
def regAndCrop(stack, obj):
    return(cropToZeros(pReg(stack, obj)))
    
def cropToZeros(stack, obj):
    minStack = np.amin(stack, axis = 0) ## ?Need to change to percentiel (eg 1%) for rare reg mistakes
    print(f'Shape: {minStack.shape}')
    plt.imshow(minStack)
    plt.show()
    boostack = minStack.astype(bool)
    min8 = boostack.astype(np.uint8)*255
    x, y, w, h = LIR.largest_interior_rectangle(min8)
    
    print(f'Bounds: {x,y,w,h}')
    
    return(stack[:,y:y+h,x:x+w,...])
        
    

def stitch(stack,obj):      ##:TODO fix
    images = []
    im1 = np.median(stack, axis = 0)
    images.append(im1)
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    templateKey, okPressed = QInputDialog.getItem(obj,"Select Template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        print(okPressed)
        print(~okPressed)
        return(stack)
    im2 = np.median(obj.DB['Animals'][obj.curAnimal][obj.curFOV][templateKey][:], axis = 0)
    images.append(im2)
    stitcher = cv2.Stitcher.create()
    (status, stitched) = stitcher.stitch(images)
    output = stitched.repeat(stack.shape[0], 0)
    return(output)
                  
def crossCorr(stack, obj):
    im = DYroi.local_correlations_fft(stack)
    im = np.expand_dims(im,0)
    im = np.array(im)
    return(im.repeat(stack.shape[0], 0))
    
def adaptiveThresh(stack, obj):
    blocksize = obj.segmentationMethods['Adaptive threshold']['Params']['Block size'][1]
    #erodeRep  = self.segmentationMethods['Adaptive threshold']['Params']['Erode cycles'][1]
    #erodeArea = self.segmentationMethods['Adaptive threshold']['Params']['Erode area'][1]
    C         = obj.segmentationMethods['Adaptive threshold']['Params']['C'][1]
    #minArea   = self.segmentationMethods['Adaptive threshold']['Params']['Min area'][1]
    #maxArea = self.segmentationMethods['Adaptive threshold']['Params']['Max area'][1]
    #img = stack[0,...]
    stack = stack - np.min(np.min(np.min(img)))
    img = stack/np.max(np.max(np.max(stack)))
    img = img * 255
    img = img.astype('uint8')
    output = np.zeros(stack.shape)
    for ii in range(stack.shape[0]):
        output[ii,:,:] = cv2.adaptiveThreshold(img[ii,:,:], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, C)
        print(ii)
    return(output.astype(np.uint16))

def adaptiveThreshInv(img, obj):
    blocksize = obj.segmentationMethods['Adaptive threshold']['Params']['Block size'][1]
    #erodeRep  = self.segmentationMethods['Adaptive threshold']['Params']['Erode cycles'][1]
    #erodeArea = self.segmentationMethods['Adaptive threshold']['Params']['Erode area'][1]
    C         = obj.segmentationMethods['Adaptive threshold']['Params']['C'][1]
    #minArea   = self.segmentationMethods['Adaptive threshold']['Params']['Min area'][1]
    #maxArea = self.segmentationMethods['Adaptive threshold']['Params']['Max area'][1]
    #img = stack[0,...]
    img = img - np.min(np.min(np.min(img)))
    img = img/np.max(np.max(np.max(img)))
    img = img * 255
    img = img.astype('uint8')
    output = np.zeros(img.shape)
    for ii in range(img.shape[0]):
        output[ii,:,:] = cv2.adaptiveThreshold(img[ii,:,:], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, C)
        print(ii)
    return(output.astype(np.uint16))




    
def erode(stack, obj):
    img = stack[0,...]
    img = stack
    img = img - np.min(np.min(img))
    img = img/np.max(np.max(img))
    img = img * 255
    img8 = img.astype('uint8')
    kernel = np.ones((3,3), dtype=np.uint8)
    newim = cv2.erode(img8, kernel)
    return(newim.repeat(stack.shape[0], 0))

def splitROI(stack, obj):
    pass

def maskToROI(stack, obj):
    pass


class FOVreporter:
    
    def __init__(self, dataArray, timeStep):
        pass

class stackRegIterator:
    
    def __init__(self,stack, refIm = 'X', method = 'RIGID_BODY'):
        self.frame = 0
        self.max = stack.shape[0]
        self.stack = stack
        if method == 'RIGID_BODY':
            self.sr = StackReg(StackReg.RIGID_BODY)
        elif method == 'AFFINE':
            self.sr = StackReg(StackReg.AFFINE)
        elif method == 'BILINEAR':
            self.sr = StackReg(StackReg.BILINEAR)
        else:
            self.sr = StackReg(StackReg.RIGID_BODY)
        if refIm != 'X':
            self.refIm = refIm
        else:     
            self.refIm = np.median(stack, axis = 0) ###
        
        
        
    def __iter__(self):
        self.frame = 0
        return(self)
    
    def __next__(self):
        if self.frame < self.max:
            result = (self.stack[self.frame,:,:], self.refIm, self.sr)
            self.frame = self.frame +1
            pctDone = (100*self.frame/self.max)  ## test if this works
            #print('Registering [%d%%] complete...\r'%pctDone, end="")
            return(result)
        else:
            raise StopIteration


def pReg(stack, obj, template='X', poolNum = None, method = 'RIGID_BODY'):
    if poolNum == None:
        poolNum = obj.segmentationMethods['Random params']['Params']['nClusters'][1]
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibrary" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        IT = stackRegIterator(stack, refIm = template, method = 'RIGID_BODY')
        result = pool.map(regImage, IT) ## this works,testing progress bar
       
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print('Time elapsed registering:' + str(endTime-startTime))
        print(poolNum)
        return(np.array(result))


def affineReg(stack, obj):
    return(pReg(stack, obj, method = 'AFFINE'))
 


def regImage(regInput):
    im =  regInput[0]
    refIm = regInput[1]
    sr = regInput[2]
    #return(np.uint16(sr.register_transform(refIm, im)))
    return(sr.register_transform(refIm, im))

def pGetTransform(stack, obj, template='X', poolNum = 23):
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibrary" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        IT = stackRegIterator(stack, template)
        result = pool.map(getTransform, IT)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(endTime-startTime)
        print(poolNum)
        return(np.array(result))
    
    
def removeBadFrames(stack,obj):
    
    L = stack.shape[0]
    diff = np.zeros(L)
    for c, frame in enumerate(stack):
        if c < 3:
            mask = [0, 1, 2, 3, 4, 5, 6]
        elif c > L-4:
            mask = [L-7, L-6, L-5, L-4, L-3, L-2, L-1]
        else:
            mask = [c-3, c-2, c-1, c, c+1, c+2, c+3]
        print(c)
        print(mask)
        template = stack[mask,...]
        diff[c] = np.absolute(np.sum(template-frame))
        
    plt.plot(diff)
    plt.show()
    plt.hist(diff)
    plt.show()
    
    minShift = int(input('Enter min shift value:'))
    mask = diff>minShift
  
    timescale =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus]
    Istart = np.searchsorted(timescale,obj.timeLUT[0])
    Iend = np.searchsorted(timescale,obj.timeLUT[-1])
    timescale = timescale[Istart:Iend]

    newStack = np.delete(stack, mask, axis = 0)
    newTime  = np.delete(timescale, mask, axis = 0)
    
    newDataKey = obj.dataFocus + '_Xframes' + '_' + str(minShift)
    
    
    newShape = list(newStack.shape)
    if len(newShape) > 1:
        newShape[0] = 1
    chunkShape = tuple(newShape)
        
    obj.DB['Animals'][obj.curAnimal][obj.curFOV].create_dataset(newDataKey, data = newStack, track_order = True, chunks=chunkShape) 
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'].create_dataset(newDataKey,  data = newTime, dtype = newTime.dtype, shape = newTime.shape, maxshape = (None,) )
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'].require_group(newDataKey)
      
    return(stack)
            
            
def downSampleSpatial(stack,obj):
    output = downscale_local_mean(stack, (1,2,2))
    return(output)

def dataTime(obj):
    output = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus][...]
    return(output)

def downSampleTemporal(stack, obj, factor = 2):
    timescale = dataTime(obj)
    Istart = np.searchsorted(timescale,obj.timeLUT[0])
    Iend = np.searchsorted(timescale,obj.timeLUT[-1])
    oldtime = timescale[Istart:Iend]
    newStack = stack[::factor]
    newTime = oldtime[::factor]
    dName = obj.dataFocus + '_dwnT'
    genericDepositTrial(obj, newStack, newTime, dName)
    return(stack)
    
    
    
def pGetTransformFromTemplate(stack, obj, template='X', poolNum = 23):
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    templateKey, okPressed = QInputDialog.getItem(obj,"Select Template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        print(okPressed)
        print(~okPressed)
        return(stack)
    templateStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][templateKey][:]
    template = np.median(templateStack, axis = 0)

    
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibrary" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        IT = stackRegIterator(stack, template)
        result = pool.map(getTransform, IT)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(endTime-startTime)
        print(poolNum)
        return(np.array(result))
    
def getTransform(regInput):
    im =  regInput[0]
    refIm = regInput[1]
    sr = regInput[2]
    return(sr.register(refIm, im))

def applyTransform(stack, obj):
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    tmatKey, okPressed = QInputDialog.getItem(obj,"Select Transformation Matrix:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        print(okPressed)
        print(~okPressed)
        return(stack)
    output = np.zeros(stack.shape)
    sr = StackReg(StackReg.RIGID_BODY)
    tmats = obj.DB['Animals'][obj.curAnimal][obj.curFOV][tmatKey][:]
    counter = 0
    for frame in stack:
        output[counter,...] = sr.transform(frame, tmat=tmats[counter])
        counter = counter+1
    return(output)
        
    
def manualRegister(stack,obj): ## not finished
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    tempKey, okPressed = QInputDialog.getItem(obj,"Select template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        return(stack)
    obj.clearLayout()
    tempStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][tempKey][:]
    templateImage = np.median(tempStack, axis = 0)
    sourceImage = np.median(stack, axis = 0)
    obj.regView.invertY(True)
    obj.regView.autoRange(padding=None)
    source = pg.plotItem()
    target = pg.plotItem()
    source.setimage(sourceImage)
    target.setImage(targetImage)
    source.setOpacity(0.5)
    obj.regView.addItem(target)
    obj.regView.addItem(source)
    

    
    
    
    

def singleMedFilt2(IMG):
    return(cv2.medianBlur(IMG, 3))

def pMedFilt2(stack, obj, poolNum = 23):
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibrary" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        result = pool.map(singleMedFilt2, stack)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(endTime-startTime)
        print(poolNum)
        return(np.array(result))

class FFTiter:
    def __init__(self,stack):
        self.frame = 0
        self.max = stack.shape[0]
        self.stack = stack
        self.LPfilt = gaussianLP(5, stack[0,:,:].shape)
    
    def __iter__(self):
        self.frame = 0
        pctDone = (100*self.frame/self.max)  ## test if this works
        #print('Filtering [%d%%] complete...\r'%pctDone, end="")
       
        return(self)
    
    def __next__(self):
        if self.frame < self.max:
            result = (self.stack[self.frame,:,:], self.LPfilt)
            self.frame = self.frame +1
            return(result)
        else:
            raise StopIteration
     
def singleFFT(inputArgs):
    img       = inputArgs[0]
    LPfilter  = inputArgs[1]
    fft       = np.fft.fft2(img)
    center    = np.fft.fftshift(fft)
    LPcenter  = center * LPfilter
    LP        = np.fft.ifftshift(LPcenter)
    invLP     = np.fft.ifft2(LP)
    blurred   = np.abs(invLP)
    corrected = np.float32(np.divide(img,blurred))
    return(corrected)
    

def pFFThighPass(stack, obj, poolNum = 23):
    startTime = time.perf_counter()
    #print(__name__)
    #print('FFT par called')
    if __name__ == "DYpreProcessingLibrary" or __name__ == '__main__':
        #print('FFT par progressed')
        pool = mp.Pool(poolNum)
        IT = FFTiter(stack)
        result = pool.map(singleFFT, IT)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(f'Filtering took:{endTime-startTime} seconds')
        #print('Collecting results...')
        #output = normalizeStack(result)
        output = np.array(result)
        
        return(output)
    else:
        print(__name__)
        
def normalizeStack(stack, obj):
    output = stack - np.min(np.min(np.min(stack)))
    output = output*(((2**16)-1)/np.max(np.max(np.max(output))))
    output = np.uint16(output)
    arraySum = np.sum(output)
    print(arraySum)
    return(output)

def convertUint16(stack, obj, minV = 0.8, maxV = 1.2): #typical range for fft filtered data
    output = stack - minV
    output = output*(((2**16)-1)/maxV)
    output = np.uint16(output)
    return(output)

        
def pDFF(stack):
    return
     
        
def transposeRGB(stack, obj):
    print(stack.shape)
    output = np.zeros(stack.shape)
    output[:,:,:,0] = stack[:,:,:,2]
    output[:,:,:,1] = stack[:,:,:,1]
    output[:,:,:,2] = stack[:,:,:,0]
    return(output)
        
def grams_to_millinewtons(stack, obj):
    return(stack*9.80665)
     
def regToFeature(stack, obj):
    for ROI in obj.transformROIlist:
        subStack = DYcrop(stack, ROI)
        ## get transformation matrix of each stack, zero out rotation, take medianm apply to full stack
    
        
def crop(stack, obj):
    for ROI in obj.transformROIlist:
        return(DYcrop(stack,ROI))
    
        
     
def DYcrop(stack, ROI):
    rect = ROI.parentBounds()
    top = round(rect.y())
    bot = round(rect.y()+rect.height())
    left = round(rect.x())
    right = round(rect.x()+rect.width())
    data = stack[:,left:right,top:bot,...]
    print(top)
    print(bot)
    print(left)
    print(right)
    return(data)
        
     
def getSubStack(obj, rect = None, FOV=None, datakey = None, Animal = None):
        if FOV == None:
            FOV = obj.curFOV
        if datakey == None:
            datakey = obj.dataFocus
        if Animal == None:
            Animal = obj.curAnimal
        
        Istart, Iend = getTimeBounds(obj, FOV=FOV, datakey=datakey, Animal = Animal)
        print(f'Istart: {Istart}')
        print(f'Iend: {Iend}')
        data = obj.DB['Animals'][Animal][FOV][datakey][Istart:Iend,...]
        time = obj.DB['Animals'][Animal][FOV]['T'][datakey][Istart:Iend]

        return(data, time)

def getTimeBounds(obj,  FOV=None, datakey = None, Animal = None): # data is key to 
    if FOV == None:
            FOV = obj.curFOV
    if datakey == None:
            datakey = obj.dataFocus
    if Animal == None:
            Animal = obj.curAnimal
    timescale =  obj.DB['Animals'][Animal][FOV]['T'][datakey][...]
    timescale =  obj.DB['Animals'][Animal][FOV]['T'][datakey][...]
    Istart = np.searchsorted(timescale,obj.timeLUT[0])
    Iend = np.searchsorted(timescale,obj.timeLUT[-1])
    return(Istart, Iend)
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
        
def findFOVfolders(rootDir, skipStr = 'zzz'):
    dirList = []
    for currentPath, folders, files in os.walk(rootDir):
        for folder in folders:
            dirList.append(os.path.join(currentPath, folder))
    FOVlist = []
    for folder in dirList:
        contents = os.listdir(folder)
        for item in contents:
            concat = os.path.join(folder, item)
            if os.path.isdir(concat) and item.isdecimal() and len(item)==10 and not skipStr in item:
                if not (skipStr in folder):
                    FOVlist.append(folder)
                break
    return(FOVlist)

class dbContainer:
    def __init__(self, DBpath):
        self.DB = h5py.File(DBpath,'a')
        self.DB.require_group('Animals')
        self.curAnimal = ''
        self.curFOV = ''
        
    def updateDataList(self):
        return
    

 
    
def autoDepositFOV(FOVfolder, depositFolder, mouseName = 'empty'):
    FOVnameBase = os.path.split(FOVfolder)[1]
    
    Trials = os.listdir(FOVfolder)
    Trials.sort()
    FOVname = 'dummy'
    for trial in Trials:
        if trial.isdecimal() and len(trial)==10:
            t = time.localtime(int(trial))
            FOVname = f'{FOVnameBase} {t[0]}-{t[1]}-{t[2]}'
            break  
    
    DBpath = os.path.join(depositFolder,(FOVname+'.h5'))
    db = dbContainer(DBpath)
    db.curAnimal = mouseName
    print('DBpath is' + DBpath)
    db.DB['Animals'].require_group(mouseName)
    db.curFOV = FOVname
    
    if FOVname in db.DB['Animals'][mouseName].keys():
        print('already deposited, skipping...')
    else:
        newFOV = db.DB['Animals'][mouseName].require_group(FOVname)
        newFOV.require_group('T')   ### Time data
        newFOV.require_group('R')    ## ROI data (masks and traces)
        newFOV.require_group('S')    ## signal data (selected 1D signals)
        newFOV['S'].require_group('Signal')
        newFOV['S'].require_group('Time')
        depositSessionToFOV(db, FOVfolder)
        newFOVs=[]
        newFOVs.append(FOVname)
        autoProcessCERNA(db, mouseName, newFOVs)
    return(DBpath)
      
def slurmDepositMouse():
    
    #Read path to target and depsoit directories from file:
    f=open('/home/ch184656/YarmoPain_GUI/targetPath.txt','r')
    targetDir = f.read()
    f.close()
    g=open('/home/ch184656/YarmoPain_GUI/depositPath.txt','r')
    depositDir = g.read()
    g.close()
    #Get list of FOV directories to process:
    fovFolders = findFOVfolders(targetDir)
    #Iterate through FOVs, deposit in indiviudal dbs and pre-preprocess:  
    DBlist = []
    mN = os.path.split(targetDir)[1]
    targetName = mN + '.h5'
    mDBpath = os.path.join(depositDir, targetName)
    mouseDB = dbContainer(mDBpath)
    mouseDB.curAnimal = mN
    mouseDB.DB.require_group('Animals')
    mouseDB.DB['Animals'].require_group(mN)
    for FOVfolder in fovFolders:
        DBlist.append(autoDepositFOV(FOVfolder, depositDir, mouseName = mN))  
    for DBpath in DBlist:
        curDB = h5py.File(DBpath,'a')
        #mouseDB.DB['Animals'][mN] = curDB['Animals'][mN]
        for FOVname in curDB['Animals'][mN].keys():
            print(f'Transfering data from {FOVname} to  {targetName}...')
            curDB.copy(curDB['Animals'][mN][FOVname], mouseDB.DB['Animals'][mN])
        curDB.close()
        os.remove(DBpath)
    mouseDB.DB.close()



def localDepositMouse():
    
    FOLDER = selectFolder()
    
    tT = time.localtime(time.time())
    mouseName = os.path.split(FOLDER)[1]
    IDstr = mouseName + '_proc_' + str(tT[2]) + str(tT[1]) + str(tT[0]) + str(tT[3]) + str(tT[4])
    depositDir = f'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Processed/{IDstr}'
    os.mkdir(depositDir)
    
    targetDir = FOLDER
    
    #Get list of FOV directories to process:
    fovFolders = findFOVfolders(targetDir)
    #Iterate through FOVs, deposit in indiviudal dbs and pre-preprocess:  
    DBlist = []
    mN = os.path.split(targetDir)[1]
    targetName = mN + '.h5'
    mDBpath = os.path.join(depositDir, targetName)
    mouseDB = dbContainer(mDBpath)
    mouseDB.curAnimal = mN
    mouseDB.DB.require_group('Animals')
    mouseDB.DB['Animals'].require_group(mN)
    for FOVfolder in fovFolders:
        DBlist.append(autoDepositFOV(FOVfolder, depositDir, mouseName = mN))  
    for DBpath in DBlist:
        curDB = h5py.File(DBpath,'a')
        #mouseDB.DB['Animals'][mN] = curDB['Animals'][mN]
        for FOVname in curDB['Animals'][mN].keys():
            print(f'Transfering data from {FOVname} to  {targetName}...')
            curDB.copy(curDB['Animals'][mN][FOVname], mouseDB.DB['Animals'][mN])
        curDB.close()
        os.remove(DBpath)
    mouseDB.DB.close()
    print('Finished deposit')

                
                
def autoProcessCERNA(db, mouseName, FOVs): #db is an object with a DB property (can be GUI instance or dbcontainer)
    for FOV in FOVs:
        print(FOV)
    #db.DB['Animals'][mouseName].keys():              ###go through data and process Ca images 
        prefix= db.DB['Animals'][mouseName][FOV]
        for key in prefix.keys():
            if 'CERNA' in key:
                print(f'Pre-processing Ca++ imaging data for FOV {FOV}')
                oldData = prefix[key]
                oldTime = prefix['T'][key]
                nFrames = oldTime.shape[0]
                nPieces = math.ceil(nFrames/1000)
                splitData = np.array_split(oldData, nPieces, axis=0)
                
                    
                
                oldTime = prefix['T'][key][:]
                print(f'Processing block {1} of {len(splitData)}, size {splitData[0].shape[0]} frames')
                
                #Register 1st block using its own median value
                newData = prep(splitData[0], db, templateIM = 'X')
                #Take median of registered first block to use as registration template for subsequent blocks:
                regTemplate = np.median(newData, axis=0)
                newDataKey = key + '_prepped'
                
                prefix['T'].require_dataset(newDataKey,  data = oldTime, dtype = oldTime.dtype, shape = oldTime.shape, maxshape = (None,) )
                
                prefix.require_dataset(newDataKey, data = newData, shape =newData.shape, dtype =newData.dtype, track_order = True, maxshape = (None, newData.shape[1], newData.shape[2]), chunks=(1,newData.shape[1],newData.shape[2])) 
                prefix['R'].require_group(newDataKey)
                prefix['R'][newDataKey].require_group('Masks')
                prefix['R'][newDataKey].require_group('Traces')
                
               
                
                if len(splitData)<2:
                    return
                counter = 2
                for DATA in splitData[1:]:
                    print(f'Processing block {counter} of {len(splitData)}, size {DATA.shape[0]} frames')
                    newData = prep(DATA, db, templateIM = regTemplate)
                    prefix[newDataKey].resize((prefix[newDataKey].shape[0]+DATA.shape[0]),axis = 0)   
                    prefix[newDataKey][-newData.shape[0]:]=newData
                    counter = counter + 1
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
def autoDepositMouse(targetDir, createDB=True, DBpath=None, preprocess = True):
    print('Depositing mouse...')
    
    mouseName = os.path.split(targetDir)[1]
    targetName = mouseName + '.h5'
    
    if DBpath is None:
        
        #Create database if option selected
        DBpath = os.path.join(targetDir, targetName)

        
    db = dbContainer(DBpath)
    db.curAnimal = mouseName
    
    FOVfolders = findFOVfolders(targetDir)  
    db.DB['Animals'].require_group(mouseName)
     
    newFOVs = []
    counter = 1
    for FOVfolder in FOVfolders:
        print(f'Depositing FOV {counter} of {len(FOVfolders)}')
        counter = counter + 1
        FOVnameBase = os.path.split(FOVfolder)[1]
        Trials = os.listdir(FOVfolder)
        Trials.sort()
        FOVname = 'dummy'
        for trial in Trials:
            if trial.isdecimal() and len(trial)==10:
                t = time.localtime(int(trial))
                FOVname = f'{FOVnameBase} {t[0]}-{t[1]}-{t[2]}'
                break
        if FOVname in db.DB['Animals'][mouseName].keys():
            print(f'{FOVname} already deposited, skipping...')
            continue
        else:
            newFOVs.append(FOVname)
        newFOV = db.DB['Animals'][mouseName].require_group(FOVname)
        db.curFOV = FOVname
        newFOV.require_group('T')   ### Time data
        newFOV.require_group('R')    ## ROI data (masks and traces)
        newFOV.require_group('S')    ## signal data (selected 1D signals)
        newFOV['S'].require_group('Signal')
        newFOV['S'].require_group('Time')
        depositSessionToFOV(db, FOVfolder)
             
    if preprocess:
        autoProcessCERNA(db, mouseName, newFOVs)
        
        
    return

    
                
def depositArchive(targetDir=None, preprocess = False, exString = 'zzz'):
    
    
    
    if targetDir is None:
        targetDir = selectFolder(msg = "Choose archive to process...")
    
    resultDir = selectFolder(msg = "Choose folder to deposit results...")
        
  #  tT = time.localtime(time.time())
  #  archiveName = os.path.split(targetDir)[1]
  #  print(f'Archive name: {archiveName}')
  #  IDstr = archiveName + '_proc_' + str(tT[2]) + str(tT[1]) + str(tT[0]) + str(tT[3]) + str(tT[4])
  #  resultDir = f'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Processed/{IDstr}'
  #  os.mkdir(resultDir)
    
    
    folders = os.listdir(targetDir)
    for folder in folders:
        if os.path.isdir(os.path.join(targetDir,folder)) and 'Mouse' in folder:
            if not exString in folder:  # exclude folders with exString in name
    
                print(folder)
                DBname = folder + '.h5'
                autoDepositMouse(os.path.join(targetDir,folder), preprocess, DBpath = os.path.join(resultDir, DBname), preprocess=preprocess)
     
        
     
def appendSessions(targetDir, obj, preprocess = False):
    
    
    print('Depositing mouse...')
    
    mouseName = obj.curAnimal
    #targetName = mouseName + '.h5'
    
    
    FOVfolders = findFOVfolders(targetDir)  

     
    newFOVs = []
    counter = 1
    for FOVfolder in FOVfolders:
        print(f'Depositing FOV {counter} of {len(FOVfolders)}')
        counter = counter + 1
        FOVnameBase = os.path.split(FOVfolder)[1]
        
        Trials = os.listdir(FOVfolder)
        Trials.sort()
        FOVname = 'dummy'
        for trial in Trials:
            if trial.isdecimal() and len(trial)==10:
                t = time.localtime(int(trial))
                FOVname = f'{FOVnameBase} {t[0]}-{t[1]}-{t[2]}'
                break
        if FOVname in obj.DB['Animals'][mouseName].keys():
            print(f'{FOVname} already deposited, skipping...')
            continue
        else:
            newFOVs.append(FOVname)
        newFOV = obj.DB['Animals'][mouseName].require_group(FOVname)
        obj.curFOV = FOVname
        newFOV.require_group('T')   ### Time data
        newFOV.require_group('R')    ## ROI data (masks and traces)
        newFOV.require_group('S')    ## signal data (selected 1D signals)
        newFOV['S'].require_group('Signal')
        newFOV['S'].require_group('Time')
        depositSessionToFOV(obj, FOVfolder)
             
    if preprocess:
        autoProcessCERNA(obj, mouseName, newFOVs)
        
    obj.updateFOVlist()
    return
     
def joinMini(ar1, ar2):
    
    if ar1.size == 0:
        arrayIn = (ar2)
        print('1st array to join is empty...')
    else:
        arrayIn  = np.concatenate((ar1, ar2), axis = 0 )
        
    output  = arrayIn
    return(output)
        
def findDBs(rootDir, pattern = 'def'): ## can find file by any pattern, default is for 'definitive' database
    DBlist = []
    for currentPath, folders, files in os.walk(rootDir):
        for file in files:
            print(file)
            if pattern in file:
                DBlist.append(os.path.join(currentPath, file))
    return(DBlist)
            
def runAnalyses(obj):
    initAnalysis(obj)
    thermAnalyses(obj)
    
    
def initAnalysis(obj):  ## Funct to create DB of stims and responses for FOV within a mouses H5 file
    
    fullArr, compactArray, timestep, stimIX, stimLabels, meanImage, ROIs, nullValue = obj.generateSummaryTable(createFiles = False)
    A  = obj.DB.require_group('Analyses')
    if obj.curFOV in A.keys(): #delete existing analysis if present; may want to amend this
        del A[obj.curFOV]
    FA = A.require_group(obj.curFOV)
    FA.require_group('stims')
    for IX, label in zip(stimIX, stimLabels):
        stim = compactArray[IX,:]
        FA['stims'].require_dataset(label, data = stim, shape = stim.shape, dtype = stim.dtype)
    
    cellRaster = np.delete(compactArray, stimIX, axis = 0)
    source = h5py.ExternalLink(obj.DBpath, f'/Animals/{obj.curAnimal}/{obj.curFOV}')
    #FA['source'] = source
    FA.require_dataset('timestep', data = timestep, shape = timestep.shape, dtype = timestep.dtype)
    FA.require_dataset('nullValue', data = nullValue, shape = nullValue.shape, dtype = nullValue.dtype)
    FA.require_dataset('raster', data = cellRaster, shape = cellRaster.shape, dtype = cellRaster.dtype)
    FA.require_dataset('ROIs', data = ROIs, shape = ROIs.shape, dtype = ROIs.dtype)
    FA.require_dataset('meanImage', data = meanImage, shape = meanImage.shape, dtype = meanImage.dtype)
    print('Analysis entered')
    
def prepAnalysisData(obj):
    data = LA.processTempStim(LA.prepAnalysis(obj.DB['Analyses'][obj.curFOV], closeAfterReading = False))
    
    return(data)
    
def thermAnalyses(obj):
    data = prepAnalysisData(obj)
    LA.TvsTcor(data)
    LA.thermoScatter(data)
    cmap = LA.thermPolarPlot(data)
    LA.thermColorCode(data)
    


def isolateFOVtonewDB(obj): ## send an FOV to a new DB to work on without endangering rest of file
    path, parent = os.path.split(obj.DBpath)
    Animal = obj.curAnimal
    FOV = obj.curFOV
    savePath = os.path.join(path, parent + '_' + obj.curFOV+ '.h5')
    print(f'Copying {FOV} to {savePath}...')
    NF = h5py.File(savePath, 'a')
    NF.require_group('Animals')
    NF['Animals'].require_group(Animal)
    start = time.time()
    obj.DB.copy(obj.DB['Animals'][Animal][FOV], NF['Animals'][Animal])
    NF.close()
    print('Done copying')
    print(f'Transfer took {time.time-start} seconds')
    
def isolateDataToNewDB(obj): ## send an FOV to a new DB to work on without endangering rest of file
    path, parent = os.path.split(obj.DBpath)
    Animal = obj.curAnimal
    FOV = obj.curFOV
    savePath = os.path.join(path, parent + '_' + obj.curFOV+ '.h5')
    print(f'Copying {FOV} to {savePath}...')
    NF = h5py.File(savePath, 'a')
    NF.require_group('Animals')
    NF['Animals'].require_group(Animal)
    NF['Animals'][Animal].require_group(FOV)
    NF['Animals'][Animal][FOV].require_group('R')
    NF['Animals'][Animal][FOV].require_group('T')
    Fstart = time.time()
    for item in obj.DataList.selectedItems():
        Dstart = time.time()
        key = item.text()
        print(f'Copying datastream {key}')
        obj.DB.copy(obj.DB['Animals'][Animal][FOV][key], NF['Animals'][Animal][FOV], name = key)
        print(f'Copying ROIs for {key}')
        obj.DB.copy(obj.DB['Animals'][Animal][FOV]['R'][key], NF['Animals'][Animal][FOV]['R'], name =key)
        print(f'Copying time scale for {key}')
        obj.DB.copy(obj.DB['Animals'][Animal][FOV]['T'][key], NF['Animals'][Animal][FOV]['T'], name = key)
        print(f'Datastream took {time.time()-Dstart} seconds to copy')
        
    NF.close()
    print('Done copying')
    print(f'Transfer took {time.time()-Fstart} seconds')
    
        
     
        