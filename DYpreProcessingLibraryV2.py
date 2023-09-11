#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:43:17 2022

@author: ch184656
"""
import pdb
from PIL import ImageFont
import cv2
import math
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
mat.rcParams['pdf.fonttype'] = 42
mat.rcParams['ps.fonttype'] = 42
from matplotlib import cm
import h5py
from pystackreg import StackReg
from skimage.transform import downscale_local_mean
#from tkinter import file
#from tkinter import *
import glob
from sklearn.cluster import KMeans
from beeswarm import beeswarm, beeswarms
#import scipy
import time
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, \
    QLineEdit, QMainWindow, QSizePolicy, QLabel, QSlider, QMenu, QAction, \
    QComboBox, QListWidget, QGridLayout, QPlainTextEdit, QDateTimeEdit, QTextEdit, QInputDialog, QColorDialog

from stim_corr_image import corrImSig
import DYroiLibrary as DYroi
import json
import pickle
import natsort

from scipy import stats
from skimage.data import colorwheel
from skimage.draw import disk
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from scipy import signal
from scipy.linalg import block_diag
#import LIRnoNumba as LIR #least inscribed rectangle  -crashing with origina lLIR library, took out numba acceleration
import LIR as LIR
import libAnalysis as LA
from libAnalysis import unpickle, sortByKmean, jitter
from libAnalysis import gen_colors
from libAnalysis import alignROIsCAIMAN, DYroi2CaimanROI, CaimanROI2dyROI, plot_contours
import stitchLib as stitch
from stim_corr_image import corrImSig
import copy

from skimage import data, util, transform, feature, measure, filters, metrics
import DYnormCOR
from summaryTable import genSumTable, dbContainer, getData
#from DHfigures import colorbar
from libAnalysis import plot_scale_bar
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






def mergeChannels(obj):
    for item in obj.DataList.selectedItems():
        dataKey = item.text()
        data = obj.DB['Animals'][obj.curAnimal][obj.curFOV][dataKey][...]
        C = QColorDialog.getColor()
        
        
        
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

def plotContours():
    pass
    
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
    CERNAfiles = findFilesFromFOV(FOVfolder, '*CERNA.tif*')
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





def max_index(vector):
    return(np.where(vector==np.amax(vector))[0][0])


def time_code(obj, data = None, TIME = None):
    if obj is None:
        IMG = data
        TIME = TIME
    else:
        IMG, TIME = getSubStack(obj)
        
    output  = IMG[0,...]
    for ii in range(output.shape[0]):
        for jj in range(output.shape[1]):
            output[ii,jj] = max_index(IMG[:,ii,jj])
    return(output, TIME)
    
def medFilt3D(obj, data = None, TIME = None):
    if obj is None:
        IMG = data
        TIME = TIME
    else:
        IMG, TIME = getSubStack(obj)
    output = median_filter(IMG, size = 3)
    return(output, TIME)
    
def medFilt2(obj, data = None, TIME = None):
    if obj is None:
        IMG = data
        TIME = TIME
    else:
        IMG, TIME = getSubStack(obj)
    print('Median filtering...')
    output = np.zeros(IMG.shape, dtype = IMG.dtype)#np.uint16)
    for ii in range(IMG.shape[0]):
        output[ii,:,:] = cv2.medianBlur(IMG[ii,:,:], 3)
        #pctDone = ii/IMG.shape[0]*100
        #print(f'\r{pctDone}', end="")
    return(output, TIME)

def flip_horizontal(obj, DATA=None, TIME = None):
    if DATA == None:
        DATA, TIME = getSubStack(obj)
    output = np.flip(DATA, axis = 1)
    return(output, TIME)

def flip_vertical(obj, DATA=None, TIME = None):
    if DATA == None:
        DATA, TIME = getSubStack(obj)
    output = np.flip(DATA, axis = 2)
    return(output, TIME)

def diff(obj, DATA=None,TIME = None):
    if DATA == None:
        DATA, TIME = getSubStack(obj)
    output = np.diff(DATA, axis = 0)
  
    return(output, TIME[:-1])


def abs_value(obj, DATA=None, TIME = None):
    if DATA == None:
        DATA, TIME = getSubStack(obj)
    output = np.absolute(DATA)
    return(output, TIME)

def reg_and_crop(obj, DATA=None, TIME = None):
    if DATA == None:
        DATA, TIME = getSubStack(obj)
    DATA, TIME = pReg(obj, stack=DATA, TIME=TIME)
    output = cropToZeros(None, stack=DATA, TIME=TIME)
    return(output)

def snapshot(figure):
    datastring = np.frombuffer(figure.canvas.tostring_rgb(), dtype = np.uint8)
    return(datastring.reshape(figure.canvas.get_width_height()[::-1] + (3,)))
    

def writeToAvi(obj, DATA=None, TIME = None, save_root = '/lab-share/Neuro-Woolf-e2/Public/Figure publishing/movies'):
    if DATA == None:
        DATA, TIME = getSubStack(obj)
    filename = f'{obj.curFOV}_{obj.dataFocus}.avi'
    avi_path = os.path.join(save_root, filename)
    rate = 1/np.median(np.diff(TIME))
    writer = imageio.get_writer(avi_path, fps=rate)
    for c, frame in enumerate(DATA):
        writer.append_data(frame.astype(np.uint8))
        print(f'Writing frame {c} of {DATA.shape[0]}')
    writer.close()
    print(f'Movie written to {avi_path}')
    
def write_movie(obj, figscale=None, acceleration = None, save_root = '/lab-share/Neuro-Woolf-e2/Public/Figure publishing/movies'):
 
    if figscale is None:
        figscale = obj.segmentationMethods['Movie']['Params']['scale'][1]
    if acceleration is None:
        acceleration = obj.segmentationMethods['Movie']['Params']['acceleration'][1]
    vmin = obj.segmentationMethods['Movie']['Params']['vmin'][1]/100 
    vmax = obj.segmentationMethods['Movie']['Params']['vmax'][1]/100
    margin = obj.segmentationMethods['Movie']['Params']['margin'][1]/1000
    filename = f'{obj.curFOV}_{obj.dataFocus}.avi'
    avi_path = os.path.join(save_root, filename)
    rate = 1/obj.timeStep
    writer = imageio.get_writer(avi_path, fps=rate*acceleration, quality=10, codec = 'ffv1')
    As = {}
    DATA = {}
    TIME = {}
    allTime = []
    num_streams = len(obj.curDataStreams)
    aspect_ratios = []
    mins = []
    maxs = []
    for d, datastream in enumerate(obj.curDataStreams):
        
        obj.dataFocus = datastream
        D, T= getSubStack(obj)
        
        mins.append(np.nanpercentile(D, vmin))
        maxs.append(np.nanpercentile(D, vmax))
        
        if len(D.shape) == 1:
            aspect_ratios.append(1)
        else:
            D = np.swapaxes(D, 1,2)
            aspect_ratios.append(D.shape[2]/D.shape[1])
        DATA[datastream] = D
        TIME[datastream ] = T                            
        allTime.extend(T)
    width = np.sum(aspect_ratios)*figscale
    F = plt.figure(figsize = (width, figscale))
    X=0
    for d, datastream in enumerate(obj.curDataStreams):
        #As[datastream] = F.add_subplot(1, num_streams, d+1)
        Y=0
        W_ = aspect_ratios[d]/np.sum(aspect_ratios)
        H=1
        W = W_*1
        if len(DATA[datastream].shape)==1:
            W = W - margin
            
            X = X + margin/2
            
        As[datastream] = F.add_axes([X,Y,W,H])
        X=X+W_
        
        
    allTime.sort()
    maxTime = allTime[-1]
    curTime = allTime[0]
    minTime = allTime[0]
    
    while curTime < maxTime:
        showed_text = False
        all_errors = []
        for d, datastream in enumerate(obj.curDataStreams):
            As[datastream].clear()
            sample_index = np.searchsorted(TIME[datastream ], curTime)
            try:
                error = curTime - TIME[datastream][sample_index]
                
                all_errors.append(abs(error))
            except:
                error = 3 * obj.timeStep
                
                all_errors.append(abs(error))
                continue
          #  print(f'Error at time {curTime-minTime} sample index {sample_index} is {error} for {datastream}')
                
            if len(DATA[datastream].shape)>2:
                if not showed_text:
                    As[datastream].text(10,10, f'{acceleration}x playback', horizontalalignment = 'left', verticalalignment = 'top', color = 'r', fontsize = 16)
                    showed_text = True
                if error < (2* obj.timeStep):
                    if len(DATA[datastream].shape) == 4:
                        As[datastream].imshow(DATA[datastream][sample_index,...].astype(np.uint8))
                    else:
                        As[datastream].imshow(DATA[datastream][sample_index,...], vmin = mins[d], vmax=maxs[d], cmap='Greys_r')
                    
                    LA.box_off(As[datastream],All=True)
                    
                #else:
                #    As[datastream].imshow(DATA[datastream][sample_index,...]*0)
            elif len(DATA[datastream].shape)==1:
                if error < (2* obj.timeStep):
                    As[datastream].plot(TIME[datastream][0:sample_index], DATA[datastream][0:sample_index]) 
                    As[datastream].set_xlim([allTime[0], maxTime])
                    As[datastream].set_ylim([np.amin(DATA[datastream]), np.amax(DATA[datastream])])
                    LA.box_off(As[datastream])
        F.canvas.draw()
        if len(all_errors)>0:
            #print(f'Min error: {np.nanmin(all_errors)} Time step: {obj.timeStep}')
            if np.nanmin(all_errors) < obj.timeStep*2:
                snap = snapshot(F)
                writer.append_data(snap)
                print(f'Writing time point {curTime-minTime} of {maxTime-minTime}')
            else:
                print(f'Not writing time point {curTime-minTime} of {maxTime-minTime}')
        else:
            print(f'Not writing time point {curTime-minTime} of {maxTime-minTime}')
            
            
        curTime = curTime + obj.timeStep
        
    writer.close()
    
    
def simpleThreshold(obj):
    img, TIME = getSubStack(obj)
    levels = obj.dLevels[obj.dataFocus]
    minV = np.min(np.min(np.min(img)))
    maxV = np.max(np.max(np.max(img)))
    img[img<levels[0]] = minV
    img[img>levels[1]] = maxV
    return(img, TIME)

    

def prep(IMG, obj, templateIM='X'):
    return(pReg(convertUint16(pFFThighPass(medFilt2(IMG, obj),obj),obj), obj, template=templateIM))



def remove_aligned_data(obj):
    for fov in obj.DB['Animals'][obj.curAnimal].keys():
        for data in obj.DB['Animals'][obj.curAnimal][fov]:
            if 'Aligned ca' in obj.DB['Animals'][obj.curAnimal][fov][data].attrs:
                if obj.DB['Animals'][obj.curAnimal][fov][data].attrs['Aligned ca']:
                    del obj.DB['Animals'][obj.curAnimal][fov][data]
                    del obj.DB['Animals'][obj.curAnimal][fov]['R'][data]
                    del obj.DB['Animals'][obj.curAnimal][fov]['T'][data]
                    print(f'Removing data {data} from FOV {fov} in animal {obj.curAnimal}')
    obj.updateDataList()
    
def switch_ca_data_flag_to_aligned(obj):
    for fov in obj.DB['Animals'][obj.curAnimal].keys():
        for data in obj.DB['Animals'][obj.curAnimal][fov]:
            ## remove calcium data flag from previous input to alignment:
            if 'Ca data' in obj.DB['Animals'][obj.curAnimal][fov][data].attrs:
                if obj.DB['Animals'][obj.curAnimal][fov][data].attrs['Ca data']:
                    for attr in obj.DB['Animals'][obj.curAnimal][fov][data].attrs:
                        obj.DB['Animals'][obj.curAnimal][fov][data].attrs[attr] = False
                    obj.DB['Animals'][obj.curAnimal][fov][data].attrs['None'] = True
                    print(f'Removing data {data} from FOV {fov} in animal {obj.curAnimal}')
                    
            ## switch flag of aligned  data to Ca data for realignment
            if 'Aligned ca' in obj.DB['Animals'][obj.curAnimal][fov][data].attrs:
                if obj.DB['Animals'][obj.curAnimal][fov][data].attrs['Aligned ca']:
                    for attr in obj.DB['Animals'][obj.curAnimal][fov][data].attrs:
                        obj.DB['Animals'][obj.curAnimal][fov][data].attrs[attr] = False
                    obj.DB['Animals'][obj.curAnimal][fov][data].attrs['Ca data'] = True
                    print(f'Removing data {data} from FOV {fov} in animal {obj.curAnimal}')
            
              
    obj.updateDataList()
    
                    
        
                    
                
        
        
def recolor(obj):
    stack, TIME = getSubStack(obj)
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
    return(output, TIME)
    
    
def stackMax(obj): 
    stack, TIME = getSubStack(obj)
    maxIm = np.amax(stack, axis = 0)
    return(maxIm, TIME[0])

def cpu_count():
    print(mp.cpu_count())

def stackMedian(obj): 
    stack, TIME = getSubStack(obj)
    maxIm = np.median(stack, axis = 0)
    return(maxIm, TIME[0])

def stackDev(stack, obj): 
    stack, TIME = getSubStack(obj)
    maxIm = np.std(stack, axis = 0)
    return(maxIm, TIME[0])

def invert_stack(obj):
    stack, TIME = getSubStack(obj)
    return(stack*-1, TIME)
    
    
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
    

def regColor(obj):
    stack, TIME = getSubStack(obj)
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
            
    return(output, TIME)

def rotateStack(obj):
    stack, TIME = getSubStack(obj)
    output = np.rot90(stack, axes=(1,2))
    return(output, TIME)
          
def createPawMap(obj):
    stack, TIME = getSubStack(obj)
    im=imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
    im = cv2.resize(im, (534,534))
    im  = np.fliplr(im)
    im = np.rot90(im)
    im = np.expand_dims(im,0)
    output = im.repeat(stack.shape[0], 0)
    return(output, TIME)


def add_annotated_track(obj):
     stack, TIME = getSubStack(obj)
     output = TIME * 0
     return(output, TIME)
    
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
    alpha[x,y] = 255#128
    RGBAwheel = np.concatenate((wheel,alpha), axis = 2)
    #plt.imshow(RGBAwheel)

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
        
  
    
    colorref = copy.copy(overlay)
    imageio.imwrite('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color ref.tif', colorref)
    #alpha = np.expand_dims(alpha, 2)
    #RGBA = np.concatenate((overlay, alpha), axis = 2)
    pawsmono = paws[:,:,0]
    
    for c in range(4):
        overlay[:,:,c] = overlay[:,:,c] * (1-(pawsmono==255))
        if c <3:
            overlay[0:140,:,c] = pawsmono[0:140,:]
            overlay[:, 0:50,c] = pawsmono[:, 0:50]
            overlay[135:145,:,c] = 255
            overlay[:,45:55,c] = 255
        else:
            overlay[0:140,:,c] = 255
            overlay[:, 0:50,c] = 255
    overlay = overlay/255
    F = plt.figure()
    A = F.add_subplot(1,1,1)
    #plt.imshow(paws)
   
    display_im = np.swapaxes(overlay,0,1)
    A.imshow(display_im)
    imageio.imwrite('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color for display.tif', display_im)
    #plt.xlim([534,0])
    A.set_ylim([534,0])
    #A.show
    
    #pdb.set_trace()
    return(colorref)



def paw_image():
    paws =imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
    paws = cv2.resize(paws, (534,534))
    paws  = np.fliplr(paws)
    paws = np.rot90(paws)
    return(paws)

def paw_bounds(A, XY):
    X = XY[0]
    Y= XY[1]
    if X >= 245:
        A.set_xlim([245, 460])
    else:
        A.set_xlim([40, 255])
    if Y >= 260:
        A.set_ylim([280,440])
    else:
        A.set_ylim([120,280])

def get_paw_color(X,Y, ref_im = None, show=False):
    if ref_im is None:
        ref_im = imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color ref.tif')
    
    color = ref_im[Y,X][0:3]/255
    if show:
        display_im = imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color for display.tif')
        paws =imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
        paws = cv2.resize(paws, (534,534))
        paws  = np.fliplr(paws)
        paws = np.rot90(paws)
        
        F = plt.figure(f'{X=} {Y=}')
        A = F.add_subplot(1,3,1)
        A.imshow(ref_im, origin='upper')
        A.scatter(X,Y, s=50, facecolor=color, edgecolor = 'k')
        B = F.add_subplot(1,3,2)
        B.imshow(display_im, origin='upper')
        B.scatter(Y,X, s=50, facecolor=color, edgecolor = 'k')
        C = F.add_subplot(1,3,3)
        C.imshow(paws, origin='upper')
        C.scatter(X,Y, s=50, facecolor=color, edgecolor = 'k')
  
    return(color)


def mapRGB(value, colormap=plt.cm.cool, start=0, stop=255):
    clipped = np.clip(value,start,stop)-start
    clipped = clipped/(stop-start)
    
    return(colormap(clipped))
                 
    
def mechThreshold(obj, cells=None, cellSource=None, stimDataSource = None, plot = False):
    if cells is None:
        cells = obj.selectedROI
    
    if cellSource is None:
        cellSource = obj.dataFocus
        
    if stimDataSource is None:
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
    
    
    minStim = 10 ## Find stims over 10 mN
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
    
        
def z_mech_heat_map(obj, cells=None, cellSource=None, stimDataSource=None, force_limit = 500, plot=True):
    if cells is None:
        if hasattr(obj, 'selectedROI'):
            cells = obj.selectedROI
    
    if cellSource is None:
        cellSource = obj.dataFocus
        
    if stimDataSource is None:
        for datastream in obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys():
            if 'paw' in datastream:
                stimDataSource = datastream
        if stimDataSource is None:
            stimDataSource = obj.getDataDialog(prompt = "Select paw map data:")
            
    if cells is None:
        
        tracearray = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][cellSource]['traceArray'][...]
        cells = []
        for n, t in enumerate(tracearray):
            cells.append(n)
    
    var_dict = {}
        
    #pdb.set_trace()
    stimTraces = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][stimDataSource]['traceArray'][...]
    stimMasks = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][stimDataSource]['floatMask']
    ## get cell and stim traces on ssame timebase:
    calciumTime = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][cellSource]
    stimTime = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][stimDataSource]
    astims = alignTimeScales(calciumTime, stimTraces, stimTime)
    
    ## eliminate duplicates :
    unique_aligned_stim, u_indices = np.unique(astims, axis = 0, return_index = True)
    
    #maxStim = np.amax(unique_aligned_stim)
    maxStim = force_limit ## make 500mM maximum stim intensity
    
    #indices.sort()
 #   unique_masks = stimMasks[:,:,u_indices] # select masks corresponding to unique stims
    unique_masks = np.zeros([stimMasks.shape[0],stimMasks.shape[1],len(u_indices)])
    for un, u in enumerate(u_indices):
        unique_masks[:,:,un] = stimMasks[:,:,u]
    #unique_masks = 
    ##sort stims by start times
    start_times = np.zeros(u_indices.shape)
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
    
    stim_diam =  10
    
    im=imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
    im = cv2.resize(im, (534,534))
    im  = np.fliplr(im)
    im = np.rot90(im)
    pawMap = im #createPawMap(np.zeros([1,1,1]), None)[0,...]
    
    stimCenters = []
    # show all stim:
    if plot:
        F_all_stim = plt.figure('Location and intensity of all stimuli')
        A_all_stim = F_all_stim.add_subplot(1,1,1)
        A_all_stim.imshow(im)
    norm_stims = []
    for ii in range(sorted_unique_masks.shape[-1]):
        omask = sorted_unique_masks[:,:,ii].astype(np.uint8)
        M = cv2.moments(omask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])  ## getting centroids of mask for stim position
        stimCenters.append((cx,cy))
        
        #ax.add_patch(stimpatch)
        stim = sorted_unique_aligned_stim[ii,...]
        normStim = int(255*np.amax(stim)/maxStim)
        if normStim > 255:
            normStim = 255
        norm_stims.append(normStim)
        
        if plot:
            patch_color = plt.cm.cool(normStim, alpha = 0.2)
            edge_color = plt.cm.cool(normStim, alpha = 0.5)
            map_patch = mat.patches.Ellipse((cx,cy), stim_diam, stim_diam, facecolor = patch_color, edgecolor=edge_color, linewidth=1.5)
            A_all_stim.add_patch(map_patch)
            A_all_stim_colorbar  = F_all_stim.add_subplot(8,4,32)
        #cbar = color_bar(plt.cm.cool(), A = A_all_stim_colorbar, F=F_all_stim, unit = 'F (mN)', min_v = 0, max_v = maxStim)
    
    ref_color_im = imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color ref.tif')
    response_points = {}
    somato_colors = {}
    convex_hulls = {}
    responses = {}
    Fs = {}
    for cCount, cell in enumerate(cells):
        cellTrace = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][cellSource]['traceArray'][cell,:]

        maxstim = np.amax(sorted_unique_aligned_stim)
        maxResponse = np.amax(cellTrace)
        #pawMap = np.squeeze(createPawMap(np.zeros([1,1,1]), None))
        mechMap = np.zeros(pawMap.shape)
    
        stimMags = np.zeros(sorted_unique_aligned_stim.shape[0])
        responseMags = np.zeros(sorted_unique_aligned_stim.shape[0])

        if plot:
            Fs[cCount] = plt.figure(str(cCount))
            F = Fs[cCount]
            ax_paw = F.add_subplot(1,2,1)
            plt.imshow(pawMap)
        
        max_response = 30
        min_response = 3.5
        response_points[cCount] = []
        responses[cCount] = {}
        responses[cCount]['stims'] = {}
        responses[cCount]['traces'] = {}
        responses[cCount]['response_points'] = []
        responses[cCount]['somatocolors'] = {}
        responses[cCount]['stim_mask_id'] = {}
        
        #somato_colors[cCount] = []
        response_counter = 0
        for c, stim in enumerate(sorted_unique_aligned_stim):
            response = np.zeros(stim.shape)
            #print(f'stim shape: {stim.shape}')
            if np.amax(stim) <= 0:
                continue
            stimstart = np.where(stim>0)[0][0]
            stimendOriginal = np.where(stim>0)[0][-1]  ## actual end of stim
            if c < sorted_unique_aligned_stim.shape[0]-1:
                stimend = np.where(sorted_unique_aligned_stim[c+1]>0)[0][0]-1 # extend stim period to start of next stim
            else:
                stimend = sorted_unique_aligned_stim.shape[1]-1  
            delay = 20 #TODO: get timestep to make this # correspond to ~1 second
            if stimend-stimendOriginal > delay: 
                stimend = stimendOriginal + delay
            response[stimstart:stimend] = cellTrace[stimstart:stimend]
            if response.size == 0:
                print('zero size response')
                continue
            stimMags[c] = np.amax(stim)
            # if c == 1:
            #     show=True
            # else:
            #     show = False
            show=False
            somato_color = get_paw_color(stimCenters[c][0], stimCenters[c][1], ref_im = ref_color_im, show=show)
            
            normStim = int(255*np.amax(stim)/maxStim)
            if normStim > 255:
                normStim = 255
            responseMags[c] = np.amax(response)
            xresponse= response - min_response
            normResponse = np.amax(xresponse)/(maxResponse-min_response)
            if normResponse>1:
                normResponse = 1
            if normResponse < 0:
                normResponse = 0
            patch_color = plt.cm.cool(normStim, alpha = normResponse)
            
            patch_color = [somato_color[0], somato_color[1], somato_color[2], np.ceil(normResponse) ]
            if normResponse <= 0:
                edge_color = (1,1,1,0.5)
            else:
                #somato_colors[cCount].append(somato_color)
                #responses[cCount]['stims'].append(stim)
                #responses[cCount]['traces'].append(response)
                edge_color = plt.cm.cool(normStim, alpha = 0.5)
                edge_color=patch_color
                edge_color = 'w'
                response_points[cCount].append(stimCenters[c])
                responses[cCount]['stims'][response_counter] = stim
                responses[cCount]['traces'][response_counter] = response
                responses[cCount]['response_points'].append(stimCenters[c])
                responses[cCount]['somatocolors'][response_counter] = somato_color
                response_counter = response_counter+1
                
                
            print(f'{patch_color=}')
            map_patch = mat.patches.Ellipse(stimCenters[c], stim_diam, stim_diam, edgecolor=edge_color, facecolor = patch_color)
            
            if plot:
                ax_paw.add_patch(map_patch)
         
        
        n_responses = len(responses[cCount]['traces'])
        if plot:
            ax_s = F.add_subplot(2,2,2)
            ax_r = F.add_subplot(2,2,4, sharex  = ax_s)
            ax_r.plot(cellTrace,'k', alpha = 0.2)                     
        for rCount, (stim, response) in enumerate(zip(responses[cCount]['stims'].values(), responses[cCount]['traces'].values())):
            #ax_s = F.add_subplot(n_responses,2,(rCount*2)+2)
            
            #ax_r = ax_s.twinx()
            if plot:
                ax_s.plot(stim, color=responses[cCount]['somatocolors'][rCount])
                
                ax_r.plot(response, 'r')
                ax_s.set_ylim([0, maxStim])
                ax_r.set_ylim([-3, maxResponse])
        
 #       var_dict['response_points'] = response_points
#        LA.dump(locals())
        
        centroid, somato_color = get_rf_centroids(response_points[cCount], show=False)
        responses[cCount]['somato_color'] = somato_color
  
        if len(response_points[cCount])>2:
            hull = ConvexHull(response_points[cCount])
            p = hull.points
            v = np.append(hull.vertices, hull.vertices[0])
            if plot:
                ax_paw.plot(p[v][:,0], p[v][:,1], color=somato_color, linewidth = 0.5)
        if plot:
            LA.box_off(ax_paw, All=True)
            LA.box_off(ax_s, left_only=True)
            LA.box_off(ax_r)
            ax_r.set_xticks(ax_r.get_xticks())
            ax_r.set_xticklabels([str(int(x)/10) for x in ax_r.get_xticks()])
            ax_r.set_xlabel('Time (s)')
            ax_r.set_ylabel('Fz')
            ax_s.set_ylabel('Force (mN)')
            title = f'Paw map {obj.curAnimal} {obj.curFOV} {cell=}'
            LA.save_fig(F, title)
            plt.close(F)
        
    FOV = obj.DB['Animals'][obj.curAnimal][obj.curFOV]
    somatotopic_map(obj, cells=cells, cellSource=cellSource, responses = responses)
    for F in Fs:
        plt.close(F)
    return(responses)
    #var_dict['response_points'] = response_points
    #LA.dump(var_dict)
    
    
    #LA.dump(locals())
    #print('Dumped log to pickle')

def somatotopic_map(obj, cells=None, cellSource=None, responses=None, mode='circles'):
    ## For input need to get image of field, selected ROIs, and somatotopic distrivution of responses for each roi
    ROIs = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][cellSource]['floatMask'][:,:,cells]
    field_image= np.median(obj.DB['Animals'][obj.curAnimal][obj.curFOV][cellSource][0:100,...], axis = 0)
    
    F = plt.figure('Somatotopic map')
    A_field = F.add_subplot(2,1,1)
    A_field.imshow(field_image.T, cmap = 'Greys', vmin = 0.9, vmax = 1.1)
    
    display_im = imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color for display.tif')
    paws = paw_image()
    ref_color_im = imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw color ref.tif')
    ROI_centroids = {}
    for cCount, cell in enumerate(cells):
        #pdb.set_trace()
        mask = ROIs[:,:,cell]
        mask = (mask/np.amax(mask))*255
        omask = mask.astype(np.uint8)
        
        M = cv2.moments(omask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])  ## getting centroids of mask for stim position
        ROI_centroids[cCount] = (cx,cy)
        if mode =='circles':
            A_field.scatter(cy,cx, color = responses[cCount]['somato_color'], s=100 )
        elif mode == 'rois':
            pass
        
        A_paw = F.add_subplot(2,len(cells), len(cells) + cCount+1)
        A_paw.imshow(paws)
        X = [r[0] for r in responses[cCount]['response_points']]
        Y = [r[1] for r in responses[cCount]['response_points']]
        Xc = int(np.mean(X))
        Yc = int(np.mean(Y))
        for x,y in zip(X,Y):
            color = get_paw_color(x,y, ref_im = ref_color_im, show=False)
            A_paw.scatter(x,y, edgecolor = color, facecolor = color, s=5)
            
            A_paw.scatter(Xc,Yc, edgecolor = 'k', facecolor = responses[cCount]['somato_color'], marker='x')
            paw_bounds(A_paw, (Xc,Yc))
            LA.box_off(A_paw, All=True)
        LA.box_off(A_field, All=True)
    title = f'Sommatotopic map {obj.curAnimal} {obj.curFOV}'
    LA.save_fig(F, title)
    G = plt.figure()
    B = G.add_subplot(1,1,1)
    B.imshow(display_im)
    LA.box_off(B, All=True)
    LA.save_fig(G, 'Paw color code')
    
    
    
    
    
    
    
def ts(cells=None, obj=None, animal=None, FOV=None, pawdata=None, DBpath=None, plot_all = True):
    if obj is None:
        if DBpath is None:
            DBpath = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7241/7241A mech.h5'
            animal = '7241'
            FOV = 'LA 2022-7-08 mech caudal shift'
            cellSource = 'calcium'
            stimDataSource = None
        obj = dbContainer(DBpath)
        obj.curFOV = FOV
        obj.curAnimal = animal
        obj.dataFocus = cellSource
        
    z_mech_heat_map(obj, cells=cells, cellSource=cellSource, stimDataSource=stimDataSource, force_limit = 500, plot=plot_all)
        
    
def get_rf_centroids(response_list, ref_color_im = None, show=False, A = None):
    
    centroid = np.mean(response_list)
    X = [r[0] for r in response_list]
    Y = [r[1] for r in response_list]
    Xc = int(np.mean(X))
    Yc = int(np.mean(Y))
    centroid_color = get_paw_color(Xc,Yc, ref_im = ref_color_im, show = show)
    if show:
        for x,y in zip(X,Y):
            color = get_paw_color(x,y, ref_im = ref_color_im, show=False)
            plt.scatter(x,y, color = color)
    return(centroid, centroid_color)
        
def mechanoHeatMap(obj, cells=None, cellSource=None, stimDataSource=None, force_limit=500, cmap = plt.cm.cool(255)): ##cell = ROI # of cell, cellSource = key of calcium imaging data
    #Plot stim/ response relationship for a cell
    #
    if cells is None:
        cells = obj.selectedROI
    
    if cellSource is None:
        cellSource = obj.dataFocus
        
    if stimDataSource is None:
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
    
    #maxStim = np.amax(unique_aligned_stim)
    maxStim = force_limit ## make 500mM maximum stim intensity
                      
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
    
    im=imageio.imread('/lab-share/Neuro-Woolf-e2/Public/DavidY/Paw chart v1.tif')
    im = cv2.resize(im, (534,534))
    im  = np.fliplr(im)
    im = np.rot90(im)
    pawMap = im #createPawMap(np.zeros([1,1,1]), None)[0,...]
    
    
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
        #pawMap = np.squeeze(createPawMap(np.zeros([1,1,1]), None))
        mechMap = np.zeros(pawMap.shape)
    
        stimMags = np.zeros(sorted_unique_aligned_stim.shape[0])
        responseMags = np.zeros(sorted_unique_aligned_stim.shape[0])


        fig, ax = plt.subplots()
        plt.imshow(pawMap)
        
        maxResponse = np.amax(cellTrace)
        for c, stim in enumerate(sorted_unique_aligned_stim):
            response = np.zeros(stim.shape)
            #print(f'stim shape: {stim.shape}')
            if np.amax(stim) <= 0:
                continue
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
            normStim = int(255*np.amax(stim)/maxStim)
            responseMags[c] = np.amax(response)
            normResponse = np.amax(response)/maxResponse
            patch_color = plt.cm.Reds(255-normStim, alpha = normResponse)
            map_patch = mat.patches.Ellipse(stimCenters[c], stimDiam, stimDiam, color = patch_color)
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
    
    
def makeGrayscale(obj):
    stack, TIME = getSubStack(obj)
    return(np.mean(stack, axis = 3), TIME)

def dynamicOverlay(stack, obj):
    pass


def inverseCorMap(obj):
    cMap, TIME = corMap(obj)
    return(cMap*-1, TIME)

def color_merge(obj, vmin = 0.0, vmax = 1, save=True, scale_bar = 100):
    vmin = obj.segmentationMethods['Merge']['Params']['vmin'][1]/100
    vmax = obj.segmentationMethods['Merge']['Params']['vmax'][1]/100
    #scaling = 1
    frames= []
    colors = []
    Fname = 'Merge_' + obj.curAnimal + obj.curFOV
    for s, source in enumerate(obj.DataList.selectedItems()):
        dKey = source.text()
        if len(obj.DB['Animals'][obj.curAnimal][obj.curFOV][dKey].shape) == 2:
            Fname  = Fname + dKey
            frame = obj.DB['Animals'][obj.curAnimal][obj.curFOV][dKey][...]
            frame[np.where(frame<vmin)] = 0
            
            frame = frame/vmax
            frame[np.where(frame>1)] = 1
            #frame = frame*scaling
            frames.append(frame)
            C = QColorDialog.getColor(title = f'Pick color for {dKey}')
       
            colors.append([C.red()/255, C.green()/255, C.blue()/255])
    output = np.zeros([frames[0].shape[0], frames[0].shape[1], len(colors[0])])
    for c, (frame, colorset) in enumerate(zip(frames, colors)):
        for cc, color in enumerate(colorset):
            output[:,:, cc] = output[:,:,cc] + (frame * color)
    output[np.where(output>1)] = 1
            
    F = plt.figure()
    A = F.add_axes([0,0,1,1])
    A.imshow(output)
    A.xaxis.set_visible(False)
    A.yaxis.set_visible(False)
    
    if not scale_bar is None:
        plot_scale_bar(A, scale_bar)
        
        
    #pdb.set_trace()m
    F.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' +  Fname + '.tif')
    F.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' +  Fname + '.pdf')
    return(output)
            

    

def corMap(obj):
        stack, TIME = getSubStack(obj)
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
        print(TIME.shape)
        print(TIME)
        print(TIME[0])
        print(TIME[0].shape)
        return(cMap, TIME[0])
        
def segmentMechTrace(forceData, thresh=None, interval=None, prom=None, base=None):
    if thresh is None:
        thresh = 20
    if interval is None:
        interval = 10
    if prom is None:
        prom = 10
    if base is None:
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
        
# def corrImSig(im,sig,imTime,sigTime):
#     # Convert time base of signal for correlation to correspond to image series:
#     regSig = np.zeros(imTime.shape[0])
#     for ii in range(0,imTime.shape[0]):
#         sampleIndex = np.searchsorted(sigTime,imTime[ii])
#         while sampleIndex >= sig.shape[0]:
#             sampleIndex=sampleIndex-1
#         print(sampleIndex)
#         regSig[ii] = sig[sampleIndex]
#     plt.plot(sigTime)
#     #plt.plot(imTime)

   
#     print(regSig.shape)
    
    
#     output = np.zeros([1,im.shape[1],im.shape[2]])
#     for ii in range(0,im.shape[1]):
#         for jj in range(0,im.shape[2]):
#             output[0,ii,jj] = np.corrcoef(im[:,ii,jj],regSig)[0,1]
#         print(ii)
#     #output = output - np.min(np.min(output))
#     #output = output/np.max(np.max(output))
#     output = np.nan_to_num(output)
#     #output = np.uint16(output*2**16)
#     #output = output.repeat(im.shape[0], 0)
#     output = np.squeeze(output)
#     return(output)  
    


def copyData(obj):
    stack, TIME = getSubStack(obj)
    return(stack, TIME)

def convertTempToC(obj):
    stack, TIME = getSubStack(obj)
    return(stack/100, TIME)

def dff(obj):
    stack, TIME = getSubStack(obj)
    Fnought  =  np.median(stack, axis = 0)
    DeltaF   =  np.float32(stack) - np.float32(Fnought)
    return (np.float32(DeltaF/Fnought), TIME)



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
    
def selectFile(existing=True, message = None):
    if existing:
        if message is None:
            message = "Choose file..."
        result = QFileDialog.getOpenFileName(None, message, "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
    else:
        if message is None:
            message = "Save as..."
        result = QFileDialog.getSaveFileName(None, message, "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
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



def removeSelected(obj): #returns selected data to be removed, also creates a trimmed version of data without selected
    stack, TIME = getSubStack(obj)

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
    
    
    return(stack, TIME)
    

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
    
# def batchRegister(FOVfolder):
#     CERNAfiles = makeFileList(FOVfolder, '*CERNAbgcorrected.tif*')
#     firstStack = imageio.volread(CERNAfiles[1])
#     first10 = firstStack[0:9,:,:]
#     first10reg = regStackFirstFrame(first10)
#     refImage = np.mean(first10reg, axis=0)
    
#     sr=StackReg(StackReg.RIGID_BODY)
#     for Cfile in CERNAfiles:
#         imdata = imageio.volread(Cfile)
#         output = np.zeros(imdata.shape, dtype = np.float32)
#         for ii in range(imdata.shape[0]):
#             output[ii,:,:] = sr.register_transform(refImage,imdata[ii,:,:])
#             print(ii)
            
#         savename = os.path.join(os.path.dirname(Cfile),'CERNAbgcorrected_registered.tif')

#         imageio.volwrite(savename, output, bigtiff=True)
#         print('Corrected file saved'+ savename)    

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



def depositTrial2(obj, trialFolder=None, data_filter=None):  
    A=obj.curAnimal
    F=obj.curFOV
    
    ## Input dictionary defines what file names to look for and
    ## function to read and process for deposit
    
    inputDict = {}
    
    if trialFolder is None:
        trialFolder = selectFolder(msg = 'Select trial to add from...')
    
    if not data_filter is None:
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
            
            if error:
                print(f'Error encountered trying to parse data in {dataPath}')
                continue
            
            TIME = timedata + trialStartTime            
        else:
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
        if 'link_method' in inputDict[input_type].keys():
            link = inputDict[input_type]['link_method'](dataPath, timePath)
            obj.DB['Animals'][A][F][dName].attrs['Link'] = link
        if 'ROI_handler' in inputDict[input_type]:
            inputDict[input_type]['ROI_handler'](obj = obj, A=A, F=F,dName = dName, dataPath = dataPath, timepath=timePath, DATA=DATA, TIME = TIME, trialStartTime = trialStartTime)
            
            
            
    print('Deposit finished')

def genericDepositTrial(obj, DATA, TIME, dName, A=None, F=None, DBpath = None): 
    print('Starting genericDepositTrial')
    if A is None:
        A=obj.curAnimal 
    if F is None:
        F=obj.curFOV
    if obj is None:
        if not (DBpath is None):
            obj = dbContainer(DBpath)
        else:
            print('No DB provided for deposit')
            return
    
    print(f'Checking if {dName} exists in {F}')
    if obj.DB['Animals'][A][F].__contains__(dName): #Check if data stream already exists for FOV
            print(f'Appending to dataset {dName}in FOV {F}...')
            #resize time scale 
            obj.DB['Animals'][A][F]['T'][dName].resize((obj.DB['Animals'][A][F]['T'][dName].shape[0]+TIME.shape[0]), axis = 0)
            obj.DB['Animals'][A][F]['T'][dName][-TIME.shape[0]:]=TIME
            #Resize dataset and add new data:
            obj.DB['Animals'][A][F][dName].resize((obj.DB['Animals'][A][F][dName].shape[0]+DATA.shape[0]),axis = 0)
            obj.DB['Animals'][A][F][dName][-DATA.shape[0]:]=DATA
    else: ## If data stream doesn't exist, create new:
            print(f'Creating dataset {dName} in FOV {F}')
            if len(DATA.shape) == 2 or len(TIME.shape) == 0:
                maxShape = DATA.shape
                chunkShape = DATA.shape
                obj.DB['Animals'][A][F]['T'].require_dataset(dName, maxshape = TIME.shape, shape = TIME.shape, data = TIME, dtype = TIME.dtype, track_order = True)
                obj.DB['Animals'][A][F].require_dataset(dName, shape =  DATA.shape, maxshape = maxShape, data = DATA, dtype = DATA.dtype, track_order = True, chunks=chunkShape)

            else:
                maxShape = (None,) + DATA.shape[1:]
                print(f'maxShape {maxShape}')
                obj.DB['Animals'][A][F]['T'].require_dataset(dName, maxshape = (None,), shape = TIME.shape, data = TIME, dtype=TIME.dtype, track_order = True)
                chunkShape = (1,) + DATA.shape[1:]
                obj.DB['Animals'][A][F].require_dataset(dName, shape =  DATA.shape, maxshape = maxShape, data = DATA, dtype = DATA.dtype, track_order = True, chunks=chunkShape)
            
            #Create ROI mask and trace groups:
            emptyArray = np.array([])
            obj.DB['Animals'][A][F]['R'].require_group(dName)
            #obj.DB['Animals'][A][F]['R'].require_dataset('floatMask', data = emptyArray)
            #obj.DB['Animals'][A][F]['R'].require_dataset('traceArray', data = emptyArray)
            
            
            
    print(f'Data {dName} deposited to FOV {F}')
    
    
def read_annotation(datapath, timepath, resolution = 0.1):
    print(timepath)
    timedata = np.genfromtxt(timepath, delimiter = ',')
    
    if timedata.shape[0] < 2:
        return(np.array([]), np.array([]), True)
    elif timedata.shape[0] == 2:
        timedata = np.array([timedata[0]])
    else:
        timedata = timedata[:,0]
    #pdb.set_trace()
    if timedata.shape[0] == 1:
        TIME = timedata
    else:
        print(f'{timedata[0]=} {timedata[-1]=}')
        TIME = np.arange(0,timedata[-1]+5, resolution)
    DATA = np.zeros([len(TIME),256,256])
    return(DATA, TIME, False)



def note_ROI_handler(obj = None, A=None, F=None,dName = None, dataPath = None, timepath=None, DATA=None, TIME = None, trialStartTime=None):
    
    
    #print(f'{dataPath=}')
    #print(f'{timepath=}')
    TIME=TIME-trialStartTime
    with open(dataPath, 'r') as File:
        S = File.read()
        S = S.replace('\n', '')
        annotation = S.split(',')[:-1]
    
    #print(f'{annotation=}')
    H = DATA.shape[1]
    W = DATA.shape[2]
    o_time = np.genfromtxt(timepath, delimiter = ',')
    if o_time.shape[0] == 2:
        o_time = np.array([o_time[0]])
    else:
        o_time = o_time[:,0]
    trace_array = np.zeros([len(annotation), len(TIME)])
    float_mask  = np.zeros([H, W, len(annotation)])
    fs = int(H/(len(annotation)*2))
    font = ImageFont.truetype('Lato-Black.ttf', fs)
    
    for c, (note, o) in enumerate(zip(annotation, o_time)):
        index = np.searchsorted(TIME, o)
        trace = TIME*0
        trace[index] = 1
        
        #trace[index+1] = 1
        trace_array[c,:] = copy.deepcopy(trace)
        
        #print(f'{note=}')
        text_bmap =font.getmask(note)
        w = text_bmap.size[0]
        h = text_bmap.size[1]
        
        text_mask = np.reshape(text_bmap,[h,w])/255
        
        float_mask[c*fs:(c*fs)+h,:w,c] = text_mask[:,:W]
        DATA[index,...] = float_mask[:,:,c]
        
        
    float_mask = np.swapaxes(float_mask, 0,1)
    DATA = np.swapaxes(DATA,1,2)
    
    FF = plt.figure()
    a1 = FF.add_subplot(1,2,1)
    a2 = FF.add_subplot(1,2,2)
    a1.imshow(np.max(float_mask, axis=2))
    a2.plot(trace_array.T)
    
    obj.DB['Animals'][A][F][dName][...] = DATA
    obj.updateROIdata(float_mask, trace_array, names = annotation, animal=A, FOV=F, datakey = dName, updateGUI=False)
    # obj.DB['Animals'][A][F]['R'][dName].require_dataset('traceArray', shape = trace_array.shape, dtype=trace_array.dtype, data = trace_array)
    # obj.DB['Animals'][A][F]['R'][dName].require_dataset('floatMask', shape = float_mask.shape, dtype = float_mask.dtype, data =float_mask)
    # print(f'{A=}')
    # print(f'{F=}')
    # print(f'{dName=}')
   
    # if not ('names' in obj.DB['Animals'][A][F]['R'][dName].keys()):
    #     dt = h5py.string_dtype(encoding = 'utf-8')
    #     nROIs= float_mask.shape[2]
       
    #     N = obj.DB['Animals'][A][F]['R'][dName].require_dataset('names',shape=(nROIs,1), dtype = dt)
    
    #     for r, n in enumerate(annotation):
    #         N[r,0] = n
 
   
    
def parse_eVF(obj = None, time_limit = 500):
    codes = {}
    codes['i'] = {}
    codes['c'] = {}
    codes['r'] = {}
    codes['n'] = {}
    codes['a'] = {}
    codes['b'] = {}
    codes['i']['name'] = 'ipsi'
    codes['c']['name'] = 'contra'
    codes['r']['name'] = 'response'
    codes['n']['name'] = 'no response'
    codes['a']['name'] = 'aborted'
    codes['b']['name'] = 'bite'
    codes['i']['type'] = 'laterality'
    codes['c']['type'] = 'laterality'
    codes['r']['type'] = 'outcome'
    codes['n']['type'] = 'outcome'
    codes['a']['type'] = 'outcome'
    codes['b']['type'] = 'outcome'
    codes['i']['color'] = 'r'
    codes['c']['color'] = 'b'
    codes['r']['color'] = 'r'
    codes['n']['color'] = 'g'
    codes['a']['color'] = [0,0,0,0]
    codes['b']['color'] = 'b'
    codes['r']['marker'] = 'o'
    codes['n']['marker'] = 's'
    codes['a']['marker'] = ''
    codes['b']['marker'] = 'x'
    codes['i']['order'] = 0
    codes['c']['order'] = 1
    
    
    results = {}
    
    vf, vf_time = getSubStack(obj)
    
    F = obj.DB['Animals'][obj.curAnimal][obj.curFOV]
    
    dataset = {}
    animals = {}
    if 'key_data' in F:
        ta = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R']['key_data']['traceArray'][...]
        names = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R']['key_data']['names'][...]
        key_time = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T']['key_data']
    #Get mouse identifiers:
    
    time_str = time.localtime(key_time[0])
    month = str(time_str[1])
    day = str(time_str[2])
    year = str(time_str[0])
    date_str = f'{month}-{day}-{year}'
    #F = plt.figure('Response array')
    #A = F.add_subplot(1,1,1)
    #A.imshow(ta, aspect = 'auto')
    for c, name in enumerate(names):
        print(f'{name.shape=}')
        name = name[0]
        dataset[name] = {}
        dataset[name]['trace'] = ta[c,:]
        if name.isnumeric():
            animals[name] = {}
            results[name] = {}
            codes[name] = {}
            codes[name]['name'] = name
            codes[name]['type'] = 'animal'
     
    with open('keyfile.txt', 'w') as keyfile:
        keyfile.write('Starting...')
        
    for a, animal in enumerate(animals):
        for t, is_animal in enumerate(dataset[animal]['trace']):
            # if key_time[t] > vf_time[0] or key_time[t] > vf_time[-1]:
            #     continue
            tt = copy.deepcopy(t)
            
            if is_animal:
                with open('keyfile.txt', 'a') as keyfile:
                    keyfile.write(animal)
                print(f'Animal {animal} keyed at point {t}, keytime = {key_time[t]-key_time[0]}')
                
               # ta_after = ta[:,t:]
                next_event = None
                curOutcome = None
                
                curLaterality = None
                stop = False
                while not stop:
                    #print(f'{tt=}')
                    if tt >= ta.shape[1]:
                        curOutcome = None
                        stop = True
                    if tt-t > time_limit:
                        stop = True
                    for n, name in enumerate(names):
                        name = name[0]
                        if ta[n, tt]:
                            with open('keyfile.txt', 'a') as keyfile:
                                keyfile.write(name)
                            last_event = name
                            print(f'Key {name} entered at point {tt} after mouse {animal} keyed in at {t}')
                            if codes[name]['type'] == 'laterality':
                                print(f'Paw {name} keyed for mouse {animal} at time {tt}')
                                curLaterality = name
                      
                            elif codes[name]['type'] == 'outcome':
                                print(f'Response {name} keyed at time {tt}, mouse is {animal}, laterality is {curLaterality}')
                                curOutcome = name
                                stop = True
                        
                            elif codes[name]['type'] == 'animal':
                                if name == animal:
                                    pass
                                else:
                                    print(f'New animal {name} keyed')
                                    curOutcome = None
                                    stop = True
                        if not curLaterality is None and not curOutcome is None:
                            if not (curLaterality in results[animal]):
                                results[animal][curLaterality] = {}
                            if not (curOutcome in results[animal][curLaterality]):
                                    results[animal][curLaterality][curOutcome] = {}
                                    results[animal][curLaterality][curOutcome]['traces'] = []
                                    results[animal][curLaterality][curOutcome]['timepoints'] = []
                                    results[animal][curLaterality][curOutcome]['peaks'] = []
                            start_time = key_time[t]
                            end_time = key_time[tt]
                            vf_start = np.searchsorted(vf_time, start_time)
                            vf_end   = np.searchsorted(vf_time, end_time)
                            print(f'{time.localtime(vf_time[0])=}')
                            print(f'{time.localtime(vf_time[-1])=}')
                            print(f'{time.localtime(start_time)=}')
                            print(f'{time.localtime(end_time)=}')
                            print(f'{vf_start=}')
                            print(f'{vf_end=}')
                            if (vf_end-vf_start)>0:
                                
                                trace = vf[vf_start:vf_end]
                                timepoints = vf_time[vf_start:vf_end]
                                results[animal][curLaterality][curOutcome]['traces'].append(trace)
                                results[animal][curLaterality][curOutcome]['timepoints'].append(timepoints)
                                results[animal][curLaterality][curOutcome]['peaks'].append(np.amax(trace))
                                break
                        if stop:
                            print(f'Will halt at time point {tt}, for mouse {animal} keyed at time {t}, last event = {last_event}')
                    # print(f'{t=} before')  
                    # print(f'{tt=} before')  
                    tt = tt + 1 
                    # print(f'{t=} after')  
                    # print(f'{tt=} after')  
                            
    n_animals = len(results)
    fig_title = f'{obj.dataFocus} {date_str} {LA.uid_string()}'
    F = plt.figure(fig_title)
    
    for a, animal in enumerate(results):
        
        A1 = F.add_subplot(2,n_animals,a+1)
        A2 = F.add_subplot(2,n_animals,a+1+n_animals)
        A1.set_title(f'Mouse {animal}')
        xticks = [x for x in range(len(results[animal]))]
        xlabels = [x for x in range(len(results[animal]))]
        for s, side in enumerate(results[animal]):
            xval = codes[side]['order']
            side_color = codes[side]['color']
            xticks[codes[side]['order']] = codes[side]['order']
            xlabels[codes[side]['order']] = codes[side]['name']
            for outcome in results[animal][side]:
                outcome_color = codes[outcome]['color']
                outcome_marker = codes[outcome]['marker']
                R = results[animal][side][outcome]['traces']
                T = results[animal][side][outcome]['timepoints']
                P = results[animal][side][outcome]['peaks']
                
                for r, t, p  in zip(R,T,P):
                    print(f'{p=}')
                    A1.plot(t,r, color=side_color)
                    A1.plot([t[0],t[-1]],[p, p], color=outcome_color )
                if len(P)>1:
                    beeswarm(P, codes[side]['order'], width = 0.125, A=A2, color = outcome_color, alpha = 0.25, marker = outcome_marker)
                elif len(P)==1:
                    A2.scatter(codes[side]['order'], P[0], color= side_color, alpha = 0.25, marker = outcome_marker)
                A2.plot([xval-0.25,xval+0.25],[np.median(P), np.median(P)], color = outcome_color)
        A2.set_ylim([0,100])    
        A2.set_xticks(xticks)
        A2.set_xticklabels(xlabels)
        if a == 0:
            A2.set_ylabel('Force (mN)')
            
            LA.box_off(A2)
        else:
            LA.box_off(A2, bot_only = True)
        LA.box_off(A1, left_only=True)
        
    
    LA.save_fig(F, fig_title)
    
    picklefile = open('temp.pickle', 'wb')
    pickle.dump([results, dataset], picklefile)
    
    picklefile.close()
    obj.DB.close()
    
    
def key_ROI_handler(obj = None, A=None, F=None, dName = None, dataPath = None, timepath=None, DATA=None, TIME = None, trialStartTime=None):
    
    
    offset = 0
    if 'traceArray' in obj.DB['Animals'][A][F]['R'][dName]:
        offset = obj.DB['Animals'][A][F]['R'][dName]['traceArray'].shape[0]
        
    TIME = TIME - trialStartTime
    with open(dataPath, 'r') as File:
        S = File.read()
        S = S.replace('\n', '')
        annotation = S.split(',')[:-1]
    
    H = DATA.shape[1]
    W = DATA.shape[2]
    
    nr_keys = []
    traces = {}
    for key in annotation:
        if key == '':
            continue
        if key not in nr_keys:
            nr_keys.append(key)
            traces[key] = copy.deepcopy(TIME * 0)
    
    
    o_time = np.genfromtxt(timepath, delimiter = ',')
    if o_time.shape[0] == 2:
        o_time = np.array([o_time[0]])
    else:
        o_time = o_time[:,0]
    trace_array = np.zeros([len(nr_keys), len(TIME)])
    float_mask  = np.zeros([H, W, len(nr_keys)])
    fs = int(H/(len(nr_keys)+offset))
    font = ImageFont.truetype('Lato-Black.ttf', fs)
    
    key_ids = {}

    for c, nr_key in enumerate(nr_keys): 
        key_ids[nr_key]=c
        text_bmap =font.getmask(nr_key)
        w = text_bmap.size[0]
        h = text_bmap.size[1]
        text_mask = np.reshape(text_bmap,[h,w])/255
        float_mask[c*fs:(c*fs)+h,:w,c] = text_mask[:,:W]

    float_mask = np.swapaxes(float_mask, 0,1)
    
    for c, (key, o) in enumerate(zip(annotation, o_time)):
        if key == '':
            continue
        index = np.searchsorted(TIME, o)
        print(f'{key=} {o=} {index=}')
        traces[key][index] = 1
        #traces[key][index+1] = 1
        
        DATA[index,...] = float_mask[:,:,key_ids[key]]
    
    
    for c, nr_key in enumerate(nr_keys): 
        trace_array[c,:] = copy.deepcopy(traces[nr_key])
    
    Figure = plt.figure(dataPath)
    A1 = Figure.add_subplot(2,2,1)
    A2 = Figure.add_subplot(2,2,2)
    A3 = Figure.add_subplot(2,1,2)
    A1.imshow(np.amax(float_mask, axis=2))
    A2.imshow(trace_array, aspect='auto')
    A3.text(0,0, ', '.join(nr_keys))
    
    
    if 'names' in obj.DB['Animals'][A][F]['R'][dName]:
        new_names = []
        old_names = []
        
        old_traces = {}
        old_masks = {}
        ## go through new keys, check if in existing dataset:
        existing_names = obj.DB['Animals'][A][F]['R'][dName]['names'][...]
        existing_trace_array = obj.DB['Animals'][A][F]['R'][dName]['traceArray'][...]
        existing_float_mask = obj.DB['Animals'][A][F]['R'][dName]['floatMask'][...]
        
        for n, name in enumerate(existing_names):
            name = name[0]
            old_names.append(name)
            old_traces[name] = existing_trace_array[n,:]
            old_masks[name] = existing_float_mask[:,:,n]
            
           
        for name in nr_keys:
            if not (name in old_names):
                new_names.append(name)
        all_names = copy.deepcopy(old_names)
        all_names.extend(new_names)
        
        
        e_width = existing_trace_array.shape[1]
        n_width = trace_array.shape[1]
        combined_trace_array = np.zeros([len(all_names), existing_trace_array.shape[1] + trace_array.shape[1]])
        combined_float_mask = np.zeros([float_mask.shape[0], float_mask.shape[1], len(all_names) ])
        for c, name in enumerate(all_names):
            if name in old_names:
                combined_trace_array[c, :e_width] = old_traces[name]
                combined_float_mask[:,:,c] = old_masks[name]
            if name in nr_keys:
                combined_trace_array[c, e_width:] = traces[name]
                combined_float_mask[:,:,c] = float_mask[:,:,key_ids[name]]
            
        # existing_data = obj.DB['Animals'][A][F][dName][...]
        # existing_time = obj.DB['Animals'][A][F]['T'][dName][...]
        # del obj.DB['Animals'][A][F][dName]
        # del obj.DB['Animals'][A][F]['T'][dName]
        # DATA = np.concatenate((existing_data, DATA), axis=0)
        # TIME = np.concatenate((existing_time, TIME+trialStartTime), axis=0)
        # chunkShape = (1,) + DATA.shape[1:]
        # maxShape = (None,) + DATA.shape[1:]
        # obj.DB['Animals'][A][F].require_dataset(dName, shape =  DATA.shape, maxshape = maxShape, data = DATA, dtype = DATA.dtype, track_order = True, chunks=chunkShape)
        # obj.DB['Animals'][A][F]['T'].require_dataset(dName, shape =  TIME.shape, maxshape = None, data = TIME, dtype = TIME.dtype, track_order = True)
        
        obj.updateROIdata(combined_float_mask, combined_trace_array, names = all_names, animal=A, FOV=F, datakey = dName, updateGUI=False)
    
    else:
        obj.DB['Animals'][A][F][dName][...] = DATA
        obj.updateROIdata(float_mask, trace_array, names = nr_keys, animal=A, FOV=F, datakey = dName, updateGUI=False)
    
    
    
    # if not ('Names' in obj.DB['Animals'][A][F]['R'][dName].keys()):
    #     dt = h5py.string_dtype(encoding = 'utf-8')
    #     nROIs= float_mask.shape[2]
    #     obj.DB['Animals'][A][F]['R'][dName].require_dataset('traceArray', shape = trace_array.shape, dtype=trace_array.dtype, data = trace_array)
    #     obj.DB['Animals'][A][F]['R'][dName].require_dataset('floatMask', shape = float_mask.shape, dtype = float_mask.dtype, data =float_mask)
    #     N = obj.DB['Animals'][A][F]['R'][dName].require_dataset('names',shape=(nROIs,1), dtype = dt)
    
    # for r, n in enumerate(nr_keys):
    #     N[r,0] = n
    
  
    
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

def readCERNA_proc(datapath, timepath): ##For re-import of data exported for external processiing in imagej
    timedata = np.genfromtxt(timepath)
    data = imageio.volread(datapath)
    return(data, timedata, False)

def transformCERNA(data, TIME):
    data, TIME = pMedFilt2(None, data=data, TIME = TIME)
    data, TIME = pFFThighPass(None, poolNum = 23, data =data, TIME=TIME)
    if data.shape[1] > 500:
        data, TIME = downSampleSpatial(None, stack=data, TIME=TIME)
    return(data, TIME, 'CERNAfiltered')

#def transformCERNA_proc(DATA, TIME): ##For re-import of data exported for external processiing in imagej
 #   return(DATA, TIME, 'CERNAfiltered_ex')

def readNIR(datapath, timepath):
    timedata = np.genfromtxt(timepath)
    try:
        data = np.array(imageio.mimread(datapath, memtest = False))[0:-1,...]
    except:
        print('fCould not read NIR data file frin {datapathh}')
        return(None, None, True)
    return(data, timedata, False)

def read_linked_cam(datapath, timepath):
    timedata = np.genfromtxt(timepath)
    data = timedata
    return(data, timedata, False)

def link_cam(datapath, timepath):
    return(datapath)

def readTRG(datapath, timepath):
    print(f'Reading TRG {datapath}')
    data = imageio.volread(datapath)
    print(f'{data.shape=}')
    timedata = np.genfromtxt(timepath)
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
            print(f'Depositing trial {c+1} of {len(directory)}')
            depositTrial2(obj, os.path.join(sessionFolder,os.path.normpath(item)), data_filter = input_filter)
    
    
 
def blockRegister(obj):
    stack, TIME = getSubStack(obj)
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
    return(output, TIME)
    

def registerTemplate(obj, stack = None, TIME = None):
    if stack is None:
        stack, TIME = getSubStack(obj)
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    templateKey, okPressed = QInputDialog.getItem(obj,"Select Template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        return(stack)
    templateStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][templateKey][...]
    template = np.median(templateStack, axis = 0)
    output, TIME = pReg(obj, stack = stack, TIME=TIME, template=template)
    return(output, TIME)
    

    
def cropToZeros(obj, stack = None, TIME = None):
    if stack is None:
        stack, TIME = getSubStack(obj)
    minStack = np.amin(stack, axis = 0) ## ?MAybe should  change to percentile (eg 1%) for rare reg mistakes
    print(f'Shape: {minStack.shape}')
    plt.imshow(minStack)
    plt.show()
    boostack = minStack.astype(bool)
    min8 = boostack.astype(np.uint8)*255
    x, y, w, h = LIR.largest_interior_rectangle(min8)
    
    print(f'Bounds: {x,y,w,h}')
    
    return(stack[:,y:y+h,x:x+w,...], TIME)
        
    


                  
def crossCorr(obj):
    stack, TIME = getSubStack(obj)
    im = DYroi.local_correlations_fft(stack)
    return(im, TIME[0])
    
def adaptiveThresh(obj):
    stack, TIME = getSubStack(obj)
    blocksize = obj.segmentationMethods['Adaptive threshold']['Params']['Block size'][1]
    #erodeRep  = self.segmentationMethods['Adaptive threshold']['Params']['Erode cycles'][1]
    #erodeArea = self.segmentationMethods['Adaptive threshold']['Params']['Erode area'][1]
    C         = obj.segmentationMethods['Adaptive threshold']['Params']['C'][1]
    #minArea   = self.segmentationMethods['Adaptive threshold']['Params']['Min area'][1]
    #maxArea = self.segmentationMethods['Adaptive threshold']['Params']['Max area'][1]
    #img = stack[0,...]
    stack = stack - np.min(np.min(np.min(stack)))
    img = stack/np.max(np.max(np.max(stack)))
    img = img * 255
    img = img.astype('uint8')
    output = np.zeros(stack.shape)
    for ii in range(stack.shape[0]):
        output[ii,:,:] = cv2.adaptiveThreshold(img[ii,:,:], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, C)
        print(ii)
    return(output.astype(np.uint16), TIME)

def adaptiveThreshInv(obj):
    img, TIME = getSubStack(obj)
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
    return(output.astype(np.uint16), TIME)


def registerUsingROIs(obj, template = 'first', reg_mode = StackReg.RIGID_BODY):
    ## Make lists containing ROI mask arrays and stacks:
    ROIs = []
    traces = []
    stacks =[]
    TIMES = []
    ROIs_out = []
    stacks_out = []
    tfs = []
    names_out = []
    
    for item in obj.DataList.selectedItems():
        datakey = item.text()
        ROIs.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][datakey]['floatMask'][...])
        traces.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][datakey]['traceArray'][...])
        stacks.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV][datakey][...])
        TIMES.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][datakey][...])
        names_out.append(datakey + '_ROIreg')
    ## Use first ROIstack flat image as template, unless template passed to func
    if template == 'first':
        template = np.sum(ROIs[0], axis = 2)
    ## Get transforms to align each ROIstack to template:
    sr = StackReg(reg_mode)
    for ROI in ROIs:
        target = np.sum(ROI, axis =2)
        tf = sr.register(template, target)
        tfs.append(tf)
    ##Apply transforms to each stack and each ROI:
    for stack, ROIstack, tf in zip(stacks, ROIs, tfs):
        stack_out = np.zeros(stack.shape)
        ROI_out = np.zeros(ROIstack.shape)
        for count, frame in enumerate(stack):
            stack_out[count,:,:] = sr.transform(frame, tmat=tf)
        print(f'ROIstack shape: {ROIstack.shape}')
        print(f'ROIout shape: {ROI_out.shape}')
        plt.figure()
        for count in range(ROIstack.shape[2]):
            print(count)
            frame = ROIstack[:,:,count]
            plt.imshow(frame)
            ROI_out[:,:,count] = sr.transform(frame, tmat=tf)
        stacks_out.append(stack_out)
        ROIs_out.append(ROI_out)
    for stack, ROI, TIME, dName, trace in zip(stacks_out, ROIs_out, TIMES, names_out, traces):
        genericDepositTrial(obj, stack, TIME, dName)
        obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dName].require_dataset('floatMask', data=ROI, shape=ROI.shape, dtype = ROI.dtype, maxshape=(ROI.shape[0], ROI.shape[1], None))
        obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dName].require_dataset('traceArray', data=trace, shape=trace.shape, dtype = trace.dtype, maxshape=(None, trace.shape[1]))
        obj.updateROImask()
        
        print('Depositing ROI aligned stack')
    print('Done aligning stacks based on ROIs')
  
    return(None, None)
        
        
    

    
def erode(obj):
    stack, TIME = getSubStack(obj)
    img = stack[0,...]
    img = stack
    img = img - np.min(np.min(img))
    img = img/np.max(np.max(img))
    img = img * 255
    img8 = img.astype('uint8')
    kernel = np.ones((3,3), dtype=np.uint8)
    newim = cv2.erode(img8, kernel)
    return(newim, TIME[0])

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


def pReg(obj, template='X', poolNum = None, method = 'RIGID_BODY', stack = None, TIME = None):
    if stack is None:
        stack, TIME = getSubStack(obj)
    if poolNum == None:
        #poolNum = mp.cpu_count()-1
        poolNum = obj.segmentationMethods['Random params']['Params']['nClusters'][1]
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibraryV2" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        IT = stackRegIterator(stack, refIm = template, method = 'RIGID_BODY')
        result = pool.map(regImage, IT) ## this works,testing progress bar
       
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print('Time elapsed registering:' + str(endTime-startTime))
        print(poolNum)
        return(np.array(result), TIME)


def affineReg(obj):
    return(pReg(obj, method = 'AFFINE'))

def bilinearReg(obj):
    return(pReg(obj, method = 'BILINEAR'))
 


def regImage(regInput):
    im =  regInput[0]
    refIm = regInput[1]
    sr = regInput[2]
    #return(np.uint16(sr.register_transform(refIm, im)))
    return(sr.register_transform(refIm, im))

def pGetTransform(obj, template='X', poolNum = 23):
    stack, TIME = getSubStack(obj)
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibraryV2" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        IT = stackRegIterator(stack, template)
        result = pool.map(getTransform, IT)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(endTime-startTime)
        print(poolNum)
        return(np.array(result), TIME)
    
    
def removeBadFrames(obj):
    stack, TIME = getSubStack(obj)
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
      
    return(stack, TIME)
            
def exp_as_raw_data(obj):
    saveParent = selectFolder()
    #DATA, TIME = getSubStack(obj)
    
    for item in obj.DataList.selectedItems():
 
        key = item.text()
        TIME =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][key][...]
        DATA =  obj.DB['Animals'][obj.curAnimal][obj.curFOV][key][...]
        trial_start_time = str(int(TIME[0]))
        
        source = obj.curAnimal + '-' + obj.curFOV + '_' + key
        folder_path = os.path.normpath(f'{saveParent}/{source}/{trial_start_time}')
        os.makedirs(folder_path, exist_ok = True)
        CERNAPath = os.path.normpath(f'{folder_path}/CERNAex.tif')
        TimePath = os.path.normpath(f'{folder_path}/CERNAex_time.txt')
        TIME = TIME - TIME[0]
        print(f'Saving data to {folder_path}')
        imageio.volwrite(CERNAPath, DATA, bigtiff = True)
        np.savetxt(TimePath, TIME)
    print('Done saving')
    
def exp_data(obj):
    saveParent = '/lab-share/Neuro-Woolf-e2/Public/Figure publishing/'
    #saveParent = selectFolder()
    #DATA, TIME = getSubStack(obj)
    
    for item in obj.DataList.selectedItems():
 
        key = item.text()
        TIME =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][key][...]
        DATA =  obj.DB['Animals'][obj.curAnimal][obj.curFOV][key][...]
        
        if len(TIME.shape)==0:
            trial_start_time = str(int(TIME))
        else:
            trial_start_time = str(int(TIME[0]))
        
        source = obj.curAnimal + '-' + obj.curFOV + '_' + key
        folder_path = os.path.normpath(f'{saveParent}/{source}/{trial_start_time}')
        os.makedirs(folder_path, exist_ok = True)
        savePath = os.path.normpath(f'{folder_path}/{key}.tif')
        TimePath = os.path.normpath(f'{folder_path}/CERNAex_time.txt')
        TIME = TIME - TIME[0]
        print(f'Saving data to {folder_path}')
        if len(DATA.shape)>2:
            imageio.volwrite(savePath, DATA, bigtiff = True)
        else:
            np.savetxt(savePath, DATA)
        if key in obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'].keys():
            if 'floatMask' in obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][key].keys():
                ROIs = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][key]
        np.savetxt(TimePath, TIME)
    print('Done saving')


def GUI_plot_traces(obj, raster=None, timestep = 0.1, stimIX = None, stimLabels = None, savePath = None, nullValue=None):
    F = plt.figure()
    A = {}
    # Count # of traces to plot
    ys = {}
    xs = {}
    FOV = obj.DB['Animals'][obj.curAnimal][obj.curFOV]
    num_stim = 0
    for item in obj.DataList.selectedItems():
        data_stream = item.text()
        key = item.text()
        ## If data is 1d, plot directly
        data_shape = FOV[data_stream].shape
        if len(data_shape) == 1:
            data_stream = data_stream + '__STIM'
            num_stim = num_stim+1
            ys = FOV[(data_stream, 0)][...]
            xs[(data_stream, 0)] = FOV['T'][data_stream][...]
        elif not ('paw' in data_stream):
            if 'FLIR' or 'rora' in data_stream:
                data_stream = data_stream + '__STIM'
                num_stim = num_stim +1
            for t, trace in enumerate(FOV['R'][key]['traceArray'][...]):
                ys[(data_stream, t)] = trace
                xs[(data_stream, t)] = FOV['T'][key][...]
        else:
            ys[(data_stream, t)] = np.sum(FOV['R'][key]['traceArray'][...], axis=0)
            xs[(data_stream, t)] = FOV['T'][key][...]
    num_traces = len(xs)
    axis_count = 1
    for k, key in enumerate(xs):
        if '__STIM' in key:
            A[key] = F.add_subplot(num_traces, 1, axis_count)
            A[key].plot(xs[key], ys[key])
            axis_count = axis_count+1
    for k, key in enumerate(xs):
        if not ('__STIM' in key):
            A[key] = F.add_subplot(num_traces, 1, axis_count)
            A[key].plot(xs[key], ys[key])
            axis_count = axis_count+1
    
    title = f'{obj.curAnimal} + {obj.curFOV} + {LA.uid_string()}'
    LA.save_fig(F, title)
        
        
             
            
        


def downSampleSpatial(obj, stack = None, TIME = None):
    if obj is not None:
        stack, TIME = getSubStack(obj)
    output = np.array(downscale_local_mean(stack, (1,2,2)))
    return(output, TIME)

def dataTime(obj):
    output = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus][...]
    return(output)

def downSampleTemporal(obj, factor = 2):
    
    timescale = dataTime(obj)
    Istart = np.searchsorted(timescale,obj.timeLUT[0])
    Iend = np.searchsorted(timescale,obj.timeLUT[-1])
    oldtime = timescale[Istart:Iend]
    newStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][obj.dataFocus][::factor]
    newTime = oldtime[::factor]

    return(newStack, newTime)
    
    
    
def pGetTransformFromTemplate(obj, template='X', poolNum = 23):
    if obj is not None:
        stack, TIME = getSubStack(obj)
    dataStreams = obj.DB['Animals'][obj.curAnimal][obj.curFOV].keys()
    templateKey, okPressed = QInputDialog.getItem(obj,"Select Template:", "Data:", dataStreams, 0, False)
    if okPressed != True:
        print(okPressed)
        print(~okPressed)
        return(None,None)
    templateStack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][templateKey][:]
    template = np.median(templateStack, axis = 0)

    
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibraryV2" or __name__ == '__main__':
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

def applyTransform(obj):
    stack, TIME = getSubStack(obj)
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
        print(counter)
    return(output, TIME)
        
    
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
    #print(f'{IMG=}')
    return(cv2.medianBlur(IMG, 3))

def pMedFilt2(obj, data = None, TIME = None, poolNum = 23):
    if obj is not None:
        stack, TIME = getSubStack(obj)
    else:
        stack = data
        TIME=TIME
    startTime = time.perf_counter()
    if __name__ == "DYpreProcessingLibraryV2" or __name__ == '__main__':
        pool = mp.Pool(poolNum)
        result = pool.map(singleMedFilt2, stack)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(endTime-startTime)
        print(poolNum)
        return(np.array(result), TIME)

class FFTiter:
    def __init__(self,stack, constant = 5):
        self.frame = 0
        self.max = stack.shape[0]
        self.stack = stack
        self.LPfilt = gaussianLP(constant, stack[0,:,:].shape)
    
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
    

def pFFThighPass(obj, poolNum = None, data = None, TIME = None, spatial_constant=5):
    if poolNum is None and not (obj is None):
        poolNum = obj.segmentationMethods['Random params']['Params']['nClusters'][1]
    if obj is None:
        stack = data
        TIME = TIME
    else:
        stack, TIME = getSubStack(obj)
        spatial_constant = obj.segmentationMethods['Random params']['Params']['spatial filter'][1]
   
    startTime = time.perf_counter()
    #print(__name__)
    #print('FFT par called')
    if __name__ == "DYpreProcessingLibraryV2" or __name__ == '__main__':
        #print('FFT par progressed')
        pool = mp.Pool(poolNum)
        IT = FFTiter(stack, constant = spatial_constant)
        result = pool.map(singleFFT, IT)
        pool.close()
        pool.join()
        endTime = time.perf_counter()
        print(f'Filtering took:{endTime-startTime} seconds')
        #print('Collecting results...')
        #output = normalizeStack(result)
        output = np.array(result)
        
        return(output, TIME)
    else:
        print(__name__)
        
def normalizeStack(obj):
    stack, TIME = getSubStack(obj)
    output = stack - np.min(np.min(np.min(stack)))
    output = output*(((2**16)-1)/np.max(np.max(np.max(output))))
    output = np.uint16(output)
    arraySum = np.sum(output)
    print(arraySum)
    return(output, TIME)

def convertUint16(stack, obj, minV = 0.8, maxV = 1.2): #typical range for fft filtered data
    output = stack - minV
    output = output*(((2**16)-1)/maxV)
    output = np.uint16(output)
    return(output)

        
def pDFF(stack):
    return
     
def mask_ROIs(obj):
    stack, TIME = getSubStack(obj)
    for ROI in obj.selectedROI:
        ROImask = np.invert( obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][obj.dataFocus]['floatMask'][:,:,ROI].astype(bool))
        stack = stack * ROImask    
    return(stack, TIME)

def mask_inverse_ROIs(obj):
    stack, TIME = getSubStack(obj)
    for ROI in obj.selectedROI:
        ROImask = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][obj.dataFocus]['floatMask'][:,:,ROI].astype(bool)
        stack = stack * ROImask    
    return(stack, TIME)
        
        
def transposeRGB(obj):
    stack, TIME = getSubStack(obj)
    print(stack.shape)
    output = np.zeros(stack.shape)
    output[:,:,:,0] = stack[:,:,:,2]
    output[:,:,:,1] = stack[:,:,:,1]
    output[:,:,:,2] = stack[:,:,:,0]
    return(output, TIME)
        
def grams_to_millinewtons(obj):
    stack, TIME = getSubStack(obj)
    return(stack*9.80665, TIME)
     
def regToFeature(stack, obj):
    for ROI in obj.transformROIlist:
        subStack = DYcrop(stack, ROI)
        ## get transformation matrix of each stack, zero out rotation, take medianm apply to full stack
    
        
def crop(obj):
    stack, TIME, floatMask, traceArray = getSubStack(obj)
    #if len(obj.transformList)==0:
       # return('STOP',0)
    return(stack, TIME)
    
def crop_with_ROIs(obj, A=None, F=None):
    if A is None:
        A=obj.curAnimal 
    if F is None:
        F=obj.curFOV
    if 'floatMask' in obj.DB['Animals'][A][F]['R'][obj.dataFocus].keys():
        ROImode = True
        DATA, TIME, floatMask, traceArray = getSubStack(obj, getROIs=True)
    else:
        ROImode = False
        DATA, TIME = getSubStack(obj, getROIs=False)
        
    #if len(obj.transformList)==0:
       # return('STOP',0)
    dName = f'{obj.dataFocus}_{str(TIME[0])}_{str(TIME[-1])}'
    genericDepositTrial(obj, DATA, TIME, dName) 
    if ROImode:
        retain = []
        for ix in range(floatMask.shape[-1]):
            if np.amax(floatMask[:,:,ix]) >0:
                retain.append(ix)
                
        floatMask = floatMask[:,:,retain]
        traceArray = traceArray[retain,:]
            
        
        if not (floatMask is None):
            obj.updateROIdata(floatMask, traceArray, animal = A, FOV=F, datakey=dName)
    return('STOP', 0)

def plot_transient_correlations(obj):
    cells = obj.selectedROI
    for cell in cells:
        ROI = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][obj.dataFocus]['floatMask'][:,:,cell]
        trace = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][obj.dataFocus]['traceArray'][cell,:]
        TIME = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus][...]
        movie_ref = obj.DB['Animals'][obj.curAnimal][obj.curFOV][obj.dataFocus]
        print(f'{movie_ref=}')
        pad_area = obj.segmentationMethods['Transients']['Params']['pad area'][1]
        clip = movie_context_for_roi(ROI, movie_ref=movie_ref, pad=pad_area)
        thresh = obj.segmentationMethods['Transients']['Params']['SNR thresh'][1]
        dur = obj.segmentationMethods['Transients']['Params']['Min duration'][1]
        pad_dur = obj.segmentationMethods['Transients']['Params']['pad transients'][1]
        transients = identify_transients(trace, threshold_SNR = thresh, duration = dur, pad = pad_dur)
        corr_transients_to_movie(obj, transients, trace, clip, TIME, ROI = cell)
        
    
    
def movie_context_for_roi(ROI, movie=None, movie_ref=None, pad=5):
    if not movie_ref is None:
        movie = movie_ref
    mask = ROI*255
    mask = mask.astype(np.uint8)
    plt.figure('mask')
    plt.imshow(mask)
    y,x,h,w = cv2.boundingRect(mask)
    y = y-pad
    x = x-pad
    h = h+ (2*pad)
    w = w+ (2*pad)
    
    if y<0:
        y=0
    if x < 0:
        x=0
    print(f'{x=},{y=},{h=},{w=}')
    sub = movie[:,x:x+w,y:y+h]
    print(f'{sub=}')
    return(sub)
                

def identify_transients(trace, **kwargs):
  
    params = {'threshold_SNR' : 5,  ## 5 x SNR threshold
              'duration' : 10, ## 10 frames (could switch to seconds if timestep known)
              'pad' : 0} 
    params.update(kwargs)
    n_trace = trace-np.amin(trace)
    n_trace = n_trace/np.amax(n_trace)
    q = stats.median_abs_deviation(trace)/0.79788456
    m = np.median(trace)
    thr = m+(params['threshold_SNR']*q)
    transients = {}
    n=0
    started = False
    for c, i in enumerate(trace):
        if i > thr:
            if not started:
                transients[n] = {}
                transients[n]['frames'] = []
                transients[n]['trace'] = []
                transients[n]['norm_trace'] = []
                started = True
            transients[n]['trace'].append(i)
            transients[n]['frames'].append(c)
            transients[n]['norm_trace'].append(n_trace[c])
        else:
            if started:
                started = False
                if len(transients[n]['trace']) >= params['duration']:
                    n=n+1
                else:
                    transients[n] ={}
    toDel = []
    for n in transients.keys():
        if not transients[n]:
            toDel.append(n)
            
    for d in toDel:
        del transients[d]
        
    if params['pad']:
        for n in transients.keys():
            start = transients[n]['frames'][0]-params['pad']
            if start <0:
                start = 0
            end  = transients[n]['frames'][-1]+params['pad']
            if end > trace.shape[0]:
                end = trace.shape[0]
            frames = np.arange(start,end)
            transients[n]['frames'] = frames
            transients[n]['trace'] = trace[frames]
            transients[n]['norm_trace'] = n_trace[frames]
                
    return(transients)  

def corr_transients_to_movie(obj, transients, trace, movie, TIME, ROI='unknown'):
    fig = plt.figure(f'transient correlation for cell {ROI}')
    row = int(np.ceil(np.sqrt(len(transients))))+1
    col = int(np.ceil(np.sqrt(len(transients))))
    print(f'{movie.shape=}')
    for c, transient in enumerate(transients.keys()):
           a = fig.add_subplot(row, col, c+1)
           t_map = np.zeros([movie.shape[1],movie.shape[2]])
           for ii in range(0,movie.shape[1]):
               for jj in range(0,movie.shape[2]):
                   t_start = transients[transient]['frames'][0]
                   t_end = transients[transient]['frames'][-1]
                   movie_sub = movie[t_start:t_end+1,ii,jj]
                   print(f'{movie.shape=}')
                   print(f'{ii=}')
                   print(f'{jj=}')
                   print(f'{movie_sub.shape=}')
            #       print(f"{transients[transient]['trace'].shape=}")
            #       print(f"{transients[transient]['frames'].shape=}")
                   print(f'{transient=}')
                   t_map[ii,jj] = np.corrcoef(movie_sub,transients[transient]['trace'])[0,1]
           a.imshow(t_map)
           a.xaxis.set_visible(False)
           a.yaxis.set_visible(False)
           
    A_trace = fig.add_subplot(row, 1, row)
    A_trace.plot(TIME,trace, color = 'k')
    for transient in transients:
        A_trace.plot(TIME[transients[transient]['frames']], transients[transient]['trace'], color = 'r')
           
def append_data_to_multi_session(obj, FOV, key_session = 0): ## Work in progress
    """
    Parameters
    ----------
    obj : DBgui instance
        
    FOV : Name of FOV to add
    
    

    Returns
    -------
    Multi-session object with new data added.
    
    First, stitch data from FOV to template from mult-session object
    -Remove FOVs trimmed out during stiching
    Create FOV with stitched data
    Generate session from FOV

    """
    ## Open alignment receiving new data:
    alignment_file = selectFile()
    M = unpickle(alignment_file)
    K = M.sessions[key_session] ## Key session to use for alignment
    template = K.fieldImage(K.get_raw_stack(start=0, stop = 10))
      
    dk = get_data_keys_from_flag(obj, 'Ca data', FOV=FOV, Animal = obj.curAnimal)[0]
    
    Stacks = [template, obj.DB['Animals'][obj.curAnimal][FOV][dk][...]]
    
    ROIstacks = [K.ROIs, obj.DB['Animals'][obj.curAnimal][FOV]['R'][dk]['floatMask'][...]]
    transform = stitch.getTransforms(Stacks)
    
    
    print('Stitching...')
 #   transformed_stacks, tfs, glob_transform, transformed_ROIs = stitch.Stitch(Stacks, mode=reg_mode, transform_ROIs = True, ROIstacks = ROIstacks, traceArrays = traceArrays, split_output = True, registration_input = align_mode)   
    print('Stitching completed')
    
    

def align_chronic_data(obj, DBpath = None, Animal = None, FOV_list = None, FOV_flags = None, activity_flag = None,  stim_flags = None, create_sessions = False, default_flags=True):

    
    ### get alignment mode:
    alignment_modes = ['rois', 'raw', 'vasc', 'last_3_rois']

    align_mode, okPressed = QInputDialog.getItem(obj,"Select alignment mode:", "Mode:", alignment_modes, 0, False)
    if not okPressed:
        return
    ## Get DB from DBgui instance
    
    
    reg_mode_dict = {'rigid' : StackReg.RIGID_BODY,
                     'affine' : StackReg.AFFINE,
                     'bilinear' : StackReg.BILINEAR}
    reg_mode, okPressed = QInputDialog.getItem(obj,"Select reg mode:", "Mode:", reg_mode_dict.keys(), 0, False)
    if not okPressed:
        return
    
    reg_mode = reg_mode_dict[reg_mode]
    
    if default_flags:
        FOV_flags = ['Chronic']
       
        activity_flag = 'Ca data'
        stim_flags = [activity_flag, 'Mech stim','Thermo stim']
        only_stim_flags = []
        for flag in stim_flags:
            if not (flag == activity_flag):
                only_stim_flags.append(flag)

    #obj = dbContainer(DBpath)
          
    DBpath = obj.DBpath
    
    if Animal is None:
        Animal = obj.curAnimal
        
    if FOV_flags is None:
        FOV_flags = ['Chronic']
    ## get list of FOVs to align ( create list from selected flags if list not supplied)
    if FOV_list is None:
        if FOV_flags is None:
            print('No FOVs specified')
            return
        else:
            FOV_list = []
            for FOV in obj.DB['Animals'][Animal].keys():
                for flag in FOV_flags:
                    if flag in obj.DB['Animals'][Animal][FOV].attrs:
                        if obj.DB['Animals'][Animal][FOV].attrs[flag]:
                            FOV_list.append(FOV)
                            print(f'{FOV} added to list')
    
    print(FOV_list)
    ## Get calcium data to align
    source_FOVs = []
    Stacks = []
    Activity_keys = []
    Stim_data_keys = []
    Stim_data_keys_without_activity = []
    Times = []
    ROIstacks = []
    traceArrays = []      
    
    for FOV in FOV_list:            ### Getting data keys to align from flags - this is done in a very convoluted way but seems to work
        print(FOV)
        for data_set in obj.DB['Animals'][Animal][FOV].keys():
    #        print(f'{data_set=}')
            if activity_flag in obj.DB['Animals'][Animal][FOV][data_set].attrs:
                    if obj.DB['Animals'][Animal][FOV][data_set].attrs[activity_flag]:
                        #Stacks.append(obj.DB['Animals'][Animal][FOV][data_set][...]) 
                        Stacks.append(obj.DB['Animals'][Animal][FOV][data_set]) 
                        Times.append(obj.DB['Animals'][Animal][FOV]['T'][data_set][...]) 
                        ROIstacks.append(obj.DB['Animals'][Animal][FOV]['R'][data_set]['floatMask'][...]) 
                        traceArrays.append(obj.DB['Animals'][Animal][FOV]['R'][data_set]['traceArray'][...]) 
                        Activity_keys.append(data_set)
                        source_FOVs.append(FOV)
                        Stim_data = []
                        Stim_data_not_activity = []
                        print(f'{data_set} added')
                        for data_set_II in obj.DB['Animals'][Animal][FOV].keys():
            #                print(f'{data_set_II=}')
                            for stim_flag in stim_flags:
                                if stim_flag in obj.DB['Animals'][Animal][FOV][data_set_II].attrs:
                                    if obj.DB['Animals'][Animal][FOV][data_set_II].attrs[stim_flag]:
                                        Stim_data.append(data_set_II)
                                        if not(Stim_data == data_set):
                                            Stim_data_not_activity.append(data_set_II)
                                        print(f'{data_set_II} added')
                        Stim_data_keys.append(Stim_data)
                        Stim_data_keys_without_activity.append(Stim_data_not_activity)
    
    
    if len(Stacks) == 0:
        a = obj.DB['Animals'][Animal].keys()
        obj.close()
        print('No data found to align')
        return(a)
    
    
    print('Stitching...')
    transformed_stacks, tfs, glob_transform, transformed_ROIs = stitch.Stitch(Stacks, mode=reg_mode, transform_ROIs = True, ROIstacks = ROIstacks, traceArrays = traceArrays, split_output = True, registration_input = align_mode)   
    print('Stitching completed')
 
    #print(f'{len(transformed_stacks)}')
    #print(f'{len(transformed_ROIs)}')
    #print(f'{len(Times)}')
    #print(f'{(source_FOVs)}')
    #print(f'{len(traceArrays)}')
    #print(f'{(Activity_keys)}')
    
    
    ## Trim transformed stacks:
    print('Trimming...')
    combined = np.zeros([len(transformed_stacks), transformed_stacks[0].shape[1], transformed_stacks[0].shape[2]])
    for count, stack in enumerate(transformed_stacks):
        combined[count,:,:] = np.amin(np.nan_to_num(stack), axis=0)
    minStack = np.amin(combined, axis = 0)
    boostack = minStack.astype(bool)
    min8 = boostack.astype(np.uint8)*255
    print('Calculating least internal rectangle...')
    x, y, w, h = LIR.largest_interior_rectangle(min8)
    
    for count, stack in enumerate(transformed_stacks):
        transformed_stacks[count] = np.nan_to_num(stack[:,y:y+h,x:x+w])
    
    for count, masks in enumerate(transformed_ROIs):
        transformed_ROIs[count] = np.nan_to_num(masks[y:y+h,x:x+w,:])
    
    print('Done trimming')




    ## Deposit processed image stack, ROIs and sample times into source FOV:
    aligned_activity_keys = []
    for transformed_stack, transformed_ROI, source_FOV, TIME, traceArray, dName in zip(transformed_stacks, transformed_ROIs, source_FOVs, Times, traceArrays, Activity_keys):
        dName = dName + '_aligned_chronic' + align_mode
        aligned_activity_keys.append(dName)
        print(f'Depositing {dName}')
        genericDepositTrial(obj, transformed_stack, TIME, dName, A=Animal, F = source_FOV)
        if 'floatMask' in obj.DB['Animals'][Animal][source_FOV]['R'][dName].keys():
            del obj.DB['Animals'][Animal][source_FOV]['R'][dName]['floatMask']
        if 'traceArray' in obj.DB['Animals'][Animal][source_FOV]['R'][dName].keys():
            del obj.DB['Animals'][Animal][source_FOV]['R'][dName]['traceArray']
            
        print(f'Transformed ROI shape {transformed_ROI.shape}')
        obj.DB['Animals'][Animal][source_FOV]['R'][dName].require_dataset('traceArray', data = traceArray, shape = traceArray.shape, dtype = traceArray.dtype)
        obj.DB['Animals'][Animal][source_FOV]['R'][dName].require_dataset('floatMask', data = transformed_ROI, shape = transformed_ROI.shape, dtype = transformed_ROI.dtype)
        
        ## Remove ROIs cropped out by alignment
        to_delete = []
        for count in range(transformed_ROI.shape[2]):
            if np.sum(transformed_ROI[:,:,count]) == 0:
                to_delete.append(count)
        if len(to_delete)>0:
            delete_ROIs(obj, Animal=Animal, FOV=source_FOV, data_key = dName, rois_to_delete = to_delete)
            print(f'Deleting ROIs {to_delete} not represented in all sessions')
        else:
            updateROIdata(obj, transformed_ROI, traceArray, appendMode = False, animal=Animal, FOV = source_FOV, datakey = dName)
        
        obj.DB['Animals'][Animal][source_FOV][dName].attrs['Aligned ca'] = True
   
    create_sessions = True   
    #if create_sessions:
    chronic_data_to_sessions(source_FOVs, Stim_data_keys_without_activity, aligned_activity_keys, Activity_keys, obj, create_sessions, Animal)

def interface_for_export_to_multi_session(obj, default_flags = True, Animal = None, FOV_list = None, FOV_flags = None, stim_flags = None):
    """
    Find and format data for export as a multi-session object  - initial purpose is to export aligned data without re-stitching
    obj - instance of DBgui object
    default_flags -use default flags to identify FOVs and datastreams to export
    Animal - select which animal data is being exported from, will use curAnimal from DBgui if none specified
    FOV_list
    """
    if default_flags:
        FOV_flags = ['Chronic']
        activity_flag = 'Aligned ca'
        stim_flags = [activity_flag, 'Mech stim','Thermo stim']
        only_stim_flags = []
        for flag in stim_flags:
            if not (flag == activity_flag):
                only_stim_flags.append(flag)

    #obj = dbContainer(DBpath)
          
    DBpath = obj.DBpath
    
    if Animal is None:
        Animal = obj.curAnimal
        
    if FOV_flags is None:
        FOV_flags = ['Chronic']
        
    ## get list of FOVs to align ( create list from selected flags if list not supplied)
    if FOV_list is None:
        FOV_list =[]
        for flag in FOV_flags:
            FOV_list.extend(get_FOV_keys_from_flag(obj, flag))
    
    convert_h5_to_multi_session(obj, FOV_list, Animal, default_flags = True, activity_flag = None, stim_flags = None, activity_key = None, all_data_keys = None, pickle_data = True, store_pickle = True)
    
    
    

def get_data_keys_from_flag(obj, flag, FOV=None, Animal = None):
    if FOV is None:
        FOV = obj.curFOV
    if Animal is None:
        Animal = obj.curAnimal
    data_keys = []
    for data_set in obj.DB['Animals'][Animal][FOV].keys():
        if flag in obj.DB['Animals'][Animal][FOV][data_set].attrs:
            if obj.DB['Animals'][Animal][FOV][data_set].attrs[flag]:
                data_keys.append(data_set)
    return(data_keys)

def get_FOV_keys_from_flag(obj, flag, Animal = None, ):
    if Animal is None:
        Animal = obj.curAnimal
    FOV_keys = []
    for FOV in obj.DB['Animals'][Animal].keys():
        if flag in obj.DB['Animals'][Animal][FOV].attrs:
            if obj.DB['Animals'][Animal][FOV].attrs[flag]:
                FOV_keys.append(FOV)
    return(FOV_keys)
            
def chronic_data_to_sessions(source_FOVs, Stim_data_keys_without_activity, aligned_activity_keys, Activity_keys, obj, create_sessions, Animal):
    """
    Transform data into session_data objects and save pickle files in folder with H5 file
    This is interface is specifically for dealing with output of stitching multiple sessions
    For generic cases use 'convert_h5_to_mulit_session' function below
    """
    sessions=[]
    
    create_sessions = True
    if create_sessions:
        for (FOV, potential_stims, activity_key, old_key) in zip(source_FOVs, Stim_data_keys_without_activity, aligned_activity_keys, Activity_keys):
   
            
            stim_data_keyset = []
            for stim in potential_stims:
                if not(stim == old_key):
                    stim_data_keyset.append(stim)
            stim_data_keyset.append(activity_key)
            
            print(f'{stim_data_keyset=}')
            print(f'{activity_key=}')
     
            DATA = getData(obj.DBpath, selectedDataStreams = stim_data_keyset, activityStream = activity_key, selectedFOV=FOV)
            #pdb.set_trace()
            session = LA.session_data(DATA)
            sessions.append(session)
            
            pickle_individual_sessions = False
            if pickle_individual_sessions:
                file_name = obj.DBpath.split('.')[0] + f'_{FOV}.pickle' 
                session.pickle(file_name)
                print(f'{file_name=}')
                obj.DB['Animals'][Animal][FOV].attrs['session file']  = file_name
            
    FOV_longitudinal = LA.multi_session(sessions)
    m_file_name = obj.DBpath.split('.')[0] + FOV.split(' ')[0] + ' longitudinal.pickle'
    FOV_longitudinal.pickle(m_file_name)
    
    ## Store multi_session file locations in database:
    if 'multi session files' in obj.DB['Animals'][Animal].attrs:
        file_list = list(obj.DB['Animals'][Animal].attrs['multi session files'])
        file_list.append(m_file_name)
        new_file_list = []
        for file in file_list:
            if os.path.exists(file):
                if not (file in new_file_list):
                    new_file_list.append(file)
        obj.DB['Animals'][Animal].attrs['multi session files'] = new_file_list
    else:
        obj.DB['Animals'][Animal].attrs['multi session files'] = [m_file_name]
    obj.closeDB()
    
    return(sessions)

def convert_h5_to_multi_session(obj, FOVs, Animal, default_flags = True, activity_flag = None, stim_flags = None, activity_key = None, all_data_keys = None, pickle_data = False, store_pickle = False):
    sessions = []
    for FOV in FOVs:
        sessions.append(convert_FOV_to_session(obj, FOV, Animal=Animal, default_flags = default_flags, activity_flag = activity_flag, stim_flags = stim_flags , activity_key = activity_key, all_data_keys  = all_data_keys))
    M = LA.multi_session(sessions)
    
    if pickle_data:
       
        m_file_name = obj.DBpath.split('.')[0] + FOVs[0].split(' ')[0] + ' longitudinal.pickle'
        M.pickle(filepath = m_file_name)
    if pickle_data and store_pickle:
        if 'multi session files' in obj.DB['Animals'][Animal].attrs:
            file_list = list(obj.DB['Animals'][Animal].attrs['multi session files'])
            file_list.append(m_file_name)
            new_file_list = []
            for file in file_list:
                if os.path.exists(file):
                    if not (file in new_file_list):
                        new_file_list.append(file)
            obj.DB['Animals'][Animal].attrs['multi session files'] = new_file_list
        else:
            obj.DB['Animals'][Animal].attrs['multi session files'] = [m_file_name]
    obj.closeDB()
    

def convert_FOV_to_session(obj, FOV, Animal=None, default_flags = True, activity_flag = None, stim_flags = None , activity_key = None, all_data_keys  = None):
    ## Generate session objects for FOV from specified data:
    #pdb.set_trace()
    if Animal is None:
        Animal = obj.curAnimal
        
    if default_flags: 
        if activity_flag is None:
            activity_flag = 'Aligned ca' ## 'Ca data'
        if stim_flags is None:
            stim_flags = ['Mech stim','Thermo stim']
    
    all_flags = copy.copy(stim_flags)
    all_flags.append(activity_flag)
    
   
    if all_data_keys is None:
        all_data_keys = []
        for flag in all_flags:
            all_data_keys.extend(get_data_keys_from_flag(obj, flag, FOV=FOV, Animal=Animal))
    if activity_key is None:
        activity_key = get_data_keys_from_flag(obj, flag, FOV=FOV, Animal=Animal)
    
    
    
    
    if not isinstance(activity_key, list):
        print('activity_key should be one element list ')
        return
    if len(activity_key) == 0:
        print('No activity data found')
        return
    if len(activity_key) > 1:
        print('Multiple activity keys found, not converting')
        return
    
    activity_key = activity_key[0]
    
    DATA = getData(obj.DBpath, selectedDataStreams = all_data_keys, activityStream = activity_key, selectedFOV=FOV)
    session = LA.session_data(DATA)
    return(session)


                
def updateROIdata(obj, masks, traces, appendMode = False, animal=None, FOV = None, datakey = None):
    if animal is None:
        animal = obj.curAnimal
    if FOV is None:
        FOV = obj.curFOV
    if datakey is None:
        datakey = obj.dataFocus
        
    
    if appendMode:
        oMasks = obj.DB['Animals'][animal][FOV]['R'][datakey]['floatMask']
        oTraces = obj.DB['Animals'][animal][FOV]['R'][datakey]['traceArray'] 
        masks = np.concatenate((oMasks,masks), axis=-1)
        traces = np.concatenate((oTraces,traces),axis = 0)
        
    else:
        pass
    
    floatMask = masks
    boolMask = masks.astype('bool')
    binaryMask = masks>0
    labelMask = np.zeros(masks.shape)
    
    if len(traces.shape)<2:
        traces = np.expand_dims(traces, axis = 1)
    for label in range(0, masks.shape[-1], 1):
        labelMask[:,:,label] = binaryMask[:,:,label]*label+1
        
    for key in obj.DB['Animals'][animal][FOV]['R'][datakey]:
        del obj.DB['Animals'][animal][FOV]['R'][datakey][key]
    obj.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('labelMask', shape = labelMask.shape, dtype = labelMask.dtype, data =labelMask) 
    obj.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('floatMask', shape = floatMask.shape, dtype = floatMask.dtype, data =floatMask) 
    obj.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('boolMask', shape = boolMask.shape, dtype = boolMask.dtype, data =boolMask)     
    obj.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('traceArray', shape = traces.shape, dtype=traces.dtype, data = traces)
  

           
   
def delete_ROIs(obj, Animal=None, FOV=None, data_key = None, rois_to_delete = None): 
    floatMask = obj.DB['Animals'][Animal][FOV]['R'][data_key]['floatMask'][...]
    traces =    obj.DB['Animals'][Animal][FOV]['R'][data_key]['traceArray'][...]

    newMask = np.delete(floatMask, rois_to_delete, axis = -1)
    newTraces = np.delete(traces, rois_to_delete, axis = 0)
    
    newMask = newMask.astype('float64')
    #obj.updateROIdata(newMask, newTraces)
    
    #if 'paintedROImap' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
    #    del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['paintedROImap']       ## Fix later -gets rid of all painted rois, should make it only delete selected
    updateROIdata(obj, newMask, newTraces, appendMode=False, animal=Animal, FOV=FOV, datakey=data_key)
   
    
def glue(obj, merge_method = 'caiman', resolve_manual = False, fill_missing_method = 'zeros'):
    ## merge imaging sessions ROIs    
    ## imaging sessions should already be stitched
    
    ## collect ROI data:
    traces_in = []
    masks_in = []
    time_in = []
    templates = []
    FOV = obj.curFOV
    TIME = np.array([])
    session_starts = []
    session_ends = []
    start = 0
    end = 0
    oNames = []
    allkeys = []
    for item in obj.DataList.selectedItems():
        key = item.text()
        allkeys.append(key)
        masks_in.append(obj.DB['Animals'][obj.curAnimal][FOV]['R'][key]['floatMask'][...])
        traces_in.append(obj.DB['Animals'][obj.curAnimal][FOV]['R'][key]['traceArray'][...])
        templates.append(np.median(obj.DB['Animals'][obj.curAnimal][FOV][key][0:50,:,:], axis =0)) #average 1st 50 frames of imaging movie for display
        session_time = obj.DB['Animals'][obj.curAnimal][FOV]['T'][key][...]
        time_in.append(session_time)
        TIME = np.concatenate([TIME, session_time])
        print(f'Shape of TIME is: {TIME.shape}')
        session_starts.append(start)
        session_ends.append(start+ session_time.shape[0])
        start = start + session_time.shape[0]
        oNames.append(key)
        
        
        
        
    ## Align ROIS:
    if merge_method == 'caiman':
        allROIs, spatial_union, assignments, matchings = alignROIsCAIMAN(masks_in, templates)
    
   
    
    
    ## assemble dataset with integrated data
    traces_out = np.zeros([assignments.shape[0], TIME.shape[0]])
    for r_count, ROI in enumerate(assignments):
        for s_count, (start, stop, trace) in enumerate(zip(session_starts, session_ends, traces_in)):
            trace_num = ROI[s_count]
            if np.isnan(trace_num):
                if fill_missing_method =='zeros':
                    pass
            else:
                print(f'{r_count=}')
                print(f'{start=}')
                print(f'{stop=}')
                
                trace_num = int(trace_num)
                print(f'{trace_num=}')
                traces_out[r_count,start:stop] = trace[trace_num,:]
                
    ## deposit data
    oNames.append('glued')
    dName = '_'.join(oNames)
    if dName in obj.DB['Animals'][obj.curAnimal][FOV].keys():
        print(f'{dName} already in {FOV}, aborting...')
        print(f'Shape is {obj.DB["Animals"][obj.curAnimal][FOV][dName].shape}')
        return(None, None)
    for key in allkeys:
        print(f'Depositing {key} to {dName}')
        DATA = obj.DB['Animals'][obj.curAnimal][FOV][key][...]
        TIME = obj.DB['Animals'][obj.curAnimal][FOV]['T'][key][...]
        genericDepositTrial(obj, DATA, TIME, dName)
        new_data = obj.DB['Animals'][obj.curAnimal][FOV][dName]
        new_time = obj.DB['Animals'][obj.curAnimal][FOV]['T'][dName]
    
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'].require_group(dName)
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dName].require_dataset('floatMask', data = allROIs, dtype = allROIs.dtype, shape = allROIs.shape)
    obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dName].require_dataset('traceArray', data = traces_out, dtype = traces_out.dtype, shape = traces_out.shape)
    
    obj.updateDataList()
    obj.transformBox.setCurrentIndex(0) 
    print('Locking database...')
        
    self.closeDB()
    return(None, None)
  
    
  
    
def Stitch(obj, split_output = True, trim = True, across_FOVs = False , transform_ROIs = False):
    
    ## Align imaging sessions (even if not matched in XY dimensions)
    ## Inputs:
    ## obj - instance of YarmoPain GUI app
    
    #May want to refactor to making splitting of output mandatory, move gluing aligned sessions to separate function
    
    
    ##  Set up options
    
    
    split_output = obj.segmentationMethods['Stitching']['Params']['split alignment'][1]
    trim = obj.segmentationMethods['Stitching']['Params']['trim alignment'][1]
    across_FOVs = obj.segmentationMethods['Stitching']['Params']['Across FOVs'][1]
    transform_ROIs = obj.segmentationMethods['Stitching']['Params']['Transform ROIs'][1]
    
    ## Collect data to align
    #Make list of source data to register (and associated ROIs)
    Stacks = []
    if split_output:
        TIMES = []
    else:
        TIMES = np.array([])
    ROIstacks = []
    traceArrays = []
    original_names = []
    
    #add image stacks to register
    ## If more than one FOV are selected in GUI, select data across FOVs
    if len(obj.FOVlist.selectedItems())>1:
        across_FOVs=True
        
    
    if across_FOVs:  
        FOVs = []
        
        ## Deprecating previous method - chooses all data with the same data key - better to use flag
        
        # for item in obj.FOVlist.selectedItems():
        #     FOV = item.text()
        #     FOVs.append(FOV)
        #     for data_key in obj.DB['Animals'][obj.curAnimal][FOV].keys():
        #         data_tags.append(data_key)
        #data_tags = list(set(data_tags))
        
        
        data_tags = []
        for flag in obj.DataFlags.keys():
            data_tags.append(flag)
        data_tags.sort()
        selected_tag, okPressed = QInputDialog.getItem(obj,"Select data to align:", "Data:", data_tags, 0, False)
        if not okPressed:
            return(None,None)
        
        for FOV in FOVs:
            for datakey in obj.DB['Animals'][obj.curAnimal][FOV].keys():
                if selected_tag in obj.DB['Animals'][obj.curAnimal][FOV].attrs:
                    if obj.DB['Animals'][obj.curAnimal][FOV].attrs[FOVtag]:
                        Stacks.append(obj.DB['Animals'][obj.curAnimal][FOV][selected_tag][...])
                        ROIstacks.append(obj.DB['Animals'][obj.curAnimal][FOV]['R'][selected_tag]['floatMask'])
                        traceArrays.append(obj.DB['Animals'][obj.curAnimal][FOV]['R'][selected_tag]['traceArray'])
                        original_names.append(FOV+'_'+ selected_tag)
                        TIME = obj.DB['Animals'][obj.curAnimal][FOV]['T'][selected_tag][...]
                        if split_output:
                            TIMES.append(TIME)
                        else:
                            TIMES = np.concatenate([TIMES, TIME], axis = 0)
           
        print(f'{Stacks=}')
        newFOVname = selected_tag + '_stitched'
        newFOV = obj.DB['Animals'][obj.curAnimal].require_group(newFOVname)
        newFOV.require_group('T')   ### Time data
        newFOV.require_group('R')    ## ROI data (masks and traces)
        obj.curFOV = newFOVname
        
                
        
    else:
        for item in obj.DataList.selectedItems():
            dataKey = item.text()
            Stacks.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV][dataKey][...])
            ROIstacks.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dataKey]['floatMask'])
            traceArrays.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dataKey]['traceArray'])
            TIME = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][dataKey][...]
            if split_output:
                TIMES.append(TIME)
            else:
                TIMES = np.concatenate([TIMES, TIME], axis = 0)
            original_names.append(dataKey)
    
   
    #mVal = np.median(Stacks[0][0,...])
    ### So reNADOEMrion
    output, tfs, glob_transform, transformed_ROIs = stitch.Stitch(Stacks, transform_ROIs = transform_ROIs, ROIstacks = ROIstacks, traceArrays = None, split_output = split_output)
    
    
    ## for now will recalculate data name and add ROIs to DBs- in future could refactor to make ROI transformation generic method
    if not split_output and not across_FOVs:
        Tkey = 'Stitch'
        timescale =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus]
        Istart = np.searchsorted(timescale,obj.timeLUT[0])
        Iend = np.searchsorted(timescale,obj.timeLUT[-1])
        dName = obj.dataFocus + obj.suffixDict[Tkey] + obj.suffixDict[Tkey] + '_' + str(Istart) + '_' + str(Iend)
    
    
    ROIoutput = np.array([])
    traceOutput = np.array([])
    #ROIoutput, traceOutput = alignROIs(transformedROIs, traceArrays)
    #ROIoutput = np.concatenate(outputROIs, axis=2)
    #traceOutput = block_diag(*traceArrays)
    
    if trim: ##Trim out NaN/zero values:
        if split_output:  
            combined = np.zeros([len(output), output[0].shape[1], output[0].shape[2]])
            for count, stack in enumerate(output):
                combined[count,:,:] = np.amin(np.nan_to_num(stack), axis=0)
            minStack = np.amin(combined, axis = 0)
        else:
            minStack = np.amin(np.nan_to_num(output), axis = 0) 

        boostack = minStack.astype(bool)
        min8 = boostack.astype(np.uint8)*255
        x, y, w, h = LIR.largest_interior_rectangle(min8)
        
        
        if split_output:
            for count, stack in enumerate(output):
                output[count] = np.nan_to_num(stack[:,y:y+h,x:x+w])
        else:
            output = np.nan_to_num(output[:,y:y+h,x:x+w])
        
    if split_output:
        
        for tStack, TIME, oName in zip(output, TIMES, original_names):
            dName = oName + '_stitched'
            print(dName)
            genericDepositTrial(obj, np.nan_to_num(tStack), TIME, dName)   
        obj.updateFOVlist()
        return('STOP', None)
    else:
    
      #  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'].require_group(dName)
      #  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dName].require_dataset('floatMask', data = ROIoutput, dtype = ROIoutput.dtype, shape = ROIoutput.shape)
      #  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][dName].require_dataset('traceArray', data = traceOutput, dtype = traceOutput.dtype, shape = traceOutput.shape)
        
        return(np.nan_to_num(output), np.array(TIMES))
    
def mask_centroid(ROI):
    ROI = ROI.astype(bool)
    eight_bit = ROI.astype(np.uint8)
    M = cv2.moments(eight_bit)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])  
    return(cx,cy)



        


    
    
    

        

    
          
    
    
def alignROIs(ROIs, traceArrays, max_shift = 13):
    nInputROIs = 0
    nSessions = 0
    centroidLists = []
    ROI_session_list = []
    for ROIstack in ROIs:
        centroidList = np.zeros([ROIstack.shape[2],2]) ## array for x, y coordinates of each ROI centroid
        nInputROIs = nInputROIs + ROIstack.shape[2]
        for count in range(ROIstack.shape[2]):
            ROI = np.squeeze(ROIstack[:,:,count])
            ROI_session_list.append(nSessions)
            cx, cy = mask_centroid(ROI)
            centroidList[count,0] = cx
            centroidList[count,1] = cy
        nSessions = nSessions + 1
        centroidLists.append(centroidList)
        
    
    distanceArray = np.zeros([nInputROIs, nInputROIs])
    nearestArray = np.full([nInputROIs, nSessions], np.nan)
    source_counter = 0
    for sourceList in centroidLists:
        for source_ROI in sourceList:
            target_counter = 0
            for target_list in centroidLists:
                for target_ROI in target_list:
                    distance = np.linalg.norm(source_ROI-target_ROI)
                    if distance == 0:
                        distance = 0.5
                    print(f'Source: {source_counter}')
                    print(f'Target: {target_counter}')
                    distanceArray[source_counter,target_counter] = distance
                    target_counter = target_counter + 1
            source_counter = source_counter + 1
    plt.figure()
    plt.imshow(1/distanceArray)
            
    return
        
        
    ROIoutput = []
    traceOutput = []
    ### get pos of centroids for each roi set
    ###
    return(ROIoutput, traceArrays)


    
        
     
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

def stack_2D(obj, deposit = True):
    out_time = []
    dName = 'stack of:'
    for c, item in enumerate(obj.DataList.selectedItems()):
        dataset = item.text()
        dName = dName + dataset
        DATA = obj.DB['Animals'][obj.curAnimal][obj.curFOV][dataset][...]
        TIME = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][dataset][...]
        if len(TIME.shape) > 0:
            print('Non image data found in {dataset}, aborting')
            return
        if c==0:
            out_data = np.expand_dims(DATA, axis = 0) 
        else:
            out_data = np.concatenate((out_data, np.expand_dims(DATA, axis = 0)), axis = 0)
            
        out_time.append(TIME)
    out_time = np.array(out_time)
    if deposit:
        genericDepositTrial(obj, out_data, out_time, dName) 
        obj.updateDataList()
    return(out_data, out_time)
            
        
        
        
        
def data_info(obj):     
    for c, item in enumerate(obj.DataList.selectedItems()):
        dataset = item.text()
        DATA = obj.DB['Animals'][obj.curAnimal][obj.curFOV][dataset]
        TIME = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][dataset][...]
        print(f'{dataset} shape is {DATA.shape} dtype is {DATA.dtype}')     
        print(f'{dataset} has {TIME.shape} samples in time field')
        if len(TIME.shape)>0:
            print(f'Data spans {TIME[-1] - TIME[0]} seconds')
        else:
            print(f'Time point is {TIME}')
        
   
    
def getSubStack(obj, rect = None, FOV=None, datakey = None, Animal = None, getROIs = False):
        if FOV is None:
            FOV = obj.curFOV
        if datakey is None:
            datakey = obj.dataFocus
        if Animal is None:
            Animal = obj.curAnimal
            
        Istart, Iend = getTimeBounds(obj, FOV=FOV, datakey=datakey, Animal = Animal)
        if len(obj.DB['Animals'][Animal][FOV][datakey].shape) < 2:
            d_shape = 'scalar'
        elif len(obj.DB['Animals'][Animal][FOV][datakey].shape) == 2:
            d_shape = 'flat image'
        else:
            d_shape = 'image series'
        
        if rect is None:
            if len(obj.transformROIlist)>0:
                ROI = obj.transformROIlist[-1]
                rect = ROI.parentBounds()
                top = round(rect.y())
                bot = round(rect.y()+rect.height())
                left = round(rect.x())
                right = round(rect.x()+rect.width())
                print('Using rectangle')
            else:
                top = 0
                left = 0
                if d_shape == 'image series':
                    bot = obj.DB['Animals'][Animal][FOV][datakey].shape[2]
                    right = obj.DB['Animals'][Animal][FOV][datakey].shape[1]
                elif d_shape == 'flat image':
                    bot = obj.DB['Animals'][Animal][FOV][datakey].shape[1]
                    right = obj.DB['Animals'][Animal][FOV][datakey].shape[0]
        else:
            left = round(rect[0])
            top = round(rect[1])
            right = left + round(rect[2])
            bot = top + round(rect[3])
            
        print(f"{obj.DB['Animals'][Animal][FOV]['T'][datakey].shape=}")
        print(f"{obj.DB['Animals'][Animal][FOV][datakey].shape=}")
       
        
        if d_shape == 'image series': # if image time series
           print('Image time series')
           DATA = obj.DB['Animals'][Animal][FOV][datakey][Istart:Iend,left:right,top:bot,...]
           TIME = obj.DB['Animals'][Animal][FOV]['T'][datakey][Istart:Iend]
        elif d_shape == 'scalar': # if scalar time seris
            print('Scalar time series')
            DATA = obj.DB['Animals'][Animal][FOV][datakey][Istart:Iend]
            TIME = obj.DB['Animals'][Animal][FOV]['T'][datakey][Istart:Iend]
        else: # if single data point         
            print('Flat image')
            DATA = obj.DB['Animals'][Animal][FOV][datakey][...]
            TIME = obj.DB['Animals'][Animal][FOV]['T'][datakey][...]
       
        
        
        if getROIs:
            floatMask = None
            traceArray = None
            if 'floatMask' in obj.DB['Animals'][Animal][FOV]['R'][datakey].keys():
                floatMask = obj.DB['Animals'][Animal][FOV]['R'][datakey]['floatMask'][left:right, top:bot, :]
            if 'traceArray' in obj.DB['Animals'][Animal][FOV]['R'][datakey].keys():
                traceArray = obj.DB['Animals'][Animal][FOV]['R'][datakey]['traceArray'][:, Istart:Iend]
            return(DATA, TIME, floatMask, traceArray)
        else:
            return(DATA, TIME)

def getTimeBounds(obj,  FOV=None, datakey = None, Animal = None): # data is key to 
    if FOV is None:
            FOV = obj.curFOV
    if datakey is None:
            datakey = obj.dataFocus
    if Animal is None:
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
    
    
def initAnalysis(obj, selectedDataStreams = None, activityStream = None, selectedFOV = None, timeStep = 0.1, trimStim = True, external=False):  ## Funct to create DB of stims and responses for FOV within a mouse's H5 file
    
    if external:
        fullArr, compactArray, timestep, stimIX, stimLabels, meanImage, ROIs, nullValue, start_time = genSumTable(obj, selectedDataStreams = selectedDataStreams, activityStream = activityStream, selectedFOV = selectedFOV, timeStep=0.1, trimStim = True)
        
    else:
        fullArr, compactArray, timestep, stimIX, stimLabels, meanImage, ROIs, nullValue, start_time = obj.generateSummaryTable(createFiles = False, selectedDataStreams = selectedDataStreams, activityStream = activityStream, selectedFOV = selectedFOV, timeStep=0.1, trimStim = True)
    
    
    A  = obj.DB.require_group('Analyses')
    if obj.curFOV in A.keys(): #delete existing analysis if present; may want to amend this
        del A[obj.curFOV]
    FA = A.require_group(obj.curFOV) ## create/require a group for analysis of the current FOV
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
    FA.require_dataset('experiment_start', data = start_time, shape = start_time.shape, dtype = start_time.dtype)
    
    print('Analysis entered')
    
def prepAnalysisData(obj, FOV = None, source = None):
    if FOV is None:
        FOV = obj.curFOV
    data = LA.processTempStim(LA.prepAnalysis(obj.DB['Analyses'][FOV], closeAfterReading = False, source=None))
    data['source'] = source
    return(data)
    
def thermAnalyses(obj):
    data = prepAnalysisData(obj, source = f'{obj.DBpath} {obj.curAnimal} {obj.curFOV}')
    LA.TvsTcor(data)
    LA.thermoScatter(data)
    #cmap = LA.thermPolarPlot(data)
    output = LA.therm_polar_plot(data)
    cmap=output['cmap']
    LA.thermColorCode(data)
    


def isolateFOVtonewDB(obj): ## send an FOV to a new DB to work on without endangering rest of file
    # path, parent = os.path.split(obj.DBpath)
    Animal = obj.curAnimal
    # savePath = os.path.join(path, Animal + '_' + obj.curFOV + '.h5')
    savePath = selectFile(existing=False)
    NF = h5py.File(savePath, 'a')
    NF.require_group('Animals')
    NF['Animals'].require_group(Animal)
    for item in obj.FOVlist.selectedItems():
        FOV = item.text()
        print(f'Copying {FOV} to {savePath}...')
        start = time.time()
        obj.DB.copy(obj.DB['Animals'][Animal][FOV], NF['Animals'][Animal])
        
        print('Done copying')
        print(f'Transfer took {time.time()-start} seconds')
    print('All FOVs transferred')
    NF.close()
    
    
def isolateDataToNewDB(obj, create_new = False): ## send an FOV to a new DB to work on without endangering rest of file
    path, parent = os.path.split(obj.DBpath)
    Animal = obj.curAnimal
    FOV = obj.curFOV
    
    if create_new:
        savePath = os.path.join(path, parent + '_' + obj.curFOV+ '.h5')
    else:
        result = QFileDialog.getOpenFileName(obj, "Choose Database...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        savePath = os.path.normpath(result[0])
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

def rectifyMech(obj):
    trace, TIME = getSubStack(obj)
    trace[trace<-10] = -10
    return(trace, TIME)


def adjust_baseline_mech(obj):
    trace, TIME = getSubStack(obj)
    output=trace*0
    ## split trace by trials
    trace[np.where(trace<0)] = 0
    separation_gap = 10 ## separate wherever there is 10s gap or more between data points
    stopIXs = np.where(np.diff(TIME)>separation_gap)[0]
    startIXs = stopIXs+1
    startIXs = np.insert(startIXs, 0, 0)
    stopIXs = np.append(stopIXs, TIME.shape[0]-1)
    for start, stop in zip (startIXs, stopIXs):
        baseline = np.quantile(trace[start:stop], 0.1)
        trace[start:stop] = trace[start:stop] - baseline

    return(trace, TIME)

def normCOR(obj):
    inputMovie, TIME = getSubStack(obj)
    correctedMovie = DYnormCOR.normCOR(inputMovie, obj)
    print(correctedMovie.shape)
    return(correctedMovie, TIME)


def mouseSetup():
    ## Organize experimental animals - link mice to genotypes, identify DB files
    #populations = ['Tacr1', 'Gpr83', 'rPbN', 'Gpr83-homo', 'Tacr1-homo', 'CAPS']
    populations = []
    Mice = {}
    Mice['Tacr1'] = ['237',
                     '414', 
                     '573', 
                     '5896', 
                     '757', 
                     '685', 
                     '696', 
                     '687', 
                     '439', 
                     '585',
                     '7243']
                     #'7278']
    Mice['Gpr83'] = ['6356',
                     '6046',
                     '6355', 
                     '6044',
                     '6048',
                     '6041',
                     '704',
                     '8248']# # #247
    Mice['rPbN'] = ['457', 
                    '594', 
                    '379', 
                    '290',
                    '291',
                    '431',  ### no useable data?
                    '794', ## no useable data?
                    '7778',
                    '7242',
                    '7241',
                    '8243',
                    '8244',
                    '3A12'] ##actually gpr83 uninudced/rpbn-cre/Ai95
    Mice['Vglut2'] = ['9736']
    Mice['Trh-germline'] = []
    Mice['Trh-php.eB'] = []
    #Mice['rPbN'] = ['7778']
    Mice['Histamine'] = ['8243'] # abridged thermo , missing 45/48
    Mice['CAPS'] = ['7778','7242','7241','8244'] ## 7278, 7243
    
    Mice['SNI'] = ['237', '414','573','594','457','379','6356','6406']
    Mice['SNIgpr'] = ['6356','6406']
    Mice['SNItac'] = ['237', '414','573']
    Mice['SNIpbn'] = ['594','457','379']
    Mice['Mech'] = ['8244','7778','7242','7241','757'] ## 6856 has comparison of aurora and evf
    Mice['Mech-rPbN'] = ['8244','7778','7242','7241']
    Mice['Mech-thermo'] = ['7241', '8244', '7242', '7778']
    Mice['More mech thermo'] = ['9736']
    Mice['Mech-basal'] = ['7241']
    Mice['longitudinal'] = ['573','379','457','594']
    Mice['All'] = []
    for key in Mice.keys():
        populations.append(key)
        for mouse in Mice[key]:
            if not (mouse in Mice['All']):
                Mice['All'].append(mouse)

    return(Mice, populations)

def data_folder_dict():
    data_folders = {}
    data_folders['Basal thermo'] = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Basal thermo tuning'
    data_folders['All'] =  '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed'
    return(data_folders)
    
def locate_DBs(popFilter = ['All'], mouse_filter = None, data_folder = 'All', non_redundant=False): #popFilter is list of groups to retrieve DB paths for, mouse filter is specific mouse IDs
    
    data_folders = data_folder_dict()
    if data_folder is None:
        data_folder = 'All'
    if data_folder in data_folders:
        data_folder = data_folders[data_folder]

    print(f'Looking in folder {data_folder}')

    Mice, populations = mouseSetup()
    if not popFilter is None:
        if isinstance(popFilter, list):
            populations = popFilter
        else:
            populations = [popFilter]
            
    if mouse_filter is None:
        mouse_filter = Mice['All']
            
    DBs = {}
    
    for pop in populations:
        DBlist=[]
        if pop in Mice.keys():
            for mouse in Mice[pop]:
                if mouse in mouse_filter:
                    temp_list = []
                    temp_list.extend(glob.glob(f'{data_folder}/*{mouse}*.h5', recursive=True))
                    #DBlist.extend(found_DBs)
                    temp_list.extend(glob.glob(f'{data_folder}/*/{mouse}*.h5', recursive=True))
                    #DBlist.extend(found_DBs)
                    temp_list.extend(glob.glob(f'{data_folder}/*/*{mouse}*.h5', recursive=True))
                    #DBlist.extend(found_DBs)
                    found_DBs = temp_list
                    print(f'{len(found_DBs)} .h5 files found found for mouse {mouse}')
                    if non_redundant and len(found_DBs) > 1:
                        found_DBs = found_DBs[0]
                    DBlist.extend(found_DBs)
        DBs[pop] = list(set(DBlist))
    
   
    return(DBs)

def SNItimes():
    SNIdates = {}
    SNIdates['237'] = (2021,5,6,12,0,0,-1,-1,-1)
    SNIdates['414'] = (2021,6,4,12,0,0,-1,-1,-1)
    SNIdates['573'] = (2021,12,14,10,33,6,-1,-1,-1)
    SNIdates['594'] = (2021,12,14,13,27,8,-1,-1,-1)
    SNIdates['457'] = (2021,12,14,14,27,11,-1,-1,-1)
    SNIdates['379'] = (2021,12,14,11,20,43,-1,-1,-1)
    SNIdates['6356'] = (2022,2,8,12,0,0,-1,-1,-1)
    SNIdates['6046'] = (2022,2,8,10,27,0,-1,-1,-1)
    SNIdates['704'] = (2021,5,6,12,0,0,-1,-1,-1)
    SNIdates['6355'] = (2022,7,26,12,0,0,-1,-1,-1)
    SNIdates['7241'] = (2022,7,26,13,31,0,-1,-1,-1)
    SNIdates['7243'] = (2022,7,26,14,3,0,-1,-1,-1)

    return(SNIdates)

def mechThermoCompare(data_folder=None):
    ## select mice thta are rPbN and are on list for mech-thermo analysis
    DBs = locate_DBs(popFilter = 'Mech-rPbN', data_folder=data_folder)
    return(DBs)
            
        

def CAPSfigs():
    thermoTuning(FOVtag='CAPS', activeTag = 'pre CAPS', dataTags=['pre CAPS', 'Thermo stim'], genotypes = ['CAPS'])
    thermoTuning(FOVtag='CAPS', activeTag = 'post CAPS', dataTags=['post CAPS', 'Thermo stim'], genotypes = ['CAPS'])
    

def get_multi_session(mouse, FOVtag = None, activeTag = None, dataTags = None, norm = False):
    if FOVtag is None:
        FOVtag = 'Chronic'
    if activeTag is None:
        activeTag = 'Aligned ca'
    if dataTags is None:
        dataTags = ['Aligned ca', 'Thermo stim', 'Mech stim']
        
    sessions = get_sessions(mouse, FOVtag = FOVtag, activeTag = activeTag, dataTags = dataTags, norm = norm)
    multi_session = LA.multi_session(sessions)
    return(multi_session)
    
def get_sessions(mice, pop_filter = None, FOVtag = None, activeTag = None, dataTags = None, norm = False, data_folder = None, return_failures=False, z_score=False):
    
    DBs = locate_DBs(mouse_filter = mice, data_folder = data_folder, popFilter = pop_filter)
    print(DBs)
    sessions = []
    failures = []
    
    for population in DBs:
        for DB in DBs[population]:
            datas = scrape(DB, FOVtag = FOVtag, activeTag = activeTag, dataTags = dataTags, z_score=z_score)
            if len(datas) == 0:
                failures.append(DB)
            for data_file in datas:
                data_file = LA.processTempStim(data_file, plot=False)
                data_file = LA.processMechStim(data_file, plot=False)
                data_file = LA.classify_cells_in_dataset(data_file)
                sessions.append(LA.session_data(data_file))
    if return_failures:
        return(sessions, failures)
    return(sessions)
   

    
def scrapePrep(dbPath, FOVtag = None, activeTag = None, dataTags = None): 
    ## From location of .h5 file, gets inputs to create summary table and prep data for further analysis
    ## Uses attribute tags of FOV and datastreams to choose what to extract
    if FOVtag == None:
        FOVtag = 'Use for basal'
    if activeTag == None:
        activeTag = 'Ca data'
    if dataTags == None:
        dataTags = ['Ca data', 'Thermo stim']
    
    obj = dbContainer(dbPath)
    outputs = {}
   
    for animal in obj.DB['Animals'].keys():
        for FOV in obj.DB['Animals'][animal].keys():
            if FOVtag in obj.DB['Animals'][animal][FOV].attrs:
                if obj.DB['Animals'][animal][FOV].attrs[FOVtag]:
                    print(f'{FOVtag} matches in {animal} {FOV}')
                    ## If FOV has a True value for FOVtag attribute, add field in dictionary:
               #     print(f'{FOVtag} matches in {animal} {FOV}')
                    outputs[FOV] = {}
                    outputs[FOV]['selectedDataStreams'] = []
                    outputs[FOV]['activityStream'] = None
                    
                    ## Look for datastreams corresponding to datatags (any data to be added)
                    ## or to activeTag (specified calcium imaging data)
                    
                    for datakey in obj.DB['Animals'][animal][FOV].keys():
                        if activeTag in obj.DB['Animals'][animal][FOV][datakey].attrs:
                            if obj.DB['Animals'][animal][FOV][datakey].attrs[activeTag]:
                                outputs[FOV]['activityStream'] = datakey
                        for dataTag in dataTags:
                            if dataTag in obj.DB['Animals'][animal][FOV][datakey].attrs:
                                if obj.DB['Animals'][animal][FOV][datakey].attrs[dataTag]:
                                    outputs[FOV]['selectedDataStreams'].append(datakey)
                       
                    ## if no activity tagged in FOV, remove FOV from output
                    if outputs[FOV]['activityStream'] == None:
                        print(f'No activity data in FOV {FOV}')
                        del outputs[FOV]
                        
                else:
                    print(f'{FOVtag} does not match in {animal} {FOV}')
                    
            
    obj.close()
    return(outputs)



def scrape(dbPath, FOVtag = None, activeTag = None, dataTags = None, source = None, z_score=False):
    try:
        selectedData = scrapePrep(dbPath, FOVtag = FOVtag, activeTag = activeTag, dataTags = dataTags)
        datas=[]
        
        for FOV in selectedData.keys():
            DATA = getData(dbPath, selectedFOV=FOV, selectedDataStreams=selectedData[FOV]['selectedDataStreams'], activityStream = selectedData[FOV]['activityStream'], z_score=z_score)
            DATA['source'] = 'DB:'+ dbPath + ' FOV:' + FOV
            DATA['activity key'] = selectedData[FOV]['activityStream']
            DATA['mouse'] = os.path.split(dbPath)[-1].split('.')[0]
            datas.append(DATA)
        return(datas)          
    except:
        return([])


        

def get_data_files(FOVtag=None, activeTag = None, split_genotypes = True, genotypes = None, dataTags = None, trimStim = True, popFilter = None, data_folder = None):
    ## retrieves processed data files for specified data 
    ## activeTag - tag of datastreams with calcium dta
    if genotypes == None:
        genotypes = ['Tacr1', 'Gpr83', 'rPbN']
    DBs=locate_DBs(popFilter = popFilter, data_folder = data_folder)
    #pdb.set_trace()
    DBs_good = {}
    DBs_bad = {}
    data_files = {}
    for genotype in genotypes:
        data_files[genotype] = {}
        DBs_good[genotype] = []
        DBs_bad[genotype] = []
        genDBs = DBs[genotype]
        foundData = False
        for DB in genDBs:
            data_files[genotype][DB] = []
            data_found =  False
            #try:
                
            datas = scrape(DB, FOVtag = FOVtag, activeTag=activeTag, dataTags=dataTags)
            if len(datas)>0:
                 data_found = True
        # except:
            else:
                 print(f'scraping failed for {DB}')
                 DBs_bad[genotype].append(DB)
                 data_found = False
                
            if data_found:
                data_files[genotype][DB].extend(datas)
                DBs_good[genotype].append(DB)
            
    output={}
    output['goodDBs'] = DBs_good
    output['badDBs'] = DBs_bad
    output['data'] = data_files
    return(output)

def track_probe(obj, stack = None, TIME = None):
    
    if stack is None:
        stack, TIME = getSubStack(obj)
    stack = np.average(stack, axis=3).astype(np.float64)
    output = np.zeros([stack.shape[0],stack.shape[1],stack.shape[2]])
    m_key = get_data_keys_from_flag(obj, 'Mech stim')[0]
    mech_trace = obj.DB['Animals'][obj.curAnimal][obj.curFOV][m_key][...]
    mech_time = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][m_key][...]
    Mstim  = LA.process_mech_trace(mech_trace, plot=True)
    for stim in Mstim.values():
        m_start = mech_time[stim.start]
        n_start = np.searchsorted(TIME, m_start)
        if n_start < 0 or n_start > stack.shape[0]-1:
            return(output, TIME)
        template = stack[n_start-6,:,:]
        for f in range(n_start-25, n_start+10):
            difference_p = np.absolute(stack[f-1,:,:] - template)
            difference_n = np.absolute(stack[f,:,:] - template)
            output[f] = np.absolute(difference_p-difference_n)
   
    return(output, TIME)
        
    
    
    
def thermal_tuning(FOVtag=None, activeTag = None,  split_genotypes = True, genotypes = None, dataTags = None, trimStim = True, plotMech = False, classify_func=None):
    
    
    if genotypes == None:
        genotypes = ['Tacr1', 'Gpr83', 'rPbN']
        
    prep = get_data_files(FOVtag=FOVtag, activeTag = activeTag, genotypes = genotypes, dataTags = dataTags, trimStim = trimStim)
    
    datas = prep['data']
    DBs = prep['goodDBs']

    
    cell_counts = {}   ### Dictionary with total # of cells for 
    tuning_angles = {}
    class_counts = {}
    class_counts['all'] = {}
    class_defs = []
    mouse_count = {}
    
    
    if classify_func == None:
        classify_func = LA.classify_cells_in_dataset
    
    for genotype in genotypes:
        cell_counts[genotype] = 0
        mouse_count[genotype] = 0
        class_counts[genotype] = {}
        tuning_angles[genotype] = np.array([])
        for mouse in datas[genotype]:
            class_counts[mouse]= {}
            mouse_datas = datas[genotype][mouse]
            mouse_count[genotype] = mouse_count[genotype] + 1
            for DATA in mouse_datas:
               
                
                n_cells = len(DATA['cells'])
                cell_counts[genotype] = cell_counts[genotype] + n_cells
                
                DATA = classify_func(DATA)
                #pdb.set_trace()
                counts = LA.count_cell_classes(DATA)
                
                for count_level in [mouse, genotype, 'all']:
    
                        for key in counts:
                            if not (key in class_defs):
                                class_defs.append(key) ## collect tags used to classify cell types
                                
                            if key in class_counts[count_level]:
                                class_counts[count_level][key] = class_counts[count_level][key] + counts[key] ## Increment counts for DB(mouse), genotype, total
                            else:
                                class_counts[count_level][key] = counts[key]
                                
                response_stats = LA.thermPolarPlot(DATA, F = None, A=None, plot_on=False)
                tuning_angles[genotype] = np.hstack([tuning_angles[genotype], response_stats['thetas']]) ## append to list of tuning angles for animal
    
    output={}
    output['data_files'] = datas
    output['cell counts'] = cell_counts
    output['tuning angles'] = tuning_angles
    output['class counts'] = class_counts
    output['class_defs'] = class_defs
    output['DBs'] = DBs
    output['genotypes'] = genotypes
    
    
    # output['data_files'] = data_files
    output['failed'] = prep['badDBs']
    output['success'] = prep['goodDBs']
   
    
    
    plot_thermo_tuning_multi_animal(output, split_genotypes= split_genotypes)
    tuning_summary(output)
    return(output)

    
    



def plot_thermo_tuning_multi_animal(input_data, split_genotypes = True, FOVtag=None, activeTag = None, dataTags = None, trimStim = True, plotMech = False):     
    
    # takes output of thermal_tuning as input data, plots histrogram of tuning angles and 
    # polar plots of thermal tuning for each genotype
    figure_list = []
    figure_names = []
    genotypes = input_data['genotypes']
    if not split_genotypes:
            F = plt.figure(f'Thermo coded {"-".join(genotypes)} {FOVtag} {activeTag}')
            figure_list.append(F)
            figure_names.append(f'Thermo coded {"-".join(genotypes)} {FOVtag} {activeTag}.pdf')
            A = F.add_axes([0, 0, 1, 1])
            
    tuning_angles = input_data['tuning angles']
    
    n_bins = 30
        
    num_plots = len(tuning_angles)
    G = plt.figure()
    figure_list.append(G)
  
    figure_names.append(f'Tuning histogram {"-".join(genotypes)} {FOVtag} {activeTag}.pdf')
    B=[]
    for count, genotype in enumerate(tuning_angles):
        B = G.add_axes([0, count/num_plots, 1, 1/(num_plots+1)])
        B.hist(np.degrees(tuning_angles[genotype]), bins = n_bins, range = [-180, 180], density=True, color='k')
        B.set_frame_on(False)
        if count>0:
            B.xaxis.set_visible(False)
        else:
            B.set_xticks([ -90, -30, 90])
            B.set_xticklabels(['Warm','Hot','Cold'])
        B.set_ylabel(genotype)
            
    for genotype in input_data['data_files']:
        draw_axes=True
        fName = f'Polar plot {genotype} {FOVtag} {activeTag}'
        F = plt.figure(fName)
        figure_list.append(F)
        figure_names.append(fName)
        A = F.add_axes([0, 0, 1, 1])
        for DB in input_data['data_files'][genotype].keys():
            LA.therm_polar_plot(input_data['data_files'][genotype][DB], F=F, A=A, draw_axes=draw_axes, alpha=0.2)
            draw_axes = False
    
    #G.savefig('/lab-share/Neuro-Woolf-e2/Public/TuningHistogram.pdf')
    for fig, figname in zip(figure_list, figure_names):
        base = '/lab-share/Neuro-Woolf-e2/Public/Figure publishing'
        filename = os.path.join(base, figname +'.pdf')
        print(f'Saving {filename}')
        fig.savefig(filename)
        
        
        
        
        


def tuning_summary(d):
    ## called by thermal_tuning to make sub_figure with bar graph of 
    ## thermal class types
    if 'total' in d['class_defs']:
        d['class_defs'].remove('total')
    if None in d['class_defs']:
        d['class_defs'].remove(None)
    genotype_class_table = np.zeros( [len(d['genotypes']), len(d['class_defs'])]) 
    
    ## Make table of counts for each genotype and each respone class
    for ii, genotype in enumerate(d['genotypes']):
        for jj, class_type in enumerate(d['class_defs']):
            if class_type in d['class counts'][genotype].keys():
                genotype_class_table[ii,jj]  = d['class counts'][genotype][class_type]
    
    genotype_class_table_normalized = genotype_class_table/genotype_class_table.sum(axis=1, keepdims=True)
    ## Make table of counts for each mouse within each genotype
    mouse_class_tables = {}
    mouse_class_tables_normalized={}
    for genotype in d['genotypes']:
        mice = d['DBs'][genotype]
        mouse_class_table = np.zeros( [len(mice), len(d['class_defs'])])
        for ii, mouse in enumerate(mice):
            if mouse in d['class counts']:
                for jj, class_type in enumerate(d['class_defs']):
                    if class_type in d['class counts'][mouse].keys():
                        mouse_class_table[ii,jj]  = d['class counts'][mouse][class_type]
        
        
        #remove any rows from table if 0 for all classes
        mouse_class_table = mouse_class_table[~np.all(mouse_class_table == 0, axis=1)]
        mouse_class_tables[genotype] = mouse_class_table
        mouse_class_tables_normalized[genotype] = mouse_class_table/mouse_class_table.sum(axis=1, keepdims=True)
                
         
        
    
    
    out={}
    out['genotype raw counts'] = genotype_class_table
    out['genotype normalized counts'] = genotype_class_table_normalized
    
    out['genotypes'] = d['genotypes']
    out['classes'] = d['class_defs']
    
    out['mouse raw counts'] = mouse_class_tables
    out['mouse normalized counts'] = mouse_class_tables_normalized
    
    num_classes = len(d['class_defs'])
    num_genotypes = len(d['genotypes'])
    
    ##Plot scatter of proportions of cell types by genotype
    
    mat.rc('font', size = 16)
    F=plt.figure()
    A = F.add_axes([0, 0, 1, 1])
    gc = gen_colors()
    Nc = out['mouse normalized counts']
    Xtick_pos=[]
    Xtick_labels=[]
    for n_class, classification in enumerate(d['class_defs']):
        for n_geno, genotype in enumerate(d['genotypes']):
            Y = Nc[genotype][:,n_class]  ## proportion in class for each mouse of given genotype
            plot_pos = n_class*(num_genotypes+1)+n_geno
            X= np.ones(Y.shape)*plot_pos
            if n_geno == int(num_genotypes/2):
                Xtick_pos.append(plot_pos)
                Xtick_labels.append(classification)
            #pdb.set_trace()
            plt.bar(plot_pos, np.mean(Y), color=gc[genotype],alpha=0.75)
            jitter(X,Y, alpha = 1, s=40, edgecolors = gc[genotype], color = 'None', linewidths=0.5)
            
    A.set_xticks(Xtick_pos)
    A.set_xticklabels(Xtick_labels)
    A.set_frame_on(False)   
    A.set_yticks([0,0.5,1])
    A.set_yticklabels(['0','0.5','1'])
    
    F.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/therm tuning by genotype.pdf')
    return(out)

# def jitter(X,Y, s=20, color=None, scale = 1, cmap=None, norm=None, vmin = None, vmax = None, alpha = 0.5, linewidths=None, edgecolors=None, A=None, **kwargs):
#     if A is None:
#         A = plt.gca()
#     rng = np.random.default_rng()
#     v = rng.uniform(low=-0.1, high = 0.1, size=X.size)
#     X = X+v/scale
#     A.scatter(X,Y, s=s, color=color, cmap=cmap, norm=norm, vmin = vmin, vmax = vmax, alpha = alpha, linewidths=linewidths, edgecolors=edgecolors, **kwargs)
    
    
# def gen_colors():
#     ## standardized coloring for genotypes/conditions
#     gc={}
#     gc['Gpr83'] = 'g'
#     gc['Tacr1'] = 'r'
#     gc['rPbN'] = 'k'
#     gc['hot'] = 'r'
#     gc['warm'] = np.array([1,0.5,0])
#     gc['cold'] = 'b'
#     gc['poly'] = 'm'
#     gc['none'] = 'k'
#     gc[None] = 'k'
#     return(gc)


def SNItrends(genotypes = None, SNItag = 'SNI', basalTag = 'Use for basal', activeTag = None, mode=None, day_range = None, data_folder = None):
    ## for each SNI mouse, calculate proportion of cells in each tuning category vs time:
    SNI_time_data = SNItimes()
    if genotypes == None:
        genotypes = ['Tacr1', 'Gpr83', 'rPbN']
    DBs=locate_DBs(data_folder = data_folder)
    allMice = []
    cell_counts = {}
    mouse_count = {}
    mouse_count['total'] = 0
    failed = []
    nodata = []
    success = []
    results = {}
    F = plt.figure()
    A = F.add_axes([0, 0, 1, 1])
    for genotype in genotypes:
        genDBs = DBs[genotype]
        results[genotype] = {}
        cell_counts[genotype] = 0
        mouse_count[genotype] = 0
        for DB in genDBs:
            ## attempt to get data:
                
            foundData = False
            if 1==1: #try:
                datasSNI = scrape(DB, FOVtag = SNItag, activeTag=activeTag)
                datasBasal = scrape(DB, FOVtag = basalTag, activeTag=activeTag)
                if len(datasSNI)>0 and len(datasBasal)>0:
                    if mode == 'basal only':
                        datas = datasBasal
                    elif mode == 'SNI only':
                        datas = datasSNI
                    else:    
                        datasBasal.extend(datasSNI)
                        datas = datasBasal
                    
                    foundData = True
                    results[genotype][DB] = {}
                    mouse_count[genotype] = mouse_count[genotype] + 1
                    mouse_count['total'] = mouse_count['total'] + 1
                    for DATA in datas:
                        n_cells = len(DATA['cells'])
                        cell_counts[genotype] = cell_counts[genotype] + n_cells
                        
                else:
                    foundData = False
                    nodata.append(DB)
            else: ## except:
                print(f'Scraping failed for DB: {DB}')
                failed.append(DB)
                foundData = False
                #raise
            if foundData:
                success.append(DB)
                SNI_time = 0
                for SNI_mouse in SNI_time_data.keys():
                    if SNI_mouse in DB:
                        SNI_time = time.mktime(SNI_time_data[SNI_mouse])
                                               
                        
                if SNI_time == 0:
                    print(f'Could not find SNI time for mouse {DB}')
                    output={}
                    #output['TagsSNI'] = 
                    output['datas'] = datas
                    output['DB'] = DB
                    return(output)
                        
                results[DB] = {}
                ### Construct results structure with post SNI time and proportion of each tuning type
                for DATA in datas:
                    start_time = DATA['experiment_start'][...]
                    print(f'start_time: {start_time}')
                    print(f'SNI_time: {SNI_time}')
                    if start_time < SNI_time:
                        SNI_day = -1
                    else:
                        SNI_day = round((start_time-SNI_time)/86400) ## get days since SNI for imaging session (per FOV)
                    
                    ## get tuning angles
                    plt.figure()
                    
                    ## restrict to time bounds:
                    if not day_range == None:
                        if SNI_day < day_range[0]:
                            continue
                        if SNI_day > day_range[1]:
                            continue
                        
                    response_stats = LA.thermPolarPlot(DATA, plot_on = True)
                    
                    if SNI_day in results[genotype][DB].keys():
                        print(f'Keys: {results[genotype][DB].keys()}')
                        print(f'SNI_day: {SNI_day}')
                        results[genotype][DB][SNI_day]['tuning angles'] = np.hstack([results[genotype][DB][SNI_day]['tuning angles'][...], response_stats['thetas']])
                    
                    else:
                        results[genotype][DB][SNI_day] = {}
                        results[genotype][DB][SNI_day]['tuning angles'] = response_stats['thetas']
                        
    output = {}
    output['results'] = results
    output['failed'] = failed
    output['nodata'] = nodata
    output['success'] = success
    
    ## Define bins for warm, cold and hot populations in polar coordinates
    warmb = (-120, -60)
    hotb = (-60, 30)
    coldb = (30, 120)
    polyb = (0,60)
    
    
    G = plt.figure()
    C = G.add_axes([0, 0, 1, 1])
    for genotype in results.keys():
        for mouse in results[genotype].keys():
            timepoints = []
            phot = []
            pcold = []
            pwarm = []
            ppoly = []
            all_angles = []
            for timepoint in results[genotype][mouse].keys():
                tuning_radians = results[genotype][mouse][timepoint]['tuning angles'][...]
                print(tuning_radians.shape)
                try:
                    angles = np.degrees(tuning_radians)
                except:
                    return(tuning_radians)
                
                phot.append(count_bounds(angles, hotb)/angles.size)
                pwarm.append(count_bounds(angles, warmb)/angles.size)
                pcold.append(count_bounds(angles, coldb)/angles.size)
                ppoly.append(count_bounds(angles, polyb)/angles.size)
                timepoints.append(timepoint)
            
            output['results'][genotype][mouse]['time'] = timepoints
            output['results'][genotype][mouse]['phot'] = phot
            output['results'][genotype][mouse]['pcold'] = pcold
            output['results'][genotype][mouse]['pwarm'] = pwarm
            output['results'][genotype][mouse]['ppoly'] = ppoly
            C.plot(timepoints, phot, 'r')
            C.plot(timepoints, pcold, 'b')
            C.plot(timepoints, pwarm, 'y')
            C.plot(timepoints, ppoly, 'm')
            
    
    return(output)
            
                    
                   
                    
             
    
def count_bounds(array, bounds):
    # counts number of elements in array between two specifed bounds
    where = np.where((array>bounds[0]) & (array<bounds[1]))
    return(where[0].size)




def prep_longitudinal():
    DBpaths = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 457A/Mouse 457A - def.h5',
               '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 379/Mouse 379 definitive.h5',
               '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 594/Mouse 594 definitive.h5',
                None,
               '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 573/Mouse 573 definitive.h5']
    
    corrCharts = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 457A/457Acorrespondence.csv',
                  '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 379/379.csv',
                  '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 594/594.csv',
                   None,
                  '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 573/573.csv']
    return(DBpaths, corrCharts)

def run_longitudinal(DBpaths = None, corrCharts = None):
    if DBpaths == None:
        DBpaths, corrCharts = prep_longitudinal()
    output = [] 
    Figures = []             
    Axes = []
    
    F1 = plt.figure()
    A1 = F1.add_axes([0, 0, 1, 1])
    Figures.append(F1)
    Axes.append(A1)
    draw_axes=True
    for DBpath, corrFile in zip(DBpaths, corrCharts):
        if DBpath is None:
            F1 = plt.figure()
            A1 = F1.add_axes([0, 0, 1, 1])
            Figures.append(F1)
            Axes.append(A1)
            draw_axes=True
            
        else:
            
            F = Figures[-1]
            A = Axes[-1]
            #draw_axes = False#
            output.append(longitudinal_compare(DBpath = DBpath, corrFile = corrFile, F=F,A=A, draw_axes=draw_axes))
            draw_axes=False
    
    return(output)
        
        
                    

def longitudinal_compare(DBpath=None, corrFile = None, FOVtag='Chronic', F=None, A=None, draw_axes=False ):
    ## For a DB and cell correspondence file, gets data nd runs match_and_compare
    datas = scrape(DBpath, FOVtag = FOVtag, activeTag = 'Ca data', dataTags = ['Ca data', 'Thermo stim'])
    print(f'Datas: {datas}')
    output = LA.match_and_compare(datas, corrFile=corrFile, F=F, A=A, draw_axes=draw_axes)
    return(output)
    
    


def get_sessions_for_alignment(obj, stim_tags = ['Therm stim', 'Mech stim'], active_tag = 'Ca data'):  ## creates longitudinal session object from open .h5database

    ## set up data containers and options
    numSessions = 0
    session_FOVs = []
    session_ca_datas = []
    session_stims = []
    session_states = []
    session_types = []
    recording_types = ['thermo', 'mech', 'thermo-mech']
    pain_state = ['nociceptive','SNI','CAPS']
    
    ## select datasets to align:
    FOVs = ['Done selecting']
    for FOV in obj.DB['Animals'][obj.curAnimal].keys():
        FOVs.append(FOV)
    selecting = True
    selected_FOVs = []
    while selecting:
        #FOVs = obj.DB['Animals'][obj.curAnimal].keys()
        selected_FOV, okPressed = QInputDialog.getItem(obj,f"Select FOV for session {numSessions+1}:", "Data:", FOVs, 0, False)
        
        if okPressed != True:
            return()
        if selected_FOV == 'Done selecting':
            selecting = False
        else:
            selected_FOVs.append(selected_FOV)
            
    FOVs = selected_FOVs   
    obj.FOVflags['temp'] =  {}
   
    for FOV in FOVs:
        obj.DB['Animals'][obj.curAnimal][FOV].attrs['temp'] = True
        
    obj.closeDB()
    dbPath = obj.DBpath
    
    datas = scrape(dbPath, FOVtag = 'temp', activeTag = active_tag, dataTags = stim_tags)
    
    obj.reOpen(DB)
    
    for FOV in FOV:
        del obj.DB['Animals'][obj.curAnimal][FOV].attrs['temp']
    del obj.FOVflags['temp'] 
        
    sessions = []
    for DATA in datas:
        sessions.append(la.session_data(DATA))
        
    multi_session = la.multi_session(sessions)
    
    for session in multi_session.sessions:
        session.show_plots()
    
    return(multi_session)

   
    
    
def correspond_manual_from_sessions(obj, FOVs = None, pickled_sessions=None, show_raster = True):
    
    ##open up interactive session to assign correspondence across trials
    
    ## Open or create multi-session FOV (list of FOVs and a correspondence table)
    
    
    sessions = []
    Animal = obj.curAnimal 
    if FOVs is None:
        FOVs = []
        pickled_sessions = []
        for item in obj.FOVlist.selectedItems():
            FOV = item.text()
            FOVs.append(FOV)
            if 'session file' in obj.DBpath['Animals'][Animal][FOV].attrs:
                pickle_sessions.append(obj.DBpath['Animals'][Animal][FOV].attrs['session file'])
            else:
                print(f'No session file found for FOV {FOV}')
    for pickled_session in pickled_sessions:
        sessions.append(unpickle(pickled_session))
        if show_raster:
            sessions[-1].show_raster()
            
    return()
    
                
                
    
    
    
    
        
        
        
        #datas=[]
        #for data_stream in obj.DB['Animals'][obj.curAnimal][selected_FOV].keys():
        #    datas.append(data_stream)
        #datas.append('Done selecing')
        #selectedData, okPressed = QInputDialog.getItem(obj,"Select data for session {numSessions+1}:", "Data:", datas, 0, False)
        #if okPressed != True:
        #    return()
        #if selectedData == 'Done selecting':
        #    selecting = False
        #    break
        
        ### can put in some dialogs to detect pain state and recording type later
        
   #     numSessions = numSessions + 1
        #session_FOVs.append(selected_FOV)
        #session_ca_datas.append(selectedData)
        
        
        
#    for session, data in zip(session_FOVs, session_ca_datas):
        
    #    pass
    
    




# def thermoTuning(FOVtag=None, activeTag = None, split_genotypes = True, genotypes = None, dataTags = None, trimStim = True, plotMech = False):
#     if genotypes == None:
#         genotypes = ['Tacr1', 'Gpr83', 'rPbN']
#     DBs=locate_DBs()
#     results = {}
#     failed = []
#     nodata = []
#     success = []
#     cell_counts = {}
#     tuning_angles = {}
#     mouse_count = {}
#     figure_list=[]
#     figure_names=[]
#     if not split_genotypes:
#             F = plt.figure(f'Thermo coded {"-".join(genotypes)} {FOVtag} {activeTag}')
#             figure_list.append(F)
#             figure_names.append(f'Thermo coded {"-".join(genotypes)} {FOVtag} {activeTag}.pdf')
#             A = F.add_axes([0, 0, 1, 1])
#     for genotype in genotypes:
#         cell_counts[genotype] = 0
#         mouse_count[genotype] = 0
#         tuning_angles[genotype] = np.array([])
#         if split_genotypes:
#             F = plt.figure(f'Thermo coded {genotype} {FOVtag} {activeTag}')
#             figure_names.append(f'Thermo coded {genotype} {FOVtag} {activeTag}.pdf')
#             figure_list.append(F)
#             A = F.add_axes([0, 0, 1, 1])
#         print(f'{genotype=}')
#         genDBs = DBs[genotype]
#         foundData = False
#         for DB in genDBs:
            
#             #print(DB)
#             try:
#                 datas = scrape(DB, FOVtag = FOVtag, activeTag=activeTag, dataTags=dataTags)
#                 #ssreturn(datas) ### for debugging purposess
#                 if len(datas)>0:
#                     foundData = True
#                     for DATA in datas:
#                         n_cells = len(DATA['cells'])
#                         cell_counts[genotype] = cell_counts[genotype] + n_cells
#                         mouse_count[genotype] = mouse_count[genotype] + 1
#                 else:
#                     foundData = False
#                     nodata.append(DB)
#             except:
#                 print(f'Scraping failed for DB: {DB}')
#                 failed.append(DB)
#                 foundData = False
#                 #raise
                
#             if foundData:
#                 success.append(DB)
                
#                 for DATA in datas: 
                    
#                     response_stats = LA.thermPolarPlot(DATA, F = F, A=A)
#                     tuning_angles[genotype] = np.hstack([tuning_angles[genotype], response_stats['thetas']])
    
#     n_bins = 30
        
        
#     num_plots = len(tuning_angles)
#     G = plt.figure()
#     figure_list.append(G)
#     figure_names.append(f'Tuning histogram {"-".join(genotypes)} {FOVtag} {activeTag}.pdf')
#     B=[]
#     for count, genotype in enumerate(tuning_angles):
#         B = G.add_axes([0, count/num_plots, 1, 1/(num_plots+1)])
#         B.hist(np.degrees(tuning_angles[genotype]), bins = n_bins, range = [-180, 180], density=True, color='k')
#         B.set_frame_on(False)
#         if count>0:
#             B.xaxis.set_visible(False)
#         else:
#             B.set_xticks([ -90, -30, 90])
#             B.set_xticklabels(['Warm','Hot','Cold'])
#         B.set_ylabel(genotype)
            
        
#     #G.savefig('/lab-share/Neuro-Woolf-e2/Public/TuningHistogram.pdf')
#     for fig, figname in zip(figure_list, figure_names):
#         base = '/lab-share/Neuro-Woolf-e2/Public/Figure publishing'
#         filename = os.path.join(base, figname)
#         print(f'Saving {filename}')
#         fig.savefig(filename)
        
#     return(failed, success, nodata, cell_counts, tuning_angles)


         
# def thermoTuning2(FOVtag=None, activeTag = None, split_genotypes = True, genotypes = None, dataTags = None, trimStim = True, plotMech = False):
#     if genotypes == None:
#         genotypes = ['Tacr1', 'Gpr83', 'rPbN']
#     DBs=locate_DBs()
#     results = {}
#     failed = []
#     nodata = []
#     success = []
#     cell_counts = {}
#     tuning_angles = {}
#     class_counts = {}
#     class_counts['all'] = {}
#     class_defs = []
#     mouse_count = {}
#     figure_list=[]
#     figure_names=[]
#     DBs_good = {}
#     data_files = {}
#     if not split_genotypes:
#             F = plt.figure(f'Thermo coded {"-".join(genotypes)} {FOVtag} {activeTag}')
#             figure_list.append(F)
#             figure_names.append(f'Thermo coded {"-".join(genotypes)} {FOVtag} {activeTag}.pdf')
#             A = F.add_axes([0, 0, 1, 1])
#     for genotype in genotypes:
#         data_files[genotype] = {}
#         DBs_good[genotype] = []
#         cell_counts[genotype] = 0
#         mouse_count[genotype] = 0
#         class_counts[genotype] = {}
#         tuning_angles[genotype] = np.array([])
#         if split_genotypes:
#             F = plt.figure(f'Thermo coded {genotype} {FOVtag} {activeTag}')
#             figure_names.append(f'Thermo coded {genotype} {FOVtag} {activeTag}.pdf')
#             figure_list.append(F)
#             A = F.add_axes([0, 0, 1, 1])
#        # print(f'{genotype=}')
#         genDBs = DBs[genotype]
#         foundData = False
#         for DB in genDBs:
#             data_files[genotype][DB] = {}
#             #print(DB)
#             try:
#                 datas = scrape(DB, FOVtag = FOVtag, activeTag=activeTag, dataTags=dataTags)
                
#                 if len(datas)>0:
#                     return(datas) #ssreturn(datas) ### for debugging purposess
#                     foundData = True
#                     for DATA in datas:
#                         data_files[genotype][DB].append(DATA)
#                         n_cells = len(DATA['cells'])
#                         cell_counts[genotype] = cell_counts[genotype] + n_cells
#                         mouse_count[genotype] = mouse_count[genotype] + 1
#                 else:
#                     foundData = False
#                     nodata.append(DB)
#             except:
#                 print(f'Scraping failed for DB: {DB}')
#                 failed.append(DB)
#                 foundData = False
#                 #raise
#             if foundData:
#                 success.append(DB)
#                 DBs_good[genotype].append(DB)
#                 results[DB] = {}
#                 class_counts[DB]= {}
#                 for count, DATA in enumerate(datas):
#                     DATA = LA.classify_cells_in_dataset(DATA) ## add thermo classifications to cellss
#                     counts = LA.count_cell_classes(DATA)
                    
#                     for count_level in [DB, genotype, 'all']:
    
#                         for key in counts:
#                             if not (key in class_defs):
#                                 class_defs.append(key) ## collect tags used to classify cell types
                                
#                             if key in class_counts[count_level]:
#                                 class_counts[count_level][key] = class_counts[count_level][key] + counts[key]
#                             else:
#                                 class_counts[count_level][key] = 0
#                     response_stats = LA.thermPolarPlot(DATA, F = F, A=A)
#                     tuning_angles[genotype] = np.hstack([tuning_angles[genotype], response_stats['thetas']])
    
#     n_bins = 30
        
        
#     num_plots = len(tuning_angles)
#     G = plt.figure()
#     figure_list.append(G)
#     figure_names.append(f'Tuning histogram {"-".join(genotypes)} {FOVtag} {activeTag}.pdf')
#     B=[]
#     for count, genotype in enumerate(tuning_angles):
#         B = G.add_axes([0, count/num_plots, 1, 1/(num_plots+1)])
#         B.hist(np.degrees(tuning_angles[genotype]), bins = n_bins, range = [-180, 180], density=True, color='k')
#         B.set_frame_on(False)
#         if count>0:
#             B.xaxis.set_visible(False)
#         else:
#             B.set_xticks([ -90, -30, 90])
#             B.set_xticklabels(['Warm','Hot','Cold'])
#         B.set_ylabel(genotype)
            
        
#     #G.savefig('/lab-share/Neuro-Woolf-e2/Public/TuningHistogram.pdf')
#     for fig, figname in zip(figure_list, figure_names):
#         base = '/lab-share/Neuro-Woolf-e2/Public/Figure publishing'
#         filename = os.path.join(base, figname)
#         print(f'Saving {filename}')
#         fig.savefig(filename)
 
#     output={}
#     output['data_files'] = data_files
#     output['failed'] = failed
#     output['success'] = success
#     output['no data'] = nodata
#     output['DBs good'] = DBs_good
#     output['cell counts'] = cell_counts
#     output['tuning angles'] = tuning_angles
#     output['class counts'] = class_counts
#     output['class_defs'] = class_defs
#     output['DBs'] = DBs
#     output['genotypes'] = genotypes
    
#     return(output)
