#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:50:03 2022

@author: ch184656
"""

import os
import numpy as np
import imageio
import cv2
from matplotlib import pyplot as plt
import time
import h5py
import libAnalysis as LA
from scipy import stats


class dbContainer:
    def __init__(self, DBpath, DB = None):
        if DB is None:
            self.DB = h5py.File(DBpath,'a')
        else:
            self.DB = DB
        self.DB.require_group('Animals')
        self.DBpath = DBpath
        self.curAnimal = ''
        self.curFOV = ''
        
    def updateDataList(self):
        return
    
    def close(self):
        self.DB.close()

def genSumTable(obj, selectedDataStreams = None, activityStream = None, selectedFOV = None, timeStep=0.1, doPlotRaster=False, trimStim = True, z_score=False): 
        ## for each FOV generate data table with stimuli and responses in common time frame
        ## obj - database container object
        ## activityStream = data key for data containing calcium imaging data
      
        
      
        for animal in obj.DB['Animals'].keys():
            obj.curAnimal = animal
            break
       
        
        if selectedFOV == None: ## if none selected, return list of FOV keys to choose form
            FOVs = []
            for FOV in obj.DB['Animals'][obj.curAnimal].keys():
                FOVs.append(FOV)
 #           print(FOVs)
            print('No FOV selected, returning list of available FOVs')
            return(FOVs)
        else:
            obj.curFOV = selectedFOV
            
        if activityStream == None: #if none selected, return list of data streams in selected FOV
            datas = []
            for datakey in obj.DB['Animals'][obj.curAnimal][selectedFOV].keys():
                datas.append(datakey)
   #         print(obj.DB)
   #         print(obj.curAnimal)
   #         print(selectedFOV)
   #         print(datas)
            return(datas)
        else:
            obj.dataFocus = activityStream
            

        ## First create paired dictionaries of Xdata(timestamps) and Ydata(measurements)
        FOV = obj.DB['Animals'][obj.curAnimal][obj.curFOV]
  
        Xdata = {}
        Ydata = {}
        Labels = {}
        
        

        
     #   selectedROI.sort()
     #   selectedROI = np.array(selectedROI, dtype=np.uint16)
     #   print(f'selected for report: {selectedROI}')
        
        
        
        for datastream in selectedDataStreams:
            #datastream = item.text()
            if len(FOV[datastream].shape) ==1:  ## If data is 1d (force data typically)
                trace = FOV[datastream][...]   
                Ydata[datastream] = np.expand_dims(trace, axis = 0)
                #plt.plot(Ydata[datastream])
                #plt.show()
            elif len(FOV[datastream].shape) >= 3:
                
                    
                    
                    
                if datastream == obj.dataFocus:
                    print('selecting cells:')
                    Ydata[datastream] = FOV['R'][datastream]['traceArray'][...]##[selectedROI,...]
                    floatMask = FOV['R'][datastream]['floatMask'][...]##[:,:,selectedROI]
                    flatLabel = np.max(FOV['R'][datastream]['labelMask'][...], axis=2)##[:,:,selectedROI], axis=2)
                else:
                    Ydata[datastream] = FOV['R'][datastream]['traceArray'][...]
                    floatMask = FOV['R'][datastream]['floatMask'][...]
                    flatLabel = np.max(FOV['R'][datastream]['labelMask'][...], axis=2)
                    
                    
                    
                if z_score and not ('FLIR' in datastream):
                    print(f'Calculating z-scored dff for datastream {datastream} in FOV {obj.curFOV} animal {obj.curAnimal}')
                    floatMask = FOV['R'][datastream]['floatMask'][...]
                    stack = FOV[datastream][...]
                    vStack = np.reshape(stack, [stack.shape[0], stack.shape[1]*stack.shape[2]], order = 'F')
                    mask = floatMask>0
                    if len(mask.shape) <3:
                        mask = np.expand_dims(mask, axis = 2)
                    oMask = np.moveaxis(mask,[0,1,2],[1,2,0])
                    vMask = np.reshape(oMask, [oMask.shape[0], oMask.shape[1]*oMask.shape[2]], order = 'F')
                    Ydata[datastream] = np.zeros([vMask.shape[0], stack.shape[0]])
                    for cell, ROI in enumerate(vMask):
                        vector = np.percentile(vStack[:,ROI], 50, axis = 1)
                        m = np.median(vector)
                        mad = stats.median_abs_deviation(vector)
                        k = 1.4826
                        Ydata[datastream][cell,:] = (vector-m)/(k*mad)
                    
                meanImage = np.median(FOV[datastream][0:10,...], axis = 0)
                meanImage = meanImage - np.amin(meanImage)
                meanImage = meanImage/np.amax(meanImage)
                meanImage = meanImage*255
                meanImage = meanImage.astype(np.uint8)
                
                if datastream == activityStream: ##obj.dataFocus:
                    outMeanImage = meanImage
                
                
                    
                
        
                for cell in range(floatMask.shape[-1]):
                    floatMask[...,cell] = floatMask[...,cell]/np.max(np.max(floatMask[...,cell]))
                

                    
                flatFloat = np.max(floatMask, axis=2)
                if datastream == obj.dataFocus: #If processing main stream (Ca++ data), send ROI array to output
                    outROIs = floatMask
                    
                flatFloat = flatFloat * 255
                flatFloat = flatFloat.astype(np.uint8)
                truncatedLabel = (flatLabel % 255).astype(np.uint8) # 
    #            print(f'Datastream: {datastream}')
    #            print(f'Data focus: {obj.dataFocus}')
                paintOn = False #obj.segmentationMethods['Paint ROIs']['Params']['Painting on'][1]
    #            print(f'Paint on {paintOn}')
                if paintOn == 1 and datastream == obj.dataFocus:
                    try:
                        RGB = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][obj.dataFocus]['paintedROImap']
                    except:
                        print('No painted ROI map')
                else:
                    label_range = np.linspace(0,1,256)
                    lut = np.uint8(plt.cm.prism(label_range)[:,2::-1]*256).reshape(256, 1, 3)
                    RGB = cv2.LUT(cv2.merge((truncatedLabel, truncatedLabel, truncatedLabel)), lut)
                        
                
                #label_range = np.linspace(0,1,256)
                #lut = np.uint8(plt.cm.prism(label_range)[:,2::-1]*256).reshape(256, 1, 3)
                #RGB = cv2.LUT(cv2.merge((truncatedLabel, truncatedLabel, truncatedLabel)), lut)
                Alpha = np.expand_dims(flatFloat, 2)
                RGBA = np.concatenate((RGB,Alpha), axis = 2)
              
                
            Xdata[datastream] = FOV['T'][datastream][...]
            Labels[datastream]  = datastream
            
            
            
        ## Next get list of all time points and create new regular time base spanning experiment:   
        allTimes = []
        for Time in Xdata:  
            allTimes = list(allTimes) + list(Xdata[Time][:])
        
 
   
        allTimes.sort()
        
        start = allTimes[0] - 1
        end = allTimes[-1] + 1
        
        if trimStim: ## Option to clip time span to first and last frames of CA data
#            print(f'{activityStream=}')
            #print(f"T keys: {self.DB['Animals'][self.curAnimal][self.curFOV]['T'].keys()}")
            
            start = obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus][0]
            end  =  obj.DB['Animals'][obj.curAnimal][obj.curFOV]['T'][obj.dataFocus][-1]
        
        ##Set time span used from time navigator rectangle:
            
      #  start = obj.timeROI.parentBounds().left()
      #  end   = obj.timeROI.parentBounds().right()
        
        length = end-start
 #       print(f'Length: {length}')
        timeBase = np.linspace(start, end, int(length/timeStep), endpoint=False)
        
        stimIX = []
        stimLabels = []
        #Now align observed data to new time base:
        rowCounter = 0
        for count, datastream in enumerate(Ydata):
            
            if 'FLIR' in datastream:
                stimIX.append(rowCounter)
                stimLabels.append('Temp (°C)')
            elif 'VFraw' in datastream or 'AuroraForce' in datastream or 'eVF_mN' in datastream or 'NIR' in datastream:
                stimIX.append(rowCounter)
                stimLabels.append('Force (mN)')
            
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
            
            

            dataArray[:,:] = raster[:,IX]  #spacing preserved
            #realTimeArray = dataArray
            
            
            error = np.absolute(timeBase - originalTimes)
            
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
        #plt.imshow(compressedArray, aspect = output.shape[1]/output.shape[0]*0.33)
        
        
        
 #       print(stimIX)
  #      print(stimLabels)
        saveDir = '/lab-share/Neuro-Woolf-e2/Public/Default reports'
        if doPlotRaster:
            plotRaster(compressedArray, timestep = timeStep, stimIX = stimIX, stimLabels = stimLabels, savePath = saveDir, nullValue=nullValue)
        
        
        
        
        
        timeStep = np.array(timeStep)
        nullValue = np.array(nullValue)
        start_time = start
        return(output, compressedArray, timeStep, stimIX, stimLabels, outMeanImage, outROIs, nullValue, start_time)
        
    
    
def plotRaster(raster, items='all', timepoints = 'all', stimIX = [], stimLabels = [], timestep = 0.05, savePath = '/home/ch184656/testSave.pdf', nullValue = 0):
    if items != 'all':
        raster = raster[items,:]
    if timepoints != 'all':
        raster = raster[:,timepoints]
    if savePath is None:
        doSave = False
    else:
        doSave = True
    doSave= False
        
    nullIX = raster == nullValue ## nullValue is a place holder value  for missing data points, converted to 0 for display
    numTrace = raster.shape[0]
    stimSpace = 2 #multiplier to give extra space to stimulus traces
    timeSpan = timestep * raster.shape[1]
    rasterVspan = 8
    
    F = plt.figure(figsize =[timeSpan*0.005, (numTrace+rasterVspan)*0.1])
    Vslots = numTrace+rasterVspan+1+(len(stimIX)*stimSpace)  # 1 slot for each trace, 1 for time bar, rasterVspan for raster, and stimSpace multiplies space for stim
    traceSpan = 1/Vslots
    rasterBottom = 1-rasterVspan/Vslots
    print(rasterBottom)
    X = 0.1 # standard x offset 
    W = 0.8 # standard width
    
    rasterMap = 'inferno'
    
    RA = F.add_axes([X, rasterBottom, W, rasterVspan/Vslots])
    raster[nullIX] = 0
    displayRaster = raster.copy()
    for count, data in enumerate(displayRaster):
        minV = np.amin(data)
        maxV = np.amax(data-minV)
        
        displayRaster[count,:] = (data-minV)/maxV
    RA.imshow(displayRaster, aspect='auto', interpolation='none', cmap=rasterMap)
    raster[nullIX] = np.nan
    RA.xaxis.set_visible(False)
    RA.yaxis.set_visible(False)
    RA.set_frame_on(False)
    
    timeScale = 20
    timeBar = np.array([0, int(timeScale/timestep)])
    
    
    colorAx =  F.add_axes([X+W, rasterBottom, (1-(X+W))/4, rasterVspan/Vslots])
    colorBar = np.linspace(np.min(raster),np.max(raster),25)
    colorBar = np.expand_dims(colorBar, axis = 1)
    #colorBar = np.concatenate((colorBar,colorBar,colorBar), axis = 1)
    colorBar = np.flipud(colorBar)
    colorAx.imshow(colorBar, cmap=rasterMap, aspect ='auto')
    colorAx.xaxis.set_visible(False)
    colorAx.yaxis.set_visible(False)
    colorAx.set_frame_on(False)
    
    
    traceAxes = {}
    timeAxes = F.add_axes([X,0,W,traceSpan])
    timeAxes.plot(timeBar,[0.5,0.5], color='k')
    timeAxes.set_xlim([0, raster.shape[1]])
    timeAxes.set_ylim([0,1])
    timeAxes.xaxis.set_visible(False)
    timeAxes.yaxis.set_visible(False)
    timeAxes.set_frame_on(False)
    extra = 1
    
    if doSave:
        os.makedirs(os.path.join(savePath,'Traces'))
    
    for c, data in enumerate(raster):
        d=c+extra
        
        
        if c in stimIX:
            traceAxes[c] = F.add_axes([X,d*traceSpan,W,traceSpan*stimSpace])
            traceAxes[c].yaxis.set_visible(True)                       
            traceAxes[c].plot(data, color='m', linewidth = 1)
            extra = extra + (stimSpace-1)
            
            T = plt.figure(figsize =[timeSpan*0.05, (1)*1])
            TA = T.add_axes([X,0.25,W,0.75])
            TT = T.add_axes([X,0,W,0.25])
            TA.plot(data, color='m', linewidth = 1)
            
            TT.plot(timeBar,[0.5,0.5], color='k')
            TT.set_xlim([0, raster.shape[1]])
            TT.set_ylim([0,1])
            TT.xaxis.set_visible(False)
            TT.yaxis.set_visible(False)
            TT.set_frame_on(False)

            TA.set_xlim([0, raster.shape[1]])
            #TA.set_ylim([0,1])
            TA.xaxis.set_visible(False)
            TA.yaxis.set_visible(True)
            TA.set_frame_on(False)
            
            if doSave:
                T.savefig(os.path.join(savePath, 'Traces','stim' + str(c) + '.pdf'), transparent = True)
            plt.close(T)
            
        else:
            traceAxes[c] = F.add_axes([X,d*traceSpan,W,traceSpan])
            traceAxes[c].yaxis.set_visible(False)
            
            traceAxes[c].plot(data, color='k', linewidth = 0.75)
            
            T = plt.figure(figsize =[timeSpan*0.05, (1)*1])
            TA = T.add_axes([X,0.25,W,0.75])
            TT = T.add_axes([X,0,W,0.25])
            TA.plot(data, color='k', linewidth = 0.75)
            
            TT.plot(timeBar,[0.5,0.5], color='k')
            TT.set_xlim([0, raster.shape[1]])
            TT.set_ylim([0,1])
            TT.xaxis.set_visible(False)
            TT.yaxis.set_visible(False)
            TT.set_frame_on(False)

            TA.set_xlim([0, raster.shape[1]])
            #TA.set_ylim([0,1])
            TA.xaxis.set_visible(False)
            TA.yaxis.set_visible(False)
            TA.set_frame_on(False)
            
            if doSave:
                T.savefig(os.path.join(savePath, 'Traces','trace_' + str(c) + '.pdf'), transparent = True)
            plt.close(T)
            
        traceAxes[c].xaxis.set_visible(False)        
        traceAxes[c].set_frame_on(False)
        traceAxes[c].set_xlim([0, raster.shape[1]])
        

    
    if doSave:
        F.savefig(os.path.join(savePath,'traces.pdf'),transparent = True)
        F.savefig(os.path.join(savePath,'traces.png'),transparent = True)
    F.show()
    #saveName = os.path.join('/home/ch184656/Default reports', self.curAnimal+self.curFOV+self.dataFocus + '_plots.png')
    #F.savefig(saveName, transparent = True)
    
    
    #analyzeRaster(raster, stimIX, stimLabels)
    #Correlate to stimuli:

def getData(source, selectedDataStreams = None, activityStream = None, selectedFOV = None, z_score=False):
        ## source is path to .h5 file with data of interest
        obj = dbContainer(source) 
        ## create a temporary .h5 file to isolate selected data
        DBpath = '/lab-share/Neuro-Woolf-e2/Public/tempDB.h5'
        if os.path.exists(DBpath):
            os.remove(DBpath)
        ## create a summary table from newly created .h5 file
        fullArr, compactArray, timestep, stimIX, stimLabels, meanImage, ROIs, nullValue, start_time = genSumTable(obj, selectedDataStreams=selectedDataStreams, activityStream=activityStream, selectedFOV=selectedFOV, z_score=z_score)
        obj.close()
        DB = h5py.File(DBpath,'a')
        DB.require_group('stims')
        for IX, label in zip(stimIX, stimLabels):
            stim = compactArray[IX,:]
            DB['stims'].require_dataset(label, data = stim, shape = stim.shape, dtype = stim.dtype)
        
        cellRaster = np.delete(compactArray, stimIX, axis = 0)
        source = h5py.ExternalLink(obj.DBpath, f'/Animals/{obj.curAnimal}/{obj.curFOV}')
        #DB['source'] = source
        DB.require_dataset('experiment_start', data = start_time, shape = start_time.shape, dtype = start_time.dtype)
        DB.require_dataset('timestep', data = timestep, shape = timestep.shape, dtype = timestep.dtype)
        DB.require_dataset('nullValue', data = nullValue, shape = nullValue.shape, dtype = nullValue.dtype)
        DB.require_dataset('raster', data = cellRaster, shape = cellRaster.shape, dtype = cellRaster.dtype)
        DB.require_dataset('ROIs', data = ROIs, shape = ROIs.shape, dtype = ROIs.dtype)
        DB.require_dataset('meanImage', data = meanImage, shape = meanImage.shape, dtype = meanImage.dtype)
        DB.close()
   #     print('DB closed')
        rawData = LA.prepAnalysis(DBpath = DBpath, source=source, activityKey = activityStream)
        data = LA.processTempStim(rawData) 
        return(data)
    
def prep_for_raster_generic(obj, animal, fov, datas):
    time=[]
    data=[]
    labels = []
    for data in datas:
        data_src = obj.DB['Animals'][animal][fov][data]
        if len(data_src.shape) ==1: ## if 1d data, use directly:
            time.append(obj.DB['Animals'][animal][fov]['T'][data][...])
            data.append(data_src[...])
            labels.append(data)
        else:
            trace_array = obj.DB['Animals'][animal][fov]['R'][data]['traceArray'][...]
            for trace in trace_array:
                time.append(obj.DB['Animals'][animal][fov]['T'][data][...])
                data.append(trace)
                labels.append(data)
    return()
                
                
                
def rasterize(time, data, labels = None, limit_to_span_of_data_IX = None, time_step = 0.1, precision = 0.5, plot=False):
    ####  input are paired lists of timepoints and datasamples
    ####  if limit_to_span_of_data is not set to None, bounds of new time scale will be 
    ####  restricted to duration of selected data
    ####
    
    #### Construct new evenly spaced time base for all data
    allTimes = []
    for t in time:  
        allTimes = list(allTimes) + list(t)
    
    allTimes.sort()
    if not limit_to_span_of_data_IX is None: ## Option to clip time span to first and last frames of selected data
        start = time[limit_to_span_of_data_IX][0]
        end  =  time[limit_to_span_of_data_IX][-1]
    else:
        start = allTimes[0] - 1
        end = allTimes[-1] + 1
  
    length = end-start
    time_base = np.linspace(start, end, int(length/time_step), endpoint=False)
    
    ### Resample data and place in a raster:
    raster = np.zeros([len(data),len(time_base)])
    for c, (t, d) in enumerate(zip(time, data)):
        ## repeat last element of data/time and append, expand dimensions for performing searchsorted
        time_ex = np.append(t, t[-1])
        data_ex = np.append(d, d[-1])
        IX = np.searchsorted(t, time_base)
        times_orig = time_ex[IX] ## actual sample times
        error = np.absolute(time_base - times_orig) ## difference between regularized time points and actual measurement times
        data_resampled = data_ex[IX]
        null_value = -2**15
        data_resampled[error>precision] = null_value
        raster[c,:] = data_resampled
    
    raster_max = np.max(raster, axis = 0)
    raster_trimmed = np.delete(raster, raster_max==null_value, axis=1)
    output={}
    output['time'] = time_base
    output['raster'] = raster_trimmed
    output['labels'] = labels
    if plot:
        plt.imshow(raster_trimmed)
    return(output)
    
    