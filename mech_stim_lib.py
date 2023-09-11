#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:24:10 2022

@author: ch184656
"""
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def processMechStim(mechTrace, timestep = None, plot=True, input_DATA=None, input_session = None):
    mode = 'trace'
    ## get trace of mechanical stimuli
    if not input_DATA is None:
        try:
            mechTrace = input_DATA['stims']['Force (mN)']
            #ts = DATA['timestep']
            mode = 'DATA'
            display_trace = mechTrace.copy()
            display_trace[np.where(display_trace == input_DATA['nullValue'])] = np.nan
            mechTrace[mechTrace == input_DATA['nullValue']] = 0
            
        except:
            print('Could not find mech data')
            return(input_DATA)
    
    if not input_session is None:
        try:
            mechTrace = input_session.mech_stim
            mode = 'session'
        except:
            print('Could not find mech data')
            return(input_session)
        if mechTrace is None:
            print('Mech data is empty')
            return(input_session)
        
    minStim = 10 ## Find stims over 10 mN
    sProm = 10
    sInt = 10
    
    mechTrace[mechTrace<minStim] = 0
    
    output = {}
    output['Mstim'] = {}
                     
  
    
    
    ### Break down mechanical stim into discrete elements (stim objects)

    maxStim = np.amax(mechTrace)
    #maxStim = 500
    normalized_stim = mechTrace/maxStim
    sPeaks = signal.find_peaks(mechTrace, distance = sInt, prominence = minStim)[0]
    
    if plot:
        print('plotting')
        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])
        plt.plot(mechTrace)
        plt.scatter(sPeaks, np.ones(sPeaks.shape)*-100)
        A.set_xlim(np.nonzero(mechTrace)[0][0], np.nonzero(mechTrace)[0][-1])
    
    stims = []
    
    for c, peak in enumerate(sPeaks[:-1]):
        ## Identify start and stop indices for each poke
        if c==0:
            offset = 0
            pre = mechTrace[0:peak]
        else:
            offset = sPeaks[c-1]
            pre = mechTrace[offset:peak]
        if c== len(sPeaks-1):
            post = mechTrace[peak:]
        else:
            post = mechTrace[peak:sPeaks[c+1]]
        
        pre_zeros = np.where(pre == 0)[0]
        post_zeros = np.where(post == 0)[0]
        
        if pre_zeros.size == 0:
            start = offset + np.argmin(pre)
        else:
            start = offset + pre_zeros[-1]
        
        if post_zeros.size == 0:
            end = peak + np.argmin(post)+1
        else:
            end = peak + post_zeros[0]+1
            
        waveform = mechTrace[start:end]
        intensity = np.amax(waveform)
        if intensity >250:
            submodality = 'High'
        elif intensity > 75:
            submodality = 'Medium'
        else:
            submodality = 'Low'
        Mstim = stim(parent = mechTrace, modality='Mechano', submodality=submodality, bounds=[start, end], waveform=waveform, intensity = intensity)
        
        output['Mstim'][c] = Mstim
        if plot:
            plt.plot(np.arange(start,end), waveform, color=rand_color())
            
            
    if mode == 'session':
        for cell in input_session.cells.keys():
            input_session.cells[cell].mechStim = display_trace
    if mode == 'DATA'      :  
        for cell in input_DATA['cells'].values():
            cell.mechStim = display_trace
        input_DATA['mech_stim'] = display_trace
        output = input_DATA
    if mode == 'trace':
        return(output)
       
    return(output)


def construct_segmented_mech_trace(trace):
    s = processMechStim(trace)
    n_stim= len(s['Mstim'])
    length = s['Mstim'][0].parent.shape[0]
    stimat = np.zeros([n_stim, length])
    for c, stim in enumerate(s['Mstim'].keys()):
        v = s['Mstim'][c]
        stimat[c, v.start:v.end] = v.waveform
    return(stimat)


class stim:
    def __init__(self, parent= [], modality=None, submodality=None, bounds=[0,0], waveform = np.array([]), intensity = 0, stim_temp=None):
        self.modality = modality # thermo, mechano, chemical, opto, etc.
        self.submodality = submodality # heating, cooling, indentation, brush, etc
        self.bounds = bounds # first and last framenumber
        self.start = bounds[0]
        self.end = bounds[1]
        self.timepoints = np.linspace(self.start, self.end, num = (self.end-self.start))
        self.waveform = waveform #intenbsity over time
        self.location = [] # save for  mechano data
        self.intensity = intensity #deviation from basal temperature, milliNewtons, etc.
        self.stim_temp = stim_temp
        self.parent = parent # trace of stimulus series stim is part of
        
    def response(self, cell, plot = False, normalize = False):
        output = {}
        trace = cell.trace[self.start:self.end]
        
        output['trace'] = trace
        output['amplitude'] = np.amax(trace) #-baseline
        
        if normalize:
            output['amplitude'] = np.amax(trace)/np.amax(cell.trace)
        output['area'] = np.sum(trace)
        output['corr'] = np.corrcoef(self.waveform, trace)[0,1]
        output['latency_on'] = np.where(trace>output['amplitude']*0.25)[0][0]
        #output['latency_peak'] = np.where(trace==output['amplitude'])[0[0]]
        #output['latency_off']  = np.where(trace>0.25)[0][-1]
        #output['duration'] = output['latency_off']-output['latency_on']
        if plot:
            F, A = blank_fig()
            A.plot(trace)
            A2 = A.twinx()
            A2.plot(self.waveform)
            plt.show()
        for o in output.keys():
            output[o] = np.nan_to_num(output[o])
        return(output)

    def show(self, F=None):
        plt.plot(self.parent)   
        plt.plot(self.timepoints, self.waveform)
        plt.show()
        
def rand_color():
    return(np.random.random_sample(3))

def blank_fig(title='Figure'):
    F = plt.figure()
    A = F.add_axes([0,0,1,1])
    return(F,A)