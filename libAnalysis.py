# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 12:42:19 2021

@author: ch184656
"""

import uuid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from stim_corr_image import corrImSig
import plotly.express as px
import multi_session_utils as ms
import copy
import itertools
from seeded_cnmf import seeded_CNMF
import cv2
from caiman.base.rois import register_multisession
from caiman.utils import visualization
import pickle
import dill
import pdb
from scipy.signal import resample, butter, filtfilt
from scipy.spatial import distance_matrix
from tkinter import filedialog
import tkinter as tk
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import os
from DY_plot_contours import plot_contours
import time
import h5py
from scipy import signal
from scipy import stats
from sklearn import cluster
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import mechano_tools as mech_lib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.use('Agg')
import math
import glob

from DYroiLibrary import local_correlations_fft
#from DYpreProcessingLibraryV2 import gen_colors


def unpickle(path):
    file = open(path, 'rb')
    out = pickle.load(file)
    file.close()
    return(out)


def get_sample_data(get=None):
    path = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 237/Mouse 237 experimental 2LA longitudinal.pickle'
    if get:
        path = selectFile()
    data = unpickle(path)
    return(data)


def selectFile():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir='/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Test pivotDBs/')
    return(file_path)


def prepAnalysis(DB=None, DBpath=None, closeAfterReading=True, plot=False, normalize_traces=False, source=None, activityKey=None):
    if DB == None:
        if DBpath == None:
            DBpath = selectFile()
        DB = h5py.File(DBpath, 'r+', rdcc_nbytes=1000000000,
                       rdcc_nslots=1000000)
    # print(DB['raster'].shape)
    # return

    cells = np.nan_to_num(DB['raster'][...])

    # Interpolate missing observations
    mask = cells == DB['nullValue']
    cells[mask] = np.interp(np.flatnonzero(
        mask), np.flatnonzero(~mask), cells[~mask])

    output = {}
    # if not (source is None):
    output['Source'] = source

    output['activity key'] = activityKey

    output['DBpath'] = DBpath

    output['cells'] = {}
    output['stims'] = {}
    output['timestep'] = DB['timestep'][...]
    if 'experiment_start' in DB.keys():
        output['experiment_start'] = DB['experiment_start'][...]
    output['nullValue'] = DB['nullValue'][...]
    output['nRaster'] = np.zeros(DB['raster'].shape)
    output['ROIs'] = DB['ROIs'][...]
    output['fieldImage'] = DB['meanImage'][...]

    for n, trace in enumerate(cells):

        ntrace = trace - np.min(trace)
        ntrace = ntrace/np.max(ntrace)
        if normalize_traces:
            trace = ntrace
        output['cells'][n] = cell(trace=trace)
        output['nRaster'][n, :] = ntrace
    output['raster'] = cells

    for stim in DB['stims']:
        output['stims'][stim] = DB['stims'][stim][...]

    for c in output['cells'].values():
        c.parent_data = output
    if closeAfterReading:
        DB.close()
    return(output)


# trace, timestep, nullvalue
def segment_temp_trace_into_stims(t, nV, plot=True, min_duration=10,  thresh=2, search_frames=80, end_fix_duration=None, adjust_for_intense_cold=820):

    traceOriginal = t.copy()

    domain = np.where(t != nV)[0]
    #domain = domain[0]
    # pull region containing non-null values
    td = t[domain[0]:domain[-1]]

    # interpolate any internal zero values
    z = np.where(td == nV)[0]

    for ix in z:
        xix = ix
        while td[xix] == nV:

            xix = xix-1
            before = td[xix]
    #    print(before)
        xix = ix
        while td[xix] == nV:

            xix = xix + 1
            after = td[xix]
   #     print(after)
        td[ix] = (before+after)/2

    traceOriginal[domain[0]:domain[-1]] = td
    delta = np.ediff1d(t, to_end=0)
    normT = t-np.nanmedian(td)
    absT = np.abs(normT)
    ramp = np.arange(0, len(t))

    modality = 'thermo'
    transients = {}
    n = 0
    started = False
    running_count = 0
    if plot:
        G = plt.figure(
            f'First pass and refined therm transients, thresh = {thresh}, min = {min_duration}, search = {search_frames}')
        B = G.add_axes([0.2, 0.2, 0.6, 0.6])
        B.plot(t)
    o_threshold = copy.copy(thresh)
    for c, (abso, norm, raw) in enumerate(zip(absT, normT, t)):
        if adjust_for_intense_cold:

            if c < adjust_for_intense_cold:
                thresh = o_threshold + 3
            else:
                thresh = o_threshold
        # pdb.set_trace()
        if abso > thresh:  # and running_count < max_duration:
            if not started:
                running_count = 0
                transients[n] = {}
                transients[n]['frames'] = []
                transients[n]['trace'] = []
                transients[n]['abs_trace'] = []
                transients[n]['norm_trace'] = []
                started = True
            transients[n]['trace'].append(raw)
            transients[n]['frames'].append(c)
            transients[n]['abs_trace'].append(abso)
            transients[n]['norm_trace'].append(norm)
            running_count = running_count+1
        else:
            if started:
                started = False
                if len(transients[n]['trace']) >= min_duration:
                    n = n+1
                else:
                    transients[n] = {}

    toDel = []
    for n in transients.keys():  # delete thermal transsiientss that start to early in recorrdiing

        start_frame_thresh = 10
       # if not end_fix_duration is None:
        #   start_frame_thresh = end_fix_duration + 10
        if not transients[n] or transients[n]['frames'][0] < start_frame_thresh:
            toDel.append(n)

    # pdb.set_trace()
    for d in toDel:
        del transients[d]

    if plot:
        B.plot(t, color='k')
        for n in transients.keys():
            B.plot(transients[n]['frames'], transients[n]['trace'], color='m')

    # refine start and endpoints by bounding with maximal delta T values
    refined_transients = {}
    # pdb.set_trace()
    for c, transient in enumerate(transients.values()):

        peak = np.where(transient['abs_trace'] ==
                        np.amax(transient['abs_trace']))[0][0]
        intensity = transient['norm_trace'][peak]
        stim_temp = transient['trace'][peak]
        if intensity < 0:
            submodality = 'cooling'
        else:
            submodality = 'heating'

        init_bounds = [transient['frames'][0], transient['frames'][-1]]
        glob_start = init_bounds[0]
        if glob_start+peak - search_frames < 0:
            start_frame = 0
        else:
            start_frame = glob_start + peak - search_frames
        start_seg = delta[start_frame: glob_start + peak]
        end_seg = delta[glob_start + peak: glob_start + peak + search_frames]
        if submodality == 'cooling':
            # pdb.set_trace()
            peak_cooling = start_frame + \
                np.where(start_seg == np.amin(start_seg))[0][0]
            peak_heating = glob_start + peak + \
                np.where(end_seg == np.amax(end_seg))[0][0]
            if not end_fix_duration is None:
                peak_cooling = peak_heating - end_fix_duration
                # if peak_cooling < 0:

            transient['trace'] = t[peak_cooling:peak_heating]
            transient['abs_trace'] = absT[peak_cooling:peak_heating]
            transient['norm_trace'] = normT[peak_cooling:peak_heating]
            transient['frames'] = ramp[peak_cooling:peak_heating]

        elif submodality == 'heating':
            peak_heating = start_frame + \
                np.where(start_seg == np.amax(start_seg))[0][0]
            peak_cooling = glob_start + peak + \
                np.where(end_seg == np.amin(end_seg))[0][0]
            if not end_fix_duration is None:
                peak_heating = peak_cooling - end_fix_duration
            transient['trace'] = t[peak_heating:peak_cooling]
            transient['abs_trace'] = absT[peak_heating:peak_cooling]
            transient['norm_trace'] = normT[peak_heating:peak_cooling]
            transient['frames'] = ramp[peak_heating:peak_cooling]

        #transient['frames']     = np.arange(glob_start, glob_start+len(transient['trace']))
        refined_transients[c] = transient

    output = {}
    # pdb.set_trace()
    for c, transient in enumerate(refined_transients.values()):

        # if transient['abs_trace'].shape[0] == 0:
        #   pdb.set_trace()
        if len(transient['frames']) < 1:
            continue
        peak = np.where(transient['abs_trace'] ==
                        np.amax(transient['abs_trace']))[0][0]
        intensity = transient['norm_trace'][peak]
        stim_temp = transient['trace'][peak]
        if intensity < 0:
            submodality = 'cooling'
        else:
            submodality = 'heating'
        bounds = [transient['frames'][0], transient['frames'][-1]]
        waveform = transient['trace']

        Tstim = stim(parent=t, modality=modality, submodality=submodality,
                     bounds=bounds, waveform=waveform, intensity=intensity, stim_temp=stim_temp)
        output[c] = Tstim
        if plot:
            if submodality == 'cooling':
                color = 'c'
            elif submodality == 'heating':
                color = 'r'

            B.plot(transient['frames'], waveform, color=color)
    return(output)


def processTempStim(data, plot=False):

    try:
        t = data['stims']['Temp (°C)']
        ts = data['timestep']
    except:
        'Could not find thermal data'
        return(data)

    fps = int(1/ts)
    #traceOriginal = t
    traceOriginal = t.copy()
    nV = data['nullValue']
    domain = np.where(t != nV)
    domain = domain[0]
    # pull region containing non-null values
    td = t[domain[0]:domain[-1]]

    # interpolate any internal zero values
    z = np.where(td == nV)[0]

    for ix in z:
        xix = ix
        while td[xix] == nV:

            xix = xix-1
            before = td[xix]
    #    print(before)
        xix = ix
        while td[xix] == nV:

            xix = xix + 1
            after = td[xix]
   #     print(after)
        td[ix] = (before+after)/2

    traceOriginal[domain[0]:domain[-1]] = td

    normT = t-np.nanmedian(td)
    absT = np.abs(normT)
    peaks = signal.find_peaks(absT, distance=23*fps)
 #   print(peaks)
    if plot:
        plt.figure()
        plt.plot(t), plt.scatter(peaks[0], np.ones(peaks[0].shape)*-1)
    data['Tstim'] = {}
    counter = 0
    modality = 'thermo'
    for peak in peaks[0]:
        if peak > 8*fps:  # peak must be at least 8 sec into recording
            if normT[peak] < 0:
                submodality = 'cooling'
                bounds = [peak-(8*fps), peak]
            else:
                submodality = 'heating'
                if peak + 8*fps > t.shape[0]:
                    bounds = [peak, t.shape[0]]
                else:
                    bounds = [peak-(8*fps), peak + (8*fps)]

            waveform = t[bounds[0]:bounds[1]]

            intensity = normT[peak]
            stim_temp = t[peak]
           # if np.absolute(intensity) >100:
            #    pdb.set_trace()
            Tstim = stim(parent=t, modality=modality, submodality=submodality,
                         bounds=bounds, waveform=waveform, intensity=intensity, stim_temp=stim_temp)
            data['Tstim'][counter] = Tstim
            counter = counter + 1

    for cell in data['cells'].values():
        displayStim = t.copy()
        displayStim[np.where(t == data['nullValue'])] = np.nan
        cell.thermStim = displayStim

    data['therm_stim'] = displayStim
    return(data)


def process_mech_session(session):
    pass


def rand_color():
    return(np.random.random_sample(3))

def process_mech_trace(mechTrace, minStim = 10, min_duration = 7, plot=False):
    out = {}
    minStim = 10  # Find stims over 10 mN
    sProm = 10
    sInt = 10
    
    display_trace = mechTrace.copy()
   
    mechTrace[mechTrace < minStim] = 0

    # Break down mechanical stim into discrete elements (stim objects)

    maxStim = np.amax(mechTrace)
    #maxStim = 500
    normalized_stim = mechTrace/maxStim
    sPeaks = signal.find_peaks(mechTrace, distance=sInt, prominence=minStim)[0]

    if plot:
        print('plotting')
        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])
        plt.plot(mechTrace)
        plt.scatter(sPeaks, np.ones(sPeaks.shape)*-100)
        A.set_xlim(np.nonzero(mechTrace)[0][0], np.nonzero(mechTrace)[0][-1])

    stims = []
    counter = 0
    for c, peak in enumerate(sPeaks[:-1]):
        # Identify start and stop indices for each poke
        if c == 0:
            offset = 0
            pre = mechTrace[0:peak]
        else:
            offset = sPeaks[c-1]
            pre = mechTrace[offset:peak]
        if c == len(sPeaks-1):
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
        if intensity > 250:
            submodality = 'High'
        elif intensity > 75:
            submodality = 'Medium'
        else:
            submodality = 'Low'
            
        if waveform.shape[0] > min_duration: # check if stim lasts at least n frames long:
            Mstim = stim(parent=mechTrace, modality='Mechano', submodality=submodality, bounds=[
                     start, end], waveform=waveform, intensity=intensity)

            out[counter] = Mstim
            counter = counter + 1
        if plot:
            # plt.figure()
            plt.plot(np.arange(start, end), waveform, color=rand_color())
    
    return(out)



def processMechStim(DATA,  plot=True):
    # get trace of mechanical stimuli
    try:
        m = DATA['stims']['Force (mN)']
        ts = DATA['timestep']
    except:
        'Could not find mech data'
        return(DATA)

    minStim = 10  # Find stims over 10 mN
    sProm = 10
    sInt = 10

    DATA['Mstim'] = {}
    mechTrace = DATA['stims']['Force (mN)']
    #mechTrace[mechTrace< -30000] = 0
    display_trace = mechTrace.copy()
    display_trace[np.where(display_trace == DATA['nullValue'])] = np.nan
    mechTrace[mechTrace == DATA['nullValue']] = 0
    mechTrace[mechTrace < minStim] = 0

    # Break down mechanical stim into discrete elements (stim objects)

    maxStim = np.amax(mechTrace)
    #maxStim = 500
    normalized_stim = mechTrace/maxStim
    sPeaks = signal.find_peaks(mechTrace, distance=sInt, prominence=minStim)[0]

    if plot:
        print('plotting')
        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])
        plt.plot(mechTrace)
        plt.scatter(sPeaks, np.ones(sPeaks.shape)*-100)
        A.set_xlim(np.nonzero(mechTrace)[0][0], np.nonzero(mechTrace)[0][-1])

    stims = []

    for c, peak in enumerate(sPeaks[:-1]):
        # Identify start and stop indices for each poke
        if c == 0:
            offset = 0
            pre = mechTrace[0:peak]
        else:
            offset = sPeaks[c-1]
            pre = mechTrace[offset:peak]
        if c == len(sPeaks-1):
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
        if intensity > 250:
            submodality = 'High'
        elif intensity > 75:
            submodality = 'Medium'
        else:
            submodality = 'Low'
        Mstim = stim(parent=mechTrace, modality='Mechano', submodality=submodality, bounds=[
                     start, end], waveform=waveform, intensity=intensity)
        DATA['Mstim'][c] = Mstim
        if plot:
            # plt.figure()
            plt.plot(np.arange(start, end), waveform, color=rand_color())
    for cell in DATA['cells'].values():
        cell.mechStim = display_trace
    DATA['mech_stim'] = display_trace
    return(DATA)


class cell:
    def __init__(self, trace=[], ROI=[], classification=None, parent_data=None, parent_session=None, thermStim=None, mechStim=None):
        self.trace = trace
        trace = trace-np.min(trace) 
        self.norm_trace = trace/np.max(trace)
        self.ROI = ROI
        self.classification = classification
        self.parent_data = parent_data
        self.parent_session = parent_session
        self.thermStim = thermStim
        self.mechStim = mechStim
        self.mech_stats = {}
        self.therm_stats = {}
        self.transients = None
        #self.mech_max = None
        #self.mech_threshold = None
    
    def directory(self, a=1):
        for i in dir(self):
            print(i)
            
    def show(self, showStim=True, showTherm=True, showMech=True, F=None, A1=None, A2=None, AT=None, dRange=None, mech_y_log=False, trace_color='k', trace_linewidth=1, trace_alpha=1, norm=True, group_label='', show_time=True, disp_transients=False, transients=None, Mbounds=None, Tbounds=None, Tstim_color='m', **kwargs):
        options = {'show_y': True,
                   'show_trace_y_label': True,
                   'trace_y_lim': None,
                   'show_stim_y': True,
                   'show_therm_transients': False,
                   'show_mech_transients': False,
                   'ybar': False,
                   'xbar': False,
                   'figName': None,
                   'mech_color_map' : 'inferno',
                   'mech_scale_max': 250}
        if hasattr(self.parent_session, 'Mstim'):
            Mstim = self.parent_session.Mstim
        else:
            Mstim = None
            
        options.update(kwargs)
        
        
        # A1 - axes for plotting trace
        # A2, AT - axes for plotting first and second stimuli
        ts = self.parent_data['timestep']
        if not (dRange is None):
            dRange = dRange/ts
            dRange = dRange.astype(np.uint64)
        else:
            dRange = [0, self.trace.shape[0]]
        if F is None:
            F = plt.figure()
            A1 = F.add_axes([0, 0, 1, 0.5])
            if showStim:
                A2 = F.add_axes([0, 0.5, 1, 0.5])

        if norm:
            trace = self.norm_trace
        else:
            trace = self.trace
        if 'therm' in trace_color:# and hasattr(self, 'Tstim'):
            if hasattr(self, 'Tstim'):
                self.thermo_tuning(plot=False)
                trace_color = self.tuning_color
            else:
                trace_color= 'k'
        if disp_transients:
            t_color = trace_color
            trace_color = 'k'
        A1.plot(trace, color=trace_color, alpha=trace_alpha,
                linewidth=trace_linewidth)  # Plot trace in axis A1

        if disp_transients:
            if not 'transients' in dir(self) or self.transients is None:
                self.transients = self.identify_transients(**kwargs)
            if transients is None:
                transients = self.transients.keys()
            for transient in transients:
                if norm:
                    trans = self.transients[transient]['norm_trace']
                else:
                    trans = self.transients[transient]['trace']
                A1.plot(self.transients[transient]['frames'], trans,
                        color=t_color, linewidth=trace_linewidth*4)

        if not options['trace_y_lim'] is None:
            A1.set_ylim(options['trace_y_lim'])
            
        #showTherm = True
        if showStim:
            mech_cmap = cm.get_cmap(options['mech_color_map'])
            
           # print(f'Therm stim shape: {self.thermStim.shape}')
           # print(f'Trace shape {self.trace.shape}')
            if showTherm and showMech:
                if not self.thermStim is None:
                    A2.plot(self.thermStim, color=Tstim_color)
                    A2.xaxis.set_visible(False)
                    A2.set_ylabel('Temp (°C)')
                    if options['show_therm_transients'] and hasattr(self.parent_session, 'Tstim'):
                        for Tstim in self.parent_session.Tstim:
                            if self.parent_session.Tstim[Tstim].stim_temp < 30:
                                stim_color = 'b'
                            elif self.parent_session.Tstim[Tstim].stim_temp > 35 and self.parent_session.Tstim[Tstim].stim_temp < 44:
                                stim_color = 'g'
                            elif self.parent_session.Tstim[Tstim].stim_temp >= 44:
                                stim_color = 'r'
                            else:
                                stim_color = [0.75, 0.75, 0.75]
                            A2.plot(self.parent_session.Tstim[Tstim].timepoints, self.parent_session.Tstim[
                                    Tstim].waveform, color=stim_color, linewidth=trace_linewidth*4)
                    if not Tbounds is None:
                        A2.set_ylim(Tbounds)
                                
                if not self.mechStim is None:
                    if AT is None:
                        AT = A2.twinx()
                    AT.plot(self.mechStim, 'k')
                    if not Mstim is None:
                        for ms in Mstim.values():
                            mech_color = mech_cmap(ms.intensity/options['mech_scale_max'])
                            AT.plot(ms.timepoints, ms.waveform, color = mech_color)
                    if mech_y_log:
                        AT.set_yscale('log')
                        AT.set_ylim(1, 1000)
                    AT.set_frame_on(False)
                    sub = self.mechStim[dRange[0]:dRange[1]]
                    ymin = np.nanmin(sub)
                    ymax = np.nanmax(sub)
                    if np.isnan(ymin):
                        ymin = np.nanmin(self.mechStim)
                    if np.isnan(ymax):
                        ymax = np.nanmax(self.mechStim)
                    AT.xaxis.set_visible(False)
                    AT.set_ylabel('Force (mN)')
                    AT.set_xlim(dRange)
                    if Mbounds is None:
                        AT.set_ylim(ymin, ymax)
                    else:
                        AT.set_ylim(Mbounds)

            elif showTherm and not (self.thermStim is None):
                A2.plot(self.thermStim, 'k')
                A2.xaxis.set_visible(False)
                A2.set_ylabel('Temp (°C)')
                if options['show_therm_transients'] and hasattr(self.parent_session, 'Tstim'):
                    for Tstim in self.parent_session.Tstim:
                        if self.parent_session.Tstim[Tstim].stim_temp < 30:
                            stim_color = 'b'
                        elif self.parent_session.Tstim[Tstim].stim_temp > 35 and self.parent_session.Tstim[Tstim].stim_temp < 44:
                            stim_color = 'g'
                        elif self.parent_session.Tstim[Tstim].stim_temp >= 44:
                            stim_color = 'r'
                        else:
                            stim_color = [0.75, 0.75, 0.75]
                        A2.plot(self.parent_session.Tstim[Tstim].timepoints, self.parent_session.Tstim[
                                Tstim].waveform, color=stim_color, linewidth=trace_linewidth*4)
                if not Tbounds is None:
                    A2.set_ylim(Tbounds)
            elif showMech and not (self.mechStim is None):
                A2.plot(self.mechStim, 'k')
                if not Mstim is None:
                    for ms in Mstim.values():
                        mech_color = mech_cmap(ms.intensity/options['mech_scale_max'])
                        A2.plot(ms.timepoints, ms.waveform, color = mech_color)
                A2.set_ylabel('Force (mN)')

                A2.set_xlim(dRange)
                sub = self.mechStim[dRange[0]:dRange[1]]
                if Mbounds is None:
                    A2.set_ylim(np.nanmin(sub), np.nanmax(sub))
                else:
                    A2.set_ylim(Mbounds)
                
            if not A2 is None:
                A2.xaxis.set_visible(False)
                A2.set_frame_on(False)
                A2.set_xlim(dRange)

        A1.set_frame_on(False)
        if options['show_trace_y_label']:
            if group_label == None:
                if norm:
                    A1.set_ylabel(f'Norm response')
                else:
                    A1.set_ylabel(f'Raw trace')
            else:
                if norm:
                    A1.set_ylabel(f'{group_label} norm')
                else:
                    A1.set_ylabel(f'{group_label} raw')

        A1.set_xlim(dRange)

       # xtick = (A1.get_xticks()-dRange[0])*ts
        # A1.set_xticks(A1.get_xticks()) ## does this resolve warning? testing
       # A1.set_xticklabels((xtick.astype(np.uint64)))

        #
        # i

        if not options['show_y']:
            A1.yaxis.set_visible(False)

        if not options['show_stim_y']:
            A2.yaxis.set_visible(False)
            if not AT is None:
                AT.yaxis.set_visible(False)

        if not show_time:
            A1.xaxis.set_visible(False)

        if options['xbar']:
            plot_x_scale_bar(A2, placement='top', color='k')

        if options['ybar']:
            plot_y_scale_bar(A1)

        handles = [F, A1, A2, AT]
        return(handles)
    
    #def get_mech_threshold(self, threshold = 3.5, limit = 1000, min_threshold = 2, unit = 'mN', print_val = True):
    #    responses = self.mech_range(max_threshold = threshold, limit = limit, min_threshold = min_threshold, unit = unit)
    #    return(responses)
        
        
    def mech_range(self, max_threshold = 3.5, limit = 500, min_threshold=2, unit = 'mN', plot=False, AC=None, AS = None, F=None, n_cells=1, calc_all=False, ylim=[-1, 20], return_cell = True):
        
        if plot:
            if AC is None:
                F = plt.figure()
                AC = F.add_subplot(2,1,1)
        
        t = self.z_score_trace()
        if plot:
            AC.plot(t, color = 'k', alpha = 0.5)
            if AS is None:
                AS = F.add_subplot(2,1,2) #A.twinx()
        
        M = self.parent_session.Mstim
        responses = []
        self.mech_threshold = np.inf
        self.mech_threshold_SNR = max_threshold
        for s in M.values():
            if s.intensity > limit:
                continue
            response = {}
            response['unit'] = unit
            if return_cell:
                response['cell'] = self
            
            if unit == 'gf':
                response['force'] = s.intensity * 0.10197
            else:
                response['force'] = s.intensity
            
            response_trace = t[s.start:s.end]
            max_r = np.amax(response_trace)
            
            '''
            Check if response has enough frames above lower threshold (to remove single frame artifacts)
            '''
            
            response['long_enough']  = False
            for nt, tt in enumerate(response_trace):
                if np.median(response_trace[nt:nt+10]) > min_threshold:
                    response['long_enough'] = True
                    
            '''
            Add Stim Force, Response SNR and whether response meets critera to responses list
            '''  

            if max_r > max_threshold and response['long_enough']:
                
                response['max_SNR'] = max_r
                response['valid_response'] = True
                if s.intensity < self.mech_threshold:
                    self.mech_threshold = s.intensity
                alpha = 1
                color = 'r'
                if plot:
                    AC.plot(s.timepoints, t[s.timepoints.astype(np.uint32)], color = 'k')
            else:
                response['valid_response'] = False
                response['max_SNR'] = max_r
                color = 'k'
                alpha = 0.2/n_cells
           
            responses.append(response)
            if plot:
                AS.plot(s.timepoints, s.waveform, color = color, alpha = alpha)
            
        if plot: 
            AS.get_shared_x_axes().join(AS,AC)
            AS.set_xlim(AC.get_xlim())
            AS.set_ylim([10,1000])
            
            if not ylim is None:
                AC.set_ylim(ylim)
            AS.set_ylabel(f'Force ({unit})')
            AS.set_yscale('log')
            AS.set_yticks([10,100,1000])
            AS.set_yticklabels(['10','100','1000'])
            Xlabels = [str(x/10) for x in AS.get_xticks()]
            AS.set_xticks(AS.get_xticks())
            AS.set_xticklabels(Xlabels)
            
            AS.set_xlabel('Time (seconds)')
            if n_cells==1:
                AC.set_ylabel('Fz')
            else:
                F.supylabel('Fz')
            if n_cells < 5:
                box_off(AC, left_only =True)
            else:
                box_off(AC, All=True)
            box_off(AS)
        return(responses)
            
    def show_corr_map(self, dummy = 0):
        m = self.parent_session.parent
        n = self.cell_n
        m.get_stim_corr_for_ROIs(selected_u_cells = [n], selected_stims='file', full_field = False, expand=0, padding=30, vmin=0.05,vmax=0.5, neighbour = False, invert_cold = True, match_patch=False, set_patch=True, patch_h = 12, patch_w=12, plot_contours = True, key_session=0, abs_corr = False, color_cycle='RGB')
    
    def show_multi_sess_traces(self, selected_sessions='All', stim_span='Auto', F=None, stim_axes=None, color_mode='therm', trace_axes=None, save=True, disp_transients=False):
        m = self.parent_session.parent
        n = self.cell_n
        m.plot_traces(selected_u_cells = [n])
        
    def mech_tuning(self, norm=False, plot = False, detail=True):
        cell_analyze_mechano(self, norm=norm, plot=plot, detail=detail)
        
    def thermo_tuning(self, testing=False, plot=True, use_abs_max=True, min_r = 0.05):
        metric = thermo_tuning_metric(
            self, stim_set=self.parent_session.Tstim,  plot=plot, use_abs_max=use_abs_max, min_r=min_r)
        self.tuning_color = metric['norm']

    def detrend(self, wavelength=2400, plot=True, remove_peaks=True, limit_q=0.75, num_segs=20, method='pad', rectify=False, update_raster = True):
        self.trace = detrend(self.trace, wavelength, plot=plot, remove_peaks=remove_peaks,
                             limit_q=limit_q, num_segs=num_segs, method=method, rectify=rectify)
        #trace = self.trace-np.min(self.trace)
        self.norm_trace = self.trace/np.max(self.trace)
        if update_raster:
            self.parent_session.update_raster()
        
        
  
    def identify_transients(self, threshold_SNR=5, duration=10, pad=0, **kwargs):
        params = {'threshold_SNR': 5,  # 5 x SNR threshold
                  # 10 frames (could switch to seconds if timestep known)
                  'duration': 10,
                  'pad': 0}
        params.update(kwargs)
        trace = self.trace.copy()
        n_trace = self.norm_trace.copy()
        q = stats.median_abs_deviation(trace)/0.79788456
        m = np.median(trace)
        thr = m+(params['threshold_SNR']*q)
        transients = {}
        n = 0
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
                        n = n+1
                    else:
                        transients[n] = {}
        toDel = []
        for n in transients.keys():
            if not transients[n]:
                toDel.append(n)

        for d in toDel:
            del transients[d]

        if params['pad']:
            for n in transients.keys():
                start = transients[n]['frames'][0]-params['pad']
                if start < 0:
                    start = 0
                end = transients[n]['frames'][-1]+params['pad']
                if end > self.trace.shape[0]:
                    end = self.trace.shape[0]
                frames = np.arange(start, end)
                transients[n]['frames'] = frames
                transients[n]['trace'] = trace[frames]
                transients[n]['norm_trace'] = n_trace[frames]

        self.transients = transients
        return(transients)
    
    def z_score_trace(self, convert = True, update_raster = True):
        z = modified_z_score(self.trace)
        if convert:
            self.trace = z
            if update_raster:
                self.parent_session.update_raster()
        return(z)
        
        
    def classify_thermo(self, mode='tuning angle'):
        bounds = temp_class_bounds()
        response_type = 'None'
        stats = cell_analyze_temp(self, self.parent_data)
        theta = map_angle(stats['angle'])  # bound angle 0->2 pi  radians
        for response_type in bounds:
            b = bounds[response_type]
            b[0] = map_angle(b[0])
            b[1] = map_angle(b[1])
            if b[0] > b[1]:
                if theta >= b[0]:
                    self.classification = response_type
            else:
                if theta >= b[0] and theta < b[1]:
                    self.classification = response_type

        print(f'Classified as {self.classification}')
        return(response_type)

    def response_by_transient(self, stim=None, plot=False, normalize=False, F=None, AS=None, AT=None):
        """

        Parameters
        ----------
        stim : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is False.
        normalize : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        Dictionary with fields ....

        """
        if plot:
            if F is None:
                F = plt.figure()
                AS = F.add_axes([0.1, 0.6, 0.8, 0.3])
                AT = F.add_axes([0.1, 0.1, 0.8, 0.3])
                AS.plot(self.parent_session.therm_stim, 'k')
                AT.plot(self.trace, 'k')
            AS.plot(stim.timepoints, stim.waveform, 'r')
        output = {}
        if not hasattr(self, 'transients'):
            self.identify_transients()
        if self.transients is None:
            self.identify_transients()
        frames = []
        trace = []

        for transient in self.transients.values():
            t_start = transient['frames'][0]
            if t_start >= stim.start and t_start <= stim.end:
                trace.extend(transient['trace'])
                frames.extend(transient['frames'])
        if plot:
            #AT.plot(self.trace, 'k')
            AT.plot(frames, trace, 'r')
            # if len(frames):
            AT.set_xlim(0, self.parent_session.therm_stim.shape[0])
            AS.set_xlim(0, self.parent_session.therm_stim.shape[0])
            #AS.set_xlim([stim.start, frames[-1]])
        if len(frames) == 0:
            #print('No transients found in this cell for stim')
            return(stim.waveform*0)
        else:

            return(trace)

    def response(self, stim, plot=False, plot_transients=False, normalize=False, use_transients=False, F=None, AS=None, AT=None):
        if use_transients:

            if F is None and plot_transients:
                F = plt.figure()
                AS = F.add_axes([0.1, 0.6, 0.8, 0.3])
                AT = F.add_axes([0.1, 0.1, 0.8, 0.3])
                AS.plot(self.parent_session.therm_stim, 'k')

            trace = self.response_by_transient(
                stim=stim, plot=plot_transients, normalize=normalize, F=F, AS=AS, AT=AT)
        else:
            trace = self.trace[stim.start:stim.end]
        output = {}

        output['trace'] = trace
        output['amplitude'] = np.amax(trace)
        if normalize:
            output['amplitude'] = np.amax(trace)/np.amax(self.trace)
        output['normamplitude'] = np.amax(trace)/np.amax(self.trace)
        output['area'] = np.sum(trace)
        if use_transients:
            pass
        else:
            pass
            #output['corr'] = np.absolute(np.corrcoef(stim.waveform, trace)[0,1])

        base_window = int(8/self.parent_data['timestep'])
        baseline = self.trace[(stim.start-base_window)-10:stim.start-10]
        q = stats.median_abs_deviation(baseline)/0.79788456
        # if q == 0:
        #    q = 0.01
        m = np.median(baseline)
        output['baseline_diff'] = np.amax(trace) - (m+q)
        output['baseline_ratio'] = (np.amax(trace) - m)/q
        # plt.figure()
        # plt.plot(trace)
        # print(output['amplitude'])
        
        try:
            output['z_latency'] = np.where(trace>3.5)[0][0]/10
            
        except:
            output['z_latency'] = np.inf
            output['z_duration'] = np.nan
        
        latency_peak = np.where(trace==np.amax(trace))[0][0]
        try:
            peak_stim_temp = stim.waveform[latency_peak]
        except:
            pdb.set_trace()
        peak_frame = latency_peak+stim.start
        post_segment = self.trace[int(peak_frame):int(peak_frame+80)]

        if any(post_segment < 3.5):
            response_end = np.where(post_segment<3.5)[0][0]+latency_peak
            response_end_global = response_end + stim.start
            output['z_duration'] = (response_end)/10
        else:
            output['z_duration'] =  (len(post_segment) + (len(stim.waveform)-latency_peak))/10
        
            
            
        try:
            output['latency_on'] = np.where(
                trace > output['amplitude']*0.8)[0][0]
        except:
            output['latency_on'] = trace.shape[0]
        
        

        if plot:
            self.parent_session.index_cells()
            
            G = plt.figure(f'Cell {self.cell_n} from {self.parent_session.source}')
            A = G.add_subplot(1,1,1)
            A.plot(self.thermStim, 'k')
            A.plot(self.trace, 'r')
            A.plot([peak_frame, response_end_global],[3.5,3.5], 'r')
            A.scatter(peak_frame, peak_stim_temp, color='r')
            
            A.plot(stim.timepoints, stim.waveform)
            
            #plt.show()
        for o in output.keys():
            output[o] = np.nan_to_num(output[o])
        return(output)

def dump(var_dict, filepath = '/home/ch184656/GUI debug/'):
    with open(filepath+  'log_' +str(str(time.time())[1:10] +'pickle'), 'wb') as file:
        output = {}
        for var in var_dict:
            if dill.pickles(var_dict[var]):
                output[var] = var_dict[var]
            else:
                print(f'Unable to pickle {var}')
        dill.dump(output, file)
        file.close()
    
def dumpTest():
    foo = 5
    bar = {}
    bar[1] = 7
    dump(locals())
    
    
    
def undump(folder = '/home/ch184656/GUI debug/'):
    file_list = glob.glob(folder+'/*')
    latest_log = max(file_list, key = os.path.getctime)
    print(latest_log)
    file = open(latest_log, 'rb')
    output = dill.load(file)
    file.close()
    return(output)

def jitter(X,Y, s=20, color=None, scale = 1, cmap=None, norm=None, vmin = None, vmax = None, alpha = 0.5, linewidths=None, edgecolors=None, A=None, **kwargs):
    if A is None:
        A = plt.gca()
    rng = np.random.default_rng()
    v = rng.uniform(low=-0.1, high = 0.1, size=X.size)
    X = X+v/scale
    A.scatter(X,Y, s=s, color=color, cmap=cmap, norm=norm, vmin = vmin, vmax = vmax, alpha = alpha, linewidths=linewidths, edgecolors=edgecolors, **kwargs)
    
        
    
def box_off(A, left_only = False, All = False, bot_only=False):
    A.spines.right.set_visible(False)
    A.spines.top.set_visible(False)
    if left_only:
        A.spines.bottom.set_visible(False)
        A.xaxis.set_visible(False)
    if bot_only:
        A.spines.left.set_visible(False)
        A.yaxis.set_visible(False)
    if All:
        A.spines.left.set_visible(False)
        A.spines.bottom.set_visible(False)
        A.xaxis.set_visible(False)
        A.yaxis.set_visible(False)
    return(A)


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (ymin, ymax, xmin, xmax)


def blank_fig(title='Figure'):
    F = plt.figure()
    A = F.add_axes([0, 0, 1, 1])
    return(F, A)


def read_T_series(path='/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Tseries'):

    Tseries = []
    with open(path) as F:
        lines = F.readlines()
    for line in lines:
        a = line.split('\n')[0]
        b = float(a.split(',')[0])
        c = float(a.split(',')[1])
        Tseries.append((b, c))
    print(f'Getting Tseries from path {path}')
    return(Tseries)


def get_temps_at_peaks(session_data, cells=None, tRange=[0, 55], plot=True, F=None, A=None):
    if cells is None:
        cells = session_data.cells.values()
    Tstims = []
    T_amplitudes = []
    peak_temps = []
    peak_latencies = []
    temperature_values = []
    output = {}
    for T in session_data.Tstim.values():
        if T.stim_temp >= tRange[0] and T.stim_temp <= tRange[1]:
            Tstims.append(T)
            T_amplitudes.append(T.stim_temp)

    for CELL in cells:
        PTs = []
        PLs = []
        Tamps = []
        for T in Tstims:

            response_trace = CELL.response(T)['trace']
            stim_trace = T.waveform
            maxIX = np.where(response_trace == np.amax(response_trace))
            temp_at_peak = stim_trace[maxIX]
            PTs.append(temp_at_peak)
            PLs.append(maxIX*session_data.timestep)
            Tamps.append(T.stim_temp)
    peak_temps.append(PTs)
    peak_latencies.append(PLs)
    temperature_values.append(Tamps)
    if plot:
        if F is None:
            F, A = blank_fig()
        for PT, T_amplitudes in zip(peak_temps, temperature_values):
            A.scatter(T_amplitudes, PT)
    return(peak_temps)


def props_vs_temp(session_data, cells=None, tRange=[0, 55], plot=True, F=None, A=None, props=None):
    if cells is None:
        cells = session_data.cells.keys()
    Tstims = []
    T_amplitudes = []

    output = {}
    if props is None:
        dT = session_data.Tstim[0]
        props = session_data.cells[0].response(dT).keys()
    F={}
    A={}  
    for prop in props:
        output[prop] = []
        #prop_mat = np.array([])
        for T in session_data.Tstim.values():
            if T.stim_temp >= tRange[0] and T.stim_temp <= tRange[1]:
                Tstims.append(T)
                T_amplitudes.append(T.stim_temp)

        for CELL in cells:
            prop_array = []
            for T in Tstims:
                response = session_data.cells[CELL].response(T)
                result = response[prop]
                prop_array.append(result)
            output[prop].append(prop_array)
            #corr_mat = vstack(corr_mat,corr_array)
        
        if plot and not prop == 'trace':
            #pdb.set_trace()
            
            
                for prop_array in output[prop]:
                    if not prop in F:
                        F[prop] = plt.figure(prop)
                        A[prop] = F[prop].add_axes([0,0,1,1])
             
                    A[prop].scatter(T_amplitudes, prop_array)
                    #A[prop].plot(T_amplitudes, prop_array)
                    # A.set_ylim(-1,1)

    return(output)


class stim:
    def __init__(self, parent=[], modality=None, submodality=None, bounds=[0, 0], waveform=np.array([]), intensity=0, stim_temp=None):
        self.modality = modality  # thermo, mechano, chemical, opto, etc.
        self.submodality = submodality  # heating, cooling, indentation, brush, etc
        self.bounds = bounds  # first and last framenumber
        self.start = bounds[0]
        self.end = bounds[1]
        self.timepoints = np.linspace(
            self.start, self.end-1, num=(self.end-self.start))
        self.waveform = waveform  # intenbsity over time
        if self.timepoints.shape[0] < self.waveform.shape[0]:
            self.timepoints = np.linspace(
                self.start, self.end, num=1+(self.end-self.start))
        self.location = []  # save for  mechano data
        # deviation from basal temperature, milliNewtons, etc.
        self.intensity = intensity
        self.stim_temp = stim_temp
        self.parent = parent  # trace of stimulus series stim is part of

    def response(self, cell, plot=False, normalize=False):
        output = {}
        trace = cell.trace[self.start:self.end]

        output['trace'] = trace
        output['amplitude'] = np.amax(trace)  # -baseline

        if normalize:
            output['amplitude'] = np.amax(trace)/np.amax(cell.trace)
        output['area'] = np.sum(trace)
        output['corr'] = np.corrcoef(self.waveform, trace)[0, 1]
        output['latency_on'] = np.where(trace > output['amplitude']*0.25)[0][0]
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

    def show(self, F=None, A=None):
        if A == None:
            F = plt.figure()
            A = F.add_axes([0, 0, 1, 1])
        A.plot(self.parent)
        A.plot(self.timepoints, self.waveform)


class session_data:

    def __init__(self, DATA, genotype=None):
        self.mech_stim = None
        self.therm_stim = None

        if not DATA is None:
            if 'Tstim' in DATA.keys():
                DATA = processTempStim(DATA)
                try:
                    DATA = classify_cells_in_dataset(DATA)
                except:
                    print('Could not run thermo classifiication')
            DATA = processMechStim(DATA)
            for key in DATA.keys():
                setattr(self, key, DATA[key])

        for cell in self.cells.values():
            cell.parent_session = self
            cell.parent_data = DATA

        self.genotype = genotype
        self.pain_state = None
    
    def index_cells(self, dummy=0):
        for cell in self.cells:
            self.cells[cell].cell_n = cell
            
    def show_source(self, print_output = True):
        source = str(self.Source).split('"')[1]
        if print_output:
            print(source)
        return(source)
    
    def mech_thresholds(self, cells = 'All', max_threshold = 3.5, limit = 500, min_threshold=2, unit = 'mN', plot=False):
        if cells == 'All':
            cells = []
            for cell in self.cells.keys():
                cells.append(cell)
        t = []
        for cell in cells:
            if not hasattr(self.cells[cell], 'mech_threshold'):
                self.cells[cell].mech_range(plot=False, max_threshold = max_threshold, limit =limit, min_threshold=min_threshold, unit = unit )
            t.append(self.cells[cell].mech_threshold)
        return(t)
        
    def process_mech_stim(self, minStim=10, min_duration = 7, plot=False):
        self.Mstim = process_mech_trace(self.mech_stim, minStim = minStim, min_duration = min_duration, plot=plot)
        
    def segment_thermo_stim(self, min_duration=10,  thresh=2, plot=True, end_fix_duration=80, adjust_for_intense_cold=820):
        if 'Temp (°C)' in self.stims.keys():
            trace = self.stims['Temp (°C)']
        else:
            print('Thermal stim not found!')
            return
        stims = segment_temp_trace_into_stims(trace, self.nullValue, min_duration=min_duration, thresh=thresh,
                                              plot=plot, end_fix_duration=end_fix_duration, adjust_for_intense_cold=adjust_for_intense_cold)
        self.Tstim = {}
        for c, stim in enumerate(stims.values()):
            self.Tstim[c] = stim
        return(self)

    def add_cell(self, cell=None):
        index = len(self.cells)
        cell.parent_session = self
        self.cells[index] = cell

    def classify_thermo_cells(self, cells='All'):
        if cells == 'all':
            cells = []
            for cell in self.cells.keys():
                cells.append(cell)

        for c, cell in enumerate(cells):
            self.cells[cell].classify_thermo()

    def show_plots_groups(self, cellIXs='all', class_filters=None, showStim=True, show_mean=True, showTherm=True, showMech=True, dRange=None, handles=None, stack=False, norm=True, show_time=True, **kwargs):  # plot groups overlaid or stacked

        if not handles is None:
            F, A1, A2, AT = handles
        else:
            F = None
            A1 = None
            A2 = None
            AT = None

        if class_filters is None:
            #stack = False
            class_filters = class_list()

        if stack:
            num_plots = len(class_filters)+1
            F = plt.figure(figsize=[6.4*4, 2.4*num_plots])
            num_plots = len(class_filters)+1
            v_size = 1/(num_plots)
            A = {}
            # pdb.set_trace()
            A[0] = F.add_axes([0, 1-v_size, 1, v_size])
            for c, group in enumerate(class_filters):
                A[c+1] = F.add_axes([0, 1-((c+2)*v_size), 1, v_size])

        for c, group in enumerate(class_filters):
            class_filter = [group]
            if stack:
                A2 = A[0]
                A1 = A[c+1]
                if group == class_filters[-1]:
                    show_time = True
                else:
                    show_time = False
            if c == 0:
                handles = self.show_plots(cellIXs=cellIXs, showStim=showStim, show_mean=show_mean, class_filter=class_filter, showTherm=showTherm, showMech=showMech, F=F, A1=A1, A2=A2,
                                          AT=None, dRange=dRange, mech_y_log=False, trace_color='k', mean_color=None, trace_alpha=None, mean_alpha=1, group_label=group, norm=norm, show_time=show_time, **kwargs)
                F, A1, A2, AT = handles
            else:
                self.show_plots(cellIXs=cellIXs, showStim=showStim, show_mean=show_mean, class_filter=class_filter, showTherm=False, showMech=False, F=F, A1=A1, A2=A2, AT=AT,
                                dRange=dRange, mech_y_log=False, trace_color='k', mean_color=None, trace_alpha=None, mean_alpha=1, group_label=group, norm=norm, show_time=show_time, **kwargs)

        return(handles)

    def pickle(self, filepath=None):
        file = open(filepath, 'wb')
        pickle.dump(self, file)
        file.close()
        self.pickle_path = filepath

 

    def mech_range(self, cells ='all', plot_cells = True, unit = 'mN', limit=500, plot_summary =True, color=None, threshold = 3.5, sort_responses=True, AS=None, AC=None, CF=None, calc_all=False, return_cells=True):
        if cells == 'all':
            cells = []
            for cell in self.cells.values():
                cells.append(cell)
        for c, cell in enumerate(cells):
            if type(cell) == int:
                cells[c] = self.cells[cell]
            
        source = str(self.Source).split('"')[1]
        title = source.split("/")[-2] + ' ' + source.split("/")[-1]
        responses = []
        if plot_cells:
            CF = plt.figure('Session responses' + ' ' + title, figsize=[4,4])
            AS = CF.add_subplot(math.ceil((len(cells)+2)/2), 1, math.ceil((len(cells)+2)/2))
        for cc, c in enumerate(cells):
            if plot_cells:
                AC = CF.add_subplot(len(cells)+2, 1, cc+1)
                
            responses.append(c.mech_range(max_threshold = threshold, limit=limit, plot=plot_cells, AS=AS, AC=AC, F=CF, n_cells=len(cells), calc_all=calc_all, return_cell = return_cells))
        
        if plot_cells:
            save_fig(CF, title + ' traces ' + uid_string())
            
        if sort_responses:
            responses = sort_mech_ranges(responses)['sorted_responses']

        if color is None:
            color='k'
        if plot_summary:
            plot_mech_ranges(responses, title = title, SNR_threshold = threshold, color=color, limit = limit, unit =  unit)
            
            # F = plt.figure('Force sensing')
            # A = F.add_subplot(1,1,1)
            # for r, response_set in enumerate(responses):
            #     for rr, response in enumerate(response_set):
            #         A.scatter(r, response[0], s = response[1]**1.5, color = 'k')

            # A.set_ylim([0,600])
        
        return(responses)        
            
    
        
    def detrend(self, cells = 'all', wavelength=2400, plot=False, remove_peaks=True, limit_q=0.75, num_segs=20, method='pad', rectify=False):
        if cells =='all':
            cells = []
            for cell in self.cells.values():
                cells.append(cell)
                
        for cell in cells:
            cell.detrend(wavelength=wavelength, plot=plot, remove_peaks=remove_peaks,
                         limit_q=limit_q, num_segs=num_segs, method=method, rectify=rectify, update_raster = False)
            self.update_raster()
    
    def z_score_traces(self, cells = 'all', convert = False, update_raster = True): ##TODO
        
        if cells =='all':
            cells = []
            for cell in self.cells.values():
                cells.append(cell)
        
        # z = modified_z_score(self.trace)
        # if convert:
        #     self.trace = z
        #     if update_raster:
        #         self.parent_session.update_raster()
        # return(z)
    
    
    def show_raster(self, cells='all', sort_raster=True, sort_by='thermo', show_stim=True, show_therm=True, show_mech=True, F=None, AS=None, AR=None, AT=None, dRange=None, cmap='inferno', norm=True, vmin=None, vmax=None):
        if cells == 'all':
            cells = []
            for cell in self.cells.values():
                cells.append(cell)
        else:
            cell_objects = []
            for cell in cells:
                cell_objects.append(self.cells[cell])
            cells = cell_objects

        if dRange is None:
            dRange = [0, self.raster.shape[1]]

        sorted_raster = np.zeros([len(cells), self.raster.shape[1]])
        if sort_raster and sort_by == 'thermo':
            counter = 0

            thermo_types = temp_class_bounds()
            sort_keys = thermo_types.keys()
            for sort_key in sort_keys:
                for Cell in cells:
                    if Cell.classification == sort_key:
                        if norm:
                            trace = Cell.norm_trace
                        else:
                            trace = Cell.trace
                        sorted_raster[counter, :] = trace
                        counter = counter+1
        else:
            #print('Not sorting raster')
            sorted_raster = np.zeros([len(cells), self.raster.shape[1]])
            for c, Cell in enumerate(cells):
                if norm:
                    trace = Cell.norm_trace
                else:
                    trace = Cell.trace
                sorted_raster[c, :] = trace

        if F is None:
            # !! gets name of animal and FOV for fig name
            source = ' '.join(self.Source.path.split('/')[2:])
            F = plt.figure(source + ' raster')
            if show_stim:
                AR = F.add_axes([0, 0, 1, 0.75])
                AS = F.add_axes([0, 0.75, 1, 0.25])
            else:
                AR = F.add_axes([0, 0, 1, 1])
        if len(cells) > 0:
            print('showing raster in auto')
            AR.imshow(sorted_raster, aspect = 'auto',
                      interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            AR.imshow(self.raster*0)
        if show_stim:
            if show_therm and show_mech:
                if not self.therm_stim is None:
                    handles = self.plot_therm(
                        F=F, AS=AS, dRange=dRange, show_time=True)
                else:
                    handles = {}
                    handles['AS'] = AS
                    handles['F'] = F
                if not self.mech_stim is None:
                    handles = self.plot_mech(
                        F=F, AS=handles['AS'], AT=None, dRange=dRange)
            elif show_therm and not (self.therm_stim is None):
                handles = self.plot_therm(
                    F=F, AS=AS, dRange=dRange, show_time=True)
            elif show_mech and not (self.mech_stim is None):
                handles = self.plot_mech(
                    F=F, AS=AS, AT=None, dRange=dRange, show_time=True)

        #AR.set_xlim([0, sorted_raster.shape[1]])
        #xtick = (AR.get_xticks()-dRange[0])*self.timestep
        #AR.set_xticks(AR.get_xticks())
        #AR.set_xticklabels((xtick.astype(np.uint64)))

        # plt.figure()
        #plt.imshow(sorted_raster, aspect='auto')
        return(handles)
    
    def update_raster(self, sort = False):
        num_cells = len(self.cells)
        if num_cells == 0:
            self.raster = np.array([])
            return()
  
        for c, cell in enumerate(self.cells):
            if c == 0:
                n_samples = self.cells[cell].trace.shape[0]
                raster = np.zeros([num_cells, n_samples])
            raster[c,:] = self.cells[cell].trace
        self.raster = raster
        return(raster)
    
    def props_vs_temp(self, cells=None, tRange=[0, 55], plot=True, F=None, A=None, props=None):
        return(props_vs_temp(self, cells=cells, tRange=tRange, plot=plot, F=F, A=A, props=props))
                      
    def get_thermal_stims_set(self, Tseries=None, min_duration=40, reps=None, get_all=False):

        # get series of thermal stimuli to align data to
        if Tseries is None:
            #Tseries = [(3,12), (19,25), (37,40), (41,43), (44,46), (46,48), (48,52)]

            Tseries = read_T_series()
        if reps is None:
            #reps = [2, 2, 2, 1, 2, 2, 2]
            reps = [1]
        if len(reps) == 1:  # If reps input is a one iteme list, look for that number of stims for each range in Tseries
            rep = []

            for t in Tseries:
                rep.append(reps[0])
            reps = rep

        if get_all:
            expanded_reps = []
            for rep in reps:
                expanded_reps.append(10)
            reps = expanded_reps

        stim_indices_dict = {}
        used_indices = []
        all_stim_indices = list(self.Tstim.keys())
       # pdb.set_trace()

        num_reps_found_list = []

        for tt, (rep, Trange) in enumerate(zip(reps, Tseries)):
            num_reps_found = 0
            for r in range(rep):
                for c, stim in enumerate(all_stim_indices):
                    stim_temp = self.Tstim[stim].stim_temp
                    if stim_temp >= Trange[0] and stim_temp <= Trange[1] and len(self.Tstim[stim].waveform > min_duration):
                        s = all_stim_indices[c]
                        if not s in used_indices and not (tt, r) in stim_indices_dict.keys():
                            stim_indices_dict[(tt, r)] = s
                            used_indices.append(s)
                            num_reps_found = num_reps_found+1  # counting available stim for each range
                            continue
            num_reps_found_list.append(num_reps_found)

        if get_all:  # if getting all available stims for each stimulus range
            reps = num_reps_found_list

        stim_list = []
        missing_stims = []
        for tt, (rep, Trange) in enumerate(zip(reps, Tseries)):
            for r in range(rep):
                if (tt, r) in stim_indices_dict:
                    stim_list.append(stim_indices_dict[(tt, r)])
                else:
                    missing_stims.append((Tseries[tt], r))
                    # pdb.set_trace()
                    print(
                        f'Session from {self.Source} missing trial {r} for temp range {Trange}')

        return(stim_list, missing_stims, stim_indices_dict)

    def regularize_raster(self, Tseries=None, prepend_frames=10, append_frames=10, plot=False, min_duration=40, end_fix_duration=None, fail_if_missing=False, reps=None, cmap = 'inferno', sort_by = 0, vmin=None, vmax = None, F=None, A1=None, A2=None):
        # pdb.set_trace()
        print(f'{fail_if_missing=}')
        # pdb.set_trace()
        self.segment_thermo_stim(
            plot=False, min_duration=min_duration, end_fix_duration=end_fix_duration)
        stims, missing, stim_dict = self.get_thermal_stims_set(
            Tseries=Tseries, reps=reps)
        if missing and fail_if_missing:
            print(f'Stims {missing} were not found')
            if fail_if_missing:
                print('session returns no data')
                return(None, None, missing)
        else:
            print(Tseries)
            print(stims)
        raster_segments = []
        stim_segments = []
        for stim in stims:
            start = self.Tstim[stim].start - prepend_frames
            if start < 0:
                #start= 0
                print(
                    f"Can't prepend enough frames for stim # {stim} starting at {start}")
                return(None)
            end = self.Tstim[stim].end + append_frames
            if end >= self.raster.shape[1]:
                #end = self.raster.shape[1]-1
                print(
                    f"Can't append enough frames for stim # {stim} ending at {end}")
                return(None)
            raster_segments.append(self.raster[:, start:end])
            stim_segments.append(self.Tstim[stim].parent[start:end])
        for c, (r, s) in enumerate(zip(raster_segments, stim_segments)):
            if c == 0:
                raster_output = r
                stim_output = s
            else:
                raster_output = np.hstack((raster_output, r))
                stim_output = np.hstack((stim_output, s))

        if plot:
            if F is None:
                F = plt.figure()
            if A1 is None:
                A1 = F.add_axes([0, 0, 1, 0.8])
            if A2 is None:
                A2 = F.add_axes([0, 0.8, 1, 0.2])
            if vmin is None:
                vmin = np.amin(raster_output)
            if vmax is None:
                vmax = np.amax(raster_output)
            A1.imshow(raster_output, aspect='auto', vmin = vmin, vmax = vmax, cmap=cmap)
            A2.plot(stim_output)
            A2.set_xlim(0, stim_output.shape[0])
      #  print(f'Number of stims is {len(stims)}' )
        # pdb.set_trace()
       # for c, stim in enumerate(stims):
       #     print(f'Stim {c} is {len(self.Tstim[stim].timepoints)} samples starting at {self.Tstim[stim].start}')

        return(raster_output, stim_output, missing)
    
   
        
        
        
        
    def plot_mech(self, F=None, AS=None, AT=None, dRange=None):
        if F is None:
            F = plt.figure()
        if AS is None:
            AS = F.add_axes([0, 0, 1, 1])
            AS.set_frame_on(False)
            AS.set_visible(False)
        if dRange == None:
            dRange = [0, self.mech_stim.shape[0]]
        if AT is None:
            AT = AS.twinx()
        AT.plot(self.mech_stim, 'k')
        AT.set_frame_on(False)
        sub = self.mech_stim[dRange[0]:dRange[1]]
        ymin = np.nanmin(sub)
        ymax = np.nanmax(sub)
        if np.isnan(ymin):
            ymin = np.nanmin(self.mech_stim)
        if np.isnan(ymax):
            ymax = np.nanmax(self.mech_stim)
        AT.xaxis.set_visible(False)
        AT.set_ylabel('Force (mN)')
        AT.set_xlim(dRange)
        AT.set_ylim(ymin, ymax)
        handles = {}
        handles['F'] = F
        handles['AS'] = AS
        handles['AT'] = AT
        return(handles)

    def plot_therm(self, F=None, AS=None, dRange=None, show_stims=False, show_time=False):
        if F is None:
            F = plt.figure()
        if AS is None:
            AS = F.add_axes([0, 0, 1, 1])
        if dRange == None:
            dRange = [0, self.therm_stim.shape[0]]

        therm_line, = AS.plot(self.therm_stim, 'k', )
        if show_stims:
            if hasattr(self, 'Tstim'):
                pass
            else:
                print('No Tstims have been extracted')

        AS.xaxis.set_visible(False)
        AS.set_ylabel('Temp (°C)')
        AS.xaxis.set_visible(False)
        AS.set_frame_on(False)
        AS.set_xlim(dRange)
        xtick = (AS.get_xticks()-dRange[0])*self.timestep
        AS.set_xticks(AS.get_xticks())
        AS.set_xticklabels((xtick.astype(np.uint64)))

        if not show_time:
            AS.xaxis.set_visible(False)

        handles = {}
        handles['F'] = F
        handles['AS'] = AS
        return(handles)

    # Return image of field with ROIs - work in progress
    def show_field(self, cellIXs='all', display=True, class_filters=None):
        if cellIXs == 'all':
            cellIXs = []
            for key in self.cells:
                if class_filters is None:
                    cellIXs.append(key)
                else:
                    if self.cells[key].classification in class_filters:
                        cellIXs.append(key)

        floatMask = self.ROIs.copy()
        for cell in range(floatMask.shape[-1]):
            floatMask[..., cell] = floatMask[..., cell] / \
                np.max(np.max(floatMask[..., cell]))
        labelSelected = floatMask[:, :, cellIXs]
        RGBA = []
        if display:
            plt.imshow(RGBA)
        return(RGBA)
    
    def directory(self, a=1):
        for i in dir(self):
            print(i)
            
    def show_plots(self, cellIXs='all', show_mean=True, class_filter=None, showTherm=True, showMech=True, F=None, A1=None, A2=None, AT=None, dRange=None, mech_y_log=False, trace_color='b', mean_color=None, trace_alpha=None, mean_alpha=1, showStim=True, group_label=None, norm=True, show_time=True, **kwargs):
        ts = self.timestep
        if not (dRange is None):
            dRange = np.array(dRange)
            sRange = dRange/ts
            sRange = sRange.astype(np.uint64)

            print(f'{ts=}')
            print(f'{dRange=}')
            print(f'{sRange=}')
        else:
            sRange = [0, self.cells[0].trace.shape[0]]

        # Determine which cells to plot - all as default, only matching class if class filter is specified
        if cellIXs == 'all':
            cellIXs = []
            for key in self.cells:
                if class_filter is None:
                    cellIXs.append(key)
                else:
                    if self.cells[key].classification in class_filter:
                        cellIXs.append(key)
                    color_set = gen_colors()
                    trace_color = color_set[class_filter[0]]

        if len(cellIXs) == 0:
            print('Cells of specified type not found')
            return

        if mean_color is None:
            mean_color = trace_color
        if trace_alpha is None:
            if show_mean:
                trace_alpha = 1/len(cellIXs)
            else:
                trace_alpha = 1/len(cellIXs)
        trace_array = np.array([])

        for c, cellIX in enumerate(cellIXs):
            if c == 0:
                if norm:
                    trace_array = self.cells[cellIX].norm_trace[sRange[0]:sRange[1]]
                else:
                    trace_array = self.cells[cellIX].trace[sRange[0]:sRange[1]]
                handles = self.cells[cellIX].show(showTherm=showTherm, showMech=showMech, F=F, A1=A1, A2=A2, AT=A2, dRange=dRange, mech_y_log=mech_y_log,
                                                  trace_color=trace_color, trace_alpha=trace_alpha, showStim=showStim, group_label=group_label, norm=norm, show_time=show_time, **kwargs)
                F = handles[0]
                A1 = handles[1]
                A2 = handles[2]
            else:
                if norm:
                    trace_array = np.vstack(
                        (trace_array, self.cells[cellIX].norm_trace[sRange[0]:sRange[1]]))
                else:
                    trace_array = np.vstack(
                        (trace_array, self.cells[cellIX].trace[sRange[0]:sRange[1]]))
                self.cells[cellIX].show(showTherm=False, showMech=False, F=F, A1=A1, A2=A2, AT=A2, dRange=dRange, mech_y_log=mech_y_log,
                                        trace_color=trace_color, trace_alpha=trace_alpha, showStim=False, group_label=group_label, norm=norm, show_time=show_time, **kwargs)
        if show_mean:

            if dRange is None:
                A1.plot(np.mean(trace_array, axis=0),
                        color=mean_color, alpha=mean_alpha)
            else:
                X_vals = np.arange(sRange[0], sRange[1], 1)
                print(f'{X_vals}')
                A1.plot(X_vals, np.mean(trace_array, axis=0),
                        color=mean_color, alpha=mean_alpha)

        handles = [F, A1, A2, AT]
        return(handles)

    def print_classification(self, cellIXs='All'):
        if cellIXs == 'all':
            for key in self.cells:
                cellIXs.append(key)
        for c in cellIXs:
            print(f'Cell {c} is classified as {self.cells[c].classification}')

    def class_count(self, classification=None):
        found = 0
        for cell in self.cells.values():
            if cell.classification == classification:
                found = found + 1
        return(found)

    def get_cells_from_class(self, class_list=None):
        if class_list is None:
            output = {}
            output['all'] = self.cells
            return(output)
        output = {}
        for CLASS in class_list:
            output[CLASS] = []
            for CELL in self.cells.values():

                if CELL.classification == CLASS:
                    output[CLASS].append(CELL)
        return(output)

    # b is bounds of stack to retrieve
    def get_raw_stack(self, start=None, stop=None, left=None, right=None, top=None, bottom=None):
        F = h5py.File('temp.h5', 'a')
        if 'source' in F.keys():
            del F['source']
        F['source'] = self.cells[0].parent_data['Source']
        dshape = F['source'][self.cells[0].parent_data['activity key']].shape
        if start is None:
            start = 0
        if stop is None:
            stop = dshape[0]
        if left is None:
            left = 0
        if right is None:
            right = dshape[1]
        if top is None:
            top = 0
        if bottom is None:
            bottom = dshape[2]
        print(f"Getting activity data from {self.cells[0].parent_data['Source']}, Frames {start}:{stop}")
        stack = F['source'][self.cells[0].parent_data['activity key']
                            ][start:stop, left:right, top:bottom]  # start-stop,left:right,top:bot
        F.close()
        return(stack)

    def get_source_time(self, flag='Thermo stim', search_key=None):
        output = []

        F = h5py.File('temp.h5', 'a')
        if 'source' in F.keys():
            del F['source']
        F['source'] = self.cells[0].parent_data['Source']

        if not search_key is None:
            output.append(F['source']['T'][search_key][...])
            return(output)

        for key in F['source'].keys():
            for attr in F['source'][key].attrs:
                if attr == flag:
                    if F['source'][key].attrs[attr]:
                        output.append(F['source']['T'][key][...])
        return(output)

    def get_source_data(self, flag='Thermo stim', search_key = None):
        output = {}
        output['data'] = []
        output['T'] = []
        F = h5py.File('temp.h5', 'a')
        if 'source' in F.keys():
            del F['source']
            
        F['source'] = self.cells[0].parent_data['Source']
        
        if not search_key is None:
            output.append(F['source']['T'][search_key][...])
            return(output)
        
        for key in F['source'].keys():
            for attr in F['source'][key].attrs:
                if attr == flag:
                    if F['source'][key].attrs[attr]:
                        if flag == 'Thermo stim':
                            traceArray = F['source']['R'][key]['traceArray'][...]
                            output['data'].append(traceArray[0, ...])
                        else:
                            output['data'].append(F['source'][key][...])
                        output['T'].append(F['source']['T'][key][...])
        return(output)

    def calc_trace_from_mask(self, mask=None, method='cnmf', detrend=False, dff_z_score=True):
        stack = self.get_raw_stack()
        if method == 'cnmf':
            extracted = seeded_CNMF(stack, masks=mask, detrend=detrend)
        elif method == 'dff':
            extracted = extract_dff_trace(stack, masks=mask, detrend=detrend, z_score = dff_z_score)
        return(extracted)

    def convert_traces_to_dff(self, cells='All', dff_maxse=False, dff_z_score=True, **kwargs):
        if cells == 'All':
            cells = list(self.cells.keys())
        extracted = self.calc_trace_from_mask(mask=self.ROIs[:, :, cells], method='dff', dff_z_score=dff_z_score)
        o_raster = copy.deepcopy(self.raster)
        t = o_raster.shape[1]
        for cell in cells:
            
            self.cells[cell].trace = resample(
                extracted['traces_out'][cell, :], t)
            Tr = copy.copy(self.cells[cell].trace)
            Tr = Tr-np.amin(Tr)
            
            self.cells[cell].norm_trace = Tr/np.amax(Tr)
            self.raster[cell, :] = self.cells[cell].trace
        self.update_raster()
        print('Converted all traces to dff')

    def convert_traces_to_cnmf(self, cells='All', **kwargs):
        if cells == 'All':
            cells = list(self.cells.keys())
        extracted = self.calc_trace_from_mask(
            mask=self.ROIs[:, :, cells], method='cnmf')
        o_raster = copy.deepcopy(self.raster)
        t = o_raster.shape[1]
        for cell in cells:
            self.cells[cell].trace = resample(
                extracted['traces_out'][cell, :], t)
            self.raster[cell, :] = self.cells[cell].trace
        self.update_raster()
        print('Converted all traces to dff')

    def calc_transients(self, cells='All', **kwargs):
        # options:
        # threshold_SNR' : 5,  ## 5 x SNR threshold
        #         'duration' : 10, ## 10 frames (could switch to seconds if timestep known)
        #         'pad' : True,
        #         'pad_n' : 10}
        if cells == 'All':
            cells = list(self.cells.keys())
        for cell in cells:
            self.cells[cell].identify_transients(**kwargs)

    def identify_transients(self, cells='All'):
        pass

    def show_transients(self, cells=None, show_trace=True, expand_bounds=0, **kwargs):
        for cell in cells:
            params = {'show_trace': True,
                      'resample': True,
                      'recalc': True,
                      'expand_bounds': 0}
            params.update(kwargs)
            if not 'transients' in dir(self.cells[cell]):
                transients = self.cells[cell].identify_transients(**kwargs)
            elif self.cells[cell].transients is None:
                transients = self.cells[cell].identify_transients(**kwargs)
            else:
                transients = self.cells[cell].transients
            if params['recalc']:
                transients = self.cells[cell].identify_transients(**kwargs)

            # pdb.set_trace()
            mask = self.ROIs[:, :, cell]*255
            y, x, h, w = cv2.boundingRect(mask.astype(np.uint8))
            p = params['expand_bounds']
            y = y-p
            x = x-p
            h = h + (2*p)
            w = w + (2*p)
            print(f'{x=},{y=},{w=},{h=},{p=}')
            if y < 0:
                y = 0
            if x < 0:
                x = 0
            print(f'{x=},{y=},{w=},{h=},{p=}')
            clipped_ROI = mask[x:x+w, y:y+h]
            clipped_ROI = np.expand_dims(clipped_ROI, axis=2)
            caiman_ROI = DYroi2CaimanROI(clipped_ROI)
            movie = self.get_raw_stack(left=x, right=x+w, top=y, bottom=y+h)
            if params['resample']:
                # Number of samples for trace (previously resampled when rasterizing)
                t = self.cells[cell].trace.shape[0]
                movie = resample(movie, t, axis=0)
            #plt.figure('Area mean trace')
            area_trace = np.average(movie, axis=(1, 2))
            f0 = np.amin(area_trace)
            delta_f = area_trace-np.amin(area_trace)

            dff = delta_f/f0
            norm_area_trace = dff-np.amin(dff)
            norm_area_trace = dff/np.amax(norm_area_trace)
            # plt.plot(area_trace)

            row = int(np.ceil(np.sqrt(len(transients))))
            col = int(np.ceil(np.sqrt(len(transients))))
            if params['show_trace']:
                row = row+2
            fig = plt.figure(
                f'transient correlation for cell {cell} from {self.Source}')
            # pdb.set_trace()

            for c, transient in enumerate(transients.keys()):
                a = fig.add_subplot(row, col, c+1)
                t_map = np.zeros([movie.shape[1], movie.shape[2]])
                for ii in range(0, movie.shape[1]):
                    for jj in range(0, movie.shape[2]):
                        t_start = transients[transient]['frames'][0]
                        t_end = transients[transient]['frames'][-1]
                        movie_sub = movie[t_start:t_end+1, ii, jj]
                        print(f'{movie.shape=}')
                        print(f'{ii=}')
                        print(f'{jj=}')
                        print(f'{movie_sub.shape=}')
    #                   print(f"{transients[transient]['trace'].shape=}")
      #                  print(f"{transients[transient]['frames'].shape=}")
                        print(f'{transient=}')
                        t_map[ii, jj] = np.corrcoef(
                            movie_sub, transients[transient]['trace'])[0, 1]
                # a.imshow(t_map)
                plot_contours(caiman_ROI, t_map, ax=a)
                a.xaxis.set_visible(False)
                a.yaxis.set_visible(False)

            if params['show_trace']:
                A2 = fig.add_subplot(row, 1, row-1)
                A1 = fig.add_subplot(row, 1, row)
                self.cells[cell].show(
                    disp_transients=True, F=fig, A1=A1, A2=A2)
                A1.plot(np.arange(
                    0, norm_area_trace.shape[0]), norm_area_trace, alpha=0.5, linewidth=1, color='r')

def sort_mech_ranges(responses):   
    thresholds = []
    for r, response_set in enumerate(responses):
        threshold = np.Inf
        for rr, response in enumerate(response_set):
            if response['valid_response']:
                threshold = np.amin([threshold, response['force']])
        thresholds.append(threshold)
    
    IX = np.argsort(thresholds)
    sorted_responses = [responses[x] for x in IX]
    responses = sorted_responses
    output = {}
    output['sorted_responses'] = sorted_responses
    output['IX'] = IX
    return(output)

def align_thermo_rasters(sessions, cohort_name = None, nClusters=4,  prepend_frames = 60, append_frames=80, end_fix_duration=80, PLOT=True, plot=False, do_sort=True, Tseries=None, **kwargs):
    # For aligning sessions across animals
    #kwargs['fail_if_missing'] = True
    rasters = []
    stims = []
    trace_session_IXs = []
    performance = {}
    performance['errors'] = []
    performance['aligned'] = []
    performance['cell_list'] = []
    filenames = []
    for c, session in enumerate(sessions):
        if kwargs['z_scored']:
            session.convert_traces_to_dff()
        raster, stim, missing = session.regularize_raster(end_fix_duration=end_fix_duration,  prepend_frames = prepend_frames, append_frames=append_frames, fail_if_missing=True, Tseries=Tseries)
        if not raster is None and not stim is None:
            rasters.append(raster)
            trace_session_IXs.append(np.zeros(raster.shape[0])+c)
            stims.append(stim)
            performance['aligned'].append(session)
            filenames.append(f"{session.Source.filename.split('/')[-1].split('.')[0]}")

            for cell in session.cells:
                performance['cell_list'].append(session.cells[cell])

            
        else:
            performance['errors'].append((session, missing))
            
    if len(rasters)==0:
        return(np.array([]),np.array([]),performance)
    
    combined_raster = np.vstack(rasters)
    print(f'{do_sort=}')
    if do_sort:
        newIX = sortByKmean(combined_raster, nClusters)
        combined_raster = combined_raster[newIX, :]
        performance['cell_list'] = [performance['cell_list'][i] for i in newIX]
    #trace_session_IXs = trace_session_IXs[newIX]
    combined_stims = np.vstack(stims).T
    if PLOT:
        if cohort_name is None:
            figname = '-'.join(filenames)
        else:
            figname = cohort_name
        F = plt.figure(figname)
        A_raster = F.add_axes([0.1, 0.1, 0.9, 0.7])
        A_stim = F.add_axes([0.1, 0.8, 0.9, 0.2])
        #A_sess_IX = F.add_axes([0.9, 0, 0.8])

        A_raster.imshow(combined_raster, aspect='auto', vmin=0, vmax=0.5)
        A_raster.set_yticks([0, combined_raster.shape[0]])
        A_raster.spines.bottom.set_visible(False)
        A_raster.set_xticks([])

        A_stim.plot(combined_stims)
        A_stim.set_xlim([0, combined_stims.shape[0]])
        A_stim.spines.top.set_visible(False)
        A_stim.spines.right.set_visible(False)
        A_stim.spines.bottom.set_visible(False)
        A_stim.set_xticks([])

    return(combined_raster, combined_stims, performance)


def extract_dff_trace(stack, masks, weighted=False, detrend=False, z_score = True):

    vStack = np.reshape(
        stack, [stack.shape[0], stack.shape[1]*stack.shape[2]], order='F')

    output = {}
    if not weighted:
        mask = masks > 0
    output['masks'] = masks
    if len(mask.shape) < 3:
        mask = np.expand_dims(mask, axis=2)
    oMask = np.moveaxis(mask, [0, 1, 2], [1, 2, 0])
    vMask = np.reshape(
        oMask, [oMask.shape[0], oMask.shape[1]*oMask.shape[2]], order='F')
    traceArray = np.zeros([vMask.shape[0], vStack.shape[0]], dtype=np.float64)
    for cell, ROI in enumerate(vMask):

        raw = np.percentile(vStack[:, ROI], 50, axis=1)
        F0 = np.percentile(raw, 10)
        df = raw-F0
        dff = df/F0
        if z_score:
            dff = modified_z_score(dff)
        traceArray[cell, :] = dff
        print(f'Extracted cell {cell} of {vMask.shape[0]}')
    output['traces_out'] = traceArray
    return(output)


def sec2day(seconds):
    return(seconds/(60*60*24))


def diag_mat(rem=[], result=np.empty((0, 0))):
    if not rem:
        return result
    m = rem.pop(0)
    result = np.block(
        [
            [result, np.nan*np.zeros((result.shape[0], m.shape[1]))],
            [np.nan*np.zeros((m.shape[0], result.shape[1])), m],
        ]
    )
    return diag_mat(rem, result)


def multi_sess_color_series():
    return([[0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0, 1],
            [1, 0, 0.5]])


class multi_session:
    # collection of imaging sessions
    # need to aligned with  same XY dimensions for cell tracking to work

    def __init__(self, data_list, genotype=None):
        self.correspondence_chart = np.array([])  # unused, depcrecate?
        self.genotype = genotype
        self.n_sessions = 0
        self.sessions = []
        self.union_ROIs = None
        self.u_cells = None  # unused, deprecate?
        self.assignments = None
        self.matchings = None  # unused, depcreccate?
        self.add_data(data_list)
        self.sess_color_series = multi_sess_color_series()

    def add_data(self, data_list=None):  # add sessions from list of sessions
        for c, session in enumerate(data_list):
            self.sessions.append(session)
            self.n_sessions = self.n_sessions + 1
            session_IX = np.expand_dims(np.arange(0, len(session.cells), 1), 1)
            if self.assignments is None:
                self.assignments = session_IX
            else:
                assign_in = self.assignments
                filler_one = np.zeros([assign_in.shape[0], 1])*np.nan
                filler_two = np.zeros(
                    [session_IX.shape[0], assign_in.shape[1]])*np.nan

                pre = np.vstack((assign_in, filler_two))
                post = np.vstack((filler_one, session_IX))

                self.assignments = np.hstack((pre, post))
            self.claim_sessions
                
    def all_sessions(self, dummy=None):
        selected_sessions = []
        for c, session in enumerate(self.sessions):
            selected_sessions.append(c)
        return(selected_sessions)
    
    def claim_sessions(self, selected_sessions = 'All'):
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
        for session in selected_sessions:
            self.sessions[session].parent = self
            
    def index_cells(self, selected_sessions='All'):
        if selected_sessions == 'All':
            selected_sessions = []
            for n, session in enumerate(self.sessions):
                selected_sessions.append(n)
        for session in selected_sessions:
            self.sessions[session].index_cells()
            
    def get_exp_days(self, start_event='SNI'):
        cohort = {}
        cohort['This'] = [self]
        get_exp_days(cohort, start_event = start_event)
    
    def show_assignments(self, cmap = 'jet'):
        print(self.assignments)
        plt.figure('Assignments')
        plt.imshow(self.assignments, cmap=cmap)
        
    def sort_u_cells(self, key_session=0, n_clusters=3, sort_method='sortByKmean', init_table=None):
        # get new indexes by clustering raster of key session:
        key_raster = self.sessions[key_session].raster

        if sort_method == 'sortByKmean':
            new_IX = sortByKmean(key_raster, n_clusters)
        else:
            print('Unknown sort method specified')
            return

        for s, session in enumerate(self.sessions):
            # update cells structure of each session
            # pdb.set_trace()
            old_cells = copy.deepcopy(session.cells)
            session.cells = {}
            for c, ix in enumerate(new_IX):
                session.cells[c] = old_cells[ix]

            # update trace structure of each session if present
            if hasattr(session, 'trace_struct'):
                old_struct = copy.deepcopy(session.trace_struct)
                session.trace_struct = {}
                for key in old_struct.keys():
                    o_ix = key[0]
                    session.trace_struct[new_IX[o_ix],
                                         session] = old_struct[key]

            # update raster and nRaster for each session
            session.raster = session.raster[new_IX]
            #session.nRaster = session.nRaster[new_IX]

            # update ROIs for each session
            session.ROIs = session.ROIs[:, :, new_IX]
            print(f'Cells sorted for data from {session.Source}')
        self.assignments = self.assignments[new_IX]
        print('Done sorting')

    def remove_sessions(self, selected_sessions=None):
        # remove sessions from self.sessions, renumber
        # remove sessions from aliginments
        # update self.n_sessioni
        # pdb.set_trace()
        print(f'Deleting {selected_sessions=}')
        if selected_sessions is None:
            return
        original_sessions = copy.deepcopy(self.sessions)

        # for session in selected_sessions:
        #   del self.sessions[session]

        new_sess = []
        for c, s in enumerate(original_sessions):
            if not (c in selected_sessions):
                new_sess.append(s)
        self.sessions = new_sess
        # remove columns of deleted ssessions from assignemtns:
        self.assignments = np.delete(
            self.assignments, selected_sessions, axis=1)
        # Remove from assignments anyrow that is all Nan (i.e. rois from deleted sessions)
        self.assignments = self.assignments[~np.isnan(
            self.assignments).all(axis=1)]
        self.n_sessions = len(new_sess)
    
    def reorder_sessions(self, new_order = None):
        original_sessions = copy.deepcopy(self.sessions)
        new_sessions = []
        original_assignments = copy.deepcopy(self.assignments)
        for c, s in enumerate(new_order):
            new_sessions.append(original_sessions[int(s)])
            self.assignments[:,c] = original_assignments[:,int(s)]
        self.sessions = new_sessions
       
    
    def set_exp_time(self, selected_sessions='All'):
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
        starts = []
        for session in selected_sessions:
            starts.append(self.sessions[session].experiment_start)

    def align_cells(self, selected_sessions='All', selected_ROIs='All', align_method='caiman', *args, **kwargs):

        # Identifies correspondence between regions of interest in multiple imaging sessions
        ####
        # Input argments:
        # selected_sessions:
        # list of integers specifying which sessions used for alignment
        # $
        # selected_ROIs:
        # Dictionary with session numbers as key and value of list of integers specifying ROIs from each session to be aligned

        o_assignments = self.assignments.copy()

        print(f'Selected sessions for alignment: {selected_sessions}')
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
        masks_in = []
        templates = []
        ROI_IXs = []
        for session in selected_sessions:
            if selected_ROIs == 'All':
                ROI_IXs.append(range(0, self.sessions[session].ROIs.shape[2]))
            else:
                ROI_IXs.append(selected_ROIs[session])
            ROI_IXs[session].sort()

            print(f'For session {session},{ROI_IXs=}')
            masks_in.append(
                self.sessions[session].ROIs[:, :, ROI_IXs[session]])
            templates.append(
                self.sessions[session].fieldImage.astype(np.float64))

        print(f'multi_session align cells: {kwargs.keys()=}')
        allROIs, spatial_union, assignments, matchings = alignROIsCAIMAN(
            masks_in, templates, *args, **kwargs)

        new_assignments = assignments*np.nan

        excluded_u_cells = []
        included_u_cells = []

        for session in selected_sessions:
            session_ROI_selection = ROI_IXs[session]

            # For all ROIs used

            for count, o_assignment in enumerate(o_assignments[:, session]):
                if o_assignment in session_ROI_selection:
                    o_assignments[count, session] = np.nan
                    # add u cell index for arranging masks
                    included_u_cells.append(count)

            for count, ass_result in enumerate(assignments[:, session]):
                if np.isnan(ass_result):
                    pass
                else:
                    new_assignments[count, session] = session_ROI_selection[int(
                        ass_result)]

        included_u_cells = list(set(included_u_cells))
        included_u_cells.sort  # list of indexes of original assignments input to alignment
        for i in range(o_assignments.shape[0]):
            if i in included_u_cells:
                pass
            else:
                # list of indexes of original assignments not used
                excluded_u_cells.append(i)

        print(f'{included_u_cells=}')
        print(f'{excluded_u_cells=}')
        unchanged_assignments = o_assignments[excluded_u_cells]

        self.assignments = np.vstack([unchanged_assignments, new_assignments])
        if self.union_ROIs is None:
            self.union_ROIs = spatial_union
        else:
            unchanged_spatial_union = self.union_ROIs[:, excluded_u_cells]
            self.union_ROIs = np.hstack(
                [unchanged_spatial_union, spatial_union])

        return(allROIs, spatial_union)

    def extract_across_sessions(self, selected_sessions='All', selected_u_cells='All', detrend=True):
        ms.extract_across_sessions(self, selected_u_cells=selected_u_cells,
                                   selected_sessions=selected_sessions, detrend=detrend)

    def show_thermo_grid(self, selected_sessions='All', selected_u_cells='All', norm=True, real_time=False, tuning_threshold=0.3, use_abs_max=True, min_r=0.05):
        thermo_grid_multi_session(self, plot=True, selected_sessions=selected_sessions, selected_u_cells=selected_u_cells,
                                  norm=norm, real_time=real_time, tuning_threshold=tuning_threshold,  min_r=min_r)

    def convert_to_unified_cells(self, selected_u_cells='All', selected_sessions='All'):
        self = ms.create_u_cells(
            self, selected_u_cells=selected_u_cells, selected_sessions=selected_sessions)
    
    
    
    
    def get_stim_corr_full_field(self, selected_u_cells = 'All', selected_sessions='All', selected_stims = 'file', vmin=0.05,vmax=0.5, neighbour = False, invert_cold = True, abs_corr = False):
        self.get_stim_corr_for_ROIs(selected_u_cells = selected_u_cells, selected_sessions = selected_sessions, selected_stims = selected_stims, full_field = True, neighbour = neighbour, invert_cold=invert_cold, abs_corr = False, color_cycle='RGB', plot_contours = False)
        
    def get_neighbour_corr(self, selected_stims = 'file', color_cycle='RGB'): 
        self.get_stim_corr_for_ROIs(selected_stims  = selected_stims, full_field = True, neighbour = True, color_cycle='RGB')
    
    def directory(self, a=1):
        for i in dir(self):
            print(i)
            
    def get_stim_corr_for_ROIs(self, selected_u_cells='All', selected_sessions='All', selected_stims='file', full_field = False, expand=0, padding=30, vmin=0.05,vmax=0.5, neighbour = False, invert_cold = True, match_patch=False, set_patch=True, patch_h = 12, patch_w=12, plot_contours = True, key_session=0, abs_corr = False, color_cycle='RGB'):
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_stims == 'file':
            selected_stims = read_T_series()
        if selected_stims == 'default':
            selected_stims = [(0, 12), (38, 43), (49, 52)]
        
        n_cells = len(selected_u_cells)
        if full_field:
            n_cells = 1
        n_sesh = len(selected_sessions)
        n_stim = len(selected_stims)
        if color_cycle == 'RGB':
            color_cycle = [[0,0,1], [0,1,0],[1,0,0],[0,1,1],[1,1,0],[1,0,1]]
        
        color_cycle = color_cycle[0:len(selected_stims)]
        A={}
        Astim = {}
        stacks = {}
        stack = {}
        if neighbour:
            title = 'Neighbour corr' +str(uuid.uuid4())[0:5]
        else:
            title = 'Stim corr' + str(uuid.uuid4())[0:5]
        F = plt.figure(title, figsize = (n_sesh*4, (n_cells+1)*2), tight_layout=True)
        for u, u_cell in enumerate(selected_u_cells):
            if u >0:
                if full_field:
                    print('ff!')
                    break
            if match_patch and not full_field:
                ref_mask = np.ceil(self.sessions[key_session].cells[u_cell].ROI).astype(np.uint8)
                
            for s, session in enumerate(selected_sessions):
                
                
                
                if u==0:
                    Astim[s] = F.add_subplot(n_cells+1, n_sesh, s+1)
                
                if full_field:
                    x = 0
                    y = 0
                    w = self.sessions[session].fieldImage.shape[0]
                    h = self.sessions[session].fieldImage.shape[1]
                else:
                    mask = np.ceil(self.sessions[session].cells[u_cell].ROI).astype(np.uint8)
                    
                    y, x, h, w = cv2.boundingRect(mask.astype(np.uint8))
                    if match_patch:
                        YY, XX, h,w = cv2.boundingRect(ref_mask.astype(np.uint8)) ## overwrite height and width if matching patch dimensions
                    if set_patch:
                        h = h+patch_h
                        w = w+patch_w
                    p = expand
                    y = y-p
                    x = x-p
                    h = h + (2*p)
                    w = w + (2*p)
             
                    if y < 0:
                        y = 0
                    if x < 0:
                        x = 0
                #stack = self.sessions[session].get_raw_stack(left=x, right=x+w, top=y, bottom=y+h)
                if not (s in stacks.keys()):
                    stacks[s] = self.sessions[session].get_raw_stack()
                if not full_field:
                    stack[s] = stacks[s][:, x-patch_w:x+w, y-patch_h:y+h]
                else:
                    stack[s] = stacks[s]
                #pdb.set_trace()
                if plot_contours:
                    mask_for_contour = np.ceil(self.sessions[session].cells[u_cell].ROI).astype(np.uint8)[x-patch_w:x+w, y-patch_h:y+h]
                    contours = cv2.findContours(mask_for_contour,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
               # stack = self.sessions[session].get_raw_stack(
               #     start=None, stop=None, left=left, right=right, top=top, bottom=bottom)
                stackTime = self.sessions[session].get_source_time(
                    search_key=self.sessions[session].cells[0].parent_data['activity key'])[0]
                stim_dict = self.sessions[session].get_source_data(
                    flag='Thermo stim')
                
                RGB = np.zeros([stack[s].shape[2], stack[s].shape[1], 3])
                #for contour in contours:
                #    RGB = cv2.drawContours(RGB,  contour, -1, (255, 255, 255), 1)
                A[u,s] = F.add_subplot(n_cells+1, n_sesh, (u+1)*(n_sesh)+s+1 )
                stim_data = stim_dict['data'][0]
                T = stim_dict['T'][0]
                # T in zip(stim_dict['data'], stim_dict['T']): ## is this an actual loop:
                    
                
                if u == 0:
                    Astim[s].plot(T, stim_data, color = 'k')
                Tstim = segment_temp_trace_into_stims(stim_data, np.nan, plot=False)

                corr_stims = []
                for bounds in selected_stims:
                    stim_group = []
                    for stim in Tstim:
                        if Tstim[stim].stim_temp > bounds[0] and Tstim[stim].stim_temp < bounds[1]:
                            stim_group.append(Tstim[stim])
                            #corr_stims.append(Tstim[stim])
                            #break
                    corr_stims.append(stim_group)
                
                
                for s_count, stim_group in enumerate(corr_stims):
                    #F = plt.figure(str(s_count)+str(u)+str(s))
                 #   A = F.add_subplot((n_cells*n_stim)+1, n_sesh, u*(n_sesh*n_stim) + (s_count*n_sesh) + s + n_sesh + 1) #u,s,s_count
                    print(f'{u=}, {s=}, {s_count=} ({selected_stims[s_count]}), {n_sesh=}, {n_stim=}')
                    
                    
                   
                    sig = np.array([])
                    sigTime = np.array([])
                    for stimulus in stim_group:
                        start = stimulus.start-padding
                        if start <0:
                            start=0
                        end = stimulus.end+padding
                        sig = np.hstack([sig, stim_data[start:end]])
                        sigTime = np.hstack([sigTime, T[start:end]])
                        
                    if len(sig)>0:
                        if neighbour:
                            frame = local_correlations_fft(stack)
                        else:      
                            frame = corrImSig(stack[s], sig, stackTime, sigTime)
                        if u == 0:
                            dc = u'\u2103'
                            Astim[s].scatter(sigTime, sig, color = color_cycle[s_count], marker='.')
                            Astim[s].set_ylim([0,55])
                            if s == 0:
                                Astim[s].set_ylabel(f'Temp {dc}')
                                box_off(Astim[s], left_only=True)
                            else:
                                box_off(Astim[s], All=True)
                        #Astim[s].plot(stackTime, np.zeros(stackTime.shape),'b')
                        #Astim.plot(imTime, np.ones(imTime.shape),'b')
                        if abs_corr:
                            frame = np.absolute(frame)
                        elif selected_stims[s_count][0] <32:
                            frame = frame * -1 
                    else:
                        frame = stack[s][0,...] * 0
                    frame[np.where(frame<vmin)] = 0
                    
                    frame = frame/vmax
                    frame[np.where(frame>1)] = 1
                    frame = frame.T
                    for cv, color_value in enumerate(color_cycle[s_count]):
                        RGB[:,:,cv] = RGB[:,:,cv] + (color_value*frame)
                    RGB[np.where(RGB>1)] = 1
                    
                bar_scale = 100
                A[u,s].imshow(RGB)
                #for contour in contours:
                #pdb.set_trace()
                if plot_contours:
                    contour_x= []
                    contour_y = []
                    for contour in contours[0]:
                        contour_x.append(contour[0][0])
                        contour_y.append(contour[0][1])
                    contour_x.append(contour_x[0])
                    contour_y.append(contour_y[0])
                    A[u,s].plot(contour_y,contour_x, 'w', linewidth = 1)
                if full_field and s == 0:
                    plot_scale_bar(A[u,s], bar_scale = bar_scale)
                #A[u,s].xaxis.set_visible(False)
                A[u,s].yaxis.set_visible(False)
                A[u,s].spines.right.set_visible(False)
                A[u,s].spines.top.set_visible(False)
                A[u,s].spines.left.set_visible(False)
                A[u,s].spines.bottom.set_visible(False)
                A[u,s].set_xticks([])
                A[u,s].set_xticklabels([])
                if u == n_cells-1:
                    A[u,s].set_xlabel(time.strftime('%b %d %Y', time.localtime(int(self.sessions[session].experiment_start))))
        fname = os.path.split(self.pickle_path)[-1].split('.')[0] + ' ' + title
        F.suptitle(fname)
        
       
        F1 = plt.figure('Legend')
        
        vmin = 0.05
        vmax = 0.5
        color_cycle = [[1,0,0], [0,1,0], [0,0,1]]
        color_scales = []
        for color in color_cycle:
            R = np.expand_dims(np.linspace(0, 1, 256)*color[0], 1)
            G = np.expand_dims(np.linspace(0, 1, 256)*color[1], 1)
            B = np.expand_dims(np.linspace(0, 1, 256)*color[2], 1)

            color_scales.append(np.stack([R,G,B], axis = 2))
        color_scale = np.hstack(color_scales)
        color_scale = np.swapaxes(color_scale,0,1)
        A_color_scale = F1.add_axes([0.1,0.1,0.8,0.05*len(color_cycle)])
        A_color_scale.imshow(color_scale, aspect = 'auto', interpolation='None')
        A_color_scale.spines.top.set_visible(False)
        A_color_scale.spines.left.set_visible(False)
        A_color_scale.spines.right.set_visible(False)
        A_color_scale.spines.bottom.set_visible(False)
        A_color_scale.set_xticks([])
        A_color_scale.set_yticks([])
        A_color_scale.text(0, len(color_scale)+0.3, f'{vmin}', horizontalalignment='left')         
        A_color_scale.text(256, len(color_scale)+0.3, f'{vmax}', horizontalalignment='right') 
        A_color_scale.set_xlabel("|r|")
        A_color_scale.text(256, -7, f'Scale bar: {bar_scale} microns', horizontalalignment='left') 
        
        
        
        
        
        save_fig(F, fname, dpi = 300)
        save_fig(F1, fname + 'scale', dpi = 300)
        #pdb.set_trace()

    def show_u_traces(self, selected_u_cells='All', selected_sessions='All'):

        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()

        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)

        F = plt.figure()
        cnmf_mins = {}
        cnmf_maxs = {}
        dff_mins = {}
        dff_maxs = {}
        for cc, session in enumerate(selected_sessions):
            for c, u_cell in enumerate(selected_u_cells):
                if cc == 0:
                    cnmf_mins[u_cell] = []
                    cnmf_maxs[u_cell] = []
                    dff_mins[u_cell] = []
                    dff_maxs[u_cell] = []

                cnmf_mins[u_cell].append(
                    np.amin(self.trace_struct[(u_cell, session)]['union cnmf']))
                cnmf_maxs[u_cell].append(
                    np.amax(self.trace_struct[(u_cell, session)]['union cnmf']))

                dff_mins[u_cell].append(
                    np.amin(self.trace_struct[(u_cell, session)]['union dff']))
                dff_maxs[u_cell].append(
                    np.amax(self.trace_struct[(u_cell, session)]['union dff']))

                if not self.trace_struct[(u_cell, session)]['session trace'] is None:
                    cnmf_mins[u_cell].append(
                        np.amin(self.trace_struct[(u_cell, session)]['session trace']))
                    cnmf_maxs[u_cell].append(
                        np.amax(self.trace_struct[(u_cell, session)]['session trace']))

        for cc, session in enumerate(selected_sessions):

            for c, u_cell in enumerate(selected_u_cells):
                IX = to_int(self.assignments[u_cell, session])
                plot_position = prc(len(selected_sessions), cc, c)
                ax = plt.subplot(len(selected_u_cells), len(
                    selected_sessions), plot_position)
                ax.plot(
                    self.trace_struct[(u_cell, session)]['union cnmf'], 'r')
                ax2 = ax.twinx()
            #     ax2.plot(self.trace_struct[(u_cell, session)]['union dff'],'b')
                if not self.trace_struct[(u_cell, session)]['session trace'] is None:
                    ax.plot(
                        self.trace_struct[(u_cell, session)]['session trace'], 'k')

                ax.xaxis.set_visible(False)
                ax2.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax2.yaxis.set_visible(False)
                ax.set_frame_on(False)
                ax2.set_frame_on(False)

                ax.set_ylim([np.amin(cnmf_mins[u_cell]),
                            np.amax(cnmf_maxs[u_cell])])
                ax2.set_ylim([np.amin(dff_mins[u_cell]),
                             np.amax(dff_maxs[u_cell])])

    def regularize_rasters(self,  Tseries=None, selected_sessions='All', prepend_frames=10, append_frames=10, plot=True, min_duration=40, end_fix_duration=None, fail_if_missing=False, reps=None, vmin=None, vmax = None, cmap = 'inferno', **kwargs):

        F = None
        A1 = None
        A2 = None
        if plot:
            F = plt.figure()

        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()

        n_ses = len(selected_sessions)

        width = 1/n_ses
        left = 0

        raster_outputs = []
        stim_outputs = []
        missings = []
        for s_n, session in enumerate(selected_sessions):
            if plot:
                
                A1 = F.add_axes([left, 0, width, 0.8])
                A2 = F.add_axes([left, 0.8, width, 0.2])
            left = left + width
            r, s, m = self.sessions[session].regularize_raster(Tseries=Tseries, prepend_frames=prepend_frames, append_frames=append_frames, plot=plot,
                                                               min_duration=min_duration, end_fix_duration=end_fix_duration, fail_if_missing=fail_if_missing, reps=reps, F=F, A1=A1, A2=A2, vmin = vmin, vmax = vmax, cmap=cmap)
            
            
            if s_n ==0:
                if vmin is None:
                    vmin = np.amin(r)
                if vmax is None:
                    vmax = np.amax(r)
                    
            raster_outputs.append(r)
            stim_outputs.append(s)
            missings.append(m)

        return(raster_outputs, stim_outputs, missings)

    # in progress - goal is to clip traces to aligned stimuli and plot
    def plot_reg_traces(self, Tseries=None, F=None, selected_u_cells='All', selected_sessions='All', prepend_frames=80, append_frames=80, min_duration=40, reps=[2], overlay_sessions=False):
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
        rasters, stims, missings = self.regularize_rasters(Tseries=Tseries, plot=False, selected_sessions=selected_sessions,
                                                           prepend_frames=prepend_frames, append_frames=append_frames, min_duration=min_duration, end_fix_duration=80, reps=reps)
        #pdb.set_trace()
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)

        if F is None:
            F = plt.figure()
            stim_axes = {}
            trace_axes = {}

        if len(selected_u_cells) == 1:
            stim_span = 1
        elif len(selected_u_cells) < 11:
            stim_span = 2
        else:
            stim_span = round(len(selected_u_cells)/6)

        n_rows = len(selected_u_cells)+stim_span
        if overlay_sessions:
            n_cols=1
        else:
            n_cols = len(selected_sessions)

        trace_mins = {}
        trace_maxs = {}

        for sc, sess in enumerate(selected_sessions):
            if overlay_sessions:
                if sc == 0:
                    stim_axes[sess] = F.add_subplot(int(n_rows/stim_span), n_cols, sc+1)
                else:
                    stim_axes[sess] = stim_axes[selected_sessions[0]]
            else:
                stim_axes[sess] = F.add_subplot(int(n_rows/stim_span), n_cols, sc+1)

            for c, u_cell in enumerate(selected_u_cells):
                if sc == 0:
                    trace_mins[u_cell] = []
                    trace_maxs[u_cell] = []

                assignment = self.assignments[u_cell, sess]
                if np.isnan(assignment):
                    trace = [0]
                else:
                   
                    trace = rasters[sess][int(assignment)]

                trace_mins[u_cell].append(np.amin(trace))
                trace_maxs[u_cell].append(np.amax(trace))

                pos = (n_cols*(c+stim_span)) + sc + 1
                if overlay_sessions:
                    if sc == 0:
                        trace_axes[(sess, u_cell)] = F.add_subplot(n_rows, n_cols, pos)
                    else:
                        trace_axes[(sess, u_cell)] =  trace_axes[(selected_sessions[0], u_cell)] 
                else:
                    trace_axes[(sess, u_cell)] = F.add_subplot(n_rows, n_cols, pos)

        for sc, session in enumerate(selected_sessions):
            # stim_num = 0
            #thermStim = self.sessions[session].therm_stim
            thermStim = stims[session]
            mechStim = self.sessions[session].mech_stim
            # if not thermStim is None:
            #     stim_num = stim_num + 1
            # if not mechStim is None:
            #     stim_num = stim_num + 1
            if hasattr(self, 'sess_color_series'):
                color = self.sess_color_series[session]
            else:
                color = 'k'
            if overlay_sessions:
                alpha = 1/len(selected_sessions)
            else:
                alpha = 1
            stim_axes[session].plot(thermStim, color=color, alpha=alpha)
            if sc == 0 or overlay_sessions:
                box_off(stim_axes[session], left_only = True)
                stim_axes[session].set_ylabel('Temp (°C)')
            else:
                box_off(stim_axes[session], All = True)
                
            stim_axes[session].set_ylim([0,60])
            stim_axes[session].set_yticks([10,30,50])
            #stim_axes[session].set_
            for c, u_cell in enumerate(selected_u_cells):

                assignment = self.assignments[u_cell, sess]
                if np.isnan(assignment):
                    trace = [0]
                else:
                    print(f'{session=} {u_cell=}')
                    trace = rasters[session][int(assignment)]

               # parent_data = self.sessions[session].cells[0].parent_data
               # v_cell = cell(trace=trace, ROI=[], classification=None, parent_data=parent_data,
                #              parent_session=self.sessions[session], thermStim=thermStim, mechStim=mechStim)
                trace_span = np.amax(
                    trace_maxs[u_cell]) - np.amin(trace_mins[u_cell])
                trace_y_lim = (np.amin(
                    trace_mins[u_cell])-trace_span/20, np.amax(trace_maxs[u_cell])+trace_span/20)
                #print(f'{trace_y_lim=} for cell {u_cell}')
                if sc == 0 and c == 0:
                    xbar = True
                    ybar = True
                    show_stim_y = True
                elif sc == 0:
                    xbar = False
                    ybar = True
                    show_stim_y = True
                else:
                    xbar = False
                    ybar = False
                    show_stim_y = False
                #print(f'{xbar=} and {ybar=} for cell {u_cell} in session {session}, {c=} {sc=}')
                
                trace_axes[(session, u_cell)].plot(trace, color=color, alpha = alpha)
                box_off(trace_axes[session, u_cell], All=True)
               # v_cell.show(F=F, A1=trace_axes[(session, u_cell)], A2=stim_axes[session], trace_color=color, show_y=False,
               #             show_stim_y=show_stim_y,  show_time=False, show_trace_y_label=False, trace_y_lim=trace_y_lim, norm=False, xbar=xbar, ybar=ybar, disp_transients=False)
        pass

    def thermo_polar_plot__across_sessions(self, selected_u_cells='All', selected_sessions='All', normalize=True, missingfifty=False, session_marker_list=None):
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()

        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])

        for uc, u_cell in enumerate(selected_u_cells):
            observations = []
            datas = []
            for sc, session in enumerate(selected_sessions):
                assignment = self.assignments[u_cell, session]
                cell = self.sessions[session].cells[assignment]
                observations.append(cell)
                datas.append(cell.parent_data)
            plot_cells_connected(observations, datas, F=F, A=A, normalize=normalize,
                                 missingfifty=missingfifty, session_marker_list=session_marker_list)
    def plot_rasters(self, selected_u_cells='All', selected_sessions='All', stim_span='Auto', F=None, trace_axes=None, Tstim_color='session', color_mode=None, save=True, vmin=2, vmax = 10, cmap = 'bwr'):
        plot_area = 0.9
        figure_area = 0.8
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
        if stim_span == 'Auto':
            if len(selected_u_cells) == 1:
                stim_span = 0.5
            elif len(selected_u_cells) < 11:
                stim_span = 0.25
            else:
                stim_span = round(len(selected_u_cells)/6)
        if F is None:
            F = plt.figure()
            stim_axes = {}
            raster_axes = {}
        F1 = plt.figure()
        A1 = F1.add_subplot(2,1,1)
        A2 = F1.add_subplot(2,1,1)
        
        n_rows = len(selected_u_cells)+stim_span
        n_cols = len(selected_sessions)

        trace_mins = {}
        trace_maxs = {}
        
        total_time = 0
        sess_t_fractions = []
        for sc, sess in enumerate(selected_sessions):
            total_time = total_time + self.sessions[sess].raster.shape[1]
            
        for sc, sess in enumerate(selected_sessions):
            fraction = self.sessions[sess].raster.shape[1]/total_time
            sess_t_fractions.append(fraction)
        
        for sc, sess in enumerate(selected_sessions):
            
            X = ((np.sum(sess_t_fractions[0:sc]))*figure_area)  + (1-figure_area)/2
            Y = ((1-stim_span)*figure_area) + (1-figure_area)/2
            W = sess_t_fractions[sc] * plot_area * figure_area
            H = stim_span * plot_area * figure_area
            stim_axes[sess] = F.add_axes([X,Y,W,H])
            Y =  (1-figure_area)/2
            H = (1-stim_span)*plot_area*figure_area
            raster_axes[sess]  = F.add_axes([X,Y,W,H])
            #stim_axes[sess] = F.add_subplot(int(n_rows/stim_span), n_cols, sc+1)
            raster = np.array([])
            for c, u_cell in enumerate(selected_u_cells):
                if sc == 0:
                    trace_mins[u_cell] = []
                    trace_maxs[u_cell] = []

                assignment = self.assignments[u_cell, sess]
                if np.isnan(assignment):
                    trace = np.zeros([self.sessions[sess].raster.shape[1]])
                else:
                    trace = self.sessions[sess].cells[int(assignment)].trace
                if c == 0:
                    raster = trace
                else:
                    raster = np.vstack([raster,trace])
                #trace_mins[u_cell].append(np.amin(trace))
                #trace_maxs[u_cell].append(np.amax(trace))

                #pos = (n_cols*(c+stim_span)) + sc + 1
                #trace_axes[(sess, u_cell)] = F.add_subplot(n_rows, n_cols, pos)
            raster_axes[sess].imshow(raster, vmin = vmin, vmax = vmax, cmap = cmap, aspect='auto', interpolation='none')
            if sc == 0:
                box_off(raster_axes[sess], left_only = True)
            else:
                box_off(raster_axes[sess], All = True)
        Mstim_mins = []
        Mstim_maxs = []
        Tstim_mins = []
        Tstim_maxs = []
        for sc, session in enumerate(selected_sessions):
            thermStim = self.sessions[session].therm_stim
            if not thermStim is None:
                Tstim_mins.append(np.amin(thermStim))
                Tstim_maxs.append(np.amax(thermStim))
                
            mechStim = self.sessions[session].mech_stim
            if not mechStim is None:
                Mstim_mins.append(np.amin(mechStim))
                Mstim_maxs.append(np.amax(mechStim))
            
            if hasattr(self.sessions[session], 'Mstim'):
                Mstim = self.sessions[session].Mstim
                if not Mstim is None:
                    Mstim_mins.append(np.amin(Mstim))
                    Mstim_maxs.append(np.amax(Mstim))
            else:
                Mstim = None
        if len(Mstim_mins):
            Mbounds = [np.amin(Mstim_mins), np.amax(Mstim_maxs)]
        else:
            Mbounds = None
        if len(Tstim_mins):
            Tbounds = [np.amin(Tstim_mins), np.amax(Tstim_maxs)]
        else:
            Mbounds = None
        
        for sc, session in enumerate(selected_sessions):
            # stim_num = 0
            thermStim = self.sessions[session].therm_stim
            mechStim = self.sessions[session].mech_stim

            if hasattr(self.sessions[session], 'Mstim'):
                Mstim = self.sessions[session].Mstim

            else:
                Mstim = None
                
            
            if hasattr(self.sessions[session], 'misc_stim'):
                pass
            # if not thermStim is None:
            #     stim_num = stim_num + 1
            # if not mechStim is None:
            #     stim_num = stim_num + 1
            for c, u_cell in enumerate(selected_u_cells):

                assignment = self.assignments[u_cell, session]
                if np.isnan(assignment):
                    trace = np.zeros([self.sessions[session].raster.shape[1]])
                else:
                    trace = self.sessions[session].cells[int(assignment)].trace

                parent_data = self.sessions[session].cells[0].parent_data
                v_cell = cell(trace=trace, ROI=[], classification=None, parent_data=parent_data,
                              parent_session=self.sessions[session], thermStim=thermStim, mechStim=mechStim)
               
                #print(f'{trace_y_lim=} for cell {u_cell}')
                if sc == 0 and c == 0:
                    xbar = True
                    ybar = True
                    show_stim_y = True
                elif sc == 0:
                    xbar = False
                    ybar = True
                    show_stim_y = True
                else:
                    xbar = False
                    ybar = False
                    show_stim_y = False
                #print(f'{xbar=} and {ybar=} for cell {u_cell} in session {session}, {c=} {sc=}')
                alpha = 1
                if Tstim_color == 'session':
                    Tcolor = self.sess_color_series[session]
                else:
                    Tcolor = Tstim_color
                if color_mode is None:
                    trace_color = 'k'
                elif color_mode == 'therm':
                    trace_color = 'therm'
                elif color_mode == 'session':
                    trace_color = self.sess_color_series[session]
                elif color_mode[0] == 'color_dict_by_session':
                    trace_color = color_mode[1][session]
                elif color_mode[0] == 'color_dict_by_cell':
                    trace_color = color_mode[1][u_cell]
                elif color_mode[0] == 'color_dict_by_cell_and_session':
                    trace_color = color_mode[1][u_cell, session]
                elif color_mode[0] == 'thermo_tuning_metric':
                    trace_color = thermo_tuning_metric(
                        v_cell, self.sessions[session].Tstim, plot=False)['norm']
                    alpha = thermo_tuning_metric(
                        v_cell, self.sessions[session].Tstim, plot=False)['amp']
                else:
                    print(f'Color mode {color_mode} not recognized')
                    trace_color = 'k'
                v_cell.show(F=F1, A1=A1, A2=stim_axes[session], trace_color=trace_color, show_y=False, show_stim_y=show_stim_y,  show_time=False,
                            show_trace_y_label=False, norm=False, xbar=xbar, ybar=ybar, trace_alpha=alpha, Mbounds = Mbounds,Tbounds=Tbounds, Tstim_color=Tcolor)
        
        
        #for sc, sess in enumerate(selected_sessions):
            
        plt.close(F1)
        fname = os.path.split(self.pickle_path)[-1].split('.')[0] + '_rasters'
        save_fig(F, fname)
        
        
        
        
        
        
        
        
        
        
        
                
    def plot_traces(self, selected_u_cells='All', selected_sessions='All', stim_span='Auto', F=None, stim_axes=None, color_mode='therm', trace_axes=None, save=True, Tstim_color='m', disp_transients=False):
        plot_area = 0.9
        figure_area = 0.8
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()

        if stim_span == 'Auto':
            if len(selected_u_cells) == 1:
                stim_span = 1
            elif len(selected_u_cells) < 11:
                stim_span = 2
            else:
                stim_span = round(len(selected_u_cells)/6)

        if F is None:
            F = plt.figure()
            stim_axes = {}
            trace_axes = {}

        n_rows = len(selected_u_cells)+stim_span
        n_cols = len(selected_sessions)

        trace_mins = {}
        trace_maxs = {}
        
        total_time = 0
        sess_t_fractions = []
        for sc, sess in enumerate(selected_sessions):
            total_time = total_time + self.sessions[sess].raster.shape[1]
            
        for sc, sess in enumerate(selected_sessions):
            fraction = self.sessions[sess].raster.shape[1]/total_time
            sess_t_fractions.append(fraction)
        
        for sc, sess in enumerate(selected_sessions):
            
            X = ((np.sum(sess_t_fractions[0:sc]))*figure_area)  + (1-figure_area)/2
            Y = ((1 -(stim_span*(1/n_rows)))*figure_area) + (1-figure_area)/2
            W = sess_t_fractions[sc] * plot_area * figure_area
            H = stim_span*(1/n_rows) * plot_area * figure_area
            stim_axes[sess] = F.add_axes([X,Y,W,H])
            
            #stim_axes[sess] = F.add_subplot(int(n_rows/stim_span), n_cols, sc+1)

            for c, u_cell in enumerate(selected_u_cells):
                if sc == 0:
                    trace_mins[u_cell] = []
                    trace_maxs[u_cell] = []

                assignment = self.assignments[u_cell, sess]
                if np.isnan(assignment):
                    trace = np.zeros([self.sessions[sess].raster.shape[1]])
                else:
                    trace = self.sessions[sess].cells[int(assignment)].trace

                trace_mins[u_cell].append(np.amin(trace))
                trace_maxs[u_cell].append(np.amax(trace))

                #pos = (n_cols*(c+stim_span)) + sc + 1
                #trace_axes[(sess, u_cell)] = F.add_subplot(n_rows, n_cols, pos)
                
                X = (np.sum(sess_t_fractions[0:sc])*figure_area ) + (1-figure_area)/2
                Y = ((1 -((stim_span+c+1)/n_rows))*figure_area) + (1-figure_area)/2
                W = sess_t_fractions[sc] * plot_area * figure_area
                H = (1/n_rows) * plot_area * figure_area
                trace_axes[(sess, u_cell)]  = F.add_axes([X,Y,W,H])
                
        Mstim_mins = []
        Mstim_maxs = []
        Tstim_mins = []
        Tstim_maxs = []
        for sc, session in enumerate(selected_sessions):
            thermStim = self.sessions[session].therm_stim
            if not thermStim is None:
                Tstim_mins.append(np.amin(thermStim))
                Tstim_maxs.append(np.amax(thermStim))
                
            mechStim = self.sessions[session].mech_stim
            if not mechStim is None:
                Mstim_mins.append(np.amin(mechStim))
                Mstim_maxs.append(np.amax(mechStim))
            
            if hasattr(self.sessions[session], 'Mstim'):
                Mstim = self.sessions[session].Mstim
                if not Mstim is None:
                    Mstim_mins.append(np.amin(Mstim))
                    Mstim_maxs.append(np.amax(Mstim))
            else:
                Mstim = None
        if len(Mstim_mins):
            Mbounds = [np.amin(Mstim_mins), np.amax(Mstim_maxs)]
        else:
            Mbounds = None
        if len(Tstim_mins):
            Tbounds = [np.amin(Tstim_mins), np.amax(Tstim_maxs)]
        else:
            Mbounds = None
        
        for sc, session in enumerate(selected_sessions):
            # stim_num = 0
            thermStim = self.sessions[session].therm_stim
            mechStim = self.sessions[session].mech_stim

            if hasattr(self.sessions[session], 'Mstim'):
                Mstim = self.sessions[session].Mstim

            else:
                Mstim = None
                
            
            if hasattr(self.sessions[session], 'misc_stim'):
                pass
            # if not thermStim is None:
            #     stim_num = stim_num + 1
            # if not mechStim is None:
            #     stim_num = stim_num + 1
            for c, u_cell in enumerate(selected_u_cells):

                assignment = self.assignments[u_cell, session]
                if np.isnan(assignment):
                    trace = np.zeros([self.sessions[session].raster.shape[1]])
                else:
                    trace = self.sessions[session].cells[int(assignment)].trace

                parent_data = self.sessions[session].cells[0].parent_data
                v_cell = cell(trace=trace, ROI=[], classification=None, parent_data=parent_data,
                              parent_session=self.sessions[session], thermStim=thermStim, mechStim=mechStim)
                trace_span = np.amax(
                    trace_maxs[u_cell]) - np.amin(trace_mins[u_cell])
                trace_y_lim = (np.amin(
                    trace_mins[u_cell])-trace_span/20, np.amax(trace_maxs[u_cell])+trace_span/20)
                #print(f'{trace_y_lim=} for cell {u_cell}')
                if sc == 0 and c == 0:
                    xbar = True
                    ybar = True
                    show_stim_y = True
                elif sc == 0:
                    xbar = False
                    ybar = True
                    show_stim_y = True
                else:
                    xbar = False
                    ybar = False
                    show_stim_y = False
                #print(f'{xbar=} and {ybar=} for cell {u_cell} in session {session}, {c=} {sc=}')
                alpha = 1
                if Tstim_color == 'session':
                    Tsim_color = self.sess_color_series[session]
                if color_mode is None:
                    trace_color = 'k'
                elif color_mode == 'therm':
                    trace_color = 'therm'
                elif color_mode == 'session':
                    trace_color = self.sess_color_series[session]
                elif color_mode[0] == 'color_dict_by_session':
                    trace_color = color_mode[1][session]
                elif color_mode[0] == 'color_dict_by_cell':
                    trace_color = color_mode[1][u_cell]
                elif color_mode[0] == 'color_dict_by_cell_and_session':
                    trace_color = color_mode[1][u_cell, session]
                elif color_mode[0] == 'thermo_tuning_metric':
                    trace_color = thermo_tuning_metric(
                        v_cell, self.sessions[session].Tstim, plot=False)['norm']
                    alpha = thermo_tuning_metric(
                        v_cell, self.sessions[session].Tstim, plot=False)['amp']
                else:
                    print(f'Color mode {color_mode} not recognized')
                    trace_color = 'k'
                v_cell.show(F=F, A1=trace_axes[(session, u_cell)], A2=stim_axes[session], trace_color=trace_color, show_y=False, show_stim_y=show_stim_y,  show_time=False,
                            show_trace_y_label=False, trace_y_lim=trace_y_lim, norm=False, xbar=xbar, ybar=ybar, disp_transients=disp_transients, trace_alpha=alpha, Mbounds = Mbounds,Tbounds=Tbounds)
        
        
        #for sc, sess in enumerate(selected_sessions):
            
        
        fname = os.path.split(self.pickle_path)[-1].split('.')[0] + '_traces'
        save_fig(F, fname)

    # under constructioni
    # def show_ROIs(self, selected_u_cells = 'All', selected_sessions = 'All', ax = None, scale_bar = False, cmap='jet', show_field = True):

        # if ax is None:
        #     F = plt.figure()
        #     ax = F.add_axe[(0,0,1,1)]
        #     ax.cla()

        # neuron_color_series = []
        # cmap = cm.get_cmap(cmap)

        # for i in range(len(u_cells)):
        #     neuron_color_series.append(cmap(i/len(u_cells)))

        # for c, sess in enumerate(selected_sessions):

        # medField = np.median(self.field_images[selected_sessions], axis=0)
        # field_for_disp = None
        # if not self.params['ROI display']['Show background']:

        #     w = (medField*0) + 1
        #     field_for_disp = np.stack([w,w,w], axis=2)

        # for c, session in enumerate(self.data.sessions):
        #     if len(self.selected_data[c]) == 0:
        #         h = session.ROIs.shape[0]
        #         w = session.ROIs.shape[1]
        #         RGBA = np.zeros([h,w,4])
        #         flatmap = np.zeros([h,w])
        #         self.ROI_overlays[c].setImage(RGBA)
        #         self.ROI_overlays[c].linkROImap(flatmap, self)
        #         continue

        #     OfloatMask = session.ROIs[:,:,self.selected_data[c]]
        #     floatMask = OfloatMask.copy()

        #     for ROI in range(floatMask.shape[-1]):
        #         floatMask[...,ROI] = floatMask[...,ROI]/np.amax(floatMask[...,ROI])

        #     ## Display contours overlaid

        #     numbers = {}
        #     for IX, i in enumerate(self.selected_data[c]):
        #         numbers[IX] = i

        #     color_mode = self.params['ROI display']['Contour labeling (session or neuron']
        #     caiman_ROIs = DYroi2CaimanROI(OfloatMask)

        #     if color_mode == 'session':
        #         color = self.sess_color_series[c%8]
        #         plot_contours(caiman_ROIs, medField, ax=ax, numbers = numbers, color=color, show_field = show_field, field_for_disp=field_for_disp)
        #     elif color_mode == 'neuron':
        #         neuron_colors = []
        #         #pdb.set_trace()
        #         for u, u_cell in enumerate(u_cells):
        #             if not np.isnan(self.data.assignments[u_cell, c]):
        #                 neuron_colors.append(neuron_color_series[u])
        #         plot_contours(caiman_ROIs, medField, ax=ax, numbers = numbers, color_series = neuron_colors, show_field = show_field, field_for_disp=field_for_disp)
        #     else:
        #         plot_contours(caiman_ROIs, medField, ax=ax, numbers = numbers, color='w', show_field = show_field)

        #     ax.set_xlim([0, medField.shape[1]])
        #     ax.set_ylim([0, medField.shape[0]])
        #     if not ax.yaxis_inverted:
        #         ax.invert_yaxis()

        #     if scale_bar:
        #         microns_per_pixel = 2.016
        #         bar_length = 100 # microns
        #         height = medField.shape[0]
        #         width = medField.shape[1]
        #         ax.plot([width*0.05, width*0.05 + (bar_length/microns_per_pixel)], [height*0.95 ,height*0.95 ], color = 'w', linewidth = 3)
        #     self.field_displays['union'].draw()

        #     ## Display ROIs in individual sessionss
        #     flatFloat = np.amax(floatMask, axis=2)
        #     flatFloat = flatFloat * 255
        #     flatFloat = flatFloat.astype(np.uint8)

        #     labelSelected = floatMask*0
        #     binaryMask = floatMask>0

        #     for label in range(0, binaryMask.shape[-1], 1):
        #        labelSelected[:,:,label] = binaryMask[:,:,label]*label+1

        #     flatLabel = np.max(labelSelected, axis=2)
        #     truncatedLabel = (flatLabel % 255).astype(np.uint8)

        #     label_range = np.linspace(0,1,256)
        #     lut = np.uint8(plt.cm.prism(label_range)[:,2::-1]*256).reshape(256, 1, 3)
        #     RGB = cv2.LUT(cv2.merge((truncatedLabel, truncatedLabel, truncatedLabel)), lut)

        #     Alpha = np.expand_dims(flatFloat, 2)
        #     RGBA = np.concatenate((RGB,Alpha), axis = 2)

        #     bgMask = np.max(floatMask, axis=2)
        #     bgMaskBool = bgMask.astype(bool)
        #     flatLabel[~bgMaskBool]= 0

        #     self.ROI_overlays[c].setImage(RGBA)
        #     self.ROI_overlays[c].linkROImap(flatLabel, self)
    def show_mech_rasters(self, selected_sessions = 'All', selected_u_cells = 'All', cmap='jet',plot=True, time_limit=50, force_limits = None, key_sess = 0, vmin=2, vmax=20, log_scale = True):
        mech_lib.get_m_response_multi_session(self, selected_sessions = selected_sessions, selected_u_cells = selected_u_cells, cmap=cmap,plot=plot, time_limit=time_limit, force_limits = force_limits, key_sess = key_sess, vmin = vmin, vmax = vmax, log_scale=log_scale)
    
    def compare_mech_ranges(self, selected_sessions = 'All', plot_non_responsive = False, key_sess=0, selected_u_cells = 'All', plot_cells = False, unit = 'mN', limit=500, color=None, threshold = 3.5, sort_responses=True, AS=None, AC=None, CF=None, calc_all=True, return_cells=True, appened=False):
     
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()
       
        F = None
        A = None
        

        offset_inc = 1/(len(selected_sessions)*2)
        #pdb.set_trace()
    
        key_response = self.sessions[key_sess].mech_range(cells = selected_u_cells, plot_cells = plot_cells, unit = unit, limit=limit, plot_summary =False, color=None, threshold = threshold, sort_responses=False, AS=None, AC=None, CF=None, calc_all=False, return_cells=False)
        IX = sort_mech_ranges(key_response)['IX']
        r={}
        for s, session in enumerate(selected_sessions):
            r[session] = self.sessions[session].mech_range(plot_summary = False, plot_cells = False, return_cells = False, sort_responses=False)
            #response = self.sessions[session].mech_range(cells = selected_u_cells, plot_cells = plot_cells, unit = unit, limit=limit, plot_summary =False, color=None, threshold = threshold, sort_responses=False, AS=None, AC=None, CF=None, calc_all=False, return_cells=False)
            #sorted_response = [response[x] for x in IX]
            #F=None
            #A = None
            #plot_mech_ranges(response, plot_non_responsive=plot_non_responsive, color = self.sess_color_series[session], offset=s/offset_inc, dexp=1.5, dm=2, F=F, A=A, only_responders = False, SNR_threshold = threshold)
        for rc, response in enumerate(r):
            offset = rc*offset_inc
            print(f'{offset=}')
            sorted_response = [r[response][x] for x in IX]
            F,A = plot_mech_ranges(sorted_response, plot_non_responsive=True, color = self.sess_color_series[selected_sessions[rc]], offset=offset, dexp=1.5, dm=2, A=A, only_responders = False, SNR_threshold = threshold)
            #F,A = plot_mech_ranges(r[response], plot_non_responsive=True, color = self.sess_color_series[selected_sessions[rc]], offset=offset, dexp=1.5, dm=2, A=A, only_responders = False, SNR_threshold = threshold)
    
        
        return(r, IX)
        
            
    
    def line_tuning_across_sessions(self, selected_u_cells='All', selected_sessions='All', scale_by_time=True, plot=True):

        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()

        if plot:
            F, A = blank_fig()

        exp_times = []

        for u_cell in selected_u_cells:

            times = []
            thetas = []
            rs = []
            colors = []

            for c, session in enumerate(selected_sessions):

                assignment = self.assignments[u_cell, session]
                CELL = self.sessions[session].cells[assignment]

                exp_start = self.sessions[session].experiment_start
                if c == 0:
                    Tzero = copy.copy(exp_start)
                exp_times.append(sec2day(exp_start-Tzero))

                if scale_by_time:
                    times.append(exp_times[-1])
                else:
                    times.append(c)
                # pdb.set_trace()
                stats = cell_analyze_temp(CELL, data=CELL.parent_data)
                thetas.append(stats['angle'])

                r = stats['r']
                if r > 1:
                    r = 1
                if r < 0:
                    r = 0
                rs.append(r)

                color = stats['color']/255
                color[np.where(color < 0)] = 0
                colors.append(color)

            for count, (theta, time, color, r) in enumerate(zip(thetas, times, colors, rs)):

                print(
                    f'Plotting color {color} for u_cell {u_cell} in session {session}')

                A.scatter(time, theta, color=color, alpha=r, s=r*20)
                if count < len(thetas)-1:
                    A.plot([time, times[count+1]], [theta,
                           thetas[count+1]], color=color, alpha=rs[count+1])

        return

    def scatter_tuning_between_sessions(self, selected_u_cells='All', selected_sessions='All', plot=True, F=None, A=None, marker='o', markersize=30):
        if selected_u_cells == 'All':
            selected_u_cells = []
            for c, u in enumerate(self.assignments):
                selected_u_cells.append(c)
        if selected_sessions == 'All':
            selected_sessions = self.all_sessions()

        if not (len(selected_sessions) == 2):
            print('Can only scatter 2 sessionss')
            return

        if F is None:
            F, A = blank_fig()

        exp_times = []
        for u_cell in selected_u_cells:
            theta = {}
            colors = {}
            rs = {}

            for c, session in enumerate(selected_sessions):

                exp_start = self.sessions[session].experiment_start
                if c == 0:
                    Tzero = copy.copy(exp_start)
                exp_times.append(sec2day(exp_start-Tzero))

                assignment = self.assignments[u_cell, session]
                CELL = self.sessions[session].cells[assignment]

                stats = cell_analyze_temp(CELL, data=CELL.parent_data)
                theta[session] = stats['angle']

                r = stats['r']
                if r > 1:
                    r = 1
                if r < 0:
                    r = 0
                rs[session] = r

                color = stats['color']/255
                color[np.where(color < 0)] = 0
                colors[session] = color

            A.plot([-np.pi, -np.pi], [np.pi, np.pi])
            A.scatter(theta[0], theta[1], facecolor=colors[0],
                      s=rs[1]*markersize, alpha=rs[0], marker=marker)
            A.set_xlim([-np.pi, 1 * np.pi])
            A.set_ylim([-np.pi, 1 * np.pi])
            A.set_aspect('equal')

    def class_counts_across_sessions(self, selected_sessions='All', f=None):
        pass

    def pickle(self, filepath=None):
        file = open(filepath, 'wb')
        pickle.dump(self, file)
        file.close()
        self.pickle_path = filepath
        return(filepath)


def save_fig(Fig, fig_name, ext='.pdf', dpi = 300):
    Fig.savefig(
        '/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' + fig_name + ext, dpi=dpi)


def to_int(text):
    text = str(text)
    if text.isnumeric():
        return(int(text))
    elif text.split('.')[0].isnumeric():
        return(int(text.split('.')[0]))
    else:
        return(np.nan)


def alignROIsCAIMAN(ROIs, templates, *args, **kwargs):
    # Arguments
    # ROIs - list of 3D (X, Y, ROI#) numpy arrays of ROI masks
    # templates - list of mean images corresponding to ROI masks
    dims = (ROIs[0].shape[0], ROIs[0].shape[1])
    H = dims[0]
    W = dims[1]
    A = []  # List of ROIs in Caiman format (2d matrix)
    plt.figure('All ROI overlay')
    for ROIstack, template in zip(ROIs, templates):
        Ac = DYroi2CaimanROI(ROIstack)
        visualization.plot_contours(Ac, template)
        # plt.figure()
        # plt.imshow(Ac)
        A.append(Ac)
    print(f'alignROIsCAIMAN: {kwargs.keys()=}')
    spatial_union, assignments, matchings = register_multisession(
        A=A, dims=dims, templates=templates, *args, **kwargs)
    n = spatial_union.shape[1]
    allROIs = CaimanROI2dyROI(spatial_union, H, W, n)
    # plt.figure()
    # plt.imshow(assignments)
    plt.figure('Spatial union')
    visualization.plot_contours(spatial_union, templates[0])
    # print(matchings)
    return(allROIs, spatial_union, assignments, matchings)


def DYroi2CaimanROI(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    n = mask.shape[2]
    A = np.zeros([h*w, n], dtype=float)
    for i in range(mask.shape[-1]):
        ROI = mask[..., i]
        ROI = ROI.astype(float)
        A[:, i] = ROI.flatten('F')
    return(A)


def CaimanROI2dyROI(A, H, W, n):
    if n is None:
        n = A.shape[1]
    out = np.reshape(A, [H, W, n], 'F')
    return(out)


def prc(n_cols, x_pos, y_pos):
    """
    Returns subplot  index # from number of columns and desired x, y position of subplot
    """
    p = (y_pos*n_cols) + x_pos + 1
    return(p)


# def plot_contours(obj, ROIstacks=None, template = None, union = False):

#     if ROIstacks == None:
#         ROIstacks = []
#         ROIstacks.append(obj.DB['Animals'][obj.curAnimal][obj.curFOV]['R'][obj.dataFocus]['floatMask'][...])
#     for ROIstack in ROIstacks:
#         plt.figure()
#         A = DYroi2CaimanROI(ROIstack)
#         if template == 'zeros':
#             template = np.zeros([ROIstack.shape[0],ROIstack.shape[1]])
#         elif template == None:
#             template = obj.DB['Animals'][obj.curAnimal][obj.curFOV][obj.dataFocus][0,...]
#         visualization.plot_contours(A, template, cmap = 'gray', colors = 'c')

def cell_analyze_temp(cell, data=None, Tstim=None, normalize=True, F=None, A=None, plot_cell=False, draw_axes=True, marker='o', alpha=0.5, missingFifty=False):
    output = {}

    theta_cold = 0+np.pi/2  # 90 deg, 1.57 radians
    theta_warm = np.pi*2/3 + np.pi/2  # 210 deg, 3.665 radians
    theta_hot = np.pi*4/3 + np.pi/2  # 330 deg, 5.5795 radians (-30)

    output['theta_cold'] = theta_cold
    output['theta_warm'] = theta_warm
    output['theta_hot'] = theta_hot

    output['angle'] = np.nan
    output['r'] = np.nan
    output['XY'] = [0, 0]
    output['color'] = np.array([0, 0, 0])
    output['cwh'] = [0, 0, 0]
    if data is None:
        data = {}
        data['source'] = 'unknown'
    if cell is None:
        print('not a valid cell')
        return(output)

    if Tstim == None:
        Tstim = data['Tstim']
    cold_rs = []
    warm_rs = []
    hot_rs = []

    if missingFifty:
        heat_min = 12
    else:
        heat_min = 14
    for s, stim in enumerate(Tstim):
        response = cell.response(Tstim[stim], normalize=normalize)
        if normalize:
            resp_amp = response['amplitude']
        else:
            resp_amp = response['normamplitude']
        if Tstim[stim].intensity < -24:
            cold_rs.append(resp_amp)
        elif Tstim[stim].intensity > 5 and Tstim[stim].intensity < 9:
            warm_rs.append(resp_amp)
        elif Tstim[stim].intensity > heat_min:
            hot_rs.append(resp_amp)

    if len(cold_rs) > 0:
        c = np.median(cold_rs)
    else:
        # pdb.set_trace()
        print(f"No cold stim found in data")
        #print(f"No cold stim found in {data['source']}")
        # pdb.set_trace()
        return(None)
    if len(warm_rs) > 0:
        w = np.median(warm_rs)
    else:
        print(f'No warm stim found in {data["source"]}')
        return(None)
    if len(hot_rs) > 0:
        h = np.median(hot_rs)
    else:
        if missingFifty:  # If have already lowered heat threshold and can't find heat stim, analysis fails
            print(f'No hot stim found in {data["source"]}')
            return(None)
        # If analysis fails due to lack of temp above 49, retry with min heat stim as ~47 (+12)
        else:
            return(cell_analyze_temp(cell, data=data, Tstim=Tstim, normalize=normalize, F=F, A=A, plot_cell=plot_cell, draw_axes=draw_axes, marker=marker, alpha=alpha, missingFifty=True))

    X = (c*np.cos(theta_cold)) + \
        (w * np.cos(theta_warm)) + (h*np.cos(theta_hot))
    Y = (c*np.sin(theta_cold)) + \
        (w * np.sin(theta_warm)) + (h*np.sin(theta_hot))

    angle = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    color = np.array([int(h*255), int(w*255), int(c*255)])

    output['angle'] = angle
    output['r'] = r
    output['XY'] = [X, Y]
    output['color'] = color
    output['cwh'] = [c, w, h]
    if plot_cell:
        F, A = cell_plot_temp(
            output, F=F, A=A, draw_axes=draw_axes, marker=marker, alpha=alpha)
    output['F'] = F
    output['A'] = A
    return(output)


def cell_plot_temp(parameters, F=None, A=None, color='default', draw_axes=True, draw_circle=True, marker='o', alpha=0.5):
    # parameters for input
    theta_cold = parameters['theta_cold']
    theta_warm = parameters['theta_warm']
    theta_hot = parameters['theta_hot']

    X = parameters['XY'][0]
    Y = parameters['XY'][1]
    if color == 'default':
        color = parameters['color']

    if F == None:
        F = plt.figure('Polar plot')
    if A == None:
        A = F.add_axes([0, 0, 1, 1])
    if draw_axes:
        A.plot([0, np.cos(theta_cold)], [
               0, np.sin(theta_cold)], 'k', linewidth=0.2)
        A.plot([0, np.cos(theta_warm)], [
               0, np.sin(theta_warm)], 'k', linewidth=0.2)
        A.plot([0, np.cos(theta_hot)], [
               0, np.sin(theta_hot)], 'k', linewidth=0.2)
    if draw_circle:  # and draw_axes:
        draw_temp_circle(F=F, A=A)

    color[np.where(color < 0)] = 0
    A.scatter(X, Y, facecolor=color/255, alpha=alpha, marker=marker)
    A.set_ylim([-1.5, 1.5])
    A.set_xlim([-1.5, 1.5])

    A.set_aspect('equal')
    A.xaxis.set_visible(False)
    A.yaxis.set_visible(False)
    A.set_frame_on(False)

    return(F, A)


def temp_class_bounds():
    bounds = {}

    bounds['cold'] = [np.pi*(1/3), np.pi*(2/3)]
    bounds['warm'] = [np.pi*7/6, np.pi * (5/3)]
    bounds['hot'] = [np.pi*(-1/3), np.pi*0]
    bounds['poly'] = [0, np.pi*(1/3)]
    return(bounds)


def draw_temp_circle(F=None, A=None):
    bounds = temp_class_bounds()
    if F is None:
        F = plt.figure()
    if A is None:
        A = F.add_axes([0, 0, 1, 1])

    color = {}
    center = {}
    color['hot'] = 'r'
    color['cold'] = 'b'
    color['warm'] = 'y'
    color['poly'] = 'm'

    r = 1
    for key, bound in bounds.items():
        arc_angles = np.linspace(bound[0], bound[1], 100)
        Xs = r*np.cos(arc_angles)
        Ys = r*np.sin(arc_angles)
        A.plot(Xs[3:-3], Ys[3:-3], color=color[key], linewidth=3)
        textpoint = [r*1.2*np.cos(arc_angles[50]),
                     r*1.2*np.sin(arc_angles[50])]
        # , rotation = (np.degrees(arc_angles[50])-90))
        A.text(textpoint[0], textpoint[1], key,
               horizontalalignment='center', verticalalignment='center')


def plot_cells_connected(cells, datas, F=None, A=None, session_color_list=None, draw_axes=False, session_marker_list=None, normalize=True, missingfifty=False):
    # Arguments:
    # cells - list of cell objects - should represent the same cell observed at different times
    # datas = list of DATA sets corresponding to the sessions for each cell - used to get stimulus information

    Xs = []
    Ys = []

    used_colors = []

    if F == None:
        F = plt.figure('Polar plot')
    if A == None:
        A = F.add_axes([0, 0, 1, 1])

    if session_marker_list is None:
        session_marker_list = []
        for c, cell in enumerate(cells):
            if c == 0:
                session_marker_list.append(".")
            elif c+1 == len(cells):
                session_marker_list.append("o")
            else:
                session_marker_list.append(" ")

    if session_color_list == None:
        session_color_list = []
        for c, cell in enumerate(cells):
            if c == 0:
                session_color_list.append("default")
            elif c+1 == len(cells):
                session_color_list.append("default")
            else:
                session_color_list.append("default")

        # session_color_list = [np.array([0.3,0.3,0.3]), 'default'] ## color by end tuning
        # session_color_list = [np.array([0.3,0.3,0.3]), 'baseline']  ## color marker by baseline tuning

    for count, (cell, DATA) in enumerate(zip(cells, datas)):
        params = cell_analyze_temp(
            cell, DATA, plot_cell=False, normalize=normalize, missingFifty=missingfifty)

        # if no Figure and Axes given, will make new and return for next cell(s)
        if count == 0:

            original_color = params['color']
            # F, A = cell_plot_temp(params, F=F, A=A, color=color, draw_axes = draw_axes, marker=session_marker_list[count])

            if draw_axes:
                # continue
                print('drawing axes')
                theta_cold = params['theta_cold']
                theta_warm = params['theta_warm']
                theta_hot = params['theta_hot']

                A.plot([0, np.cos(theta_cold)], [0, np.sin(theta_cold)],
                       'k', linewidth=0.3, alpha=0.25)
                A.plot([0, np.cos(theta_warm)], [0, np.sin(theta_warm)],
                       'k', linewidth=0.3, alpha=0.25)
                A.plot([0, np.cos(theta_hot)], [0, np.sin(theta_hot)],
                       'k', linewidth=0.3, alpha=0.25)

        if session_color_list[count] == 'default':
            color = params['color']
        elif session_color_list[count] == 'baseline':
            color = original_color
        else:
            color = session_color_list[count]

        cell_plot_temp(params, F=F, A=A, color=color,
                       draw_axes=False, marker=session_marker_list[count])
        used_colors.append(color)

        Xs.append(params['XY'][0])
        Ys.append(params['XY'][1])

    for ii in range(len(Xs)-1):
        A.plot([Xs[ii], Xs[ii+1]], [Ys[ii], Ys[ii+1]],
               color=used_colors[-1]/255, alpha=0.25)

        #A.plot([0, Xs[ii+1]],[0, Ys[ii+1]], 'k')

    # plt.ylim([-1,1])
    # plt.xlim([-1,1])

    A.set_ylim([-1.5, 1.5])
    A.set_xlim([-1.5, 1.5])

    A.set_aspect('equal')
    A.xaxis.set_visible(False)
    A.yaxis.set_visible(False)
    A.set_frame_on(False)

    XYs = [Xs, Ys]
    return(F, XYs)


def therm_polar_plot(datas, plot_on=True, F=None, A=None, normalize=False, draw_axes=False, alpha=0.5):
    if F == None:
        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])

    if not isinstance(datas, list):
        datas = [datas]
    if len(datas) == 0:
        return()

    num_cells = 0
    for data in datas:
        num_cells = num_cells + len(data['cells'])

    theta_data = np.zeros([num_cells])
    r_data = np.zeros([num_cells])
    cmap = np.zeros([num_cells, 3], np.uint8)

    counter = 0
    for data in datas:
        for n, cell in data['cells'].items():
            if draw_axes:
                if counter == 0:
                    draw_axes = True
                else:
                    draw_axes = False
            result = cell_analyze_temp(
                cell, data=data, F=F, A=A, plot_cell=True, draw_axes=draw_axes, alpha=alpha)

            # pdb.set_trace()
            theta_data[counter] = result['angle']
            r_data[counter] = result['r']
            cmap[counter, :] = result['color']
            counter = counter+1

    output = {}

    output['cmap'] = cmap
    output['thetas'] = theta_data
    output['rs'] = r_data

    output['F'] = F
    output['A'] = A
    return(output)

def normalize_traces(traces):
    
    normTraces = np.zeros(traces.shape)
    for cell, trace in enumerate(traces):
        trace = trace-np.quantile(trace, 0.1)
        if np.max(trace) == 0:
            normTraces[cell, :] = trace
        else:
            normTraces[cell, :] = trace/np.max(trace)
    return(normTraces)

def sortByKmean(traces, nClusters, plot=False, return_labels = False):
    
    if nClusters > traces.shape[0]:
        nClusters = traces.shape[0]

    normTraces = normalize_traces(traces)
    kmeans = KMeans(n_clusters=nClusters).fit(normTraces)
    
    IX = np.linspace(0, traces.shape[0]-1, traces.shape[0])
    newIX = np.array([], dtype=np.uint32)
    labels_out = []

    for label in range(nClusters):
        newIX = np.concatenate((newIX, IX[kmeans.labels_ == label]), axis=0)

    sorted_raster = normTraces[newIX.astype(np.uint32)]
    distance_mat = np.zeros([sorted_raster.shape[0], kmeans.cluster_centers_.shape[0]])
    for tt, trace in enumerate(sorted_raster):
        for cc, center in enumerate(kmeans.cluster_centers_):
            distance_mat[tt,cc] = 1/np.linalg.norm(trace-center)

    if plot:
        plt.figure()
        plt.imshow(distance_mat, aspect='auto')
    
    if return_labels:
        return(newIX.astype(np.uint32), kmeans.labels_)
    else:
        return(newIX.astype(np.uint32))


    

def assemble_rasters(results, genotypes = None, nClusters = 3):
    if genotypes is None:
        genotypes = results.keys()
    all_rasters = []
    all_stims = []
    ncells = 0
    combined_rasters_sources = []
    for cc, genotype in enumerate(genotypes):
        raster = results[genotype]['r']
        stim = results[genotype]['s']
        #A_stim.plot(stim, color = la.gen_colors()[genotype], alpha=0.1)
        all_rasters.append(raster)
        for c, row in enumerate(raster):
            combined_rasters_sources.append(cc) ## saving cluster identity for each row
        all_stims.append(stim)
        ncells = ncells + raster.shape[0]
        
    combined_raster = np.vstack(all_rasters)
    results['combined_stims']  = np.hstack(all_stims)
    
    # If not provided, optimize cluster number using silhouette score:
    if nClusters is None:
        nClusters, o, F_opt_clust = get_kmeans_clust_num(combined_raster, repeats=50)
        save_fig(F_opt_clust, 'K means optimization')
    
   
    """
    Cluster traces and sort raster by cluster identity:
    """
    
    newIX = sortByKmean(combined_raster, nClusters)
    combined_raster = combined_raster[newIX,:]
    combined_rasters_sources  = np.array(combined_rasters_sources)
    combined_rasters_sources = combined_rasters_sources[(newIX)]
 
    for c, genotype in enumerate(genotypes):
        results[genotype]['r'] = combined_raster[combined_rasters_sources==c]
        
    return(results)
    
    
def show_cluster_centers(results, key='rPbN', n_clusters=3):
    """ 
    Takes results of rasters_by_genotype from DHfigures.py as input
    """
    F = plt.figure()
    raster = results[key]['r']
    stim = results[key]['s']
    for c, trace in enumerate(raster):
        trace = trace-np.min(trace)
        trace = trace/np.max(trace)
        trace = np.nan_to_num(trace)
        raster[c, :] = trace

    kmeans = KMeans(n_clusters=n_clusters).fit(raster)
    A = F.add_subplot(len(kmeans.cluster_centers_)+1, 1, 1)
    box_off(A)
    A.plot(stim, 'm', alpha=0.25)
    A.spines.bottom.set_visible(False)
    A.set_xticks([])
    for c, center in enumerate(kmeans.cluster_centers_):
        A = F.add_subplot(len(kmeans.cluster_centers_)+1, 1, c+2)
        A.plot(center, 'k')
        box_off(A)
        if c+1 < len(kmeans.cluster_centers_):
            A.spines.bottom.set_visible(False)
            A.set_xticks([])
    
    save_fig(F, f'K_means centers {n_clusters}')
    return(kmeans.cluster_centers_, stim)

def show_raster_PCs(raster, n_PCs=3, cut_off=2, plot=True):
    plt.figure('Original raster')
    plt.imshow(raster, aspect = 'auto')
    raster = normalize_raster(raster).T
    P = plt.figure('Norm raster')
    Q = P.add_subplot(1,1,1)
    Q.imshow(raster, aspect='auto')
    PCs = get_PCs(raster).T[0:n_PCs]
    if plot:
        F = plt.figure('PCs')
        for c, pc in enumerate(PCs):
            A =F.add_subplot(n_PCs+1,1,c+1)
            A.plot(pc)
            box_off(A, left_only = True)
    return(PCs)

def normalize_raster(raster):
    if len(raster.shape) != 2:
        print('Raster dimensions are {raster.shape}, this function requires 2d raster')
        return()
    out = raster * 0
    for c, trace in enumerate(raster):
        trace = trace-np.min(trace)
        trace = trace/np.max(trace)
        trace = np.nan_to_num(trace)
        out[c, :] = trace
    
    return(out)
        

def show_PCs(results, key='rPbN', n_PCs=3, cut_off=2):
    F = plt.figure()
    raster = results[key]['r']
    stim = results[key]['s']
    for c, trace in enumerate(raster):
        trace = trace-np.min(trace)
        trace = trace/np.max(trace)
        trace = np.nan_to_num(trace)
        raster[c, :] = trace
    try:
        PCs = get_PCs(raster.T)
    except:
        pdb.set_trace()
    PCs = PCs.T[0:n_PCs]
    # PCs = PCs[]
    A = F.add_subplot(n_PCs+1, 1, 1)
    box_off(A)
    A.plot(stim, 'm', alpha=0.25)
    A.spines.bottom.set_visible(False)
    A.set_xticks([])
    for c, pc in enumerate(PCs):
        A = F.add_subplot(n_PCs+1, 1, c+2)
        if c > cut_off:
            color = 'k'
        else:
            color = 'r'
        A.plot(pc, color)
        box_off(A)
        if c+1 < n_PCs:
            A.spines.bottom.set_visible(False)
            A.set_xticks([])
            A.set_yticks([])

    save_fig(F, f'PCAs for {n_PCs}')


def get_kmeans_clust_num(traces, c_range=(2, 10), norm=True, repeats=50, plot=True):
    if norm:
        normTraces = np.zeros(traces.shape)
        for cell, trace in enumerate(traces):
            trace = trace-np.quantile(trace, 0.1)
            if np.max(trace) == 0:
                normTraces[cell, :] = trace
            else:
                normTraces[cell, :] = trace/np.max(trace)
        traces = normTraces
    if plot:
        F = plt.figure('Silhouette score')
        A = F.add_axes([0, 0, 1, 1])
    else:
        F = None
    opt = []
    for r in range(repeats):
        scores = []
        n_clusts = []
        for n in range(*c_range):
            k = KMeans(n_clusters=n).fit_predict(traces)
            score = silhouette_score(traces, k)
            print(f'For {n} clusters, silhouette score is {score}')
            scores.append(score)
            n_clusts.append(n)

        max_IX = np.where(scores == np.amax(scores))[0][0]
        best_n = n_clusts[max_IX]
        opt.append(best_n)
        if plot:
            A.plot(n_clusts, scores, color='k', alpha=2/repeats)
            A.scatter(best_n, np.amax(scores), color='r', alpha=1/repeats)
            box_off(A)
    n_clust = round(np.average(opt))
    return(n_clust, opt, F)


def hierarch_clustering(traces, n_clusters=3, **kwargs):
    linkage_data = linkage(traces, method='ward', metric='euclidean')

    hierarchical_cluster = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(traces)
    IX = []
    for i in range(n_clusters):
        IX.extend(np.where(labels == i)[0])
    reordered = traces[IX]
    plt.figure()
    plt.imshow(reordered, vmin=0, vmax=0.5)
    plt.figure()
    dendrogram(linkage_data)
    return(reordered)

def lighten_color(color, amount = .5):
    RGB= matplotlib.colors.to_rgb(color)
    L = [0,0,0]
    for c, val in enumerate(RGB):
        
        L[c] = (1-amount) * val + amount
    return(L)
    
    
def plot_mech_ranges(responses, color='k', SNR_threshold = 3.5, unit = 'mN', plot_non_responsive = True, offset=0, limit = 1000, dm=2, dexp = 1, save_plot=True, F=None, A=None, log=True, title='', show_legend=True, only_responders = True, append=False):
    uid = str(uuid.uuid4())[0:5]
    if F is None:
        
        F = plt.figure('Force sensing' + ' ' + title + uid)
        
    X0 = 0
    if A is None:
        A = F.add_subplot(1,1,1)
        X0 = 0
    elif append:
        X0 = A.get_xlim()[1]
    
    if only_responders:
        validated = []
        for r, response_set in enumerate(responses):
            #valid = False
            for rr, response in enumerate(response_set):
                if response['valid_response'] and response['force']<=limit:
                    #valid = True
                    validated.append(response_set)
                    break
    else:
        validated = responses
                
    responses = validated
    #pdb.set_trace()
    for r, response_set in enumerate(responses):
        for rr, response in enumerate(response_set):
            if not response['valid_response']: 
                if plot_non_responsive:
                    A.scatter(r+X0+offset, response['force'], s = 1, color = lighten_color(color, amount = 0.8))
                
    for r, response_set in enumerate(responses):
        for rr, response in enumerate(response_set):
            if response['valid_response'] and response['force']<=limit: 
                A.scatter(r+X0+offset, response['force'], s = response['max_SNR']**dexp, color = color)
            
    
                
    
    box_off(A)
    if log:
        A.set_yscale('log')
        if unit == 'gf':
            yticks = [1,10,100]
        else:
            yticks = [10,100,1000]
        A.set_ylim([yticks[0],limit * 1.3])
        labels = [str(x) for x in yticks]
        A.set_yticks(yticks)
        A.set_yticklabels(labels)
    # if show_legend:
    #     FL = plt.figure(title + 'legend' + uid)
    #     AL = FL.add_subplot(1,1,1)
    #     X = [1,3,4,5]
    #     S = [1, (dm*SNR_threshold)**dexp, (dm*5)**dexp,(dm*10)**dexp, (dm*15)**dexp, (dm*20)**dexp]
        
    #     for xx, ss in zip(X,S):
    #         if xx == 1:
    #             AL.scatter(xx, limit*1.15, s=ss, color=lighten_color(color, amount = 0.7))
    #         else:
    #             AL.scatter(xx, limit*1.15, s=ss, color = color)
        
    if save_plot:
        save_fig(F, 'Force sensing' + ' ' + title + uid)
    return(F,A)
    
def thermPolarPlot(data, plot_on=True, F=None, A=None, normalize=False):
    # plots all cells for a data set
    response_array = np.zeros([len(data['cells']), 3])
    for c, cell in enumerate(data['cells']):
        # get responses to cold stim(<10 deg):
        cold_rs = []
        warm_rs = []
        hot_rs = []
        for s, stim in enumerate(data['Tstim']):
            response = data['cells'][cell].response(
                data['Tstim'][stim], normalize=normalize)
            resp_amp = response['amplitude']
            #resp_amp = response['normamplitude']
            if data['Tstim'][stim].intensity < -24:
                cold_rs.append(resp_amp)
            elif data['Tstim'][stim].intensity > 5 and data['Tstim'][stim].intensity < 8:
                warm_rs.append(resp_amp)
            elif data['Tstim'][stim].intensity > 14:
                hot_rs.append(resp_amp)
        if len(cold_rs) > 0:
            response_array[c, 0] = np.median(cold_rs)
        else:
            print('No cold stim found')
            response_array[c, 0] = 0
        if len(warm_rs) > 0:
            response_array[c, 1] = np.median(warm_rs)
        else:
            print('No warm stim found')
            response_array[c, 1] = 0
        if len(hot_rs) > 0:
            response_array[c, 2] = np.median(hot_rs)
        else:
            print('No hot stim found')
            response_array[c, 2] = 0

    #    response_array[c,0] = np.mean(cold_rs)
    #    response_array[c,1] = np.mean(warm_rs)
    #    response_array[c,2] = np.mean(hot_rs)

    theta_cold = 0+np.pi/2  # 90 deg, 1.57 radians
    theta_warm = np.pi*2/3 + np.pi/2  # 210 deg, 3.665 radians
    theta_hot = np.pi*4/3 + np.pi/2  # 330 deg, 5.5795 radians (-30)

    if plot_on:
        if F == None:
            F = plt.figure('Polar plot')
        if A == None:
            A = F.add_axes([0, 0, 1, 1])
            print(A)

        A.plot([0, np.cos(theta_cold)], [
               0, np.sin(theta_cold)], 'k', linewidth=0.3)
        A.plot([0, np.cos(theta_warm)], [
               0, np.sin(theta_warm)], 'k', linewidth=0.3)
        A.plot([0, np.cos(theta_hot)], [
               0, np.sin(theta_hot)], 'k', linewidth=0.3)

    polar_data = np.zeros([len(data['cells']), 2])
    theta_data = np.zeros([len(data['cells'])])
    r_data = np.zeros([len(data['cells'])])
    cmap = np.zeros([len(data['cells']), 3], np.uint8)

    for count, cell in enumerate(response_array):

        c = response_array[count, 0]
        w = response_array[count, 1]
        h = response_array[count, 2]

        cmap[count, 0] = int(h*255)
        cmap[count, 1] = int(w*255)
        cmap[count, 2] = int(c*255)

        polar_data[count, 0] = (c*np.cos(theta_cold)) + \
            (w * np.cos(theta_warm)) + (h*np.cos(theta_hot))
        polar_data[count, 1] = (c*np.sin(theta_cold)) + \
            (w * np.sin(theta_warm)) + (h*np.sin(theta_hot))

        #theta_data[count] = np.arctan(polar_data[count,1]/polar_data[count,0])
        theta_data[count] = np.arctan2(
            polar_data[count, 1], polar_data[count, 0])
        r_data[count] = np.hypot(polar_data[count, 1], polar_data[count, 0])

        if plot_on:
            A.scatter(polar_data[count, 0], polar_data[count, 1],
                      facecolor=cmap[count, :]/255, alpha=0.5)

    #plt.scatter(polar_data[:,0], polar_data[:,1], cmap=cmap.T)
    if plot_on:

        plt.ylim([-1, 1])
        plt.xlim([-1, 1])

        A.set_aspect('equal')
        A.xaxis.set_visible(False)
        A.yaxis.set_visible(False)
        A.set_frame_on(False)

        F.show()
    output = {}
    output['cmap'] = cmap
    output['thetas'] = theta_data
    output['rs'] = r_data
    return(output)


def thermColorCode(data, plot=True, F=None):
   # cmap_dict = thermPolarPlot(data, plot_on = False)
    cmap_dict = therm_polar_plot(data)

    cmap = cmap_dict['cmap']
    ROIs = data['ROIs']
    im = data['fieldImage']

    R = np.zeros([im.shape[0], im.shape[1], 1])
    G = np.zeros([im.shape[0], im.shape[1], 1])
    B = np.zeros([im.shape[0], im.shape[1], 1])
    # dlevels = obj.dLevels[obj.dataFocus] # get gray levels from display

    for r in range(ROIs.shape[2]):
        R[ROIs[:, :, r] > 0] = cmap[r, 0]
        G[ROIs[:, :, r] > 0] = cmap[r, 1]
        B[ROIs[:, :, r] > 0] = cmap[r, 2]

    A = np.max(ROIs, axis=2)
    A = A.astype(bool)
    A = A.astype(np.uint8)*200
    A = np.expand_dims(A, axis=2)
    RGBA = np.concatenate((R, G, B, A), axis=2)/255

    if F == None:
        F = plt.figure(f'Thermo coded {data["source"]}')
    A = F.add_axes([0, 0, 1, 1])
    A.imshow(im.T, aspect='auto', interpolation='none',
             cmap='gist_gray', vmin=0, vmax=255)
    A.xaxis.set_visible(False)
    A.yaxis.set_visible(False)
    A.set_frame_on(False)

    if RGBA != np.array([]):
        R = np.swapaxes(RGBA, 0, 1)
        A.imshow(R)

    plt.show()
    #saveName = os.path.join('/home/ch184656/Default reports', self.curAnimal+self.curFOV+self.dataFocus + '.png')
    #F.savefig(saveName, transparent = True)
    return(F)


def thermoScatter(data, cells='all', plot=False):
    if cells == 'all':
        pass

    spectrum = np.zeros(len(data['Tstim']))

    for s, stim in enumerate(data['Tstim']):
        spectrum[s] = data['Tstim'][stim].intensity
    spectrum = np.sort(spectrum)
    output = np.zeros([len(data['Tstim']), len(data['cells'])])
    output2 = np.zeros(len(data['cells']))
    for c, cell in enumerate(data['cells']):
      #   if cells != 'all':
      #       if not(c in cells):
      #           continue
        for s, stim in enumerate(data['Tstim']):
            stimresponse = data['cells'][cell].response(data['Tstim'][stim])
            # using amplitude of signal
            output[s, c] = stimresponse['amplitude']

        pattern = output[:, c]
        maxT = spectrum[np.argmax(pattern)]
        # temperature at which cell gives max amplitude responses
        output2[c] = maxT
    if plot:
        plt.figure()
        for c, cell in enumerate(data['cells']):
            plt.subplot(1, 2, 1)
            plt.scatter(spectrum, output[:, c], alpha=0.5)
            plt.scatter(output2[c], 1)
            plt.ylim([0, 1])
            plt.subplot(1, 2, 2)
            data['cells'][cell].show()
            plt.show()

        plt.figure()
        plt.imshow(output)
   # plt.figure()
   # plt.plot(spectrum)
   # plt.figure()
   # plt.plot(output2)
   # plt.show()
    plt.figure()
    for ii in range(output.shape[1]):
        plt.scatter(spectrum, output[:, ii])
    return(output, spectrum, output2)


def TvsTcor(data, cells='all',  Trange=None, Tzero=34, minStimMag=3, nClusters=3, plot=False):

    if cells == 'all':
        response_raster = data['raster'][...]
    else:
        response_raster = data['raster'][cells, :]

    if len(response_raster.shape) == 1:
        response_raster = np.expand_dims(response_raster, 0)

    Tstim = data['stims']['Temp (°C)'][...]

    if Tstim.shape[0] != response_raster.shape[1]:
        print(f'Stim shape: {Tstim.shape[0]}')
        print(f'Response shape: {response_raster.shape}')
        print('Stimulus not aligned to responses')
        return(None, None)

    if Trange == None:
        Trange = np.linspace(6, 52, 47, endpoint=True)

    Trange = Trange - Tzero
    Tstim = Tstim-Tzero

    output = np.zeros([response_raster.shape[0], Trange.shape[0]])
    for cc, response in enumerate(response_raster):

        for c, T in enumerate(Trange):
            if T < -minStimMag:
                IX = np.where((Tstim > T) & (Tstim < minStimMag))[0]
            elif T > minStimMag:
                IX = np.where((Tstim < T) & (Tstim > minStimMag))[0]
            else:
                output[cc, c] = 0
                continue

            stimS = Tstim[IX]

            respS = response[IX]
            output[cc, c] = np.corrcoef(stimS, respS)[0, 1]
            # plt.plot(stimS)
            # plt.plot(respS/1000)
            # plt.show()

 #       plt.plot(Trange, output[cc,:], color = 'gray')

#    plt.plot([Trange[0],Trange[-1]],[0,0], color = 'r')
#    plt.ylim(-1, 1)
  #  plt.show()

   # plt.figure()

    if nClusters > output.shape[0]:
        nClusters = output.shape[0]

    output = np.nan_to_num(output)

    kmeans = KMeans(n_clusters=nClusters).fit(output)

    IX = np.linspace(0, output.shape[0]-1, output.shape[0])
    newIX = np.array([], dtype=np.uint32)

    for label in range(nClusters):
        newIX = np.concatenate((newIX, IX[kmeans.labels_ == label]), axis=0)

    newIX = newIX.astype(np.uint16)
    output = output[newIX]
 #   plt.imshow(output)
  #  plt.imshow(distance_matrix(output,output))
    orderedraster = response_raster[newIX]
    if plot:
        plt.figure()
        plt.imshow(output, vmin=-1, vmax=1)
        plt.set_cmap('bwr')
        plt.figure()
        plt.imshow(distance_matrix(output, output))
        plt.set_cmap('viridis')
        plt.show()
        plt.imshow(orderedraster, cmap='inferno', aspect=0.5 *
                   orderedraster.shape[1]/orderedraster.shape[0])
    return(output, orderedraster)


def getCor(path, nClusters=3, Tzero=35, minStimMag=0):
    data = processTempStim(prepAnalysis((path)))
    c, raster = TvsTcor(data, nClusters=nClusters,
                        Tzero=Tzero, minStimMag=minStimMag)
    return(c, raster)


def match_and_compare(data_list, corrFile=None, corr=None, doPlot=True, plot_mice_separate=False, F=None, A=None, draw_axes=False):

    # Create union of cells for (# total cells x number of sessions)
    if corr is None:
        corr = np.genfromtxt(corrFile, delimiter=',')
   # print(corr)
    # remove any cell unions with no entries (typically disqualified because some sessionsn don't include in field of view)
    corr = corr[~np.all(np.isnan(corr), axis=1)]
    nCells = corr.shape[0]
    nSessions = corr.shape[1]
    if len(data_list) != nSessions:
        print(
            f'{len(data_list)} data files generated but {nSessions} in correspondence file')
        print(f'Corr file is {corrFile}')
    else:
        print(f'Matching across {nSessions} sessions...')

    cellArray = {}
    stimArray = {}
    for session, sdata in enumerate(data_list):
        cellArray[session] = {}
        stimArray[session] = {}
        for stim in sdata['stims']:
            stimArray[session][stim] = sdata['stims'][stim]
        for uCell, sCell in enumerate(corr[:, session]):
            if np.isnan(sCell):
                cellArray[session][uCell] = None
            else:
                cellArray[session][uCell] = sdata['cells'][sCell]

    output = {}
    output['corr'] = corr
    output['cells'] = cellArray
    output['stims'] = stimArray
    output['data'] = data_list

    if doPlot:
        p_v = match_plot(output, F=F, A=A, draw_axes=draw_axes)
        output['paired_vectors'] = p_v
    return(output)


def gen_colors():
    # standardized coloring for genotypes/conditions
    gc = {}
    gc['Gpr83'] = 'g'
    gc['Tacr1'] = 'r'
    gc['rPbN'] = 'k'
    gc['hot'] = np.array([1, 0, 0.5])  # rose
    gc['warm'] = np.array([1, 0.5, 0])  # orange
    gc['cold'] = 'c'
    gc['poly'] = np.array([0.5, 0, 1])  # purple
    gc['none'] = 'k'
    gc[None] = 'k'
    return(gc)

def gen_colors_II():
    # standardized coloring for genotypes/conditions
    gc = {}
    gc['Gpr83'] = np.array([50/255,206/255,50/255]) #lime
    gc['Tacr1'] = np.array([1, 0, 0.5])  # rose
    gc['rPbN'] = 'k'
    gc['hot'] = 'r'  # rose
    gc['warm'] = np.array([1, 0.5, 0])  # orange
    gc['cold'] = 'c'
    gc['poly'] = np.array([0.5, 0, 1])  # purple
    gc['none'] = 'k'
    gc[None] = 'k'
    return(gc)


def match_plot(dataset, F=None, A=None, draw_axes=False):
    paired_vectors = []
    corr = dataset['corr']
    cells = dataset['cells']
    #stims = dataset['stims']
    #nStims = len(stims.keys())
    nCells = corr.shape[0]
    nSessions = corr.shape[1]

    # Plot traces and sessions arranged in grid
    G = plt.figure()
    IX = 1
    for ii in range(nCells):
        for jj in range(nSessions):
            B = G.add_subplot(nCells, nSessions, IX)
            if cells[jj][ii] is None:
                plt.plot([0, 1], [0, 0], 'r')
                plt.ylim([-0.1, 1])
            else:
                plt.plot(cells[jj][ii].trace, 'k')
            plt.axis('off')
            IX = IX + 1

    # return() ## test

    if F is None:
        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])
    for ii in range(nCells):
        cell_series = []
        data_series = []
        for jj in range(nSessions):
            cell_series.append(cells[jj][ii])
            # print(cell_series)
        if ii == 0:
            draw_axes = draw_axes
        else:
            draw_axes = False
        F, V = plot_cells_connected(
            cell_series, dataset['data'], F=F, A=A, draw_axes=draw_axes)

        paired_vectors.append(V)
    return(paired_vectors)


def map_angle(angle):
    # restricts value of angle to between 0 and 2pi radians
    return(((angle+(2*np.pi)) % (2*np.pi)))


def class_list():
    return(['cold', 'warm', 'hot', 'poly'])


def classify_cells_in_dataset(DATA):  # TODO

    bounds = temp_class_bounds()

    for key in DATA['cells']:
        response_type = 'None'
        CELL = DATA['cells'][key]

        stats = cell_analyze_temp(CELL, DATA)
        theta = map_angle(stats['angle'])  # bound angle 0->2 pi  radians
        for response_type in bounds:
            b = bounds[response_type]
            b[0] = map_angle(b[0])
            b[1] = map_angle(b[1])
            if b[0] > b[1]:
                if theta >= b[0]:
                    DATA['cells'][key].classification = response_type
            else:
                if theta >= b[0] and theta < b[1]:
                    DATA['cells'][key].classification = response_type

    return(DATA)


def count_cell_classes(DATA):
    counts = {}
    counts['total'] = 0
    for key in DATA['cells']:
        c = DATA['cells'][key].classification
        if not (c in counts):
            counts[c] = 0
        else:
            counts[c] = counts[c] + 1
            counts['total'] = counts['total']+1
    return(counts)


def recruitment_analysis():
    # For gorups of observations (datasets), calculate proportion of cells recruited at different stim intensities
    pass


def rundown_analyis():
    pass


def cell_analyze_mechano(cell=None, norm=True, plot = False, detail=False):
    sr = mech_tuning_cell(cell, plot=plot, detail = detail, norm=norm)
    cell.mech_stats['mech_max'] = np.amax(sr[:, 1])
    cell.mech_stats['stim-resp'] = sr
    return(cell)


    
    
def mech_tuning_cell(cell, F=None, A1=None, A2=None, A3=None, plot=False, detail=False, norm=True):

    if F is None and plot:
        F = plt.figure()
        A1 = F.add_axes([0, 0, 1, 0.5])
        A2 = F.add_axes([0, 0.5, 1, 0.25])
        A3 = F.add_axes([0, 0.75, 1, 0.25])

    DATA = cell.parent_data
    Mstim = DATA['Mstim']

    stim_response = np.zeros([len(Mstim), 2])
    for c, mech_stim in enumerate(Mstim.values()):
        # pdb.set_trace()
        stim_response[c, 0] = mech_stim.intensity
        # pdb.set_trace()
        if norm:
            stim_response[c, 1] = cell.response(mech_stim)['amplitude']
        else:
            stim_response[c, 1] = cell.response(mech_stim)['normamplitude']
        if detail:
            start = mech_stim.start-100
            if start < 0:
                start = 0
            end = mech_stim.end+100
            if end > len(mech_stim.parent):
                end = len(mech_stim.parent)-1
            G = plt.figure()
            A = G.add_axes([0, 0, 1, 1])
            A.plot(mech_stim.parent[start:end], 'k')
            B = A.twinx()
            B.plot(cell.trace[start:end], 'r')
            B.set_ylim([0, 1.1])
            A.set_ylim([-10, np.amax(mech_stim.parent)])

    if plot:
        A1.scatter(stim_response[:, 0], stim_response[:, 1])
        A1.set_ylim([0, 1])
        cell.show(F=F, A1=A2, A2=A3)
        if detail:
            cell.show(showMech=True)

    return(stim_response)


def mech_tuning_data_file(DATA, F=None, A=None, plot=True):
    if F is None:
        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])
    for cell in DATA['cells'].values():
        mech_tuning_cell(cell, DATA=DATA, F=F, A=A, plot=plot)


def plot_y_scale_bar(A, size=0.05, color=[0.5, 0.5, 0.5], linewidth=2):
    yspan = A.get_ylim()[1]-A.get_ylim()[0]
    top = A.get_ylim()[1] - yspan/1000
    bottom = top-size
    xspan = A.get_xlim()[1]-A.get_xlim()[0]

    left = A.get_xlim()[0] + xspan/1000
    A.plot([left, left], [bottom, top], color=color, linewidth=linewidth)


def plot_x_scale_bar(A, size=100, color='k', linewidth=2, placement='top', text='seconds'):
    bottom = A.get_ylim()[0]
    top = A.get_ylim()[1]
    yspan = top-bottom
    xspan = A.get_xlim()[1]-A.get_xlim()[0]
    offset = yspan/1000
    if placement == 'top':
        y = top - yspan/1000
    elif placement == 'bottom' or placement == 'bot':
        offset = yspan/1000
        y = bottom + yspan/1000
    left = A.get_xlim()[0] + xspan/1000
    right = left + size
    A.plot([left, right], [y, y], color=color, linewidth=linewidth)
    if not text is None:
        A.text(left, y+(offset*50), f'{int(size/10)} {text}')


def get_PCs(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)
    pca = PCA().fit(scaled)

    pc = pca.transform(scaled)
    return(pc)


def detrend(trace, wavelength, plot=True, remove_peaks=True, limit_q=0.75, num_segs=20, method='pad', rectify=False):
    # pdb.set_trace()
    if remove_peaks:
        trace_to_filter = []

        window_size = int(len(trace)/num_segs)
        # pdb.set_trace()
        for s in range(num_segs+1):
            start = s * window_size
            end = (s+1) * window_size
            segment = copy.copy(trace[start:end])
            if segment.shape[0] == 0:
                continue
            limit = np.quantile(segment, limit_q)
            #limit = 0.1
            segment[np.where(segment > limit)] = np.mean(segment)
            trace_to_filter.extend(list(segment))
        trace_to_filter = np.array(trace_to_filter)
    else:
        trace_to_filter = copy.deepcopy(trace)

    b, a = butter(3, 1/wavelength, 'highpass')
    if remove_peaks:
        ft = filtfilt(b, a, trace_to_filter,   method=method)
        diff = ft - trace_to_filter
        diff = ft - trace_to_filter
        ft = trace + diff
    else:
        ft = filtfilt(b, a, trace_to_filter,  method=method)
        diff = ft - trace_to_filter
    if rectify:
        ft[np.where(ft < 0)] = 0
    if plot:
        plt.figure()
        plt.plot(trace, alpha=0.3, color='r')
        plt.plot(diff, 'r')
        plt.plot(ft, color='k')
    return(ft)


def thermo_grid_multi_session(m, plot=True, selected_sessions='All', selected_u_cells='All', split_metric = False, tuning_threshold=0.3, norm=True, real_time=False, sort=True, use_abs_max=False, min_r=0.05):
    F = plt.figure()
    A = F.add_axes([0, 0, 0.1, 1])
    if selected_sessions == 'All':
        selected_sessions = m.all_sessions()

    for session in selected_sessions:
        m.sessions[session].segment_thermo_stim(plot=False)

    if selected_u_cells == 'All':
        selected_u_cells = []
        for c, u in enumerate(m.assignments):
            selected_u_cells.append(c)

    tuning = {}

    maxes = {}
    max_ = {}
    for c, u in enumerate(selected_u_cells):
        # To do - normlize each celll to max response
        # Iterate through sessions for each cell to get maximum amplitude
        #cwhs = []
        #amps = []
        maxes[u] = []

        for session in selected_sessions:
            m.sessions[session].cells[u].identify_transients()
            tuning[u, session] = thermo_tuning_metric(
                m.sessions[session].cells[u], stim_set=m.sessions[session].Tstim, plot=False, use_abs_max=use_abs_max, min_r=min_r)
            tuning[u, session]['u_cell'] = u
            tuning[u, session]['Source'] = m.sessions[session].Source
            tuning[u, session]['session'] = m.sessions[session]
            tuning[u, session]['cell'] = m.sessions[session].cells[u]
            tuning[u, session]['parent'] = m
            if hasattr(m.sessions[session], 'relative_day'):
                tuning[u, session]['day'] = m.sessions[session].relative_day
            else:
                if real_time:
                    d = {}
                    d[''] = [m]
                    e = get_exp_days(d)
                    m = e[''][0]

            maxes[u].append(tuning[u, session]['amp'])
        max_[u] = np.amax(maxes[u])
        if max_[u] == 0:
            max_[u] = 1

    if norm:
        for key in tuning:
            print(tuning[key]['Source'])
            print(f' Cell {key[0]} session{key[1]}')
            print(f'{max_[key[0]]=}')
            tuning[key]['amp'] = tuning[key]['amp'] / \
                max_[key[0]]  # normalizing alpha
            # for n, norm in tuning[key]['norm']:
            #    tuning[key]['norm'][n] = norm/max_[key[0]]
            if np.isnan(tuning[key]['amp']):
                print('Nan alert')
                tuning[key]['cell'].show()

    if sort:
        k = selected_sessions[0]  # sort by tuning in first session

        # get tuning of each neuron in first session
        hot_cells = []
        hot_int = []
        warm_cells = []
        warm_int = []
        cold_cells = []
        cold_int = []
        untuned_cells = []
        untuned_int = []
        for key in tuning:
            if key[1] == k:
                max_signal = tuning[key]['amp']
                preference = np.where(
                    tuning[key]['norm'] == np.amax(tuning[key]['norm']))[0]
                if max_signal < tuning_threshold:
                    untuned_cells.append(key[0])
                    untuned_int.append(max_signal)
                elif preference == 0:
                    hot_cells.append(key[0])
                    hot_int.append(np.double(tuning[key]['norm'][preference]))
                elif preference == 1:
                    warm_cells.append(key[0])
                    warm_int.append(np.double(tuning[key]['norm'][preference]))
                elif preference == 2:
                    cold_cells.append(key[0])
                    cold_int.append(np.double(tuning[key]['norm'][preference]))

        plot_ixs = np.concatenate([np.array(untuned_cells)[np.argsort(untuned_int)],
                                  np.array(cold_cells)[np.argsort(cold_int)],
                                  np.array(hot_cells)[np.argsort(hot_int)],
                                  np.array(warm_cells)[np.argsort(warm_int)],
                                   ])

        for p, ix in enumerate(plot_ixs):
            for key in tuning:
                if key[0] == ix:
                    tuning[key]['plot_ix'] = p

    else:
        for key in tuning:
            tuning[key]['plot_ix'] = tuning[key][0]

    if plot:
        for item in tuning:
            if np.isnan(tuning[item]['amp']):
                alpha = 0
            elif tuning[item]['amp'] < 0:
                alpha = 0
            elif tuning[item]['amp'] > 1:
                alpha = 1
            else:
                alpha = tuning[item]['amp']

            if real_time:
                X = tuning[item]['day']
            else:
                X = item[1]
            
            Y = tuning[item]['plot_ix']
            
            if split_metric:
                A.scatter(X-0.25, Y+0.25 ,s=25, color=[ tuning[item]['norm'][0], 0 ,0], alpha=alpha)
                A.scatter(X, Y ,s=25, color=[ 0, tuning[item]['norm'][1] ,0], alpha=alpha)
                A.scatter(X, Y ,s=25, color=[ 0,0, tuning[item]['norm'][2]], alpha=alpha)
                
                A.scatter(X, Y ,s=25, color=tuning[item]['norm'], alpha=alpha)
                A.scatter(X, Y ,s=25, color=tuning[item]['norm'], alpha=alpha)
            else:
                A.scatter(X, Y ,s=25, color=tuning[item]['norm'], alpha=alpha)

            
            if item[1] == 0:
                A.text(8, tuning[item]['plot_ix'], str(item[0]))
        box_off(A)
        if not real_time:
            A.set_xlim([-1, len(selected_sessions)])
        save_fig(F, m.pickle_path.split('/')
                 [-1].split('.')[0]+'_grid', ext='.png')
    return(tuning)


def retrieve_class_from_tuning(tuning_item):
    return(tuning_item['class'])



def modified_z_score(vector):
    m = np.median(vector)
    mad = stats.median_abs_deviation(vector)
    k = 1.4826
    output = (vector-m)/(k*mad)
    return(output)

def tuning_grid_across_mice(multi_sessions, norm=False, real_time=False, min_r=0.05):
    """
    get tuning across sessions

    """
    master_tuning_matrix = {}
    for i, m in enumerate(multi_sessions):
        m_s_tuning = thermo_grid_multi_session(
            m, norm=norm, real_time=real_time, min_r=min_r)

        for key in m_s_tuning:

            master_tuning_matrix[i, key[0], key[1]] = m_s_tuning[key]
            master_tuning_matrix[i, key[0], key[1]
                                 ]['cell_index'] = ((i+1)*1000) + key[0]
    return(master_tuning_matrix)

def uid_string(length=5):
    return(str(uuid.uuid4())[0:5])
    
def thermo_tuning_metric(cell, stim_set=None,  F=None, AS=None, AT=None, plot=False, use_abs_max=False, min_r=0.05):
    """"
    parameters:

    cell: a libAnalysis cell object
    stim_set: a list(or dictionary? not sure) of libAnalysis stim objects, assumed to be thermal stimuli
    min_r: minimum response amplitude to be used in calculating tuning (i.e. response set to 0 if lower)



    """
    if plot:
        F = plt.figure('Thermo tuning'+str(uuid.uuid4())[0:5])
        AS = F.add_axes([0.1, 0.6, 0.8, 0.3])
        AT = F.add_axes([0.1, 0.1, 0.8, 0.3])
        AS.plot(cell.parent_session.therm_stim, 'k')
        AT.plot(cell.trace, 'k')

    if stim_set is None:
        stim_set = cell.parent_session.Tstim

    # get cold tuning:
    cold_responses = []
    for t in stim_set.values():
        if t.stim_temp < 12:
            response = cell.response(
                t, use_transients=False, plot_transients=False, F=F, AS=AS, AT=AT)
            # t.show()
            cold_responses.append(response['baseline_diff'])
            if plot:
                AS.plot(t.timepoints, t.waveform, color='b')
                AS.text(t.timepoints[0], 38, str(
                    round(response['baseline_diff'], 2)))
                AT.plot(t.timepoints, cell.trace[int(
                    t.start):int(t.end)+1], color='b')
    cold_med = np.median(np.array(cold_responses))

    # get warm tuning:
    warm_responses = []
    for t in stim_set.values():
        if t.stim_temp > 40 and t.stim_temp < 43.1:
            response = cell.response(
                t, use_transients=False, plot_transients=False, F=F, AS=AS, AT=AT)
            # t.show()
            warm_responses.append(response['baseline_diff'])
            if plot:
                AS.plot(t.timepoints, t.waveform, color='g')
                AS.text(t.timepoints[0], 46, str(
                    round(response['baseline_diff'], 2)))
                AT.plot(t.timepoints, cell.trace[int(
                    t.start):int(t.end)+1], color='g')
    warm_med = np.median(np.array(warm_responses))

    # get hot tuning:
    max_stim = max([stim_set[x].stim_temp for x in stim_set])
    if max_stim < 49.1:
        hot_T = 47
    else:
        hot_T = 49.1

    hot_responses = []
    for t in stim_set.values():
        if t.stim_temp > hot_T:
            response = cell.response(
                t, use_transients=False, plot_transients=plot, F=F, AS=AS, AT=AT)
            # t.show()
            hot_responses.append(response['baseline_diff'])
            if plot:
                AS.plot(t.timepoints, t.waveform, color='r')
                AS.text(t.timepoints[0], t.stim_temp+5,
                        str(round(response['baseline_diff'], 2)))
                AT.plot(t.timepoints, cell.trace[int(
                    t.start):int(t.end)+1], color='r')
    hot_med = np.median(np.array(hot_responses))

    abs_max = np.amax(np.concatenate(
        [cold_responses, hot_responses, warm_responses]))

    hot_med = hot_med - warm_med

    if cold_med < min_r:
        cold_med = 0
    if np.isnan(cold_med):
        cold_med = 0
    if warm_med < min_r:
        warm_med = 0
    if np.isnan(warm_med):
        warm_med = 0
    if hot_med < min_r:
        hot_med = 0
    if np.isnan(hot_med):
        hot_med = 0

    vector = np.array([hot_med, warm_med, cold_med])
    amplitude = np.amax(vector)
    output = {}
    output['hwc'] = vector
    output['norm'] = vector/np.sum(vector)
    if use_abs_max:
        output['amp'] = abs_max
    else:
        output['amp'] = amplitude
    if plot:
        AS.text(5, 50, f'Norm = {output["norm"]}', color=output['norm'])
        AS.text(5, 45, f'Amp = {output["amp"]}')
        box_off(AS)
        box_off(AT)
    # for i in output:
      #   print(f'{i}: {output[i]}')
    return(output)

    # print(f'{cold_responses=}')
    # print(f'{cold_responses=}')


def get_exp_days(dataset, start_event='SNI'):
    """
    dataset - dictionary containing lists of multisession objects, with cohort keys

    returns dataset input, adding attributes 'relative_time' and 'relative_day'
    """

    if start_event == 'SNI':
        event_times = SNItimes()
    elif start_event == 'CAPS':
        event_times = CAPStimes()

    # get SNI times for each mouse (multiple)
    # pdb.set_trace()
    for cohort in dataset:
        for m, multi_session in enumerate(dataset[cohort]):
            start_time = None
            source = str(dataset[cohort][m].sessions[0].Source)
            
            for mouse in event_times.keys():
                if mouse in source:
                    start_time = event_times[mouse]
                    break
            if start_time is None:
                start_times = []
                for sessions in dataset[cohort][m].sessions:
                    start_times.append(
                        dataset[cohort][m].sessions[0].experiment_start)
                start_times = np.array(start_times)
                start_time = time.localtime(np.amin(start_times))

            #pdb.set_trace()
            if hasattr(multi_session, 'pickle_path'):
                print(f'Mouse {mouse} is from pickle file {multi_session.pickle_path}')
            else:
                print(f'Mouse {mouse} from {source} needs path saved to object')
                
            multi_session.event_time = time.mktime(start_time)
            multi_session.event = start_event
            for s, session in enumerate(multi_session.sessions):
                session.relative_time = session.experiment_start - multi_session.event_time
                if session.relative_time < 0:
                    session.relative_time = 0
                session.relative_day = np.round(
                    session.relative_time/(60*60*24))
                if session.relative_day < 1:
                    session.relative_day = session.relative_time/(60*60*24)
                print(
                    f'Mouse {mouse} session {s} day is {session.relative_day}')

    return(dataset)

def plot_scale_bar(axes, bar_scale = 100, mag = 10, acq_binning = 2, downsampled = 2, pix_size = 5.04):
    
    if axes is None:
        F = plt.figure()
        axes = F.add_axes([0,0,1,1])
        axes.imshow(np.zeros([256,512]))
    microns_per_pixel = (acq_binning*pix_size*downsampled)/mag
    length = bar_scale/microns_per_pixel
    width  =  axes.get_xlim()[1] - axes.get_xlim()[0]
    height =  axes.get_ylim()[0] - axes.get_ylim()[1]
    axes.plot([width*0.05, width*0.05 + length], [height*0.95 ,height*0.95 ], color = 'w', linewidth = 3)   

def CAPStimes():
    CAPSdates = {}
    CAPSdates['7241'] = (2022, 6, 21, 11, 3, 33, -1, -1, -1)
    CAPSdates['7242'] = (2022, 6, 21, 12, 2, 39, -1, -1, -1)
    CAPSdates['7778'] = (2022, 5, 26, 14, 17, 51, -1, -1, -1)
    CAPSdates['8244'] = (2022, 6, 17, 10, 48, 3, -1, -1, -1)
    return(CAPSdates)
    
    
    
def SNItimes():
    SNIdates = {}
    SNIdates['237'] = (2021, 5, 6, 12, 0, 0, -1, -1, -1)
    SNIdates['414'] = (2021, 6, 4, 12, 0, 0, -1, -1, -1)
    SNIdates['573'] = (2021, 12, 14, 10, 33, 6, -1, -1, -1)
    SNIdates['594'] = (2021, 12, 14, 13, 27, 8, -1, -1, -1)
    SNIdates['457'] = (2021, 12, 14, 14, 27, 11, -1, -1, -1)
    SNIdates['379'] = (2021, 12, 14, 11, 20, 43, -1, -1, -1)
    SNIdates['6356'] = (2022, 2, 8, 12, 0, 0, -1, -1, -1)
    SNIdates['6046'] = (2022, 2, 8, 10, 27, 0, -1, -1, -1)
    SNIdates['704'] = (2020, 12, 10, 13, 50, 0, -1, -1, -1)
    SNIdates['6355'] = (2022, 7, 26, 12, 0, 0, -1, -1, -1)
    SNIdates['7241'] = (2022, 7, 26, 13, 31, 0, -1, -1, -1)
    SNIdates['7243'] = (2022, 7, 26, 14, 3, 0, -1, -1, -1)

    return(SNIdates)
