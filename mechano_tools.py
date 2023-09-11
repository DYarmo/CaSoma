#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:15:26 2023

@author: ch184656
"""

import numpy as np
import pickle
from scipy import stats
from matplotlib import pyplot as plt
import copy

def ix(array):
    lin = list(np.linspace(0, len(array)-1, len(array)).astype(np.uint32))
    return(lin)

def unpickle(path):
    file = open(path, 'rb')
    out = pickle.load(file)
    file.close()
    return(out)

def get_m_sample_data(file = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Mechano scoring/6356 LA selectedLA longitudinal all dff.pickle'):
    m = unpickle(file)
    return(m)

def rank_m_stims(Mstims):
    intensities = get_m_intensities(Mstims)
    ix = np.argsort(intensities)
    return(ix)

def get_m_intensities(Mstims, sort = False):
    intensities = []
    for stim in Mstims:
        intensities.append(Mstims[stim].intensity)
    if sort:
        intensities.sort()
    return(intensities)

def rank_m_responses(r):
    activities = []
    for c in range(r.shape[1]):
        activities.append(-1*np.sum(r[:,c]))
    ix = np.argsort(activities)
    return(ix)

def get_m_response(session, stim_num, cell_num, plot=False, time_limit = 50):
    
    ## get trace of cells activity from stim start until either time limit, next stim, or end of experiment
    n_total_stims = len(session.Mstim)-1
    trace = session.cells[cell_num].trace
    stim_start = session.Mstim[stim_num].start
    stim_end = session.Mstim[stim_num].end
    stim_duration = stim_end-stim_start
    if stim_num == n_total_stims:
        end = session.mech_stim.shape[0]
    else:
        end = session.Mstim[stim_num+1].start
    if end > stim_end + time_limit:
        end = stim_end + time_limit
    
    response_trace = session.cells[cell_num].trace[stim_start:end]
    
    if stim_num == 0:
        baseline_start = 0
    else:
        baseline_start = session.Mstim[stim_num-1].end
        
    if baseline_start < stim_start-stim_duration:
        baseline_start = stim_start-stim_duration
        
    # baseline_trace = session.cells[cell_num].trace[baseline_start:stim_start]
    
    # #q = stats.median_abs_deviation(baseline_trace)/0.79788456
    # baseline_mean = np.mean(baseline_trace)
    # baseline_q = stats.median_abs_deviation(baseline_trace)/0.79788456
    
    
    baseline_q = stats.median_abs_deviation(session.cells[cell_num].trace)/0.79788456
    response_max = np.amax(response_trace)/baseline_q
    if plot:
        F = plt.figure(f'Cell {cell_num} response to {stim_num} is {response_max}')
        A = F.add_axes([0,0,1,1])
        r = session.cells[cell_num].trace[baseline_start:end]
        A.plot(ix(r),r, 'r')
        A.set_ylim([0, 0.1])
        B = A.twinx()
        m = session.mech_stim[baseline_start:end]
        B.plot(ix(m), m, 'k')
    return(response_max)


def get_all_m_responses_for_cell(session, cell_num, plot=False, time_limit = 50):
    responses = np.zeros([len(session.Mstim)])
    for i, stim in enumerate(session.Mstim):
        responses.append(get_m_response(session, stim, cell_num, plot=plot, time_limit = time_limit))
    return(responses)

def filter_m_stims(stims, force_limits = [0, np.inf]):
    filtered = {}
    counter = 0
    for stim in stims:
        if stims[stim].intensity >= force_limits[0] and stims[stim].intensity<=force_limits[1]:
            filtered[counter] = stims[stim]
            counter = counter+1
            
    return(filtered)
        

def get_m_responses_for_session(session, plot = False, time_limit = 50, force_limits = [0, np.inf], sort_stim = False, sort_response = True, vmin = 2, vmax = 10):
    session = copy.deepcopy(session)
    session.Mstim = filter_m_stims(session.Mstim, force_limits=force_limits)
    responses = np.zeros([len(session.Mstim), len(session.cells)])
    for i, stim in enumerate(session.Mstim):
        for j, cell in enumerate(session.cells):
            responses[i,j] = get_m_response(session, stim, cell, plot=False, time_limit=time_limit)
    if sort_stim:
        responses = responses[rank_m_stims(session.Mstim)]
        
    response_ranks = rank_m_responses(responses)
    responses = responses.T
    if sort_response:  
        responses = responses[response_ranks]      

    #intensities = rank_m_stims(session.Mstim)
    M = get_m_intensities(session.Mstim, sort = sort_stim)
    if plot:
        F= plt.figure()
        A = F.add_axes([0.1,0.1,0.8, 0.65])
        plt.imshow(responses, aspect = 'auto', cmap = 'jet', vmin = vmin, vmax = vmax)
        box_off(A, All=True)
        
        B = F.add_axes([0.1,0.75,0.8,0.15])
        
        B.set_xlim(0, len(M))
        B.plot(M, color = 'm')
        B.set_ylabel('Force(mN)')
        box_off(B, left_only = True)
        
    return(responses, M, response_ranks)
        
def get_m_response_multi_session(m, selected_sessions = 'All', selected_u_cells = 'All', cmap='jet',plot=True, time_limit=50, force_limits = None, key_sess = 0, vmin = 2, vmax = 10, log_scale=True):
    
    if selected_u_cells == 'All':
        selected_u_cells = []
        for c, u in enumerate(m.assignments):
            selected_u_cells.append(c)
    if selected_sessions == 'All':
        selected_sessions = m.all_sessions()
    
    n_sess = len(selected_sessions)
    
    if force_limits is None:
        max_intensities = []
        for s, session in enumerate(selected_sessions):
            intensities = get_m_intensities(m.sessions[session].Mstim, sort = False)
            max_intensities.append(max(intensities))
        force_limits = [0, min(max_intensities)]
    
    
    rasters = []
    stims = []
    
    for s, session in enumerate(selected_sessions):
        raster, M, rank = get_m_responses_for_session(m.sessions[session], time_limit=time_limit, plot=False, force_limits= force_limits, sort_stim = True, sort_response = False)
        rasters.append(raster)
        stims.append(M)
        if s == key_sess:
            cell_ordering = rank
    print(f'{cell_ordering=}')
    h_margin = 0.1
    v_margin = 0.1
    plot_width = 1- 2*h_margin
    plot_height =1-2*v_margin
    ax_w = plot_width/n_sess
    r_h = plot_height *0.75
    s_h = plot_height * 0.25
    
    if plot:
        F = plt.figure()
        for s, (raster, stim) in enumerate(zip(rasters, stims)):
            A = F.add_axes([h_margin+(s*ax_w), v_margin+s_h, ax_w*0.9, r_h])
            A.imshow(raster[cell_ordering], aspect = 'auto', cmap = cmap, interpolation='none', vmin = vmin, vmax=vmax)
            B = F.add_axes([h_margin+(s*ax_w), v_margin, ax_w*0.9, s_h])
            B.plot(stim, 'm', marker = '.', linestyle='')
            
            box_off(A, All=True)
            B.set_xlim(0, len(stim))
            B.set_ylim(force_limits)
            if log_scale:
                B.set_yscale('log')
            if not(s):
                B.set_ylabel('Force(mN)')
                box_off(B, left_only = True)
            else:
                box_off(B, All = True)
    
        


def box_off(A, left_only = False, All = False):
    A.spines.right.set_visible(False)
    A.spines.top.set_visible(False)
    if left_only:
        A.spines.bottom.set_visible(False)
        A.xaxis.set_visible(False)
    if All:
        A.spines.left.set_visible(False)
        A.spines.bottom.set_visible(False)
        A.xaxis.set_visible(False)
        A.yaxis.set_visible(False)
    return(A)