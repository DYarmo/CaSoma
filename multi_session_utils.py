#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:21:49 2022

@author: ch184656
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from scipy.signal import resample
#from libAnalysis import toInt
import libAnalysis as la
#from libAnalysis import CaimanROI2dyROI
import copy
import pdb
from DY_plot_contours import plot_contours

def CaimanROI2dyROI(A, H,W,n):
    if n is None:
        n = A.shape[1]
    out = np.reshape(A, [H,W,n], 'F')
    return(out)

def extract_across_sessions(m, selected_u_cells='All', selected_sessions='All', detrend=False, wavelength=2400, dff_z_score = True):
    #### calculate traces from combined ROIs
    ## Input m is multi-session object from libAnalysis
    if selected_u_cells == 'All':
        selected_u_cells = []
        for c, u in enumerate(m.assignments):
            selected_u_cells.append(c)
    if selected_sessions == 'All':
        selected_sessions = m.all_sessions()
    
   
    m.trace_struct = {}
        
    for session in selected_sessions:
        t = m.sessions[session].raster.shape[1]  ## # of time steps in original traces
        
        h = m.sessions[0].fieldImage.shape[0]
        w = m.sessions[1].fieldImage.shape[1]
        n = len(selected_u_cells)
        u_caiman_masks = m.union_ROIs[:, selected_u_cells]
        u_masks = np.reshape(u_caiman_masks, [h,w,n], 'F')
        u_cnmf_extraction = m.sessions[session].calc_trace_from_mask(u_masks, method = 'cnmf', detrend=detrend)
        num_cells_extracted = u_cnmf_extraction['traces_out'].shape[0]
        #pdb.set_trace()
        print
        if n != num_cells_extracted:
            print('Could not extract all input ROIS')
            return
        u_dff_extraction = m.sessions[session].calc_trace_from_mask(u_masks, method = 'dff', dff_z_score = dff_z_score)
        
     
        for c, u_cell in enumerate(selected_u_cells):
            IX = to_int(m.assignments[u_cell, session])
            m.trace_struct[(u_cell, session)] = {}
            cnmf_trace = resample(u_cnmf_extraction['traces_out'][c,:], t)
            dff_trace = resample(u_dff_extraction['traces_out'][c,:], t)
            
            if detrend:#resample:
                cnmf_trace = la.detrend(cnmf_trace, wavelength, plot=False)
                dff_trace = la.detrend(dff_trace, wavelength, plot = False)
            
            m.trace_struct[(u_cell, session)]['union cnmf'] = cnmf_trace
            m.trace_struct[(u_cell, session)]['union dff']  = dff_trace
            m.trace_struct[(u_cell, session)]['re-extracted mask'] = u_cnmf_extraction['masks'][:,:,c]
            #m.trace_struct[(u_cell, session)]['union cnmf'] = resample(u_cnmf_extraction['traces_out'][c,:], t)
            #m.trace_struct[(u_cell, session)]['union dff']  = resample(u_dff_extraction['traces_out'][c,:], t)
            if np.isnan(IX):
                m.trace_struct[(u_cell, session)]['session trace'] = None
            else:
                m.trace_struct[(u_cell, session)]['session trace'] = m.sessions[session].cells[IX].trace

    
    #m.show_u_traces()
    return(m.trace_struct)

def plot_tuning_across_sessions(m, selected_u_cells='All', selected_sessions='All', missing_mode = 'dff', show_stim = True, stim_span = 'Auto', show_field = False):
    ####
    
    if selected_u_cells == 'All':
        selected_u_cells = []
        for c, u in enumerate(m.assignments):
            selected_u_cells.append(c)
    if selected_sessions == 'All':
        selected_sessions = m.all_sessions()
        
    if not hasattr(m, 'trace_struct'):
        print('You need to extract traces!')
        return()
        #print('Extracting traces...')
        #m.trace_struct = extract_across_sessions(m, selected_u_cells='All', selected_sessions='All')   
    
    if stim_span == 'Auto':
        if len(selected_u_cells) == 1:
            stim_span = 1
        elif len(selected_u_cells) < 11:
            stim_span = 2
        else:
            stim_span = round(len(selected_u_cells)/6)
    
    F = plt.figure()
    stim_axes = {}
    trace_axes={}
    
    field_plot = 0
    if show_field:
        field_plot = 1
    
    n_rows = len(selected_u_cells)+stim_span
    n_cols = len(selected_sessions) +field_plot


    trace_mins = {}
    trace_maxs = {}
  
    #stim_mins = {}
    
    for sc, sess in enumerate(selected_sessions):
        stim_axes[sess] = F.add_subplot(int(n_rows/stim_span), n_cols, sc+1)
        
        for c, u_cell in enumerate(selected_u_cells):
            if sc == 0:
                trace_mins[u_cell] = []
                trace_maxs[u_cell] = []
                
            if m.trace_struct[(u_cell, sess)]['session trace'] is None:
                if missing_mode =='cnmf':
                    trace = m.trace_struct[(u_cell, sess)]['union cnmf']
                elif missing_mode =='dff':
                    trace = m.trace_struct[(u_cell, sess)]['union dff']
            else:
                trace = m.trace_struct[(u_cell, sess)]['session trace']
                
            trace_mins[u_cell].append(np.amin(trace))
            trace_maxs[u_cell].append(np.amax(trace))
            
            pos = (n_cols*(c+stim_span)) + sc + 1

            trace_axes[(sess, u_cell)] = F.add_subplot(n_rows, n_cols, pos)
        
       
    
    for sc, session in enumerate(selected_sessions):
        # stim_num = 0
        thermStim = m.sessions[session].therm_stim
        mechStim = m.sessions[session].mech_stim
        # if not thermStim is None:
        #     stim_num = stim_num + 1
        # if not mechStim is None:
        #     stim_num = stim_num + 1
        for c, u_cell in enumerate(selected_u_cells):
            if m.trace_struct[(u_cell, session)]['session trace'] is None:
                trace = m.trace_struct[(u_cell, session)]['union cnmf']
            else:
                trace = m.trace_struct[(u_cell, session)]['session trace']
            parent_data = m.sessions[session].cells[0].parent_data
            v_cell = la.cell(trace = trace, ROI = [], classification = None, parent_data=parent_data, parent_session= m.sessions[session], thermStim = thermStim, mechStim = mechStim)
            trace_span =  np.amax(trace_maxs[u_cell]) - np.amin(trace_mins[u_cell])
            trace_y_lim = (np.amin(trace_mins[u_cell])-trace_span/20, np.amax(trace_maxs[u_cell])+trace_span/20)
            #print(f'{trace_y_lim=} for cell {u_cell}')
            if sc==0 and c == 0:
                xbar = True
                ybar = True
                show_stim_y = True
            elif sc==0:
                xbar = False
                ybar = True
                show_stim_y = True
            else:
                xbar = False
                ybar = False
                show_stim_y = False
            #print(f'{xbar=} and {ybar=} for cell {u_cell} in session {session}, {c=} {sc=}')
            v_cell.show(F = F, A1 = trace_axes[(session, u_cell)], A2 = stim_axes[session], trace_color = m.sess_color_series[session], show_y = False, show_stim_y = show_stim_y,  show_time = False, show_trace_y_label = False, trace_y_lim = trace_y_lim, norm=False, xbar=xbar, ybar = ybar, disp_transients=False)
    if field_plot:
        FA = F.add_subplot(int(1, n_cols, n_cols))

        

  

def create_u_cells(n, selected_u_cells='All', missing_mode = 'dff', selected_sessions='All'):
    m = copy.deepcopy(n)
    
    H = m.sessions[0].fieldImage.shape[0]
    W = m.sessions[0].fieldImage.shape[1]
    union_ROIs = CaimanROI2dyROI(m.union_ROIs, H,W, None)
    if selected_u_cells == 'All':
        selected_u_cells = []
        for c, u in enumerate(m.assignments):
            selected_u_cells.append(c)
    if selected_sessions == 'All':
        selected_sessions = m.all_sessions()
    
    if not hasattr(m, 'trace_struct'):
        print('Extracting traces...')
        m.trace_struct = extract_across_sessions(m, selected_u_cells=selected_u_cells, selected_sessions=selected_sessions) 
    
    #pdb.set_trace()
    for sc, session in enumerate(selected_sessions):
        m.sessions[session].ROIs = np.zeros([H,W,len(selected_u_cells)])
        sess_v_cells = []
        thermStim = m.sessions[session].therm_stim
        mechStim = m.sessions[session].mech_stim
        for c, u_cell in enumerate(selected_u_cells):
             if m.trace_struct[(u_cell, session)]['session trace'] is None:
                 if missing_mode == 'dff':
                     trace = m.trace_struct[(u_cell, session)]['union dff']
                 elif missing_mode == 'cnmf':
                     trace = m.trace_struct[(u_cell, session)]['union cnmf']
             else:
                 trace = m.trace_struct[(u_cell, session)]['session trace']
             parent_data = m.sessions[session].cells[0].parent_data
             
             orig_cell = m.assignments[u_cell, session]
             print(f'Session {session} u_cell {u_cell} orig cell is {orig_cell}')
             
             
             if not np.isnan(orig_cell):
 
                print(f'Using original ROI for u_cell {u_cell} in session {session}')
                ROI = n.sessions[session].ROIs[:,:,int(orig_cell)]
                
             else:
              #  print(f'Using session ROI for u_cell {u_cell} in session {session}')
              #  ROI = union_ROIs[:,:,u_cell]

                print(f'Using re-extracted ROI for u_cell {u_cell} in session {session}')
                ROI = m.trace_struct[(u_cell, session)]['re-extracted mask']
             
             m.sessions[session].ROIs[:,:,c] = ROI
             sess_v_cells.append(la.cell(trace = trace, ROI = ROI, classification = None, parent_data=parent_data, parent_session= m.sessions[session], thermStim = thermStim, mechStim = mechStim))
        m.sessions[sc].cells = {}
        raster = np.zeros([len(sess_v_cells), m.sessions[sc].raster.shape[1]])
        m.sessions[sc].ROIs = np.zeros([H, W, len(sess_v_cells)])
        for c, cell in enumerate(sess_v_cells):
            m.sessions[sc].add_cell(cell)
            raster[c,:] = cell.trace
            m.sessions[sc].ROIs[:,:,c] =cell.ROI
        m.sessions[sc].raster = raster
        
        #pdb.set_trace()
        #print(f'Session count {sc} session {session}')
        #m.sessions[sc].classify_thermo_cells()
        #m.union_cells = True
    new_assignments = m.assignments*0
    lin = np.arange(m.assignments.shape[0])
    for col in range(new_assignments.shape[1]):
        new_assignments[:, col] = lin
    m.assignments = new_assignments
    return(m)
                
def to_int(text):
    text=str(text)
    if text.isnumeric(): ## 
        return(int(text))
    elif text.split('.')[0].isnumeric(): ### take whole part of decimal as int
        return(int(text.split('.')[0]))
    else:
        return(np.nan) 