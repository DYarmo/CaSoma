
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:49:01 2022

@author: ch184656
"""
import DYpreProcessingLibraryV2 as pp
import libAnalysis as la
from libAnalysis import get_sample_data, box_off, get_exp_days, SNItimes
import numpy as np
import pdb
import matplotlib
#matplotlib.use('Agg')
from Alignment_GUI import Alignment_GUI
import time
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import scipy
from scipy import stats
import pandas
import pdb
import os
import thermalRingLib as tr
import itertools
import behaviorLib as beh
import printcolor as printc
from sklearn.cluster import KMeans
import uuid
import h5py
from beeswarm import beeswarm, beeswarms
import pickle
from statistics import mode
import copy
#####

def main_figures():
    start_t = time.time()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    all_mech_responses = compile_mech_range()
    SNI_raster_panel()
    CAPS_raster_panel()
    SNI_mech_data = SNI_mech(from_pickle=False)
    stats = SNI_mech_stats(SNI_mech_data)
    SNI_mech_recruitment_plot(stats)
    base_therm_sessions, bas_therm_results = rasters_by_genotype()
    
    
def save_fig(Fig, fig_name, ext = '.pdf'):
    Fig.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' + fig_name + ext)

def save_fig_dict(d):
    for key in d.keys():
        save_fig(d[key], key, ext = '.pdf')
        save_fig(d[key], key, ext = '.png')
        
# def get_sample_data():
#     data = pp.unpickle('/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 237/Mouse 237 experimental 2LA longitudinal.pickle')
#     return(data)
#######

def fig_approach(foo=5, bar = 'asf'):
    print(locals())




def SNI_behavior(x_file = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/SNI Behavioral Data_Master Sheet.xlsx', group_tags = None, group_colors = None):
    
    #x_file = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Thermal Ring.xlsx'
    b = pandas.read_excel(x_file, sheet_name=None)
    assays = []
    n_mice = len(b)
    A = {}
    if group_colors == None:
        plot_color = None
        
    for c, mouse in enumerate(b.keys()):
        b[mouse] = b[mouse].set_index([b[mouse]['Day']])
        for assay in b[mouse].keys():
            check_assays = assays.copy()
            check_assays.extend(['Date', 'Day'])
            if not (assay in check_assays):
                assays.append(assay)
                F = plt.figure(f'{assay}')
                F.suptitle(f'{assay}')
                A[assay] = F.add_axes([0,0,1,1])
                
            if not assay in ['Date','Day']:
                #plot_color = 'k'
                if not group_tags is None:
                    for d, tag in enumerate(group_tags):
                        if tag in mouse:
                            print(mouse)
                            plot_color = group_colors[d]
                
                
                A[assay].plot(b[mouse]['Day'], b[mouse][assay].interpolate(method='values'), color=plot_color, marker = 'o')
                A[assay].set_frame_on(False)
    
    assay_data = {}           
    
        
 
    return(b)
    
def thermal_ring(tags = ['A','B'], conditions = ['SNI','SHAM']):  ## work in progress
    data_key = {}
    for tag, condition in zip(tags, conditions):
        data_key[condition] = tag
        
    raw_data = []
    days = []
    x_folder = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Thermal Ring raw data'
    for currentPath, folders, files in os.walk(x_folder):
        for x_file in files:
            print(x_file)
            if '.csv' in x_file:
                days.append(x_file.split('.')[0])
                raw_data.append(pandas.read_csv(os.path.join(x_folder, x_file)))
    
    
    
    return(raw_data, days)
    
def SNI_thermal_ring():
    a=tr.load_trg_folder(tr.default_exp_folder(0))
    Treatments = [['A baseline', 'B baseline'],
                  ['A day 3', 'B day 3'],
                  ['A day 7', 'B day 7'],
                  ['A day 14', 'B day 14'],
                  ['A day 28', 'B day 28']]
    #bigF = plt.figure()
    
    speed_data={}
    pref_data={}
    for treatment in Treatments:
        sub = a.pull(Treatment = treatment)
        sub.show_grouped('Treatment', show_err = 1, separate = False)
        for condition in treatment:
            sub_sub = sub.pull(Treatment = [condition])
            speed_data[condition] = []
            pref_data[condition] = []
            for trial, x in enumerate(sub_sub.trials):
                speed_data[condition].append(sub_sub.trials[trial].__dict__['Mean speed'])
                pref_data[condition].append(sub_sub.trials[trial].__dict__['preferred temp'])

                
    F= plt.figure()
    A = F.add_axes([0,0,1,1])
    xloc = []
    for count, key in enumerate(speed_data.keys()):
        xloc.append(count)
        y_data = np.array(speed_data[key])
        x_data = (y_data*0)+count
        A.scatter(x_data,y_data)
    A.set_xticks(xloc)
    A.set_xticklabels(speed_data.keys(), rotation=45)
    A.spines.right.set_visible(False)
    A.spines.top.set_visible(False)
    
    FF= plt.figure()
    A = FF.add_axes([0,0,1,1])
    xloc = []
    for count, key in enumerate(pref_data.keys()):
        xloc.append(count)
        y_data = np.array(pref_data[key])
        x_data = (y_data*0)+count
        A.scatter(x_data,y_data, alpha = 0.25)
    A.set_xticks(xloc)
    A.set_xticklabels(pref_data.keys(), rotation=45)
    A.spines.right.set_visible(False)
    A.spines.top.set_visible(False)
    return(a)
        
                
            
            
            
        
    
    
###########
def fig_basal_thermo_tuning():
    pp.thermal_tuning()

def fig_therm_response_diversity(mouse = '8244'):
    a = pp.get_sessions([mouse], FOVtag = 'Use for basal')[0]
    Fig = a.show_plots_groups(class_filters = ['cold','warm','hot', 'poly'], showMech=True, show_mean = True, stack =True, norm=True)[0]
    save_fig(Fig, f'Response diversity for {mouse}')

def get_example_sesh(mouse = 7242, FOVtag = 'Use for basal', session = 0):
    a = pp.get_sessions([mouse], FOVtag = FOVtag)[session]

    return(a)


def rasters_by_genotype(genotypes = None, Tseries = None, reps=[1], n_reps=1, nClusters=3, plot=True, cmap='viridis', end_fix_duration = 80, prepend_frames = 60, append_frames=80, min_f=0, max_f = 0.5, fontsize=10, z_scored = False, z_score=False): #zscore is for doing proper trace conversion at source
    if genotypes is None:
        genotypes = ['rPbN', 'Tacr1', 'Gpr83']
        
    if Tseries is None:
        Tseries = [(3,12), (19,24), (37,40), (41,43), (43,46), (46,48), (48,52)]
    if type(Tseries) is str:
        Tseries = la.read_T_series(Tseries)
    # if reps is None:
    #     reps = []
    #     for item in Tseries:
    #         reps.append(n_reps)
    
    if z_score or z_scored:
        min_f = 0
        max_f = 20
    
    results = {}
    sessions = {}
    labels={}
    """
    Get sets of session objects, generate aligned rasters (and get errors)
    """
    
    for genotype in genotypes:
        results[genotype] = {}
        sessions[genotype], results[genotype]['Empty dbs'] = pp.get_sessions(None, pop_filter = [genotype], FOVtag = 'Use for basal', data_folder = 'Basal thermo', return_failures = True, z_score=z_score)
        results[genotype]['r'], results[genotype]['s'], results[genotype]['p'] = la.align_thermo_rasters(sessions[genotype], nClusters = nClusters, end_fix_duration=end_fix_duration, prepend_frames = prepend_frames, append_frames=append_frames, Tseries=Tseries, reps=reps, z_scored = z_scored)
    
    """
    Assemble rasters together
    """
    
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
        nClusters, o, F_opt_clust = la.get_kmeans_clust_num(combined_raster, repeats=50)
        save_fig(F_opt_clust, 'K means optimization')
    
    colors = [[1,0.5,0],[0,1,1], [1,0,1],[1,1,0],[0,0,1]]
    colors = colors[0:nClusters]
    """
    Cluster traces and sort raster by cluster identity:
    """
    
    newIX, newLabels = la.sortByKmean(combined_raster, nClusters, return_labels=True)
    
    
    
    combined_raster = combined_raster[newIX,:]
    combined_rasters_sources  = np.array(combined_rasters_sources)
    combined_rasters_sources = combined_rasters_sources[(newIX)]
    
    results['combined_raster'] = combined_raster
    results['ix'] = newIX
    results['labels'] = newLabels
 
    for c, genotype in enumerate(genotypes):
        results[genotype]['r'] = combined_raster[combined_rasters_sources==c]
        labels[genotype] = newLabels[combined_rasters_sources==c]
        
  
        
    """
    Plot results:
    """
    
    
    if plot:
        P = plt.figure('Traces')
        PS = P.add_subplot(len(genotypes)+1,1, 1)
        F = plt.figure()
        
        r_top = 0.8 ## top of raster plotting area
        r_bot = 0.1 ## bottom of raster plotting area
        
        A_stim = F.add_axes([0.1, r_top, 0.9, 1-r_top])
        
        raster_ax_top = r_top
        A_rasters={}
        P_traces = {}
        for c, genotype in enumerate(genotypes):
            stim = results[genotype]['s']
            A_stim.plot(stim, color = la.gen_colors()[genotype], alpha=0.1)
            PS.plot(stim, color = la.gen_colors()[genotype], alpha=0.1)
            P_traces[c] = P.add_subplot(len(genotypes)+1, 1 ,c+2)
            
            
            raster = results[genotype]['r']
            n = raster.shape[0]
            frac = n/ncells
            height = (r_top-r_bot) * frac
            ax_top= raster_ax_top
            raster_ax_top = raster_ax_top - height
            ax_bot = ax_top - height
            A_rasters[c] = F.add_axes([0.1, ax_bot, 0.9, height-0.01])
            A_rasters[(c, 'twin')] = A_rasters[c].twinx()
            A_rasters[c].imshow(raster, aspect='auto', vmin=min_f, vmax=max_f, cmap=cmap)
            for trace, label in zip(raster.T, labels[genotype]):
                P_traces[c].plot(raster.T, color = colors[label])
            A_rasters[(c, 'twin')].set_ylim(A_rasters[c].get_ylim())
            A_rasters[(c, 'twin')].set_yticks([0, raster.shape[0]])
            A_rasters[(c, 'twin')].set_yticklabels(['', str(raster.shape[0])], fontsize=fontsize)
            
            A_rasters[c].set_yticks([])
            A_rasters[c].spines.bottom.set_visible(False)
            A_rasters[c].set_xticks([])
            A_rasters[c].set_ylabel(f'{genotype}', color = la.gen_colors()[genotype], fontsize=fontsize)
            
  
        A_stim.set_xlim([0, results['combined_stims'].shape[0]])
        A_stim.spines.top.set_visible(False)
        A_stim.spines.right.set_visible(False)
        A_stim.spines.bottom.set_visible(False)
        A_stim.set_xticks([])
        A_stim.set_yticks([0,25,50])
        A_stim.set_ylabel(f'Temp ({tr.degC()})', fontsize=fontsize)
    
        A_tscale = F.add_axes([0.1, 0, 0.9, r_bot])
        A_tscale.set_xlim(A_stim.get_xlim())
        A_tscale.set_ylim([-10,2])
        timebar = 10
        ts = sessions[genotypes[0]][0].timestep
        A_tscale.plot([0,timebar/ts], [0, 0], color='k', linewidth = 2)
        A_tscale.text(0,-6, f'{timebar} sec', fontsize = fontsize) 
        A_tscale.spines.top.set_visible(False)
        A_tscale.spines.left.set_visible(False)
        A_tscale.spines.right.set_visible(False)
        A_tscale.spines.bottom.set_visible(False)
        A_tscale.set_xticks([])
        A_tscale.set_yticks([])
        
        A_color_scale = F.add_axes([0.8, 0.005+r_bot/2, 0.2, r_bot/3])
        A_color_scale.imshow(np.expand_dims(np.linspace(min_f, max_f, 256), axis=0), cmap=cmap, aspect='auto')
        A_color_scale.spines.top.set_visible(False)
        A_color_scale.spines.left.set_visible(False)
        A_color_scale.spines.right.set_visible(False)
        A_color_scale.spines.bottom.set_visible(False)
        A_color_scale.set_xticks([])
        A_color_scale.set_yticks([])
        A_color_scale.text(0, 1.7, f'{min_f}', fontsize = fontsize)
        A_color_scale.text(256, 1.7, f'{max_f}', fontsize = fontsize, horizontalalignment='right') 
        if z_scored:
            A_color_scale.set_xlabel("Zf")
        else:
            A_color_scale.set_xlabel("F'")
        
       
     
            
    for genotype in genotypes:
        
        for failure in results[genotype]['p']['errors']:
            print(f'DB {os.path.split(failure[0].source)[-1]} failed due to missing stim(s): {failure[1]}')
        for empty in results[genotype]['Empty dbs']:
            print(f'No data obtained from DB {os.path.split(empty)[-1]}')
    
    results['combined_raster'] = combined_raster
    if plot:
        save_fig(F, f'Rasters by genotype {cmap} {z_scored=} n_groups = {nClusters}', ext = '.png')
        save_fig(F, f'Rasters by genotype {cmap} {z_scored=} n_groups = {nClusters}')
    
    
    
        la.show_PCs(results, key='rPbN', n_PCs = 8, cut_off = nClusters-1)
        centers, stim = la.show_cluster_centers(results, key = 'rPbN', n_clusters = nClusters)
    
    return(sessions, results)


def color_bar(cmap, orientation  = 'horizontal', color = 'k', legend = None, unit = 'F', fontsize = 12, F = None, A = None, min_v=0, max_v=255, save = True):
    if F is None:
        F = plt.figure()
    
    if A is None:
        if orientation == 'horizontal':
            A = F.add_axes([0,0,1,0.1])
        else:
            A = F.add_axes([0,0,0.1,1])
    span = max_v-min_v
    data = np.expand_dims(np.linspace(min_v, max_v, span), axis=0)
    if orientation == 'vertical':
        data = np.flipud(data.T)
    A.imshow(data, cmap=cmap, aspect='auto')
    A.spines.top.set_visible(False)
    A.spines.left.set_visible(False)
    A.spines.right.set_visible(False)
    A.spines.bottom.set_visible(False)
    A.set_xticks([])
    A.set_yticks([])
    if legend is None:
        legend = [str(min_v), str(max_v)]
    if orientation == 'horizontal':
        A.text(0, 0.75, legend[0], fontsize = fontsize, verticalalignment='top', horizontalalignment='left', color=color)
        A.text(span, 0.75, legend[1], fontsize = fontsize, verticalalignment='top', horizontalalignment='right', color=color) 
        A.text(span/2, 0.75, unit, fontsize = fontsize, verticalalignment='top', horizontalalignment='center', color=color)
    elif orientation == 'vertical':
        A.text(0.6, span, legend[0], fontsize = fontsize, verticalalignment='bottom', horizontalalignment='left', color=color)
        A.text(0.6, 0, legend[1], fontsize = fontsize, verticalalignment='top', horizontalalignment='left', color=color) 
        A.text(0, span, unit, fontsize = fontsize, verticalalignment='top', horizontalalignment='center', color=color)
    if not save is None:
        save_fig(F, f'{cmap} color_bar {min_v} {max_v} {unit} {color}')
    #A.set_xlabel(unit)

def comp_resp_stats():
    pass

class fig_group():
    
    def __init__(self):
        self.figs = {}
        self.current_fig = None
        
    def add(self, name, size=None):
        self.figs[name] = plt.figure(name, figsize = size)
        self.F = self.figs[name]
    
    def save(self, ext = '.pdf'):
        for name in self.figs.keys():
            save_fig(self.figs[name], name, ext = ext)
        
def panel_heat_warm_stats(plot=True, z_score=False, norm=True):
    #sessions, results = rasters_by_genotype(genotypes=['rPbN'], Tseries = [(41,43),[47,49], (49.1,52)], reps = [2,2,2])
    #centers, stim = la.show_cluster_centers(results, key = 'rPbN', n_clusters = 4)
    #pdb.set_trace()
    sessions,results= rasters_by_genotype(genotypes=['rPbN'], z_scored=False, append_frames=100, prepend_frames=63, plot=False, z_score=True)
    raster = results['rPbN']['r']
    cells = results['rPbN']['p']['cell_list']
  
    IX = results['ix']
    labels = results['labels']
    
  
    F=plt.figure('raster')
    plt.imshow(raster, aspect='auto')
    Fs=fig_group()
    Fs.add('PbN hot-warm raster')
    F = Fs.F
    clust_cells= {}
    classes = {}
    
    # for c, (row, ix) in enumerate(zip(raster, IX)):
    #     if c>300 and c <320:
    #         A = F.add_subplot(20,1,c-300)
    #         plt.plot(row, 'r')
    #         A2 = A.twinx()
    #         plt.plot(cells[ix].trace, 'b')
    #         print(f'{c=}, {cells[ix].classification} {labels[ix]}=')
            
    # pdb.set_trace()  
    
    for c, ix in enumerate(IX):
        label = labels[ix]
        CELL = cells[ix]
        if not label in clust_cells.keys():
            clust_cells[label] = []
            classes[label] = []
        clust_cells[label].append(CELL)
        classes[label].append(CELL.classification)
        
    for label in range(np.amax(labels)+1):
        name = mode(classes[label])
        clust_cells[name] = clust_cells[label]
        del clust_cells[label]
    #A=F.add_subplot(1,1,1)
    #A.imshow(raster)
    for group in clust_cells:
        for g in clust_cells[group]:
            g.cold_stims = []
            g.warm_stims = []
            g.hot_stims = []
            g.very_hot_stims = []
            g.pain_stims = []
            Tstim = g.parent_session.Tstim
            l=1000
            for c, T in enumerate(Tstim.values()):
                if T.stim_temp > 12:
                    if len(g.cold_stims) < 3:
                        g.cold_stims.append(T)
                if T.stim_temp >40.1 and T.stim_temp < 43.15:
                    if len(g.warm_stims) <3:
                        g.warm_stims.append(T)
                if T.stim_temp > 43.15 and T.stim_temp < 45.95:
                    if c-l < 3:
                        g.hot_stims.append(T)
                        l=c
                if T.stim_temp >46 and  T.stim_temp < 48.5:
                    g.very_hot_stims.append(T)
                if T.stim_temp > 48.5:
                    if T.stim_temp > 50:
                        g.pain_stims.append(T)
                    elif c >= len(Tstim)-1:
                        g.pain_stims.append(T)
                    elif Tstim[c+1].stim_temp > 48.5:
                        g.pain_stims.append(T)
    if plot:
        plot_thermo_kinetics(clust_cells)
        
    return(clust_cells) 

def plot_thermo_kinetics(cells, stim_types=None):
    Fs={}
    Fs[0] = plt.figure('Hysteresis - raw')
    Fs[1] = plt.figure('Hysteresis - norm')
    Fs[2] = plt.figure('Latency')
    Fs[3] = plt.figure('Duration')
    bounds = [(0,30), (0,1.05), (0,10), (0,12)]
    ## format data for hysteresis analsyis:
    if stim_types is None:
        stim_types = ['cold_stims','warm_stims','hot_stims','very_hot_stims','pain_stims']
        #stim_types = ['warm_stims','hot_stims','very_hot_stims','pain_stims']
        #stim_types = ['hot_stims', 'very_hot_stims']
    n_groups = len(cells)
    ns = len(stim_types)
    rawresponses = {}
    normresponses = {}
    durations = {}
    latencies = {}
    stim_intensities = {}
    for st, stim_type in enumerate(stim_types):
        rawresponses[stim_type] = {}
        normresponses[stim_type] = {}
        durations[stim_type] = {}
        latencies[stim_type] = {}
        stim_intensities[stim_type] = {}
        for g, group in enumerate(cells):
            rawresponses[stim_type][group] = {}
            normresponses[stim_type][group] = {}
            durations[stim_type][group] = {}
            latencies[stim_type][group] = {}
            for c, cell in enumerate(cells[group]):
                stims = getattr(cell, stim_type)
                for sn, stim in enumerate(stims):
                    
                    if sn > 2:
                        continue
                    if not sn in stim_intensities[stim_type]:
                        stim_intensities[stim_type][sn]=[]
                    stim_intensities[stim_type][sn].append(stim.stim_temp)
                    if sn not in rawresponses[stim_type][group]:
                        rawresponses[stim_type][group][sn] = []
                        normresponses[stim_type][group][sn] = []
                        durations[stim_type][group][sn] = []
                        latencies[stim_type][group][sn] = []
                        
                    rawresponses[stim_type][group][sn].append(cell.response(stim)['amplitude'])
                    normresponses[stim_type][group][sn].append(cell.response(stim)['normamplitude'])
                    if cell.response(stim)['amplitude'] >=3.5:
                        
                        
                        
                        durations[stim_type][group][sn].append(cell.response(stim)['z_duration'])
                        latencies[stim_type][group][sn].append(cell.response(stim)['z_latency'])
    #pdb.set_trace()
    data = [rawresponses, normresponses, durations, latencies]
    for f, responses in enumerate(data):
        for st, stim_type in enumerate(stim_types):
    
            A = Fs[f].add_subplot(1,ns,st+1)
           
    
            for g, group in enumerate(responses[stim_type]):
                color = pp.gen_colors()[group]
                for sn, response_list in enumerate(responses[stim_type][group]):
                    X = (np.array(responses[stim_type][group][sn]) *0)+(g+(sn/4))
                    R = np.array(responses[stim_type][group][sn])
                    
                    
                    #beeswarm(np.array(responses[stim_type][group][sn]), X, color=color, alpha=0.2, A=A)
                    if not len(X)==0:
                        A.bar(X[0],np.mean(R), width = 0.2, edgecolor=color, facecolor=color, alpha  = 1, zorder=0)
                        pp.jitter(X, R, color='k', alpha=0.05, scale=2, A=A, zorder=100)
                    
                    # if g ==0:
                        
                    #     S = np.array(stim_intensities[stim_type][sn])
                    #     XX = (S*0)+sn
                    #     B.bar(XX[0],np.mean(S), width = 0.2, edgecolor='k', facecolor="None", alpha  = 1)
                    #     pp.jitter(XX, S, color='k', alpha=0.05, scale=2, A=B)
            #A.set_ylim([0,1.05])
            # bounds = [0,1.05]
            # if g == 0:
            #     bounds = A.get_ylim()
            # else:
            A.set_ylim(bounds[f])
            # # B.set_ylim([30,50])
            if st == 0:
                box_off(A)
                # box_off(B, left_only = True)
            else:
                box_off(A, bot_only=True)
                # box_off(B, All = True)
    heat_warm_example_traces(cells)
    calc_therm_stats_II(cells)
    return(responses)

def heat_warm_example_traces(cells):
    Fs = {}
    ATs = {}
    ASs = {}
    traces = {}
    xbounds = {}
    for group in cells:
        
        if group == 'cold':
            continue
        traces[group] = {}
        for cell in cells[group]:
            source = str(cell.parent_session.source)
            if source not in Fs:
             #   if '3A12' in source:
                    Fs[source] = plt.figure(source+str(uuid.uuid4())[0:5])
                    ATs[source] = Fs[source].add_subplot(2,1,2)
                    ASs[source] = Fs[source].add_subplot(2,1,1, sharex=ATs[source])
                    ASs[source].plot(cell.thermStim)
                    xbounds[source] = [0,1]
                    xbounds[source][0] = cell.very_hot_stims[0].start-80
                    xbounds[source][1] = cell.very_hot_stims[-1].end+80
                    ASs[source].set_xlim(xbounds[source])
            #ATs[source].plot(cell.trace, color = pp.gen_colors()[group], alpha = 0.1)
            if source not in traces[group]:
                traces[group][source] = []
            traces[group][source].append(cell.trace)
    for group in traces:
        for source in traces[group]:
            #if '3A12' in source:
                array = np.vstack(traces[group][source])
                mean = np.mean(array, axis=0)
                #pdb.set_trace()
                ATs[source].plot(mean, color = pp.gen_colors()[group], alpha = 1)
                ATs[source].plot(array.T, color = pp.gen_colors()[group], alpha = 0.1)
                ATs[source].set_xlim(xbounds[source])
               #ATs[source].set_xlim([4100, 4900])
                #ATs[source].set_ylim([-3,20])
            
                box_off(ATs[source], left_only=True)
                box_off(ASs[source], left_only=True)
    for fig in Fs:
        save_fig(Fs[fig], source.split('/')[-1].split('.h5')[0] + str(uuid.uuid4())[0:5])
                
                
def calc_therm_stats_II(cells, response_threshold = 3.5, after=130, plot=False, STIM = 48, crit = 0.9):
    ## Calculate characteristics of cells in groups sorted by ke means clusters:
    ## 1. latency and temp of peak and onset in response to 1st 48 deg stim (if sup-threshold)
    peak_latencies = {}
    stim_at_peaks = {}
    onset_latencies = {}
    stim_at_onsets = {}
    ## 2. after stimulus discharge (time until reaches baseline)
    after_discharges = {}
    desensitizations = {}
    groups = ['warm', 'hot']
    for group in groups:
        peak_latencies[group] = []
        stim_at_peaks[group] = []
        after_discharges[group] = []
        onset_latencies[group] = []
        stim_at_onsets[group] = []
        desensitizations[group] = []
        for c, cell in enumerate(cells[group]):
          
            if STIM == 48:
                v = cell.very_hot_stims[0]
                if len(cell.very_hot_stims)>2:
                    hyst = [cell.very_hot_stims[0], cell.very_hot_stims[2]]
                else:
                    hyst=None
            elif STIM ==45:
                v = cell.hot_stims[0]
                hyst = [cell.hot_stims[0], cell.hot_stims[-1]]
            elif STIM == 51:
                v = cell.pain_stims[0]
                hyst = [cell.pain_stims[0], cell.pain_stims[-1]]
            stim_full_trace = copy.copy(v.parent)
            stim_full_trace[0:v.start] = 35
            stim_full_trace[v.end+after:] = 35
            
            stim_max = np.amax(stim_full_trace)
            stim_max_frame = np.where(stim_full_trace==stim_max)[0][0]
            #stim_max_frame = np.where(stim_full_trace>(crit*stim_max))[0][0]
            
            trace = copy.copy(cell.trace)
            trace[0:v.start] = 0
            trace[v.end+after:] = 0
            
            peak_amplitude = np.amax(trace)
            if not (hyst is None):
                r1 = cell.response(hyst[0])['amplitude']
                r2 = cell.response(hyst[1])['amplitude']
                desensitization = (r1-r2)/r1
            
            
            if peak_amplitude >= response_threshold:
                peak_frame = np.where(trace==peak_amplitude)[0][0]
                peak_frame = np.where(trace> (crit*peak_amplitude))[0][0]
                
               
                
                if peak_frame > stim_max_frame:
                    stim_at_peak = stim_max
                else:
                    stim_at_peak = v.parent[peak_frame]
                peak_latency= peak_frame - v.start
                
                
                onset_frame = np.where(trace> response_threshold)[0][0]
                if onset_frame > stim_max_frame:
                    stim_at_onset = stim_max
                else:
                    stim_at_onset = v.parent[onset_frame]
                onset_latency = onset_frame - v.start
                
                if stim_at_onset <36:
                    continue
                
                onset_latencies[group].append(onset_latency/10)
                peak_latencies[group].append(peak_latency/10)
                stim_at_peaks[group].append(stim_at_peak)
                stim_at_onsets[group].append(stim_at_onset)
                if not (hyst is None):
                    desensitizations[group].append(desensitization)
                last_frame_over_threshold = np.where(trace>response_threshold)[0][-1]
                if last_frame_over_threshold < peak_frame:
                    after_discharge = after/10
                else:
                    after_discharge = (last_frame_over_threshold-stim_max_frame)/10
                if group == 'hot' and stim_at_onset < 39:
                    plot=True
                   #title=str(after_discharge)
                    cell.show()
                after_discharges[group].append(after_discharge)
                
                if plot:
                    plt.plot(stim_full_trace, 'k')
                    #plt.plot(v.timepoints,v.waveform, linewidth = 2, color='k')
                    plt.plot(trace, color = pp.gen_colors()[group])
                    plt.scatter(peak_frame, stim_at_peak, s=30, color = pp.gen_colors()[group]) 
                    #plt.text(title, last_frame_over_threshold, stim_full_trace[last_frame_over_threshold])
                    plot=False
            
                    
    results = [peak_latencies, stim_at_peaks, after_discharges, onset_latencies, stim_at_onsets, desensitizations]
    F = plt.figure(f'Response properties by class at {STIM}C', [6,3])
    F.tight_layout()
    # A = F.add_subplot(3,1,1)
    # for group in groups:
    #     A.hist(peak_latencies[group], color=pp.gen_colors()[group], alpha=0.5, density=True)
        
    # F = plt.figure('Peak stim')
    # A = F.add_subplot(3,1,1)
    # for group in groups:
    #     A.hist(stim_at_peaks[group], color=pp.gen_colors()[group], alpha=0.5, density=True)
        
    #F = plt.figure('Temp at onset')
    A = F.add_subplot(1,3,1)
    A.set_title('Stim at response onset')
    for group in groups:
        A.hist(stim_at_onsets[group], range=(35,50), bins=24, color=pp.gen_colors()[group], alpha=0.5, density=True)
        A.set_xlim([38, 52])
        A.set_xticks(A.get_xticks())
        A.set_xticklabels([str(int(x)) for x in A.get_xticks()])
        A.set_xlabel(f'Temp (C)')
        box_off(A)
        
   # F = plt.figure('Response decay')
    A = F.add_subplot(1,3,2)
    A.set_title('Response decay')
    for group in groups:
        A.hist(after_discharges[group], range=(-10,20), bins=30, color=pp.gen_colors()[group], alpha=0.5, density=True)
        A.set_xlim([-10, 20])
        box_off(A)
        
    A = F.add_subplot(1,3,3)
    A.set_title('Desensitization')
    for group in groups:
        A.hist(desensitizations[group], range=(-1,1), bins=20, color=pp.gen_colors()[group], alpha=0.5, density=True)
        A.set_xlim([-1.5, 1.5])
        box_off(A)
    save_fig(F, f'Response properties by class at {STIM}C {str(uuid.uuid4())[0:5]}')
    return(results)         
            
        
        
    
def fig_heat_warm_comp():
    
    ### show examples of hot warm and cold cells, with inset to show kinetics of heat and warm responses at 50 deg
    example_mouse = '7242'
    inset_range = [575, 605]
    
    #example_mouse = '8244'
    #inset_range = [575, 605]
    
    a = pp.get_sessions([example_mouse], FOVtag = 'Use for basal')[0]
    Fig = a.show_plots_groups(class_filters = ['warm','hot'], showMech=False, show_mean = True, dRange = inset_range)[0]
    save_fig(Fig, 'Warm-Heat comp 3')
    Fig = a.show_plots(class_filter = ['hot'], showMech=False, show_mean = True, dRange = None)[0]
    save_fig(Fig, 'Warm-Heat comp 2')
    Fig = a.show_plots(class_filter = ['warm'], showMech=False, show_mean = True, dRange = None)[0]
    save_fig(Fig, 'Warm-Heat comp 1')
    Fig = a.show_plots(class_filter = ['cold'], showMech=False, show_mean = True, dRange = None)[0]
    save_fig(Fig, 'Warm-Heat comp 4')
    
    return(a)

    ##### plot temp at stim peak versus stim temp for hot wnad warm cells
    mice = ['7242']
    a = pp.get_sessions(mice, FOVtag = 'Use for basal')[0]
    warm_cells = a.get_cells_from_class(['warm'])
    hot_cells = a.get_cells_from_class(['hot'])
    cold_cells = a.get_cells_from_class(['cold'])
    
    props = ['corr','area','latency_on']
    P = la.props_vs_temp(a, cells = warm_cells['warm'], tRange = [35, 55], props=props)
    P = la.props_vs_temp(a, cells = hot_cells['hot'], tRange = [35, 55], props=props)
    P = la.props_vs_temp(a, cells = cold_cells['cold'], tRange = [0, 35], props=props)
    
    return(a)

def print_all_rasters(mouse_filter = None, FOVtag = 'Use for basal'):
    if mouse_filter == None:
        Mice, populations = pp.mouseSetup()
    else:
        Mice = {}
        Mice['All'] = mouse_filter
    for mouse in Mice['All']:
        sessions = pp.get_sessions([mouse], FOVtag = FOVtag)
        for c, session in enumerate(sessions):
            H = session.show_raster()
            F = H['F']
            save_fig(F, F.properties()['label'])

def fig_cold_poly_comp(): ## work in progress
    example_mouse = '8244'
    
    a = pp.get_sessions([example_mouse], FOVtag = 'Use for basal')[0]
    cold_cells = a.get_cells_from_class(['cold'])
    poly_cells = a.get_cells_from_class(['poly'])
    
    
    
     
             
##########
def fig_basal_mech_tuning():
    pass

def mech_base_stats(threshold = 3.5, c = 0, plot_cells = True, sort_responses=True, calc_all=True):
    data = la.unpickle('/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7241/7241A mechLA longitudinal.pickle')
    sess = data.sessions[0]
    #c = sess.cells[c]
    #c.mech_range(max_threshold=threshold)
    data.sessions[0].mech_range(threshold = threshold, plot_cells = plot_cells, sort_responses = sort_responses, calc_all=calc_all)
    data.sessions[1].mech_range(threshold = threshold, plot_cells = plot_cells, sort_responses = sort_responses, calc_all=calc_all)

def collect_basal_mech_data():
    groups = ['rPbN','Gpr83','Tacr1']
    files = {}
    files['rPbN'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7241/7241A mechLA longitudinal.pickle',
                     '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7241/7241A mechLB longitudinal.pickle',
                     '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7241/7241A mechLC longitudinal.pickle',
                     '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/290/290L1 longitudinal.pickle',
                     '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/291/291L2 longitudinal.pickle',
                     '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/457A/457RA longitudinal.pickle',
                     '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7778/7778 mechmech longitudinal.pickle']
                     
                     
    
    files['Gpr83']  = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/6355/6355LA longitudinal.pickle',
                       '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/6356/6356LA longitudinal.pickle',
                       '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/6048/6048L1 longitudinal.pickle',
                       '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/6046/6046LA longitudinal.pickle'
                       ]
    
    files['Tacr1'] =  ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/237/237LA longitudinal.pickle',
                      '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/5896/5896RA longitudinal.pickle',
                      '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/414/414LC longitudinal.pickle',
                      '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/685/685R1 longitudinal.pickle',
                      '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/696/696L2 longitudinal.pickle',
                      '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Basal Mech/7243/7243RA longitudinal.pickle'
                      ]
    
    return(groups, files)

def review_basal_mech_data(files = None, groups = None):
    if files is None:
        groups, files = collect_basal_mech_data()
    for group in groups:
        for file in files[group]:
            m = la.unpickle(file)
            ## Check if mouse has glabrous, hairy, or both data:
            dtype = 'None'
            has_hair = False
            has_glab = False
            for session in m.sessions:
                session.process_mech_stim()
                source = str(session.Source).split('"')[1]
                if 'hair' in source:
                    has_hair = True
                elif 'glab' in source:
                    has_glab = True
            if has_hair and has_glab:
                dtype = 'both'
            elif has_hair:
                dtype = 'hairy'
            elif has_glab:
                dtype = 'glabrous'
            print(f'{source} type is {dtype}')
                
def compile_mech_range(threshold = 3.5, plot=True, limit = 500, unit = 'mN', plot_together = True, show_legend=True, plot_cells = False, plot_summary = False, sort_responses=True, calc_all=True, return_cells = True):
    start = time.time()
    all_responses = {}
    groups, files = collect_basal_mech_data()
    if plot_together:
        Fh = plt.figure('Hairy', figsize = [4,4], tight_layout = True)
        Fg = plt.figure('Glabrous', figsize = [4,4], tight_layout = True)
        
        Ah= Fh.add_subplot(1,1,1)
       
        Ag= Fg.add_subplot(1,1,1)
        colors = pp.gen_colors()
    
    example = {}
    example['rPbN'] = {}
    example['rPbN']['session'] = [0,1,2,3,4,5,6]
    example['rPbN']['file'] = [0,1,2,3,4,5,6,7,8]
        
    for g, group in enumerate(groups):
       
        responses = {}
        responses['glabrous'] = []
        responses['hairy'] = []
        for f, file in enumerate(files[group]):
            m = la.unpickle(file)
            for s, session in enumerate(m.sessions):
                source = str(session.Source).split('"')[1]
                
                if group in example:
                    if s in example[group]['session'] and f in example[group]['file']:
                        plot_cells = True
                    else: 
                        plot_cells = False
                
                response_set = session.mech_range(threshold = threshold, unit = unit,  plot_cells = plot_cells, plot_summary = plot_summary, sort_responses = sort_responses, calc_all=calc_all, return_cells = return_cells)
                
                if 'hair' in source:
                    responses['hairy'].extend(response_set)
                elif 'glab' in source:
                    responses['glabrous'].extend(response_set)
                else:
                    print(source + ' is not tagged as hairy or glabrous')
                    return()
    
        
        hsorting = la.sort_mech_ranges(responses['hairy'])
        responses['hairy']  = [responses['hairy'][x] for x in hsorting['IX']]
        
        gsorting = la.sort_mech_ranges(responses['glabrous'])
        responses['glabrous']  = [responses['glabrous'][x] for x in gsorting['IX']]
        
        all_responses[group] = responses                
        if plot:
            if not plot_together:
                Fh = plt.figure(group+' hairy')
                Ah= Fh.add_subplot(1,1,1)
                Fg = plt.figure(group+' glabrous')
                Ag= Fg.add_subplot(1,1,1)
            la.plot_mech_ranges(responses['glabrous'], color = colors[group], unit=unit, F=Fg, A=Ag, SNR_threshold = threshold, append = plot_together)
            la.plot_mech_ranges(responses['hairy'], color = colors[group], unit=unit, F=Fh, A=Ah, SNR_threshold = threshold, append = plot_together)
            
            if show_legend:
                if g == len(groups)-1:
                    dm=2 
                    dexp = 1
                    n_cells = Ag.get_xlim()[1]
                    X = (np.array([1,3,4,5, 6]) *(n_cells/30)) + 10
                    SNRs = [1, 5, 10, 15, 20]
                    S = [(g*dm)**dexp for g in SNRs]
                    
                    
                    color=[0,0,0]
                    for xx, ss, snr in zip(X,S, SNRs):
                        if snr == 1:
                            Ag.scatter(xx, limit*1.25, s=ss, color=la.lighten_color(color, amount = 0.7))
                            Ag.text(xx, limit * 1.3, f'Z: < {threshold}', verticalalignment = 'bottom', horizontalalignment = 'right', fontsize=4)
                        else:
                            Ag.scatter(xx, limit*1.25, s=ss, color = color)
                            Ag.text(xx, limit * 1.3, f'{snr}', verticalalignment = 'bottom', horizontalalignment = 'center', fontsize = 4)
                            
                            
                    
            #Ag.scatter([0])
            
            
            Ah.set_ylabel(f'Force ({unit})')
            Ah.set_xlabel('Cell #')
            Ag.set_ylabel(f'Force ({unit})')
            Ag.set_xlabel('Cell #')
            save_fig(Fh, f'Basal mech {group} hairy SNR = {threshold} {str(uuid.uuid4())[0:5]}')
            save_fig(Fg, f'Basal mech {group} glabrous SNR = {threshold} {str(uuid.uuid4())[0:5]}')
        
    end = time.time()
    print(f'execution took {end-start} seconds')
    
    if return_cells:
        plot_mech_range_thresholds(all_responses)
    return(all_responses)

def plot_mech_range_thresholds(data):
    F = plt.figure('Thresholds by group'+str(uuid.uuid4())[0:5], figsize = [2.5,3], tight_layout = True)
    G = plt.figure('Most effective stimulus by group'+str(uuid.uuid4())[0:5],figsize = [2.5,3], tight_layout = True)
    A = F.add_subplot(1,1,1)
    B = G.add_subplot(1,1,1)
    for g, group in enumerate(data):
        best = get_mech_best_stims(data[group]['glabrous'])
        thresholds = [x[0]['cell'].mech_threshold for x in data[group]['glabrous'] if np.isfinite(x[0]['cell'].mech_threshold)]
        #thresholds = np.log(thresholds)
        beeswarm(thresholds, g, width = 0.125, A=A, color = pp.gen_colors()[group], alpha = 0.25)
        b = A.violinplot(thresholds, [g], showextrema = False, showmedians=True, widths=0.5)
        b['cmedians'].set_color('w')#pp.gen_colors()[group])
        b['cmedians'].set_linestyle('dotted')
        for bb in b['bodies']:
            bb.set_facecolor(pp.gen_colors()[group])
        #pdb.set_trace()
        A.set_yscale('log')
        A.set_yticks([10,100,1000])
        A.set_ylim([10,1000])
        A.set_ylabel('Force (mN)')
        A.set_title('Threshold stim')
        beeswarm(best, g, width = 0.125, A=B, color = pp.gen_colors()[group], alpha=0.25)
        b = B.violinplot(best, [g], showextrema = False, showmedians=True, widths=0.5)
        for bb in b['bodies']:
            bb.set_facecolor(pp.gen_colors()[group])
        b['cmedians'].set_color('w')#(pp.gen_colors()[group])
        b['cmedians'].set_linestyle((0,(1,1)))
        
        B.set_yscale('log')
        B.set_yticks([10,100,1000])
        B.set_ylim([10,1000])
        B.set_ylabel('Force (mN)')
        B.set_title('Optimal stim')
    A.set_xticks([x for x, v in enumerate(data.keys())])
    A.set_xticklabels([x for x in data.keys()])
    B.set_xticks([x for x, v in enumerate(data.keys())])
    B.set_xticklabels([x for x in data.keys()])
    box_off(A)
    box_off(B)
    save_fig(F, 'Mech thresholds')
    save_fig(G, 'Mech best stim')
    return(best)
    
    

def get_mech_best_stim(responses):
    SNR = [r['max_SNR'] for r in responses]
    force = [r['force'] for r in responses]
    ix = np.where(SNR == np.amax(SNR))[0][0]
    return(force[ix])

def get_mech_best_stims(response_group):
    b = []
    for responses in response_group:
        b.append(get_mech_best_stim(responses))
    return(b)

def plot_mech_best_stim(data):
    pass


def mech_basal_recruitment(responses=None, threshold=3.5, SNRs = [3,5,7], group = 'rPbN', side = 'glabrous', plot_threshold=50):
    if responses is None:
        responses = compile_mech_range(threshold=threshold, plot=False, plot_summary=False)
        responses = responses[group][side]
    results={}
    for SNR in SNRs:
        results[SNR] = {}
        results[SNR]['force'] = []
        results[SNR]['recruited'] = []
        for force in range(0, 500, 10):
            results[SNR]['force'].append(force)
            recruited = 0
            for responder in responses:
                if responder[0]['cell'].mech_threshold < force:
                    recruited = recruited+1
               # if responder[0]['cell'].mech_threshold < plot_threshold:
                #    responder[0]['cell'].mech_range(plot=True)
            results[SNR]['recruited'].append(recruited)
    F = plt.figure()
    for SNR in SNRs:
        plt.plot(results[SNR]['force'], results[SNR]['recruited'])
    
       
    return(responses)

def mech_thermo_overlap_map():
    ##Example map showing overaly of mech and diff thermo responses
    ##Example traces for mech and thermo
    ##
    
    ## Get example data:
    DBpath = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/All windows/Processed/Mouse 7241/Mouse 7241 def.h5'
    FOV = 'FOV L1 2022-6-14 PI Basal'
    Animal = '7241'
    ca_data = 'calcium'
    FLIR_data = 'FLIR_degC'
    mech_data = 'eVF_mN'
    
    DB = pp.h5py.File(DBpath, 'a')
    

    
def mech_thermo_correlation(datas = None, norm = True):
    figs = {}
  
    ## Rank cellss by mech sensistivity(max response as proportion o fmax)
    ## Plot maximal response to warmth, heat, cold
    ###
    results = pp.get_data_files(popFilter = 'Mech-thermo', FOVtag = 'Mech-thermo', activeTag = 'Ca data', dataTags = ['Ca data', 'Thermo stim', 'Mech stim'], genotypes = ['Mech-thermo'])
    if datas is None:
        datas=[]
        DBs = results['data']['Mech-thermo'].keys()
        for DB in DBs:
            datalist = results['data']['Mech-thermo'][DB]
            for DATA in datalist:
                #pdb.set_trace()
                #DATA['source'] = DB
                DATA = la.processMechStim(DATA, plot=False)
                
                DATA = la.classify_cells_in_dataset(DATA)
                datas.append(DATA)
        
    mtCells = {} ## a dictionary of dictionaries storing maximal thermal and mechanical responses for each cell #
    mech_maxes = []
    cell_count = 0
    sessions = []
    for DATA in datas:
        sessions.append(la.session_data(DATA)) ## create session_data obj for each session
        
        for c, cell in DATA['cells'].items():
            #pdb.set_trace()
            cell.therm_stats = la.cell_analyze_temp(cell, DATA)
            #pdb.set_trace()
            cell = la.cell_analyze_mechano(cell, norm = norm)
            
            
            #sr = la.mech_tuning_cell(cell, plot=False)
            
            mtCells[cell_count] = {}
            
            mtCells[cell_count]['mech stim-resp'] = cell.mech_stats['stim-resp']
            mtCells[cell_count]['mech max'] = cell.mech_stats['mech_max'] 
            
            mech_maxes.append(cell.mech_stats['mech_max'] )
            
            
            mtCells[cell_count]['cell'] = cell
            
            cell_count = cell_count + 1
   # pdb.set_trace()     
    new_order = np.argsort(mech_maxes)
  
    
    #make list of heat responsses ordered on max mech respones amplitude (normalized)
    heat_ordered_on_mech = [] 
    sorted_maxes = []
    sorted_cells = []
    for c in new_order:
        heat_ordered_on_mech.append(mtCells[c]['cell'].therm_stats['cwh'][2]) ## retrieve median response to 51 deg
        sorted_maxes.append(mech_maxes[c])
        sorted_cells.append(mtCells[c])
    plt.plot(heat_ordered_on_mech)
    plt.plot(sorted_maxes)
    mech_responses_all = []
    mech_responses_by_thermo_class = {}
    thermo_types = la.temp_class_bounds()
    for thermo_type in thermo_types.keys():
        mech_responses_by_thermo_class[thermo_type] = []
        for cell in sorted_cells:
            if cell['cell'].classification == thermo_type:  
                mech_responses_by_thermo_class[thermo_type].append(cell['cell'].mech_stats['mech_max'])
                mech_responses_all.append(cell['cell'].mech_stats['mech_max'])
    
    ## histogram of mech responses each category:
    F = plt.figure('Mech histogram all')
    
    #plt.hist(mech_responses_all, bins=12)
    colors = pp.gen_colors()
    A = F.add_subplot(1,1,1)
    for n, distribution in enumerate(mech_responses_by_thermo_class):
        A# = F.add_subplot(1,len(mech_responses_by_thermo_class),n+1)
        A.hist(mech_responses_by_thermo_class[distribution], color = colors[distribution], range=[0,1.5], bins=15, alpha = 0.25)
        
    figs['Mech histogram all'] = F
    
        
    ##Plot bar and scatter of mech responses for each class
    F= plt.figure('Mech vs thermo class')
    figs['Mech vs thermo class'] = F
    A = F.add_axes([0,0,1,1])
    colors = pp.gen_colors()
    Xtick_pos=[]
    Xtick_labels=[]
    for c, thermo_type in enumerate(thermo_types.keys()):
        barcolor = colors[thermo_type]
        Y = mech_responses_by_thermo_class[thermo_type]
        X = np.ones(len(Y))*c
        Ym = np.mean(Y)  ## maybe median is better?
        Ym = np.median(Y)
        A.bar(c, np.mean(Y), color=barcolor,alpha=0.75, width = 0.4)
        pp.jitter(X,Y, alpha = 1, s=40, edgecolors = 'None', color = 'k', linewidths=0.5)
        Xtick_pos.append(c)
        Xtick_labels.append(thermo_type)
        
    A.set_xticks(Xtick_pos)
    A.set_xticklabels(Xtick_labels)
    A.set_frame_on(False)   
    A.set_yticks([0,0.5,1])
    A.set_yticklabels(['0','0.5','1'])
    
    F= plt.figure('Ranking by mech color by class')
    figs['Ranking by mech color by class'] = F
    
    ranks = np.arange(len(sorted_cells))
    A = F.add_axes([0,0,1,1])
    for c, cell, in enumerate(sorted_cells):
        color = colors[cell['cell'].classification]
        A.scatter(c, cell['cell'].mech_stats['mech_max'], color=color, s=20, alpha = 1)
        A.set_frame_on(False)
        A.set_xlabel('Cell #')
        if norm:
            A.set_ylabel('Mech max/all max')
        else:
            A.set_ylabel('Mech max response')
    
    show_traces = False
    if show_traces:
        for s in sessions:
            for thermo_type in thermo_types:
                if s.class_count(thermo_type) > 0:
                    handles = s.show_plots(class_filter = [thermo_type], show_mean = False)
                    figs[s.mouse + ' ' + thermo_type] = handles[0]
                
    for save_name in figs.keys():
        figs[save_name].savefig(f'/lab-share/Neuro-Woolf-e2/Public/Figure publishing/{save_name}.pdf')
    
    
    return(datas, mtCells, sorted_cells, sessions)

def CAPS_raster_panel(baseline_correct = True):
    data=SNI_raster_panel(CAPS=True, vmin=0.01, vmax = 0.5, baseline_correct = baseline_correct)
    return(data)

def SNI_raster_panel(vmin = 0.01, vmax = 0.3, cmap = 'plasma', stim_v=0.15, figsize = None, v_gap = 0.002, warm_factor=1, dff=False, dff_z=False, CAPS=False, baseline_correct = True):
    if CAPS:
        r,s,e,c,a, d =longitudinal_aligned_rasters( files = 'CAPS', time_points = [(0,0),(0,0.001)], plot=False, warm_factor=warm_factor, dff=dff, dff_z=dff_z, baseline_correct=baseline_correct)
        perturbation = 'CAPS'
    else:
        perturbation = 'SNI'
        r,s,e,c,a, d =longitudinal_aligned_rasters( time_points = [(0,0),(14,21)], plot=False, warm_factor=warm_factor, dff=dff, dff_z=dff_z)
    title = 'Thermal responses baseline vs 2 weeks' +str(uuid.uuid4())[0:5]
    condition_labels = ['Baseline', perturbation]
    if figsize is None:
        figsize = (4,5)
    F = plt.figure(title, figsize = figsize)
    n_cohorts = len(a.keys())
    cohorts = list(a.keys())
    populations = list(a[cohorts[0]].keys())
    if perturbation == 'CAPS':
        populations = populations[:-1]
    n_pops = len(populations)
    time_points = list(a[cohorts[0]][0].keys())
    n_times = len(time_points)
    baseline = time_points[0]
    h_margins = 0.15
    v_margins = 0.1
    if v_gap is None:
        v_gap = 0.002
    cohort_gap = 0.01
    time_gap = 0.15
    if stim_v is None:
        stim_v = 0.15
    active_v = 1 - (2*v_margins)
    active_h = 1 - (2*h_margins)
    color_set = la.gen_colors()
    ## plot stimuli:
    stim_axes = {}
    # plot stimuli
    for t, tp in enumerate(time_points):
        stim_bot = 1-v_margins  - (active_v*stim_v)
        stim_left = h_margins + (t*active_h/n_times)
        stim_width = (active_h/n_times)*(1-time_gap)
        stim_height = (stim_v*active_v)
        stim_axes[tp] = F.add_axes([stim_left, stim_bot, stim_width, stim_height])
        for cohort in cohorts:
            S = s[cohort, tp]
            #pdb.set_trace()
            stim_axes[tp].plot(S, color = color_set[cohort], alpha=0.5)
        stim_axes[tp].set_xlim([0, S.shape[0]])
        stim_axes[tp].set_ylim([0, 60])
        if t==0:
            box_off(stim_axes[tp], left_only=True)
            stim_axes[tp].set_yticks([0,30,60])
            stim_axes[tp].set_ylabel(f'Temp {tr.degC()}')
        else:
            box_off(stim_axes[tp], All = True)
        stim_axes[tp].title.set_text(condition_labels[t])
            
    #plot responses
    r_axes = {}
    # get num cells for each cohort and stim, and 
    counts = {}
    counts['all'] = 0
    for cohort in cohorts:
        counts[cohort] = {}
        counts[cohort]['total'] = 0
        for population in a[cohort].keys():
            counts[cohort][population] = len(a[cohort][population][baseline])
            counts[cohort]['total'] = counts[cohort]['total'] + counts[cohort][population]
            counts['all'] = counts['all'] + counts[cohort][population]
    
    init_top = stim_bot-0.02
    top = init_top*1
    active_v = init_top - v_margins - (cohort_gap * n_cohorts)
    cohort_tops = {}
    cohort_bottoms = {}
    
    for c, cohort in enumerate(cohorts):
        
        for p, pop, in enumerate(populations):
            if counts[cohort][pop] == 0:
                continue
            r_extent = counts[cohort][pop]/counts['all'] * active_v
            r_height = r_extent-v_gap
            r_bot = top-r_extent
            if p==0:
                cohort_tops[cohort] = top
            top = r_bot
            if p == len(populations)-1:
                cohort_bottoms[cohort] = r_bot
                top = top - cohort_gap
            
            for t, tp in enumerate(time_points):
                #pdb.set_trace()
                sub_raster = r[(cohort, tp)][d[cohort][pop]]
                #cluster_n = sub_raster.shape[0]
                
                r_left = h_margins + (t*active_h/n_times)
                r_width = (active_h/n_times)*(1-time_gap)
                raster = np.zeros([10,10])
                r_axes = F.add_axes([r_left, r_bot, r_width, r_height])
                r_axes.imshow(sub_raster, aspect = 'auto', cmap = cmap, vmin = vmin, vmax = vmax)
                box_off(r_axes, All=True)
    for top, bottom in zip(cohort_tops, cohort_bottoms):
        A = F.add_axes([0, cohort_bottoms[bottom], h_margins, cohort_tops[top]-cohort_bottoms[bottom]])
        A.plot([1,1],[0,1], c = color_set[top], linewidth=2)
        A.set_xlim([0,1.1])
        A.set_ylim([0,1])
        A.patch.set_alpha(0)
        A.text(1, 0.5, top, rotation = 'vertical', color = color_set[top], horizontalalignment='right', verticalalignment='center')
        B = F.add_axes([0.95-h_margins, cohort_bottoms[bottom], h_margins, cohort_tops[top]-cohort_bottoms[bottom]])
        B.text(0, 0, counts[top]['total'], color='k', horizontalalignment='left', verticalalignment='bottom')
        B.set_xlim([0,10])
        B.set_ylim([0,1])
        B.patch.set_alpha(0)
        box_off(A,All=True)
        box_off(B,All=True)
    A = F.add_axes([h_margins, v_margins/2, 0.3,v_margins/4])
    color_bar(cmap, A=A, min_v = int(vmin*100), max_v = int(vmax*100), fontsize=9, unit="% F'", save=False)
    save_fig(F, title)
    save_fig(F, title, ext='.png')
    #pdb.set_trace()
    data =compare_traces_SNI(r,s,e,c,a,d, cohorts, populations, baseline, time_points, title, perturbation=perturbation)
    return(data)
    # pdb.set_trace()


def DREADD_inspect():
    #%%
    file1 = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Cold plate DREDD/All DREDD experiments 3-10-2023.h5'
    file2a = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Cold Plate DREADD Cohort 2 (3mgkg)/Dreadd Cohort 2 3mgkg'
    file2b  = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Cold Plate DREADD Cohort 2 (3mgkg)/Dreadd Continued'
    file_h = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Hot plate CAPS VAS/hot plate caps vas.h5'
    
    
    with h5py.File(file1, mode='r') as F1:
        
        for key in F1['Animals'].keys():
            A = key
        for trial in F1['Animals'][A].keys():
            print(trial)
            FLIR = False
            cam = False
            if 'FLIR' in F1['Animals'][A][trial].keys():
                print(f'{trial} has FLIR')
                FLIR = True
            if 'camera' in F1['Animals'][A][trial].keys():
                print(f'{trial} has camera')
                cam = True
            #if cam and FLIR:
                #if 'traceArray' in F1['Animals'][A][trial]['R']['camera'].keys():
            
            plt.figure(trial)
            if cam:
                if 'traceArray' in F1['Animals'][A][trial]['R']['camera'].keys(): 
                    lifting = F1['Animals'][A][trial]['R']['camera']['traceArray'][...].T
                    lifting_T = F1['Animals'][A][trial]['T']['camera'][...]
                    plt.plot(lifting_T,lifting)
            if FLIR:
                temp = F1['Animals'][A][trial]['R']['FLIR']['traceArray'][...].T
                temp_T = F1['Animals'][A][trial]['T']['FLIR'][...]
                plt.plot(temp_T, temp)
            #ipsi = F1['Animals'][A][trial]['R']['camera']['traceArray'][0,...]
            #ipsi = F1['Animals'][A][trial]['R']['camera']['traceArray'][1,...]
                
        F1.close()
        #%%

def compare_traces_SNI(r,s,e,c,a,d, cohorts, populations, baseline, time_points, title, figsize=[4.5,5], perturbation = 'SNI'):
    F = plt.figure('Trace comparison' + title, figsize=figsize)
    color_set = la.gen_colors()
    lines =  {}
    #if perturbation == 'CAPS':
       # pdb.set_trace()
    lines['rPbN']  = 'solid'
    lines['Tacr1']  = 'dashed'
    lines['Gpr83'] = 'dotted'
    tags = ['Cold','Warm','Hot','Silent']
    time_tag = ['Basal', 'SNI']
    colors= {}
    colors[0] = 'k'
    colors[1] = 'r'
    
    width = r[(cohorts[0], time_points[0])].shape[1]
    for p, population in enumerate(populations):
        A = F.add_subplot(len(populations),1, p+1)
        if p == 0:
            for l, cohort in enumerate(cohorts):
                A.plot([width*0.86, width], [0.49-(l*0.08), 0.49-(l*0.08)], linestyle = lines[cohort], color = 'r')
                A.plot([width*0.7, width*0.84], [0.49-(l*0.08), 0.49-(l*0.08)], linestyle = lines[cohort], color = 'k')
                A.text(width*0.68, 0.49-(l*0.08), cohort, horizontalalignment='right', verticalalignment='center')
            A.text(width*0.77,0.52, 'Basal', horizontalalignment='center', verticalalignment='bottom')
            
            A.text(width*0.93,0.52, perturbation, horizontalalignment='center', verticalalignment='bottom', color='r')
                
        if p == len(populations)-1:
            A.plot([60,140], [-0.09, -0.09],color = 'b', linewidth =3)
            A.plot([280,360], [-0.09, -0.09],color = [1,0.5,0], linewidth =3)
            A.plot([500,580], [-0.09, -0.09],color = 'm', linewidth =3)
            A.text(100,-0.1, 'cooling', color='b', horizontalalignment='center', verticalalignment='top')
            A.text(320,-0.1, 'warming', color= [1,0.5,0], horizontalalignment='center', verticalalignment='top')
            A.text(540,-0.1, 'heating', color='m', horizontalalignment='center', verticalalignment='top')
                
        for t, tp in enumerate(time_points):
            if t ==0:
                linestyle = 'dotted'
            else:
                linestyle = 'solid'
            for cohort in cohorts:
                response = r[(cohort, tp)][d[cohort][population]].T
                A.plot(response, color = colors[t], linestyle = lines[cohort], alpha = 0.025)
                A.plot(np.mean(response, axis=1), color = colors[t], linestyle = lines[cohort], alpha = 1)
                A.set_ylim([-0.1,0.5])
                A.set_ylabel(tags[p])
                box_off(A, left_only=True)
    save_fig(F, 'Comp traces' + title)
    save_fig(F, 'Comp traces' + title, ext='.png')
    data = SNI_summary_stats(r,s,e,c,a,d, cohorts, populations, baseline, time_points, title, perturbation = perturbation)
    return(data)
        
    
def SNI_summary_stats(r,s,e,c,a,d, cohorts, populations, baseline, time_points, title, perturbation ='SNI'):
    stims = ['cooling', 'warming', 'heating']
    stim_colors = ['b', [1,0.5,0], 'm']
    data = {}
    
    if perturbation == 'SNI':
        population_names = ['Cold', 'Warm', 'Hot', 'Silent']
    elif perturbation =='CAPS':
        population_names = ['Cold', 'Warm', 'Hot']
    
    nc = len(cohorts)
    n_p = len(population_names)
    ns = len(stims)
    nt = len(time_points)
    color_set = la.gen_colors()
    for pop in populations:
            for cohort in cohorts:
                for t, tp in enumerate(time_points):
                    raster = r[(cohort, tp)][[d[cohort][pop]]]
                    r_seg = np.split(raster, ns, axis=1)
                    for s, stim in enumerate(stims):
                        data[population_names[pop], stim, cohort, tp] = r_seg[s]
    F = plt.figure(title+' bars', figsize = [9,3])
    A = F.add_axes([0.15,0.1,0.7,0.8])
    
    X_labels = []
    X_ticks = []
    for p, pop in enumerate(population_names):
        for s, stim in enumerate(stims):
            for c, cohort in enumerate(cohorts):
                
                before_data = data[pop, stim, cohort, time_points[0]]
                after_data = data[pop, stim, cohort, time_points[-1]]
                cohort_w = nt
                stim_w = 2# (cohort_w * nc) + 1
                pop_w = (stim_w * ns) + 2
                offset = (p* pop_w) + (s*stim_w) #+ (c*cohort_w)
                
                #offset = ((nt+1)*c) + ((nc+1) *s) + ((ns+1)*p)
                if s == 1 and c == 1:
                    X_ticks.append(offset+0.5)
                    X_labels.append(pop)
                if s== 0:
                    pop_left = offset
                if s == len(stims)-1:
                    pop_right = offset+1
                    plt.plot([pop_left,pop_right],[-0.1,-0.1], color='k', linewidth = 2)
                n_cells = before_data.shape[1]
                # before_x = np.ones([n_cells])* offset
                # after_x = before_x + 1
                # pdb.set_trace()
                b_maxes = []
                a_maxes = []
                for by, ay in zip(before_data, after_data):
                    b_max = np.amax(by)
                    a_max = np.amax(ay)
                    b_maxes.append(b_max)
                    a_maxes.append(a_max)
                    A.plot([offset, offset+1],[b_max, a_max], color='k', alpha = 0.025)
                #A.bar([offset, offset+1],[np.mean(b_maxes), np.mean(a_maxes)], color=color_set[cohort], alpha = 1)
                #A.plot([offset, offset+1],[np.mean(b_maxes), np.mean(a_maxes)], color=color_set[cohort], alpha = 1)
                b_err = 2*np.std(b_maxes)/np.sqrt(n_cells)
                a_err = 2*np.std(a_maxes)/np.sqrt(n_cells)    
                A.errorbar([offset, offset+1],[np.mean(b_maxes), np.mean(a_maxes)], yerr = [b_err, a_err], color=color_set[cohort], alpha = 1, capsize=1)
                plt.plot([offset, offset+1],[-0.05,-0.05], color = stim_colors[s])
                #if s == 0:
                #    stim_line_left = offset
                #elif s == len(stims)-1:
                #    stim_line_right = offset
                #    plt.plot([stim_line_left, stim_line_right],[0,0], color = stim_colors[s])
    for n, (color, stim) in enumerate(zip(stim_colors, stims)):
        A.text(offset+2, 0-(n*0.05), stim, color = color, horizontalalignment='left', verticalalignment='top')
    
    for n, cohort in enumerate(cohorts):
        A.text(offset+1, 0.75-(n*0.05), cohort, color = color_set[cohort], horizontalalignment='right', verticalalignment='top')
        
    A.set_xticks(X_ticks)
    A.set_xticklabels(X_labels) 
    A.set_ylim([-0.1,0.75])
    A.set_yticks([0,0.5])
    A.spines['bottom'].set_visible(False)
    A.set_ylabel("F'")
    box_off(A)
    save_fig(F, title+'_summary')
    save_fig(F, title+'_summary', ext='.png')
    return(data)
        
class container:
    
    def __init__(self, pickle_path=None):
        self.pickle_path = pickle_path
        
    def pic__kle(self, pickle_path = None):
        
        if pickle_path is None:
            pickle_path = self.pickle_path
            
        if pickle_path is None:   
            print('No location given to store pickle!')
            return
        else:
            self.get_ke__ys()
            file = open(pickle_path, 'wb')
            pickle.dump(self, file)
            file.close()
            print(f'Saved to {pickle_path}')
            self.pickle_path = pickle_path
    
    def get_ke__ys(self):
        keys = []
        for key in dir(self):
            if not '__' in key:
                keys.append(key)
        self.keys = keys
        return(keys)
    
    
def SNI_mech(threshold = 3.5, limit = 500, plot=True, pickle_result=True, from_pickle = True, pickle_loc = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/'):
    
    start_time = time.time()
   

    if from_pickle:
        try:
            data = la.unpickle(pickle_loc + 'SNI_mech data.pickle')
            glabrous, hairy, combined = collect_SNI_mech_alignment()
        except:
            print(f'Could not unpickle data from {pickle_loc+"SNI_mech data.pickle"}')
            return()
    else:
        glabrous, hairy, combined = collect_SNI_mech_alignment()
        before_cells = {}
        after_cells = {}
        for group in glabrous:
            before_cells[group + '_glab'] = []
            after_cells[group + '_glab'] = []
            for file in glabrous[group]:
                m = la.unpickle(file)
                m.claim_sessions()
                m.index_cells()
                for n, u in enumerate(m.assignments[:,0]):
                    before_cells[group + '_glab'].append(m.sessions[0].cells[n])
                    after_cells[group + '_glab'].append(m.sessions[1].cells[n])
                    
        data = container(pickle_loc + 'SNI_mech data.pickle')
        data.keys = []
        for g, group in enumerate(glabrous):
            data.keys.append(group)
            dataset = container()
            #dataset.all_cells = []
            dataset.lost_cells_before= []
            dataset.lost_cells_after= []
            
            dataset.new_cells_after = []
            dataset.new_cells_before = []
            
            dataset.conserved_cells_before =[]
            dataset.conserved_cells_after = []
            
            dataset.all_cells_before =[]
            dataset.all_cells_after = []
            dataset.lost_responses_before = []
            dataset.lost_responses_after = []
            dataset.new_responses_before = []
            dataset.new_responses_after = []
            dataset.all_responses_before = []
            dataset.all_responses_after = []
            dataset.conserved_responses_before = []
            dataset.conserved_responses_after = []
            dataset.null_cells =[]
            
            for b, a in zip(before_cells[group + '_glab'], after_cells[group + '_glab']):
                dataset.rb = b.mech_range(max_threshold = threshold, limit=limit)
                dataset.ra = a.mech_range(max_threshold = threshold, limit=limit)
                if b.mech_threshold == np.inf and a.mech_threshold == np.inf:
                    print('This cell has no responses either session at this threshold')
                    dataset.null_cells.append((b,a))
                elif b.mech_threshold == np.inf:
                    dataset.new_cells_before.append(b)
                    dataset.new_cells_after.append(a)
                    dataset.new_responses_before.append(dataset.rb)
                    dataset.new_responses_after.append(dataset.ra)
                elif a.mech_threshold == np.inf:
                    dataset.lost_cells_before.append(b)
                    dataset.lost_cells_after.append(a)
                    dataset.lost_responses_before.append(dataset.rb)
                    dataset.lost_responses_after.append(dataset.ra)
                else: 
                    dataset.conserved_cells_before.append(b)
                    dataset.conserved_cells_after.append(a)
                    dataset.conserved_responses_before.append(dataset.rb)
                    dataset.conserved_responses_after.append(dataset.ra)
                    
                dataset.all_cells_before.append(b)
                dataset.all_cells_after.append(a)
                dataset.all_responses_before.append(dataset.rb)
                dataset.all_responses_after.append(dataset.ra)
            
            setattr(data, group, dataset)
            
    middle = time.time()
    print(f'Calculation took {middle-start_time} seconds, from pickle is {from_pickle}')       
    if plot:       
        
        plot_SNI_mech(data, limit=limit)
            
        
      
    if pickle_result:
        try:
            data.pic__kle(pickle_loc + 'SNI_mech data.pickle')  
        except:
            pdb.set_trace()
    
    end = time.time()
    duration = end-start_time
    print(f'Execution took {duration} seconds, from pickle is {from_pickle}')
    return(data)
                
def plot_SNI_mech(data, limit=500):
    F = plt.figure('Force vs Ca++ Baseline and SNI 2 weeks'+str(limit))
    A = F.add_subplot(2,1,2)
    B = F.add_subplot(2,1,1) 
    for g, group in enumerate(data.keys):  
        
        dataset = getattr(data, group)
        
        conserved_and_lost_responses_before = dataset.conserved_responses_before + dataset.lost_responses_before
        conserved_and_lost_responses_after = dataset.conserved_responses_after + dataset.lost_responses_after
        
        conserved_and_lost_cells_before = dataset.conserved_cells_before + dataset.lost_cells_before
        conserved_and_lost_cells_after = dataset.conserved_cells_after + dataset.lost_cells_after
        
        static_IX = la.sort_mech_ranges(conserved_and_lost_responses_before)['IX']
        conserved_responses_before = [conserved_and_lost_responses_before[ix] for ix in static_IX]
        conserved_responses_after = [conserved_and_lost_responses_after[ix] for ix in static_IX]
        
        conserved_cells_before = [conserved_and_lost_cells_before[ix] for ix in static_IX]
        conserved_cells_after = [conserved_and_lost_cells_after[ix] for ix in static_IX]
        
        
        new_IX = la.sort_mech_ranges(dataset.new_responses_after)['IX']
        new_responses_before = [dataset.new_responses_before[ix] for ix in new_IX]
        new_responses_after = [dataset.new_responses_after[ix] for ix in new_IX]
        
        new_cells_before = [dataset.new_cells_before[ix] for ix in new_IX]
        new_cells_after = [dataset.new_cells_after[ix] for ix in new_IX]
        
        dataset.all_responses_before = conserved_responses_before + new_responses_before
        dataset.all_responses_after = conserved_responses_after + new_responses_after
        dataset.all_cells_before = conserved_cells_before + new_cells_before
        dataset.all_cells_after = conserved_cells_after + new_cells_after
        
        la.plot_mech_ranges(dataset.all_responses_before, F=F, A=B, only_responders=False, color = la.gen_colors()[group], append = True, save_plot=False, limit=limit)
        
        la.plot_mech_ranges(dataset.all_responses_after, F=F, A=A, only_responders=False, color = la.gen_colors()[group], append = True, save_plot=False, limit=limit)
        setattr(data, group, dataset)
    save_fig(F, 'Force vs Ca++ Baseline and SNI 2 weeks')
    return(data)
        #best_stim_before = get_mech_best_stims(dataset.responses_before)



def SNI_mech_recruitment_plot(stats=None, keys = ['rPbN', 'Gpr83', 'Tacr1']):
    if stats is None:
        SNI_mech_data = SNI_mech(from_pickle=False)
        stats = SNI_mech_stats(SNI_mech_data)
        
    for k, key in enumerate(keys):
        data = getattr(stats, key)
        X = []
        thresh_count_before = []
        thresh_count_after = []
        for force in range(10,500):
            X.append(force)
            count = 0
            for c in data.all_cells_before:
                if c.mech_threshold  <= force:
                    count = count + 1
            thresh_count_before.append(count)
            
            for c in data.all_cells_after:
                if c.mech_threshold  <= force:
                    count = count + 1
            thresh_count_after.append(count)
        
        F = plt.figure('Recruitment plots')
        A = F.add_subplot(1,len(keys),k+1)
        A.set_title(key)
        A.plot(X,thresh_count_before, 'k')
        A.plot(X,thresh_count_after, 'r')
        box_off(A)
        
        
        

def SNI_mech_stats(data = None, pickle_loc = '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/'):
    if data is None:
        try:
            data = la.unpickle(pickle_loc + 'SNI_mech data.pickle')
            #glabrous, hairy, combined = collect_SNI_mech_alignment()
        except:
            print(f'Could not unpickle data from {pickle_loc+"SNI_mech data.pickle"}')
            return()
        

    F = plt.figure('Thresholds by group', figsize = [2.5,3], tight_layout = True)
    G = plt.figure('Most effective stimulus by group' + str(uuid.uuid4())[0:5],figsize = [2.5,3], tight_layout = True)
    A = F.add_subplot(1,1,1)
    B = G.add_subplot(1,1,1)
    for g, group in enumerate(data.keys):
        dataset = getattr(data, group)
        try:
            best_after = get_mech_best_stims(dataset.all_responses_after)
        except:
            continue
        thresholds_before = [c.mech_threshold for c in dataset.all_cells_before if np.isfinite(c.mech_threshold)]
        thresholds_after = [c.mech_threshold for c in dataset.all_cells_after if np.isfinite(c.mech_threshold)]
        best_before = get_mech_best_stims(dataset.all_responses_before)
        #best_after = get_mech_best_stims(dataset.all_cells_after)
        #thresholds = np.log(thresholds)
        beeswarm(thresholds_before, (g*2), width = 0.125, A=A, color = pp.gen_colors()[group], alpha = 0.25)
        bb = A.violinplot(thresholds_before, [g*2], showextrema = False, showmedians=True, widths=0.5)
        bb['cmedians'].set_color('w')#pp.gen_colors()[group])
        bb['cmedians'].set_linestyle('dotted')
        for bb in bb['bodies']:
            bb.set_facecolor(pp.gen_colors()[group])
            
        beeswarm(thresholds_after, (g*2)+1, width = 0.125, A=A, color = pp.gen_colors()[group], alpha = 0.25)
        ba = A.violinplot(thresholds_after, [(g*2)+1], showextrema = False, showmedians=True, widths=0.5)
        ba['cmedians'].set_color('w')#pp.gen_colors()[group])
        ba['cmedians'].set_linestyle('dotted')
        for bb in ba['bodies']:
            bb.set_facecolor(pp.gen_colors()[group])    
            
            
        ##pdb.set_trace()
        A.set_yscale('log')
        A.set_yticks([10,100,1000])
        A.set_ylim([10,1000])
        A.set_ylabel('Force (mN)')
        A.set_title('Threshold stim')
        
        #thresholds = np.log(thresholds)
        beeswarm(best_before, (g*2), width = 0.125, A=B, color = pp.gen_colors()[group], alpha = 0.25)
        cb = B.violinplot(best_before, [g*2], showextrema = False, showmedians=True, widths=0.5)
        cb['cmedians'].set_color('w')#pp.gen_colors()[group])
        cb['cmedians'].set_linestyle('dotted')
        for bb in cb['bodies']:
            bb.set_facecolor(pp.gen_colors()[group])
            
        beeswarm(best_after, (g*2)+1, width = 0.125, A=B, color = pp.gen_colors()[group], alpha = 0.25)
        ca = B.violinplot(best_after, [(g*2)+1], showextrema = False, showmedians=True, widths=0.5)
        ca['cmedians'].set_color('w')#pp.gen_colors()[group])
        ca['cmedians'].set_linestyle('dotted')
        for bb in ca['bodies']:
            bb.set_facecolor(pp.gen_colors()[group])    
            
            
       
        B.set_yscale('log')
        B.set_yticks([10,100,1000])
        B.set_ylim([10,1000])
        B.set_ylabel('Force (mN)')
        B.set_title('Optimal stim')
    #A.set_xticks([x for x, v in enumerate(data.keys())])
    #A.set_xticklabels([x for x in data.keys()])
    #B.set_xticks([x for x, v in enumerate(data.keys())])
    #B.set_xticklabels([x for x in data.keys()])
    box_off(A)
    box_off(B)
    save_fig(F, 'Mech thresholds')
    save_fig(G, 'Mech best stim')
  #  return(best)
    
    
    
    
    
    
    
    
    
    
    
    
    
    return(data)

    
    
    
    
    
    
def longitudinal_aligned_rasters(files = 'SNI', time_points = [(0,0) , (14, 21)], reps=[1], Tseries=None, use_first = True, time_precision = 2, check_class = True, key_session = 0, skip_missing = True, sort_mode = 'stim_preference', nClusters = 3, detrend=False, dff = False, dff_z = True, baseline_correct= True, crit = 0.8, minv = 0.1, vmin = None, vmax = None, plot=True, warm_factor = 3, check_no_response=True):
    # if files == 'SNI':
    #     files = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/273/Mouse 273 LA for thermo grid.pickle',
    #             '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/573/573_LA detrended.pickle']
    results = {}
    
    round_days = True
    if files == 'SNI':
        a = get_exp_days(collect_SNI_alignments())
        perturbation = 'SNI'
    elif files == 'CAPS':
        perturbation = 'CAPS'
        a = get_exp_days(collect_CAPS_alignments(), start_event='CAPS')
        time_points = [(0,0), (0.001,1)]
        round_days = False
    
    if Tseries is None:
        Tseries = [(0,12),(41,43.5),(48.5,52)]
    elif Tseries == 'file':
        Tseries = la.read_T_series('/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Tseries SNI raster')
    elif type(Tseries) is str: 
        Tseries = la.read_T_series(Tseries)
        
        # Tseries = [(3,12), (19,24), (37,40), (41,43), (43,46), (46,48), (48,52)]
    if vmin is None:
        if dff:
            vmin = 0
            if dff_z:
                vmin = 1
        else:
            vmin = 0.025
    if vmax is None:
        if not dff:
            vmax = 0.3
        elif dff and dff_z:
            vmax = 20
    
    
    
        
    ## include multisessions that have observations at desired time points, get session indices:
    session_indices_dict = {}
    session_dict = {}
    files={}
    cells = {}
    for cohort in a:
        if check_no_response:
            delete_from_raster = []
        files[cohort] = {}
        cells[cohort] = {}
        #cohort_n_cells = 0
        session_indices_dict[cohort] = {}
        session_dict[cohort] = {}
        for tp in time_points:
            session_dict[cohort][tp] = []
            cells[cohort][tp] = []
        for m, multi_session in enumerate(a[cohort]):
            multi_session.claim_sessions()
            multi_session.index_cells()
            db_name = os.path.split(multi_session.pickle_path)[-1].split('.')[0]
            files[cohort][db_name] = multi_session
            session_indices_dict[cohort][db_name] = {}
            start_time = multi_session.event_time
            for s, session in enumerate(multi_session.sessions):
                
                delta_t = session.experiment_start - start_time
                print(f'{session.experiment_start=} {start_time=} {delta_t=}')
                if delta_t <=0:
                    delta_t = 0
                if round_days:
                    experiment_day = round(delta_t/(60*60*24))
                else:
                    experiment_day = delta_t/(60*60*24)
                    if experiment_day <0 and experiment_day > -1:
                        experiment_day = 0
                for tt, tp in enumerate(time_points):
                    if experiment_day>=tp[0] and experiment_day<= tp[1]:   #add session index to dictionray
                        printc.g(f'{experiment_day} is between {tp[0]} and {tp[1]}')
                        if not (tp in session_indices_dict[cohort][db_name]): ## check that timepoint in range has not already been added
                            printc.g('Adding session to experiment')    
                            session_indices_dict[cohort][db_name][tp] = s
                            if dff:
                                session.convert_traces_to_dff(dff_z_score=dff_z)
                            if detrend:
                                
                                session.detrend(rectify=False)
                              
                            for c in session.cells:
                                cells[cohort][tp].append(session.cells[c])
                                
                            session_dict[cohort][tp].append(session)
                            
                        else:
                            printc.y('A session has already been added for this interval')
                    else:
                        printc.r(f'{experiment_day} is not between {tp[0]} and {tp[1]}')
    
    
    
    
    cell_counts = {}
    
    #check that number of neurons at each time point within multisession match
    for cohort in session_indices_dict:
        for db_name in files[cohort]:
            mult = la.unpickle(files[cohort][db_name].pickle_path)
            
            cell_counts[db_name] = []
            for n_session, session_index in enumerate(session_indices_dict[cohort][db_name].values()):
                n_cells = mult.sessions[session_index].raster.shape[0]
                cell_counts[db_name].append(n_cells)
                if n_session: ## If not looking at firssts session:
                    if cell_counts[db_name][-1] != cell_counts[db_name][0]: ## check that n neurons is same with prev session
                        input_data = {}
                        input_data['pickle file path'] = files[cohort][db_name].pickle_path
                        G = Alignment_GUI(input_data)
                        G.show()
                        
                        printc.y('Opening DB <{db_name}> because mismatch in cell count')
                        
                
              
    ## check that all multisession objects have a session entered at each time point
    for cohort in session_indices_dict:
        for tp in time_points:
            for db_name in files[cohort]:
                
                if not tp in session_indices_dict[cohort][db_name]:
                    
                    input_data = {}
                    input_data['pickle file path'] = files[cohort][db_name].pickle_path
                    G = Alignment_GUI(input_data)
                    G.show()
                    printc.y(f'Opening DB <{db_name}> because missing time point {tp}')
                
                    
                   # return(session_indices_dict)
    
    
    
    ## get rasters and stims for each cohort and time poiint
    end_fix_duration = 80-1
    prepend_frames = 60
    append_frames = 80
    raster_width = end_fix_duration+prepend_frames+append_frames
  
    time_point_str = ''
    for tp in time_points:
        time_point_str = time_point_str + ' ' + str(tp)
    rasters = {}
    sorted_rasters = {}
    stims = {}
    errors ={}
    #sum_check = {}
    #cell_dict = {}
    for cohort in a:
        for tp in time_points:
            rasters[(cohort, tp)], stims[(cohort,tp)], errors[(cohort,tp)] = la.align_thermo_rasters(session_dict[cohort][tp], cohort_name = cohort + '-' + str(tp), nClusters = nClusters, PLOT=False, end_fix_duration=end_fix_duration, prepend_frames = prepend_frames, append_frames=append_frames, Tseries=Tseries, reps=reps, do_sort=False)
            #cell_dict[(cohort, tp)] = errors[(cohort, tp)]['cell_list']
            
    ## Adjust responses to baseline:
    if baseline_correct:
        for cohort in a:
            for tp in time_points:
                r = rasters[cohort, tp]
                for tt, trace in enumerate(r):
                    for ss, stim in enumerate(Tseries):
                        baseline_start = (ss * raster_width)
                        baseline_end = (ss*raster_width) + prepend_frames
                        baseline_mean = np.mean(trace[baseline_start:baseline_end])
                        stim_end = (ss+1)*raster_width
                        corrected_trace = trace[baseline_start:stim_end] - baseline_mean
                        rasters[cohort, tp][tt,baseline_start:stim_end] = corrected_trace
    
    if check_no_response:
        r= {}
        for cohort in a:
            keep_traces = []
            for tp in time_points:
                r[tp] = rasters[cohort, tp]
            for tt, trace in enumerate(r[time_points[0]]):
                maxV = 0
                for tp in time_points:
                    maxV = max([maxV, np.amax(r[tp][tt])])
                if maxV >= minv:
                    keep_traces.append(tt)
            for tp in time_points:
                rasters[cohort, tp] = rasters[cohort, tp][keep_traces]
                cells[cohort][tp] = [cells[cohort][tp][i] for i in keep_traces]
    
           
    
    ### Sort rasters by initial response type, then by change in response
    #nClusters = 3
    max_vals={}
    assignment_cells = {}
    assignment_dict = {}
    for cohort in a:
        max_vals[cohort] = []
        if plot:
            F = plt.figure(cohort)
        key_raster = rasters[(cohort, time_points[0])]
        normalized_key_raster = key_raster*0
        for rr, trace in enumerate(key_raster):
            normalized_key_raster[rr,:] = np.nan_to_num(trace/np.amax(trace))
        
        assignment_dict[cohort] = {}
        assignment_cells[cohort] = {}
    
        assignments = []
        
        if sort_mode == 'kmeans':
            k = KMeans(n_clusters = nClusters).fit(normalized_key_raster) 
            centers = k.cluster_centers_
            labels = k.labels_
            centers = np.vstack([centers, centers[-1,:]*0]) ## add cluster center with no activity
    
            CF = plt.figure(f'{cohort} initial centers')
            for cc, center in enumerate(centers):
                assignment_dict[cohort][cc] = []
                assignment_cells[cohort][cc] = {}
                for tp in time_points:
                    assignment_cells[cohort][cc][tp] = []
                CF.add_subplot(1,nClusters+1, cc+1)
                plt.plot(center)
        
            ## assign traces to clusters measuring distances to all clusters
            
            for tt, trace in enumerate(normalized_key_raster):
                distances = []
                for center in centers:
                    distance = np.linalg.norm(center-trace)
                    distances.append(distance)
                assignment = np.where(distances == np.amin(distances))[0][0]
                assignments.append(assignment) ## 
                assignment_dict[cohort][assignment].append(tt)
                for tp in time_points:
                    assignment_cells[cohort][assignment][tp].append(cells[cohort][tp][tt])
        
        # crit = 0.65
        #minv =0.1
        elif sort_mode =='stim_preference':
            for s, stim in enumerate(Tseries):
                assignment_dict[cohort][s] = []
                assignment_cells[cohort][s] = {}
                for tp in time_points:
                    assignment_cells[cohort][s][tp] = []
            assignment_cells[cohort][len(Tseries)] = {}
            for tp in time_points:
                assignment_cells[cohort][len(Tseries)][tp] = []
            assignment_dict[cohort][len(Tseries)] = [] 
            for tt, (n_trace, trace) in enumerate(zip(normalized_key_raster, key_raster)):
                max_val = np.amax(trace)#*crit
                max_vals[cohort].append(max_val)#print(f'{max_val=}')
                if max_val < minv: ## If max value of trace in key session less than minv, classify as nonresponsive
                    
                    assignment_dict[cohort][len(Tseries)].append(tt)
                    for tp in time_points:
                        try:
                            assignment_cells[cohort][len(Tseries)][tp].append(cells[cohort][tp][tt])
                        except:
                            pdb.set_trace()
                else:
                    summed_responses = np.zeros(3)
                    for stim_num, s_range in enumerate(Tseries):
                        stim_start = (stim_num * raster_width) + prepend_frames
                        stim_end = ((stim_num+1)* raster_width)-1
                        if perturbation == 'CAPS':
                            stim_end = stim_start + end_fix_duration
                        response = trace[stim_start:stim_end]
                        summed_responses[stim_num] = np.sum(response)
                    max_response = np.amax(summed_responses)
                    #db.set_trace()
                    for summed_response_n, summed_response in enumerate(summed_responses):
                        if summed_response >= (max_response*crit):
                            max_stim = summed_response_n
                            break
                    
                            
                    if check_class:
                        if max_stim == 2: ## if classed as hot check if it should be warm
                            key_cell = cells[cohort][time_points[key_session]][tt]
                            if summed_responses[1] >= (max_response*crit)/warm_factor:
                                max_stim = 1
                            key_cell.classify_thermo()
                            if key_cell.classification == 'warm':
                                max_stim = 1
                      #  first_max = np.where(trace[50:]>max_val)[0][0] + 50 ## this is to make sure peak is not before stim
                      #  max_stim = np.floor(first_max/raster_width)
                    assignment_dict[cohort][max_stim].append(tt)
                    for tp in time_points:
                        try:
                            assignment_cells[cohort][max_stim][tp].append(cells[cohort][tp][tt])
                        except:
                            pdb.set_trace()
                            
        elif sort_mode == 'classification':
            thermo_classes = la.temp_class_bounds()
            for thermo_class in thermo_classes:
                if thermo_class != 'poly':
                    assignment_dict[cohort][thermo_class] = []
                    assignment_cells[cohort][thermo_class] = {}
                    for tp in time_points:
                        assignment_cells[cohort][thermo_class][tp] = []
            assignment_dict[cohort]['none'] = []
            assignment_cells[cohort]['none'] = {}
            for tp in time_points:
                assignment_cells[cohort]['none'][tp] = []
            for tt, (n_trace, trace) in enumerate(zip(normalized_key_raster, key_raster)):
                max_val = np.amax(trace)#*crit
                if max_val < minv: ## If max value of trace in key session less than minv, classify as nonresponsive
                    
                    assignment_dict[cohort]['none'].append(tt)
                    for tp in time_points:
                        assignment_cells[cohort]['none'][tp].append(cells[cohort][tp][tt])
                key_cell = cells[cohort][time_points[key_session]][tt]
                key_cell.classify_thermo()
                if key_cell.classification is None:
                    assignment_dict[cohort]['none'].append(tt)
                    for tp in time_points:
                        assignment_cells[cohort]['none'][tp].append(cells[cohort][tp][tt])
                elif key_cell.classification == 'poly':
                    pass
                else:
                    assignment_dict[cohort][key_cell.classification].append(tt)
                    for tp in time_points:
                        assignment_cells[cohort][key_cell.classification][tp].append(cells[cohort][tp][tt])
                        
        
                    
        todel = []
        for cluster in assignment_dict[cohort]:
            if len(assignment_dict[cohort][cluster]) == 0:
                todel.append(cluster)
        #if perturbation == 'CAPS':
         #   if not 3 in todel:
          #      todel.append(3)
        for d in todel:
            print(f'Deleting assignment_dict cohort{cohort} cluster{d}')
            del assignment_dict[cohort][d]
            
        
                        
        ## sort withiin raaaster by amount and direction of change in total response
       # G = plt.figure(f'{cohort} first lasts raster')
        for cluster in assignment_dict[cohort]:
            
            try:
                baseline_raster = rasters[cohort, time_points[0]][assignment_dict[cohort][cluster]]
            except:
                pdb.set_trace()
            try:
                final_raster = rasters[cohort, time_points[-1]][assignment_dict[cohort][cluster]]
            except:
                pdb.set_trace()
                
            ratios = []
            for b_trace, f_trace in zip(baseline_raster, final_raster):
                ratios.append(np.sum(b_trace)-np.sum(f_trace))
            ratios = np.array(ratios)
            order = np.argsort(ratios).astype(np.uint32)
            assignment_dict[cohort][cluster] = np.array(assignment_dict[cohort][cluster])[order]
            for tp in time_points:
                try:
                    assignment_cells[cohort][cluster][tp] = [assignment_cells[cohort][cluster][tp][i] for i in order]
                except:
                    pdb.set_trace()
            
            
       #     G.add_subplot(cluster+1,2,(2*cluster)+1)
       #     plt.imshow(baseline_raster, vmin=0,vmax=0.5)
       #     G.add_subplot(cluster+1,2,(2*cluster)+2)
       #     plt.imshow(final_raster,  vmin=0,vmax=0.5)
            
        
        ### concatenate new indices
        newIX = []
        for cluster in assignment_dict[cohort]:
            newIX.extend(assignment_dict[cohort][cluster])
        
        
        # sort all rasters by key session cluster assignment
        for tt, tp in enumerate(time_points):
            sorted_rasters[(cohort, tp)] = rasters[(cohort, tp)][newIX]
            cells[cohort][tp, 'sorted'] = []
            for ix in newIX:
                cells[cohort][tp, 'sorted'].append(cells[cohort][tp][ix])
            #cells[cohort][tp] = cells[cohort][tp]
            
       # l_margin = 0.1
       # r_margin = 0.1
        """
        Plot rasters
        """
        if plot:
            h_margin = 0.1
            v_margin = 0.025
            h_frac = (1-(2*h_margin))/len(time_points)   
            
            s_top = 0.8
            v_gap = 0.005
            h_gap = 0.05
            # plot separating each cluster in separate axes
            for tt, tp in enumerate(time_points):
                A = F.add_axes([h_margin+(h_frac*tt), s_top+v_gap, h_frac - h_gap, (1-s_top)-v_gap-0.05])
                
                A.plot(stims[cohort, tp])
                if tt==0:
                    box_off(A, left_only=True)
                    A.set_yticks([0,30,60])
                    A.set_ylabel('Temp')
                else:
                    box_off(A, All = True)
                A.set_xlim([0, stims[cohort, tp].shape[0]])
                A.set_ylim([0, 60])
                
                top = s_top*1
                for cluster in assignment_dict[cohort]:
                    sub_raster = rasters[(cohort, tp)][assignment_dict[cohort][cluster]]
                    cluster_n = sub_raster.shape[0]
                   
                    v_frac = (cluster_n/rasters[(cohort, tp)].shape[0])*(s_top-v_margin)
                    
                    left = h_margin+(h_frac*tt)
                    bot = (top-v_frac)
                    width = h_frac-h_gap
                    height = v_frac - v_gap
                    if height <0:
                        height = 0
                    #print(f'{left=}')
                    A = F.add_axes([left, bot, width, height])
                    top = top - v_frac
                    A.imshow(sub_raster, vmin = vmin, vmax = vmax, aspect = 'auto', cmap='inferno', interpolation='None')
                    box_off(A, All = True)
            # for tt, tp in enumerate(time_points):
    
            #     A=F.add_subplot(1,2,tt+1)
               
            #     A.imshow(sorted_rasters[(cohort, tp)], vmin = vmin, vmax = vmax, aspect = 'auto', cmap='inferno')
            save_fig(F, cohort + time_point_str)
    
    ## Plot Average traces for each cohort and baseline type, baseline against latest time point:
    #for cohort in a:
        
    
        #plt.figure(f'{cohort} max hist')
        #plt.hist(max_vals[cohort], bins = 40, cumulative = True)
   #return(session_indices_dict)  
    return(rasters, stims, errors, cells, assignment_cells, assignment_dict)
                        
    
                
                
            
        
        
    
    
    
        
        
    
#########
def panel_thermo_stability_scatter(markers = ['h','o','s'], files = 'default', markersize=30, fontsize = 24):
    
    if files == 'default':
        files = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Stability FOV H5 files/5896 stability/5896 for stability fig.pickle',
                 '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Stability FOV H5 files/7241 stability/7241 stability for fig.pickle',
                 '/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/Stability FOV H5 files/7242 stability/7242 for stability.pickle']
       
        
    F=plt.figure()
    A = F.add_axes([0,0,1,1])
    
    for file, marker in zip(files, markers):
       
        m =la.unpickle(file)
        m.scatter_tuning_between_sessions(F=F,A=A,marker=marker, markersize=markersize)
    
    bounds = la.temp_class_bounds()
    ticks = []
    labels = []
    for name, bound_pair in bounds.items():
        if name == 'warm':
            bound_pair = -0.5*np.pi
        ticks.append(np.mean(bound_pair))
        labels.append(name)
        
    A.set_xticks(ticks)
    A.set_yticks(ticks)
    A.set_xticklabels(labels, fontsize=fontsize)
    A.set_yticklabels(labels, fontsize=fontsize)
    
    box_off(A)
    plt.plot([-2*np.pi,-2*np.pi],[2*np.pi,2*np.pi], alpha = 0.5, color='k')
    
    save_fig(F, 'Stability scatter')
    
    
def fig_caps_all():
    pass

def panels_CAPS_imaging(*args, **kwargs):
    pass
    
    
def panels_CAPS_behavior(*args, **kwargs):
    p = {
         'trial_flags' : ['vas', 'caps'],
         'bins' : 15,
         'cum' :False,
         'return_line_plot' : False,
         'Al' : None
          }
    
    p.update(kwargs)
    colors = {}
    colors['vas'] = 'c'
    colors['caps'] = 'm'
    data = la.unpickle('/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/TRG.pickle')
    num_trials = len(data['caps']['Ts'])  + len(data['vas']['Ts'])
    hist_alpha = 2/num_trials
    
    F={}
    G={}
    fname = 'Preference'
    F[fname] = plt.figure(fname)
    A = F[fname].add_axes([0,0,1,1])
    for c, condition in enumerate(p['trial_flags']):
        print(condition)
        x = np.ones(len(data[condition]['prefs']))*c
        A.scatter(x, data[condition]['prefs'], color = colors[condition])
        A.scatter(x[0], np.average(data[condition]['prefs']), color = colors[condition], marker='_', s=200)
    A.set_ylim([15,45])
   
    A.set_yticks([15, 30, 45])
    A.set_yticklabels(['15', '30', '45'], fontsize=32)
    A.set_xlim([-1, 2])
    A.set_xticks([0,1])
    A.set_xticklabels(['Vehicle', 'Caps'], fontsize=32)
    box_off(A)
    
    cmean = np.average(data['caps']['prefs'])
    vmean = np.average(data['vas']['prefs'])
    
    t, pv = scipy.stats.ttest_ind(data['caps']['prefs'], data['vas']['prefs'])
    A.text(0.5,40, str(f'p = {pv}'))
    A.text(-0.5, vmean, str(round(vmean,2)))
    A.text(0.5, cmean, str(round(cmean,2))  )    
    A.set_ylabel(f'Temp {tr.degC()}', fontsize=32)
    
    
    fname = 'Histograms overalid'
    F[fname] = plt.figure(fname, figsize = (2,4))
    A1 = F[fname].add_subplot(1,1,1)
    fname= 'Thermal zone occupancy'
    F[fname] = plt.figure(fname)
    A2 = F[fname].add_subplot(1,1,1)
    #Ah = F.add_axes([0,0,1,1])
    Tconcat = {}
    
    for c, condition in enumerate(data.keys()):
        data[condition]['Ts'][c] = data[condition]['Ts'][c][2000:]
        data[condition]['counts_bins'] = []
    for c, condition in enumerate(data.keys()):
        Tconcat[condition] = list(itertools.chain.from_iterable(data[condition]['Ts']))
        for T in data[condition]['Ts']:
            counts, bins, patches = A1.hist(T, color = colors[condition], alpha = hist_alpha, range = (10,60), density=True, bins=p['bins'], cumulative = p['cum'])
            data[condition]['counts_bins'].append( (counts, bins) )
        A2.hist(Tconcat[condition], color = colors[condition], alpha = 0.5, range = (10,60), density =True, bins=p['bins'], cumulative = p['cum'])
    box_off(A1)
    
    
    box_off(A2)
    
    
    fname = 'Hist line'
    F[fname] = plt.figure(fname)
    if p['return_line_plot']:
        for fig in F:
            plt.close(F[fig])
        A3 = p['Al']
    else:
        A3 = F[fname].add_subplot(2,1,1)
        
    #construct matrices to calculate mean and std error of bin densities
    hist_mat = {}
    #pdb.set_trace()
    for c, condition in enumerate(data.keys()):
        hist_mat[condition] = np.zeros([len(data[condition]['Ts']), p['bins']])
        for ct, T in enumerate(data[condition]['Ts']):
            hist_mat[condition][ct,:] = data[condition]['counts_bins'][ct][0]
        plt.figure()
        plt.imshow(hist_mat[condition])
    
    #get mean and std error at each bin
    edges = data[condition]['counts_bins'][0][1]
    centers = (edges[1:]+edges[:-1])/2
    for c, condition in enumerate(hist_mat):
        line_mean = np.average(hist_mat[condition], axis=0)
        line_std  = np.std(hist_mat[condition], axis=0)
        SE = line_std/np.sqrt(hist_mat[condition].shape[0])
        
        #A3.plot(centers, line_mean, colors[condition])
        A3.errorbar(centers, line_mean, color=colors[condition], yerr = SE)
    box_off(A3)
    X = [10,20,30,40,50,60]
    Xs = [str(x) for x in X]
    A3.set_xticks(X)
    A3.set_xticklabels(Xs, fontsize=9)
    #A3.set_yticks([0,0.12])
    #A3.set_yticklabels(['0', '0.08'])
    A3.set_xlabel(f'Temp {tr.degC()}', fontsize = 9)
    A3.set_ylabel('Occupancy', fontsize = 9)
    if p['return_line_plot']:
        return()
    save_fig_dict(F)
    return(data)

def caps_behavior_recovery(files = None, figsize = (6,6), *args, **kwargs):
    p = {
         'trial_flags' : ['vas', 'caps'],
         'bins' : 15,
         'cum' :False
          }
    
    p.update(kwargs)
    colors = {}
    colors['vas'] = 'c'
    colors['caps'] = 'm'
    VF = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Behavior for Figures/Caps VF time series.xlsx'
    H = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Behavior for Figures/CAPS hargreaves time series.xlsx'
    Hc = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Behavior for Figures/Hargreaves contra.xlsx'
    
    VFdata = pandas.read_excel(VF)
    Hdata = pandas.read_excel(H)
    Hc_data = pandas.read_excel(Hc)
    F = plt.figure('CAPS behavior', figsize = figsize, tight_layout = True)
    
    A1 = F.add_subplot(2,1,1)
    panels_CAPS_behavior(return_line_plot =True, Al = A1)
    A1.set_yticks([0,0.1])
    A1.set_yticklabels(['0', '0.1'], fontsize=9)
    A1.set_ylim([0,0.12])
    A1.set_title('Thermal gradient ring', fontsize=9)
    
    A1.text(55, 0.1, 'Caps', color = 'r', fontsize=9, horizontalalignment = 'right')
    A1.text(55, 0.08, 'Veh', color = 'c', fontsize=9,  horizontalalignment = 'right')
    
    A2 = F.add_subplot(2,2,3)
    
    vf = {}
    for column in VFdata:
        vf[column] = VFdata[column][:-1]
    
    h = {}
    for column in Hdata:
        h[column] = Hdata[column]
    for column in Hc_data:
        h[column] = Hc_data[column]
        
    vf_ipsi_keys = ['Baseline I', 'I 0', 'I 1', 'I 2']
    vf_ipsi_x = [0,1,2,3]
    
    vf_contra_keys = ['Baseline C', 'C day 1', 'C day 2']
    vf_contra_x = [0,2,3]
    
    h_ipsi_keys = ['Hlat Base Ipsi', 'Hlat D0 Ipsi', 'Hlat D1 Ipsi', 'Hlat D2 Ipsi']
    h_ipsi_x = [0,1,2,3]
    
    
    h_contra_keys = ['Hlat Base Contra', 'Hlat D1 Contra', 'Hlat D2 Contra']
    h_contra_x  =[0,2,3]
    
    
    #pdb.set_trace()
    
    ## plot hargreaves (ipsi o
    means =[]
    sterrs = []
    for cell in range(Hdata.shape[0]):
        ys = []
        
        for k in h_ipsi_keys:
            ys.append(h[k][cell])
            if cell == 0:
                means.append(np.average(h[k]))
                sterrs.append(  np.std(h[k])/np.sqrt(Hdata.shape[0]))
        A2.plot(h_ipsi_x, ys, color = 'r', alpha = 0.1)
    A2.errorbar(h_ipsi_x, means, yerr = sterrs, color = 'r')
    
    means =[]
    sterrs = []
    for cell in range(Hdata.shape[0]):
        ys = []
        
        for k in h_contra_keys:
            ys.append(h[k][cell])
            if cell == 0:
                means.append(np.average(h[k]))
                sterrs.append(  np.std(h[k])/np.sqrt(Hdata.shape[0]))
        A2.plot(h_contra_x, ys, color = 'k', alpha = 0.1)
    A2.errorbar(h_contra_x, means, yerr = sterrs, color = 'k')
    
    
    
    A2.set_yticks([0,10,20])
    A2.set_xticks(h_ipsi_x)
    A2.set_ylabel('Withdrawal latency (s)', fontsize=9)
    A2.set_xticklabels(['Base', '0-15min', '24h', '48h'], rotation=45, fontsize=9)
    A2.set_xlim([-0.5,3.5])
    A2.set_title('Hargreaves', fontsize=9)
    box_off(A2)
    
    
    A3 = F.add_subplot(2,2,4)
    ## plot VF ipsi:
    means =[]
    sterrs = []
    for cell in range(VFdata.shape[0]-1):
        ys = []
        
        for k in vf_ipsi_keys:
            ys.append(vf[k][cell])
            if cell == 0:
                means.append(np.average(vf[k]))
                sterrs.append(  np.std(vf[k])/np.sqrt(VFdata.shape[0]))
        A3.plot(vf_ipsi_x, ys, color = 'r', alpha = 0.1)
    A3.errorbar(vf_ipsi_x, means, yerr = sterrs, color = 'r')
    
    means =[]
    sterrs = []
    for cell in range(VFdata.shape[0]-1):
        ys = []
        
        for k in vf_contra_keys:
            ys.append(vf[k][cell])
            if cell == 0:
                means.append(np.average(vf[k]))
                sterrs.append(  np.std(vf[k])/np.sqrt(VFdata.shape[0]))
        A3.plot(vf_contra_x, ys, color = 'k', alpha = 0.1)
    A3.errorbar(vf_contra_x, means, yerr = sterrs, color = 'k')
    
    A3.set_ylabel('50% withdrawal (gf)', fontsize=9)
    A3.set_yticks([0,0.5,1])
    A3.set_xticks(h_ipsi_x)
    A3.set_xticklabels(['Base', '0-15min', '24h', '48h'], rotation=45,  fontsize=9)
    A3.set_xlim([-0.5,3.5])
    A3.set_title('Von frey', fontsize=9)
    A3.text(3,0.92, 'Ipsi', color = 'r', fontsize=9, horizontalalignment = 'right')
    A3.text(3,0.8, 'Contra', color = 'k', fontsize=9,  horizontalalignment = 'right')
    box_off(A3)
    plt.tight_layout()
    #pdb.set_trace()
    save_fig(F, 'Caps behavior'+str(figsize))
        
    return(vf, h)
    
def caps_hargreaves(files = None):
    if files is None:
        caps_file = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/CAPS-VAS behaviors/CAPS_VF_HAGREAVES.csv'
        vas_file = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/CAPS-VAS behaviors/capsaicin data_master_sheet.xlsx - Vas data preliminary.csv'
    data = pandas.read_csv(caps_file)
    
    ## plot all data in F, plot VF CPAS onyl on F1
    F1 = plt.figure(figsize = [2,4])
    B = F1.add_subplot(1,1,1)
    F = plt.figure(figsize = [8, 4])
    ## Plot caps onlly in F1
    #F1 = plt.figure(figsize = [4,4])
    num_plots = 3
    
    d = {}
    for column in data:
        d[column] = data[column]
    
    data = pandas.read_csv(vas_file)
    v = {}
    for column in data:
        v[column] = data[column]
    
    p = {}
    #return(d,v)
    """
    Plot contra and ipsi VF thresholds before and after caps, Vas
    """

    pre = 'L Base VF'
    post = 'L Contra Caps VF '
    
    A1 = F.add_subplot(1, num_plots, 1)
    for base, CAPS in zip(d[pre], d[post]):
        plt.plot([1,2], [base, CAPS], color = [0.85,0.85,0.85])
        B.plot([1,2], [base, CAPS], color = [0.85,0.85,0.85])
    B_mean = np.average(d[pre])    
    C_mean = np.average(d[post])
    plt.plot([1,2], [B_mean, C_mean], color = 'k')
    B.plot([1,2], [B_mean, C_mean], color = 'k')
    s, p[pre,post] = stats.ttest_rel(d[pre], d[post])
   
    
    pre = 'R Base VF'
    post = 'R Caps VF'
    

    for base, CAPS in zip(d[pre], d[post]):
        plt.plot([1,2], [base, CAPS], color = [1,0.85,1])
        B.plot([1,2], [base, CAPS], color = [1,0.85,1])
    s, p[pre,post] = stats.ttest_rel(d[pre], d[post])
    B_mean = np.average(d[pre])    
    C_mean = np.average(d[post])
    plt.plot([1,2], [B_mean, C_mean], color = 'm')
    B.plot([1,2], [B_mean, C_mean], color = 'm')
    
    
    """
    Plot VAS controls
    """
    
    pre = 'L Base VF'
    post = 'L Contra Vas VF '
    for base, CAPS in zip(v[pre], v[post]):
        plt.plot([3,4], [base, CAPS], color = [0.85,0.85,0.85])
    B_mean = np.average(v[pre])    
    C_mean = np.average(v[post])
    plt.plot([3,4], [B_mean, C_mean], color = 'k')
    s, p[pre,post, 'vas'] = stats.ttest_rel(v[pre], v[post])
    
    pre = 'R Base VF'
    post = 'R Vas VF'
    for base, CAPS in zip(v[pre], v[post]):
        plt.plot([3,4], [base, CAPS], color = [0.75,1,1])
    B_mean = np.average(v[pre])    
    C_mean = np.average(v[post])
    plt.plot([3,4], [B_mean, C_mean], color = 'c')
    s, p[pre,post, 'vas'] = stats.ttest_rel(v[pre], v[post])

    A1.set_ylim([0,1.2])
    box_off(A1)
    A1.set_xticks([1,2,3,4])
    A1.set_xticklabels(['Pre','CAPS','Pre', 'VAS'])
    A1.set_xlabel('Von Frey')
    A1.set_xlim([0.8,4.2])
    
    B.set_ylim([0,1.2])
    box_off(B)
    B.set_xticks([1,2])
    B.set_xticklabels(['Pre','CAPS'])
    B.set_xlabel('Von Frey')
    B.set_xlim([0.8,2.2])
    #return
   

    """
    Plot contra and ipsi hargreaves latencies before and after caps, VAS
    """
    pre = 'L Base Hlat'
    post = 'L Contra Caps Hlat'
    
    A2 = F.add_subplot(1, num_plots, 2)
    for base, CAPS in zip(d[pre], d[post]):
        plt.plot([1,2], [base, CAPS], color = [0.85,0.85,0.85])
    B_mean = np.average(d[pre])    
    C_mean = np.average(d[post])
    plt.plot([1,2], [B_mean, C_mean], color = 'k')
    s, p[pre, post] = stats.ttest_rel(d[pre], d[post])
   
    
    pre = 'R Base Hlat'
    post = 'R Caps Hlat'
    for base, CAPS in zip(d[pre], d[post]):
        plt.plot([1,2], [base, CAPS], color = [1,0.85,1])
    s, p[pre, post] = stats.ttest_rel(d[pre], d[post])
    B_mean = np.average(d[pre])    
    C_mean = np.average(d[post])
    plt.plot([1,2], [B_mean, C_mean], color = 'm')
 
    
  
    """
    Plot VAS controls for hargreaves
    """
    
    pre = 'L Base Hlat'
    post = 'L Contra Vas Hlat'
    for base, CAPS in zip(v[pre], v[post]):
        plt.plot([3,4], [base, CAPS], color = [0.85,0.85,0.85])
    B_mean = np.average(v[pre])    
    C_mean = np.average(v[post])
    plt.plot([3,4], [B_mean, C_mean], color = 'k')
    s, p[pre,post, 'vas'] = stats.ttest_rel(v[pre], v[post])
    
    pre = 'R Base Hlat'
    post = 'R Vas Hlat'
    for base, CAPS in zip(v[pre], v[post]):
        plt.plot([3,4], [base, CAPS], color = [0.75,1,1])
    B_mean = np.average(v[pre])    
    C_mean = np.average(v[post])
    plt.plot([3,4], [B_mean, C_mean], color = 'c')
    s, p[pre,post, 'vas'] = stats.ttest_rel(v[pre], v[post])
    
    
    A2.set_ylim([0,20])
    box_off(A2)
    A2.set_xticks([1,2,3,4])
    A2.set_xticklabels(['Pre', 'CAPS','Pre', 'VAS'])
    A2.set_xlabel('Hargreaves')
    A2.set_xlim([0.8,4.2])
    
    F.tight_layout()
    
    for pval in p:
        print(f'{pval}: {p[pval]}')
    save_fig(F, 'CAP VAS VF & HG')
    save_fig(F1, 'CAPS VF')
    return(d, v)
    

def panels_cold_guarding():
    F = beh.analyze_cold_plate_data(None)
    save_fig_dict(F)

def fig_SNI_thermo(real_time = False, norm = True, cohorts = None):
    ## get SNI data set, label sessions by day relative to SNI
    
    ds = get_exp_days(collect_SNI_alignments())
    #### Get basal cells of ecah clalss, plot tuning over time - traces, polar, scatter
    ##
    if cohorts is None:
        cohorts = ds.keys()
        
    master_tuning_matrices = {}
    for cohort in cohorts:
        master_tuning_matrices[cohort] = la.tuning_grid_across_mice(ds[cohort], real_time=real_time)
       
      
        for m in ds[cohort]: # m is multisession object
            la.thermo_grid_multi_session(m, real_time = real_time, norm = norm)
            
    return(master_tuning_matrices)
            
def trace_vs_color(mtm, mouse_ix, cells, cohort = 'Tacr1'): ## want to visualize how trace correspond to coloring scheme for grid array
    ###
    ### get multisession object
    for item in mtm[cohort]:
        if item[0] == mouse_ix:
            m = mtm[cohort][item]['parent']
            break
    ### retrieve colors for each cell and session to be plotted
    color_dict = {}
    for u_cell in cells:
        for session in range(m.n_sessions):
            norm = mtm[cohort][mouse_ix, u_cell, session]['norm']
            #norm.extend(mtm[cohort][mouse_ix, u_cell, session]['amp'])
            color_dict[u_cell, session] = norm
    color_mode = {}        
    color_mode[0] = 'color_dict_by_cell_and_session'
    color_mode[1] = color_dict
    m.plot_traces(selected_u_cells = cells, selected_sessions = 'All', color_mode = color_mode)
    
    
  
def collect_CAPS_alignments():
    files = {}
    files['rPbN'] = ["/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/CAPS FOV H5 files/Mouse 7241/7241 CAPS aligned and extracted.pickle"
                     ,"/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/CAPS FOV H5 files/Mouse 7242/7242 CAPS2-post unified.pickle"
                     ,"/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/CAPS FOV H5 files/Mouse 7778/7778 unified for fig.pickle"
                     ,"/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/CAPS FOV H5 files/Mouse 8244/Mousse 8244 CAPS2 unified.pickle"
                     ]
    

                    
    data = {}
    for cohort in files.keys():
        data[cohort] = []
        for file in files[cohort]:
            data[cohort].append(la.unpickle(file))
    return(data)

def collect_SNI_mech_alignment():
    glabrous = {}
    glabrous['rPbN'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/379 SNI mech/379LA longitudinal.pickle'
                     ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/457A SNI mech/457ARA aligned glabrous.pickle'
                     ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/7241 LB glabrous aligned.pickle'
                     ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/7241 LC glabrous aligned.pickle'
                     ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/7241 LC glabrous aligned.pickle'
                     ]
    glabrous['Gpr83'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6046 SNI mech/6046LA aligned glabrous.pickle'
                      ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6356 SNI mech/6356LA glabrous aligned.pickle'
                      ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6355 SNI mech/6355 aligned glabrous.pickle'
                      ]
    glabrous['Tacr1'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/237 SNI mech/237 SNI mechLA longitudinal.pickle'
                      ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7243 SNI mech/7243 SNI medhRA baseline 2 weeks glabrous.pickle'
                      ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/414 SNI mech/414LC longitudinal.pickle'
                      ]
   

    
    hairy = {}
    hairy['rPbN'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/7241 LA hairy aligned.pickle'
                     ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/7241 LB hairy aligned.pickle'
                     ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/7241 LC hairy aligned.pickle'
                     ]
    hairy['Gpr83'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6046 SNI mech/6046LA aligned hairy.pickle'
                      ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6355 SNI mech/6355 aligned hairy.pickle'
                      ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6356 SNI mech/6356LA hairy aligned.pickle'
                      ]
    hairy['Tacr1'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7243 SNI mech/7243 SNI medhRA baseline 2 weeks hairy dff.pickle']
    
    combined = {}
    combined['Tacr1'] = []
    combined['Gpr83'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6046 SNI mech/6046LA aligned hairy and glabrous.pickle'
                         ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6355 SNI mech/6355sni aligned dff.pickle'
                         ,'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/6356 SNI mech/global alignment/6356 allLA longitudinal.pickle'
                         
                         
                         
                         ]
    combined['rPbN'] = ['/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/ SNI mech only/7241 SNI mech/LC all mech basal and two weeks.pickle']
    
    
    
    for group in glabrous.values():
        for file in group:
            m = la.unpickle(file)
            print(f'{file} has {len(m.sessions)} sessions')
    return(glabrous, hairy, combined)
    
    
def collect_SNI_alignments():
    #genotypes = ['Tacr1', 'rPbN', 'Gpr83']
    files = {}
    files['rPbN'] = ["/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/379/Mouse 379 unified.pickle"
                     , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/457/Mouse 457RA aligned  extracted unified detrended.pickle"
                     , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/7241/Moue 7241 LCLC unified.pickle"
                     , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/594/Mouse 594LA 0 and 2 weeks aligned unified.pickle"
                     ## 7241A needs to be checked, may be included
                        ]
    
    files['Tacr1'] = ["/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/273/Mouse 273 LA for thermo grid.pickle"
                     , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/573/573_LB unified detrended.pickle"
                     , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/573/573_LA detrended.pickle"
                     , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/7243/7243 RA SNIRA unified detrended.pickle"
                     ## 414 may not be well aligned enough
                     ]
    
    files['Gpr83'] = ["/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/6356/Mouse 6356 LA SNI unified.pickle"
                      , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/6046/Mouse 6046 LA SNI unified.pickle"
                      , "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Alignments/SNI FOV H5 files/704/Mouse 704 SNILA aliigned.pickle"
                      ]
                      ## 6355 not sufficient quality, cna be used for stability
                          
    del files['Gpr83'][0] ## basal data looks like acquired with inflammation - lots of 39 responses
    
    data = {}
    for cohort in files.keys():
        data[cohort] = []
        for file in files[cohort]:
            data[cohort].append(la.unpickle(file))
            
            
                

    
    return(data)
        

def fig_SNI_mechano():
    pass

def fig_spont():
    pass


def fig_awake():
    '''
    Assemble data from awake vs anesthetized mice
    Compare thermal and mech responses under both conditions
    '''
    awake_mice = ['1210',
                  '1211',
                  '379',
                  '5896',
                  '6356',
                  '6046',
                  '9960',
                  '1346']
    pass




























    