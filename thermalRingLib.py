#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:04:54 2022

@author: ch184656
"""

import pandas
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
import numpy as np
import cv2
import imageio
import pdb
from scipy import signal
from scipy.stats import ttest_ind
import itertools
import pickle
from libAnalysis import unpickle

class trial:
    def __init__(self, data):
        for key in data.keys():
            setattr(self, key, data[key])
            
    def flip_zones(self):
        self.temp = self.temp[::-1]
        self.temp.index = self.temp.index[::-1]
    
    def show(self, F=None, A= None, color = 'b', alpha = 0.5):
        if F is None:
            F = plt.figure(f'{self.Animal} {self.Treatment} {self.Trial}')
            A = F.add_axes([0,0,1,1])
            
   
        plt.plot(self.temp, self.occupancy, color = color, alpha = alpha)
        ax = plt.gca()
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        return(F,A)
        
    
class trg_set:
    def __init__(self, trials):
        self.trials = []
        self.conditions = {}
        self.add_data(trials)
        # for trial in trials:
        #     self.trials.append(trial)
        #     for attribute in trial.__dict__.keys():
        #         ### if condition is registered
        #         if attribute in ['occupancy', 'temp', 'preferred temp', 'Time', 'Distance', 'Mean speed', 'Max speed']:
        #             continue
        #         if attribute in self.conditions.keys():
        #             value = vars(trial)[attribute]
        #             print(value)
        #             if not value in self.conditions[attribute]:
        #                 self.conditions[attribute].append(value)
        #         else:
        #             self.conditions[attribute] = []
                    
    def add_data(self, trials):
        for trial in trials:
            self.trials.append(trial)
            for attribute in trial.__dict__.keys():
                ### if condition is registered
                if attribute in ['occupancy', 'temp', 'preferred temp', 'Time', 'Distance', 'Mean speed', 'Max speed']:
                    continue
                if attribute in self.conditions.keys():
                    value = vars(trial)[attribute]
                    print(value)
                    if not value in self.conditions[attribute]:
                        self.conditions[attribute].append(value)
                else:
                    self.conditions[attribute] = []
                    
    def pull(self, trials = None, **kwargs):
        
        ### put in pair of condition with list of acceptable values
        ## e.g. 'Treatment'  = ['caps, 'CFA]
        ### returns list of 
        T = []
        if trials is None:
            trials = []
            for i in range(len(self.trials)):
                trials.append(i)
            
        for trial in trials:
            for var, values in kwargs.items():
                if vars(self.trials[trial])[var] in values:
                    T.append(self.trials[trial])
                    
        return(trg_set(T))
                    
    def show(self, factor, cond_filter=None, Axes = None, separate=True, color_dict = None, group = False, raw_alpha=0.5, **kwargs):
        ## factor chooses which sets of measurments to compare  - condition filter can be used to specify  a subset of  conditions to plot
        ### for kwargs put in pair of condition with list of acceptable values to further filter which data is plotted
        ## e.g. 'Treatment'  = ['caps, 'CFA]
       
        if cond_filter is None:
            condition_list = self.conditions[factor]
        else:
            condition_list = cond_filter

        if color_dict is None:
            color_dict = gen_color_dict(len(condition_list))
            
        color_count = 0
     
        if group:
            F=self.show_grouped(factor = factor, cond_filter = condition_list, Axes = Axes, separate= separate, color_dict = color_dict, **kwargs)
            return(F)
        
        F={}
        if not separate:
            F['all'] = plt.figure(' vs '.join(str(x) for x in condition_list))
            FF = F['all']
            

        for condition in condition_list:
            if separate:
                print(condition)
                F[condition] = plt.figure(str(condition))
                plt.title(condition)
                FF = F[condition]
                
            for trial in self.trials:
                print(f'{str(vars(trial)[factor])=}')
                print(f'{condition=}')
                plotOn = True
                for var, value in kwargs.items():
                    if not vars(trial)[var] in value:
                        plotOn = False
                if vars(trial)[factor] == condition and plotOn:
                    if condition in color_dict.keys():
                        color = color_dict[condition]
                    else:
                        color = color_dict[color_count]
                    trial.show(F=FF, color = color, alpha = raw_alpha)
                    trace =trial.occupancy.to_numpy()
            color_count = color_count + 1
        plt.title(' vs '.join(str(x) for x in condition_list))
        return(F)
        
    def show_grouped(self, factor, cond_filter=None, Axes = None, separate=True, show_err = 2, color_dict = None, **kwargs):
        
        if cond_filter is None:
            condition_list = self.conditions[factor]
        else:
            condition_list = cond_filter
            
        if color_dict is None:
            color_dict = gen_color_dict(len(condition_list))
             
        
        color_count = 0
        F={}
        traces = {}
        temps = {}
        if not separate:
            F['all'] = plt.figure(' vs '.join(str(x) for x in condition_list))
            FF = F['all']
            

        for c, condition in enumerate(condition_list):
            
            if separate:
                print(condition)
                F[condition] = plt.figure(str(condition))
                plt.title(condition)
                FF = F[condition]
                
            for trial in self.trials:
                print(f'{str(vars(trial)[factor])=}')
                print(f'{condition=}')
                plotOn = True
                for var, value in kwargs.items():
                    if not vars(trial)[var] in value:
                        plotOn = False
                if vars(trial)[factor] == condition and plotOn:
                    if condition in color_dict.keys():
                        color = color_dict[condition]
                    else:
                        color = color_dict[color_count]
                    #trial.show(F=FF, color = color, alpha = raw_alpha)
                    trace =trial.occupancy.to_numpy()
                    temp = trial.temp
                    if condition in traces.keys():
                        traces[condition] = np.vstack([traces[condition], trace])
                        temps[condition]  = np.vstack([temps[condition], temp])
                    else:
                        traces[condition] = trace
                        temps[condition] = temp
                    
                    
        for condition in condition_list:
            m_trace = np.mean(traces[condition], axis = 0)
            
            m_temp = np.mean(temps[condition], axis=0)
            if separate:
                FF = F[condition]
            else:
                FF = F['all']
              #  A = F['All'].add_axes([0,0,1,1])
            plt.figure(FF)
            if condition in color_dict.keys():
                color = color_dict[condition]
            else:
                color = color_dict[color_count]
            plt.plot(m_temp, m_trace, color = color)
            if not show_err is None:
          
                m_dev = np.std(traces[condition].astype(np.float64), axis = 0)
                n = traces[condition].shape[0]
                se = m_dev/(np.sqrt(n))
                top = m_trace + (se*show_err)
                bot = m_trace - (se*show_err)
                #pdb.set_trace()
                for b,t,temp in zip(bot,top,m_temp):
                    
                    plt.plot([temp, temp], [b,t], color = color, marker = "_")
            ax = plt.gca()
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            
            
            color_count = color_count + 1
            if separate:
                plt.title('condition')
                
        if not separate:       
            plt.title(' vs '.join(str(x) for x in condition_list))
            
            
        return(F)
            
def gen_color_dict(n_color = 24):
    d = {}
    d['Cage A'] = 'b'
    d['Cage B']  = 'g'
    d['Group A'] = 'b' ## SNI for SNI pilot
    d['Group B'] = 'g'  ## sham for sni pilot
    d['A']   = 'b'
    d['B']   = 'g'
    d['0'] = 'k'
    d['A baseline']   = 'b'
    d['A day 3']   = 'b'
    d['A day 7']   = 'b'
    d['A day 14']   = 'b'
    d['A day 28']   = 'b'
    d['B baseline']   = 'g'
    d['B day 3']   = 'g'
    d['B day 7']   = 'g'
    d['B day 14']   = 'g'
    d['B day 28']   = 'g'
    d['3'] =  [1, 0.5, 0 ]
    d['7'] =  [1, 0.75, 0]
    d['14'] = [1, 0, 0.25]
    d['28'] = [1,0,0.5]
    d['vas'] = 'g'
    d['caps'] = 'r'
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    for i in range(n_color):
        d[i] = cmap(1/n_color)
        
        
    return(d)
         
        
    
def load_trg(file='/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Thermal ring data 2/21sept22_caps vs vas 10 min bins trial.csv', return_list = False):
    df = pandas.read_csv(file)
    data = {}
    Trials = []
    for reading in range(df.shape[0]):
        for column in range(1,11):
            key  = df.columns[column]
            value = df.iloc[reading, column]
            data[key] = value
        data['occupancy'] = df.iloc[reading,-12:]
        data['temp'] = df.iloc[reading, 12:24]
        data['preferred temp'] = df.iloc[reading, 36]
        data['fName'] = ''
        T = trial(data)
        Trials.append(T)
        #pdb.set_trace()
    if return_list: ## return a list of trial objects, otherswise return trg_set object
        return(Trials)
    return(trg_set(Trials))         

def default_exp_folder(IX):
    if IX == 0:
        folder = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Thermal ring data 2/SNI/'
    else:
        folder = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/Thermal ring data 2/CAPS - VAS/10 minute bins/'
        
    return(folder)

def load_trg_folder(x_folder):
    T = trg_set([])
    for currentPath, folders, files in os.walk(x_folder):
        for c, x_file in enumerate(files):
            print(x_file)
            if '.csv' in x_file:
                Trials = load_trg(os.path.join(x_folder, x_file), return_list = True)
                for Trial in Trials:
                    Trial.fName = x_file.split('.')[0]
                T.add_data(Trials)
              #  pdb.set_trace()   
    return(T)
                    

    
    
    
    
    
def abs_diff(stack):
    print(f'abs diff {stack.shape=}')
    difference = stack - np.median(stack, axis=0)
    output = np.absolute(difference)
    return(output)
    
def convert_stack_uint8(stack):
    stack = stack-np.amin(stack)
    stack = stack/np.amax(stack)
    stack = stack*255
    return(stack.astype(np.uint8))

def adaptive_threshold(stack, **kwargs):
    output = np.zeros(stack.shape, dtype = np.uint8)
    params = {'blocksize' : 11,
              'C' : 0}
    params.update(kwargs)
    for c, frame in enumerate(stack):
        output[c,:,:] = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, params['blocksize'], params['C'])
    return(output)

def locate_mouse(stack, **kwargs):
    output = np.zeros(stack.shape, dtype = np.uint8)
    params = {'minSize' : 1,
              'maxSize' : 50}
    params.update(kwargs)
    Xlist = []
    Ylist = []
    #loc=[0, 0]
    for c, frame in enumerate(stack):
        loc = [np.nan, np.nan]
        numCells, roimap, stats, centroids = cv2.connectedComponentsWithStats(frame)
       
        areas = stats[:,4]
        if numCells >1:
            IX = np.argsort(areas)[-2] #second biggest connected component (first is background)
            if areas[IX] >params['minSize'] and areas[IX] < params['maxSize']:
               # pdb.set_trace()
                loc = centroids[IX,:]
                output[c, np.where(roimap==IX)[0],np.where(roimap==IX)[1]] = 255
        Xlist.append(loc[1])
        Ylist.append(loc[0])
        
    return(Xlist, Ylist, output)
    
    
def diff_movie(stack):
    diff = np.diff(stack, axis=0)
    adiff = np.absolute(diff)
    return(adiff)

def GUI_plot_TRG(obj, stack = None, TIME = None, make_movie = True):
   # stack = obj.DB['Animals'][obj.curAnimal][obj.curFOV][obj.dataFocus][...]
    if stack is None:
        stack, TIME = obj.getSubStack()
    print(f'substack.shape is {stack.shape}')
    T, RGB = analyze_FLIR_TRG(stack, fig_name = f' {obj.curFOV} {obj.dataFocus}', make_movie = make_movie)
    return(RGB, TIME)

def analyze_FLIR_TRG(stack=None, plot_color='c', s=10, alpha = 25, vmin=10, vmax=60, fig_name = '', make_movie=False, hist_alpha=1):
    if stack is None:
        stack =sample_data()
    b = abs_diff(stack)
    c= convert_stack_uint8(b)
    d = adaptive_threshold(c, C=-10, blocksize=7)
    y,x, masked_stack = locate_mouse(d)
    T = get_T_from_pos(x,y,stack)
    t_colors = []
    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.Normalize(vmin,vmax)
    scaMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    for st in T:
        if st is np.nan:
            t_colors.append([0,0,0,0])
        else:
            RGBA = scaMap.to_rgba(st)
            R = RGBA[0]
            G = RGBA[1]
            B = RGBA[2]
            A = alpha * RGBA[3]/len(x)
            t_colors.append([R,G,B,A])
    f = plt.figure('TRG'+fig_name, figsize = [9,3])
    A1 = plt.subplot(2,3,1)
    #A1.imshow(np.median(a, axis=0), cmap = 'Greys_r', vmin = vmin, vmax = vmax)
    A1.scatter(x,y,  color = t_colors, s=s)
    A1.set_aspect('equal')
    A1.xaxis.set_visible(False)
    A1.yaxis.set_visible(False)
    A1.spines.right.set_visible(False)
    A1.spines.top.set_visible(False)
    A1.spines.left.set_visible(False)
    A1.spines.bottom.set_visible(False)
    
    
    #A1.set_title('Observed locations')
    r = A1.get_position()
    A1cbar = f.add_axes([r.xmin, r.ymin-0.1, r.width, 0.05])
    A1cbar = colorbar(vmin=vmin, vmax=vmax, axis = A1cbar, title = f'Temp {degC()}')   
    A2 = plt.subplot(2,3,2)
   
    h = A2.hist(T, color= plot_color, alpha = hist_alpha, density = True)
    #A2.yaxis.set_visible(False)
    A2.set_ylabel('Pref ratio')
    A2.set_yticks([np.round(0, decimals=0) ,np.round(np.amax(h[0]), decimals=1)])
    A2.spines.right.set_visible(False)
    A2.spines.top.set_visible(False)
    A2.set_xlim(vmin,vmax)
    
    A
    pT = np.round(np.nanmedian(T), decimals=1)
    print(f'{pT=}')
    pTs = str(pT)
    print (pTs)
    A2.set_xlabel(r'$T_{pref}$ =' + pTs + f' {degC()}')
    
    
    A3 = plt.subplot(2,3,3)
    A3.plot(T, color = plot_color)
    A3.set_xlabel('Frame #')
    A3.set_ylabel(f'Temp {degC()}')
    A3.set_ylim(vmin,vmax)
    A3.spines.right.set_visible(False)
    A3.spines.top.set_visible(False)
    save_fig(f, fig_name)
    stack = stack-vmin
    stack = stack/(vmax-vmin)
    stack = stack*255
    if make_movie:
        left_movie_RGB = np.stack([stack,stack,stack,(0*stack)+255], axis=-1).astype(np.uint8)
       # right_movie_RGB = np.zeros([stack.shape[0],stack.shape[1],stack.shape[2],4])
       # right_movie_RGB[...,-1] = right_movie_RGB[...,-1] + 255
        for c, (x_, y_) in enumerate(zip(x,y)):
            if not np.isnan(x_) and not np.isnan(y_):
                left_movie_RGB[c,round(y_),round(x_),0] = 255
                mask = np.where(masked_stack[c,...]>0)
                for color in [0,1,2]:
                    #pdb.set_trace()
                   # right_movie_RGB[c,mask[0],mask[1],color] = int(t_colors[c][color]*255)
                    left_movie_RGB[c,mask[0],mask[1],color] = int(t_colors[c][color]*255)
                    #right_movie_RGB[c,round(y_),round(x_),0:2] = t_colors[c][0:2]
        #RGB = np.hstack([left_movie_RGB, right_movie_RGB.astype(np.uint8)])
        return(T, left_movie_RGB)
    else:
        return(T)

def degC():
    return(u'\u00B0C')
    return(u'\u2103')



def colorbar(cmap='jet', vmin=None, vmax=None, axis=None, aspect = 20, title='Arbitrary units'):
    axis.imshow(np.expand_dims(np.arange(0,255),0), cmap = cmap, aspect = aspect, vmin = 0, vmax = 255)
    axis.set_xticks([0,255])
    axis.set_xticklabels([vmin,vmax])
    axis.yaxis.set_visible(False)
    axis.set_xlabel(title)
    return(axis)

def get_T_from_pos(x,y,stack, do_fill_nans=True):
    T=[]
    ref = np.median(stack, axis=0)
    for x,y in zip(x,y):
        if x is np.nan or y is np.nan:
            reading = np.nan
        else:
            reading = ref[int(y),int(x)]
        if reading == 0:
            T.append(np.nan)
        else:
            T.append(reading)
    if do_fill_nans: ## TODO interpolate nan values from prev/next non-nan values
        T = fill_nans(T)
    return(T)

def fill_nans(t):
    nIX = np.where(np.isnan(t))[0]
    IX = np.where(~np.isnan(t))[0]
    for nix in nIX:
        previous = IX[IX<nix]
        nex = IX[IX>nix]
        if len(previous) and len(nex):
            a = t[previous[-1]]
            b = t[nex[0]]
        elif len(previous):
            a = t[previous[-1]]
            b = a
        elif len(nex):
            b = t[nex[0]]
            a = b
        else:
            return(t)
        t[nix] = np.average([a,b])
    return(t)
    
    
def analyze_multi_trial(obj, **kwargs):
    ## scatter of prefs for 2 conditionss with p values
    p = {'trials' : obj.DB['Animals'][obj.curAnimal].keys(),
         'trial_flags' : ['caps', 'vas'],
         'data_flag' : 'TRG',
          }
    p.update(kwargs)
    colors = {}
    colors['vas'] = 'c'
    colors['caps'] = 'm'
    data = {}
    hist_alpha = 1/len(p['trials'])
    for condition in p['trial_flags']:
        data[condition] = {}
        data[condition]['Ts'] = []
        data[condition]['prefs'] = []
        for trial in p['trials']:
            if not condition in trial:
                continue
            FOV   = obj.DB['Animals'][obj.curAnimal][trial]
            for datastream in FOV.keys():
                if 'TRG' in FOV[datastream].attrs:
                    print(f'{FOV} {datastream} has attribute TRG and condition {condition}')
                    stack = FOV[datastream][...]
                    print(colors[condition])
                    T = analyze_FLIR_TRG(stack=stack, plot_color = colors[condition], hist_alpha = hist_alpha)
                    data[condition]['Ts'].append(T)
                    data[condition]['prefs'].append(np.nanmedian(T))
      
    filepath = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Behavior data/TRG.pickle'
    file = open(filepath, 'wb')
    pickle.dump(data, file)
    file.close()      
      
    F = plt.figure(f'Preferred temperature {" vs ".join(p["trial_flags"])}')
    A = F.add_axes([0,0,1,1])
    
    for c, condition in enumerate(p['trial_flags']):
        print(condition)
        x = np.ones(len(data[condition]['prefs']))*c
        A.scatter(x, data[condition]['prefs'])
    
    Fh = plt.figure('histograms')
    A1 = Fh.add_subplot(1,2,1)
    A2 = Fh.add_subplot(1,2,2)
    #Ah = F.add_axes([0,0,1,1])
    for c, condition in enumerate(data.keys()):
        Tconcat = list(itertools.chain.from_iterable(data[condition]['Ts']))
        for T in data[condition]['Ts']:
            A1.hist(data[condition], color = colors[condition], alpha = hist_alpha)
        A2.hist(Tconcat, color = colors[condition], alpha = 0.5)
        
 
    return(data)
                    
        
        
    
    



def save_fig(Fig, fig_name):
    Fig.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' + fig_name + '.pdf')
    Fig.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' + fig_name + '.png')          

    
# def detect_circles(frame=None, **kwargs):
#     if frame is None:
#         stack = sample_data()
#         frame = stack[1000,:,:]
#     h = frame.shape[0]
#     w = frame.shape[1]
#     params = {'min_dist' : np.amax([h, w]),
#               'acc_thresh' : 1.5}
#     params.update(kwargs)
#     frame = frame.astype(np.uint8)
#     circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, params['acc_thresh'], params['min_dist']) 
#     return(circles, frame)
    


def sample_data(path = '/lab-share/Neuro-Woolf-e2/Public/DavidY/Transfer temporary/TGR 11-1-2022/731/1667327064/FLIR.tif', crop=True):
    stack = np.array(imageio.mimread(path, memtest=False))
    if crop is None:
        return(stack)
    else:
        return(stack[:,25:95,42:120])




















             