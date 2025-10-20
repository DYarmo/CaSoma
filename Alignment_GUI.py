#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:57:43 2022

@author: ch184656
"""

from PyQt5.QtWidgets import QApplication, QWidget, QMenuBar, QMenu, QAction, QComboBox, QTableWidget, QTableWidgetItem, QMainWindow, QGridLayout, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QInputDialog, QColorDialog, QTableWidgetSelectionRange               
from PyQt5.QtCore import pyqtSlot, QRectF, Qt
import pickle
import copy
import DYpreProcessingLibraryV2 as DY
import numpy as np

import pyqtgraph as pg
from libAnalysis import DYroi2CaimanROI, CaimanROI2dyROI, multi_sess_color_series, align_thermo_rasters, read_T_series
import libAnalysis as la
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from caiman.utils import visualization

from multi_session_utils import extract_across_sessions, plot_tuning_across_sessions, create_u_cells
from multi_session_utils import to_int
from pg_mpl import MplCanvas, Mpl_multi_canvas


from random import randint
import cv2
from clickLib import clickImage
import sys
import os
import pdb
import inspect

from functools import partial


class alignTable(QTableWidget):
    def dragMoveEvent(self, e):
        e.accept()
        
    def dropEvent(self,e):
        print(e.mimeData())


class Alignment_GUI(QMainWindow):

    ## Initialize GUI with provided dataset, or ask for user selection

    def __init__(self, input_data=None):
        super().__init__()
        self.initUI(input_data=input_data) 
        
    def initUI(self, input_data=None):
        
        # Figure appearance setup:
        

        self.setGeometry(0, 30, 1800, 600)
        self.setWindowTitle('Multi session alignment')
        p=self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)


        self.data_table = alignTable(self)  ###Main data table displayes ROI-Neuron assignments across experiments
        self.data_table.setAcceptDrops(False)##True) 
        self.data_table.setDragEnabled(False)##True)
  #      self.data_table.cellChanged.connect(self.update_data_model)
        
        self.clickActions = {}
        self.clickActions['None'] = self.doNothing
        self.clickActions['Select ROI'] = self.selectROI
        self.clickAction = 'Select ROI'

        ### Set up dict containing default action and linked functions
        self.dataFuncs = {}
        self.dataFuncs['Align cells'] = self.align_ROIs
        self.dataFuncs['Align manually'] = self.align_manually
        self.dataFuncs['Remove neuron'] = self.remove_u_cells
        #self.dataFuncs['Apply spatial union'] = self.get_spatial_union_traces
        #self.dataFuncs['Fill missing'] = self.fill_missing_observations_from_spatial_union
        self.dataFuncs['Reset spatial union'] = self.reset_spatial_union
        self.dataFuncs['Calculate union traces'] = self.compare_union_traces
        self.dataFuncs['Convert to unified multi-session'] = self.convert_to_unified
        self.dataFuncs['Sort cells'] = self.sort_cells
        self.dataFuncs['Set parameters'] = self.set_params
        self.dataFuncs['Reset parameters'] = self.set_default_params
        #self.dataFuncs['Debug...'] = self.debug
        
        self.displayFuncs = {}
        self.displayFuncs['Show spatial union'] = self.display_spatial_union
        self.displayFuncs['Show union traces'] = self.show_union_traces
        self.displayFuncs['Plot cells across sessions']  = self.plot_tuning_across_sessions
        self.displayFuncs['Pick session color...'] = self.pick_session_color
        self.displayFuncs['Plot rasters across sessions'] = self.show_aligned_rasters
        
        self.exportFuncs = {}
        self.exportFuncs['Export combined ROI map'] = self.publish_ROIs
        
        
        if input_data is None:
            self.run_mode = 'independent'
            self.data_path = DY.selectFile()
        else:
            self.run_mode = 'child'
            self.data_path = input_data['pickle file path']
            
        self.original_data = DY.unpickle(self.data_path)
        self.data = copy.deepcopy(self.original_data)

        ##Set color code for experiment ID:
        if hasattr(self.data, 'sess_color_series'):
            self.sess_color_series = self.data.sess_color_series
        else:
            self.sess_color_series= multi_sess_color_series()
            self.data.sess_color_series = self.sess_color_series
        print(f'{self.sess_color_series=}')

        ##Populate menus
        self.main_menu = QMenuBar(self)
        
        file_menu = self.main_menu.addMenu('&File')
        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.save_alignment)
        
        saveAsAction = QAction('Save as...', self)
        saveAsAction.triggered.connect(self.save_alignment_as)

        openAction = QAction('Open...', self)
        openAction.triggered.connect(self.open_multi_session)
        
        saveConfigAction = QAction('Save config...', self)
        saveConfigAction.triggered.connect(self.save_params)
        
        loadConfigAction = QAction('Load config...', self)
        loadConfigAction.triggered.connect(self.load_params)
       
        file_menu.addAction(openAction)
        file_menu.addAction(saveAction)
        file_menu.addAction(saveAsAction)
        file_menu.addAction(saveConfigAction)
        file_menu.addAction(loadConfigAction)
        
        self.analyze_menu = self.main_menu.addMenu('&Analyze')
        for key in self.dataFuncs.keys():
            action = QAction(key, self)
            action.triggered.connect(self.dataFuncs[key])
            self.analyze_menu.addAction(action)
            
        self.set_default_params()
        
        self.initialize_layout()
        
        ################

        ## Connect changes in selected data to update display
        self.data_table.itemSelectionChanged.connect(self.update_views)

        display_menu = self.main_menu.addMenu('&Display')
        for key in self.displayFuncs.keys():
            action = QAction(key, self)
            action.triggered.connect(self.displayFuncs[key])
            display_menu.addAction(action)
            
        export_menu = self.main_menu.addMenu('&Export')
        for key in self.exportFuncs.keys():
            action = QAction(key, self)
            action.triggered.connect(self.exportFuncs[key])
            export_menu.addAction(action)
            
        self.setMenuBar(self.main_menu)
        
        self.caiman_reg_params = {}
        #self.caiman_reg_params
        
        
        
        self.data_table.cellChanged.connect(self.update_data_model)  
        if self.run_mode == 'independent':
            self.show()
        #menubar = self.menuBar()
       
    def initialize_layout(self):
        self.vis_layout = pg.GraphicsLayoutWidget(self)
        self.setWindowTitle(f'{self.data_path}')
        self.field_displays = {}
        self.view_boxes = {}
        self.ROI_overlays = {}
        self.field_images = np.zeros([self.data.n_sessions,self.data.sessions[0].fieldImage.shape[0],self.data.sessions[0].fieldImage.shape[1]])
        for c, session in enumerate(self.data.sessions):
            self.field_displays[c]  = pg.ImageItem(border=self.sess_color_series[c%8])
            self.view_boxes[c] = self.vis_layout.addViewBox(row=0, col = c, lockAspect=1, invertY=True)
            self.view_boxes[c].addItem(self.field_displays[c])
            self.field_displays[c].setImage(session.fieldImage)
            self.field_images[c,:,:] = session.fieldImage
            self.ROI_overlays[c] = clickImage()
            self.view_boxes[c].addItem(self.ROI_overlays[c])
            
            
        self.main_layout  = QGridLayout(self)
        
        widget = QWidget()
        widget.setLayout(self.main_layout)
        self.setCentralWidget(widget)
         
         
        self.field_displays['union'] = MplCanvas(self)
        med_image = np.median(self.field_images, axis=0)
        self.field_displays['union'].axes.imshow(med_image)
        self.field_displays['union'].axes.axis('off')
        
        field_toolbar = NavigationToolbar(self.field_displays['union'], self)
        
        self.trace_canvas = Mpl_multi_canvas(self, n_rows = 2, n_columns = self.data.n_sessions, num_items = self.data.n_sessions*2)
        
        self.main_layout.addWidget(self.trace_canvas, 1,1 )
        
        self.main_layout.addWidget(self.data_table, 0, 0, 3, 1)
        self.main_layout.addWidget(self.vis_layout, 0, 1 )
        
        
        self.main_layout.addWidget(self.field_displays['union'], 0, 2, 2,1)
        
        self.main_layout.addWidget(field_toolbar, 2,2)
        
        self.main_layout.setColumnStretch(0,2)
        self.main_layout.setColumnStretch(1,4)
        self.main_layout.setColumnStretch(2,1)
        self.main_layout.setRowStretch(0,4)
        self.main_layout.setRowStretch(1,4)
        self.main_layout.setRowStretch(2,1)
        
        self.update_data_view()
        
        
    def set_default_params(self):
        self.params = {}
        self.params['caiman'] = {}
        self.params['caiman']['align_flag']=True
        self.params['caiman']['thresh_cost'] = 0.95
        self.params['caiman']['use_opt_flow'] = True
        self.params['caiman']['max_thr']  = 0
        self.params['caiman']['max_dist'] = 15
        self.params['caiman']['enclosed_thr'] = None
        
        self.params['ROI display'] = {}
        self.params['ROI display']['Contour labeling (session or neuron'] = 'session'
        self.params['ROI display']['Show background'] = True
        self.params['ROI display']['Color map'] = 'gist_rainbow'
        
        self.params['Raster diplay'] = {}
        self.params['Raster diplay']['Normalize'] = True
        self.params['Raster diplay']['vmin'] = None
        self.params['Raster diplay']['vmax'] = None
        self.params['Raster diplay']['cmap'] = 'inferno'
        
        
        self.params['Clustering'] = {}
        self.params['Clustering']['n_clusters'] = 3
        
        
        
        """
        Here we are adding all of the methods for cell, session_data, and multi_session objects to the analyze menu,
        and then adding each argument for each of the methods to the params structure so they can be set interactively 
        from 'set params' menu action
        """
        
        self.multi_session_methods = inspect.getmembers(la.multi_session, predicate=inspect.isfunction)
        multi_sess_menu = self.analyze_menu.addMenu('Multi-session methods...')
        
        for c, method in enumerate(self.multi_session_methods):
            action = QAction(method[0], self)
            func = partial(self.run_multi_session_method, method=method[1], method_name = method[0])
            action.triggered.connect(func )
            multi_sess_menu.addAction(action)
        
        
        self.session_methods = inspect.getmembers(la.session_data, predicate=inspect.isfunction)
        sess_menu = self.analyze_menu.addMenu('Session methods...')
        
        for c, method in enumerate(self.session_methods):
            action = QAction(method[0], self)
            func = partial(self.run_session_method, method=method[1], method_name = method[0])
            action.triggered.connect(func )
            sess_menu.addAction(action)
        
        self.cell_methods = inspect.getmembers(la.cell, predicate=inspect.isfunction)
        cell_menu = self.analyze_menu.addMenu('Cell methods...')
        
        for c, method in enumerate(self.cell_methods):
            action = QAction(method[0], self)
            func = partial(self.run_cell_method, method=method[1], method_name = method[0])
            action.triggered.connect(func )
            cell_menu.addAction(action)   
            
        for method in self.session_methods:
            key = 'session_method.' +  method[0]
            param_dict = inspect.getfullargspec(method[1])._asdict()
            param_dict['args'].pop(0) ## remove 'self' argument
            self.params[key] = {}
            try:
                for arg, default in zip(param_dict['args'], param_dict['defaults']):
                    self.params[key][arg] = default
            except:
                print(f'Cannot set params for {method}')
                #pdb.set_trace()
                
        for method in self.multi_session_methods:
            key = 'multi_session_method.' +  method[0]
            param_dict = inspect.getfullargspec(method[1])._asdict()
            param_dict['args'].pop(0) ## remove 'self' argument
            self.params[key] = {}
            print(key)
            
            for arg, default in zip(param_dict['args'], param_dict['defaults']):
                self.params[key][arg] = default
                
        for method in self.cell_methods:
            key = 'cell_method.' +  method[0]
            param_dict = inspect.getfullargspec(method[1])._asdict()
            param_dict['args'].pop(0) ## remove 'self' argument
            self.params[key] = {}
            print(key)
            for arg, default in zip(param_dict['args'], param_dict['defaults']):
                self.params[key][arg] = default
                
          
    def keyPressEvent(self, e):
        print (e.key())

        if e.key() == 82:  ## r key
            self.toggle_ROI_mode()
            
    def toggle_ROI_mode(self):
        if self.params['ROI display']['Contour labeling (session or neuron'] == 'session':
            self.params['ROI display']['Contour labeling (session or neuron'] = 'neuron'
        else:
            self.params['ROI display']['Contour labeling (session or neuron'] = 'session'
        self.update_views()
            
    def run_multi_session_method(self, method_name = None, method = None):   

        param_key = 'multi_session_method.' + method_name
        param_dict = self.params[param_key]
        param_dict['self'] = self.data
        print(f'Arguments for function are {param_dict.keys()}')
        
        u_cells, s = self.get_u_cells_and_sessions()
        
        if 'selected_u_cells' in param_dict.keys():
            param_dict['selected_u_cells'] = u_cells
        if 'selected_sessions' in  param_dict.keys():
            param_dict['selected_sessions'] = s

        method(**param_dict)
        if 'init_table' in param_dict:
            self.initialize_layout()
                
                
                    
    def run_session_method(self, method_name=None, method=None):   
        print(f'Method name is {method_name}')
        param_key = 'session_method.' + method_name
        param_dict = self.params[param_key]
        param_dict['self'] = self.data
        print(f'Arguments for function are {param_dict.keys()}')
        for session in self.selected_sessions:
            print(f'running {method_name} on session {session}')
            u_cells, s = self.get_u_cells_and_sessions()
            
            param_dict['self'] = self.data.sessions[session]
            
            if 'selected_u_cells'in param_dict:
                param_dict['selected_u_cells'] = u_cells
            if 'selected_sessions' in param_dict:
                param_dict['selected_sessions'] = s
            if 'key_session' in param_dict:
                param_dict['key_session'] = s[0]
            if 'file_path' in param_dict:
                param_dict['file_path'] = DY.selectFile(existing=False)
            if 'cellIXs' in param_dict:
                param_dict['cellIXs'] = self.selected_data[session]
            if 'cells' in param_dict:
                param_dict['cells'] = self.selected_data[session]
                
            
            method(**param_dict)
    
    def run_cell_method(self, method_name=None, method=None):   
        print(f'Method name is {method_name}')
        param_key = 'cell_method.' + method_name
        param_dict = self.params[param_key]
        param_dict['self'] = self.data
        print(f'Arguments for function are {param_dict.keys()}')
        for session in self.selected_sessions:
            for cell in self.selected_data[session]:
                print(f'running {method_name} on cell {cell}, session {session}')
                param_dict['self'] = self.data.sessions[session].cells[cell]
 
                method(**param_dict)
            
            
    
    
    @pyqtSlot()
    def respondClick(self, X, Y, labelMap, session = None, mod = None):
        self.clickActions[self.clickAction](X,Y, labelMap, session=session, mod = mod)
        
    @pyqtSlot()  
    def setClickAction(self):   
        self.clickAction = self.clickActionBox.currentText()
       
    @pyqtSlot()
    def doNothing(self, *args, **kwargs):
        pass
    
    def selectROI(self, X, Y, labelMap, **kwargs):
        ##
        #plt.figure('labelMap')
        #plt.imshow(labelMap)
        append_key = 'shift'
        toggle_key = 'ctrl'
        mod = kwargs['mod']
        session= kwargs['session']
        print(mod)
        print(f'{session=}')
        IX = [int(labelMap[X,Y])-1]
        print(f'{IX=}')
        append = False
        if append_key in mod:
            append = True
        self.update_ROI_selections_software(IX=IX, session=session, append = append)
        
    @pyqtSlot()
    def align_ROIs(self, method='caiman'):
        param_set = self.params[method]
        print(f'ROIs selected for align_ROIs: {self.selected_data}')
        #param_set = {} # test
        self.data.align_cells(method=method, selected_sessions = self.selected_sessions, selected_ROIs = self.selected_data, **param_set)
        self.update_data_view()
        
        #plt.figure('assignmentsss')
        #plt.imshow(self.data.assignments)
        
        
        
    def reset_spatial_union(self):
        self.data.union_ROIs = None
        
        
        
    def display_spatial_union(self):
        if self.data.union_ROIs is None:
            print('No spatial union defined')
        elif len(self.data.union_ROIs.shape) == 2:  ### If spatial union is in caiman format (time x cell)
            medField = np.median(self.field_images, axis=0)
            plot_contours(self.data.union_ROIs, medField, display_numbers=False )
        elif len(self.data.union_ROIs.shape) == 3:
            print('Converting DY format ROIs')  ### If spatial union is in dy format (h x w x cell)
            Ac = DYroi2CaimanROI(self.data.union_ROIs)
            medField = np.median(self.field_images, axis=0)
            plot_contours(Ac, medField, display_numbers=False )
            
        
    @pyqtSlot()
    def update_data_view(self): 
        """
        Updates the table for data selection to match underlying data structure
        (multi_session object stored as self.data)
        """
        print('updating data view')
        try:
            self.data_table.cellChanged.disconnect(self.update_data_model) 
        except:
            print('not connected')
        self.data_table.clear()
        self.data_table.setColumnCount(self.data.assignments.shape[1])
        self.data_table.setRowCount(self.data.assignments.shape[0])
        headers = []
        for I in range(0, self.data.assignments.shape[1]):
            headers.append(f'Session {I}')
        self.data_table.setHorizontalHeaderLabels(headers)
        for U_IX, session_IXs in enumerate(self.data.assignments):
            for session_num, session_IX  in enumerate(session_IXs):
                self.data_table.setItem(U_IX, session_num, QTableWidgetItem(str(session_IX)))
        self.update_views()
        self.data_table.cellChanged.connect(self.update_data_model) 
    
    
      
        
    @pyqtSlot()
    def update_data_model(self):
        """
        Updates cell assignments when edited manually in table
        (manual editing not recommended but available if required)
        """
        
        print(f'{self.data_table.rowCount()=}')
        print(f'{self.data_table.columnCount()=}')
        assignments = np.zeros([self.data_table.rowCount(), self.data_table.columnCount()])*np.nan
        for ii in range(0, self.data_table.rowCount()):
            for jj in range(0, self.data_table.columnCount()):
                it = self.data_table.item(ii,jj)
                ## perform checks:
                if it is None:
                    print(f'Entry not found at {ii}, {jj}')
                    continue
                item = it.text()
                #old_value = assignments[ii,jj].copy()
                new_value = to_int(item)
                if new_value <0:
                    print('Negative ROI value')
                    continue
                if new_value > (len(self.data.sessions[jj].cells)-1):
                    print(f'ROI index {new_value} greater than # of ROIs in session {jj}')
                    continue
                
                assignments[ii,jj] = new_value
        
        #plt.figure('updated asssignments')
        #plt.imshow(assignments)
        self.data.assignments = assignments
    
    
    def update_views(self):
        
        self.update_ROI_selections()
        self.update_ROI_view()
        self.update_plot_view()
        
    def update_ROI_selections_software(self, IX=None, session=None, append=False, toggle = False):
        print(IX)
        print(session)
        
        self.data_table.itemSelectionChanged.disconnect(self.update_views)
        #self.data_table.cellChanged.disconnect(self.update_data_model) 
        
        ### remove
        # selected_in_session = self.selected_data[session]
        # for s in self.selected_data.keys():
        #     for ix in self.selected_data[s]:
        #         c = QTableWidgetSelectionRange(ix, s, ix, s)
        #         self.data_table.setRangeSelected(c, False)
        if append == False:
            self.data_table.clearSelection()
        if IX[0] > -1:
            sessionIX = np.where(self.data.assignments[:,session]==IX[0])[0][0]
            print(f'{IX=}, {sessionIX=}')
            c = QTableWidgetSelectionRange(sessionIX, session, sessionIX, session)
            self.data_table.setRangeSelected(c, True)
        
        
        
        self.update_views()
        self.data_table.itemSelectionChanged.connect(self.update_views)
        #self.data_table.cellChanged.connect(self.update_data_model) 
        
    def update_ROI_selections(self):
        self.selected_data = {}
        self.selected_sessions = []
        for c, session in enumerate(self.data.sessions):
            self.selected_data[c] = []
        selected = self.data_table.selectedItems()
        for item in selected:
            val = item.text().split('.')[0]
            if val.isnumeric():
                self.selected_data[item.column()].append(int(val)) 
                self.selected_sessions.append(item.column())
        self.selected_sessions = list(set(self.selected_sessions))
        print(f'Selected: {self.selected_data}')
                
    def get_u_cells_and_sessions(self):
        u_cells = []
        sessions=[]
        for item in self.data_table.selectedItems():
            session = item.column()
            u_cell = item.row()
            if not u_cell in u_cells:
                u_cells.append(u_cell)
            if not session in sessions:
                sessions.append(session)
        return(u_cells, sessions)
    
    def get_spatial_union_traces(self):   # u_cells = None, selected_sessions=None):
        u_cells, sessions = self.get_u_cells_and_sessions()
        
        self.data.get_spatial_union_traces(selected_u_cells = u_cells, selected_sessions = sessions)
    
        
    def match_neighbours(self, max_distance = 10):
        pass
        #get list of unmatched cells for each session
        
        
        
    def compare_union_traces(self):   # u_cells = None, selected_sessions=None):
        u_cells, sessions = self.get_u_cells_and_sessions()    
        extract_across_sessions(self.data, selected_u_cells = u_cells, selected_sessions = sessions)
        #self.data.fill_missing_observations_from_spatial_union(selected_u_cells = u_cells, selected_sessions = sessions, missing_only = False )
        
    def show_union_traces(self):   # u_cells = None, selected_sessions=None):
        u_cells, sessions = self.get_u_cells_and_sessions()    
        self.data.show_u_traces(selected_u_cells = u_cells, selected_sessions = sessions)    
    
    
    def plot_tuning_across_sessions(self):
        u_cells, sessions = self.get_u_cells_and_sessions()
        plot_tuning_across_sessions(self.data, selected_u_cells = u_cells, selected_sessions = sessions)
    
   
        
        
    def publish_ROIs(self):
        figname = f'ROI map for {os.path.split(self.data_path)[-1]}'
        F = plt.figure(figname)
        oParams = copy.deepcopy(self.params)
        
        self.params['ROI display']['Contour labeling (session or neuron'] = 'session'
        self.params['ROI display']['Show background'] = True
        ax = F.add_axes([0,0,0.5,1])
        self.update_ROI_view(ax = ax, scale_bar = True)
        

        
        self.params['ROI display']['Contour labeling (session or neuron'] = 'neuron'
        self.params['ROI display']['Show background'] = False
        ax3 = F.add_axes([0.5,0, 0.5,1])
        self.update_ROI_view(ax = ax3)
        
        self.params = oParams
        
        ext = '.pdf'
        F.savefig('/lab-share/Neuro-Woolf-e2/Public/Figure publishing/' + figname + ext)
     
    def show_aligned_rasters(self, f_min = 0, f_max = 0.5, **kwargs):
        
        u_cells, selected_sessions = self.get_u_cells_and_sessions()
        sessions = []
        for s in selected_sessions:
            sessions.append(self.data.sessions[s])
        rasters=[]
        stims = []
        trace_session_IXs = []
        performance={}
        performance['errors'] = []
        performance['aligned'] = []
        Tseries = read_T_series()
        #Tseries = [(0,12), (19,25), (37,40), (41,43), (44,46), (46,48), (49,52)]
        for c, session in enumerate(sessions):
            raster, stim, missing = session.regularize_raster(Tseries=Tseries, end_fix_duration=80, prepend_frames = 60, append_frames=80, fail_if_missing=True, **kwargs)
           
            if not raster is None and not stim is None:
                rasters.append(raster)
                trace_session_IXs.append(np.zeros(raster.shape[0])+c)
                stims.append(stim)
                performance['aligned'].append(session)
            else:
                performance['errors'].append((session, missing))
        
        figname = 'Aligned rasters'
        F = plt.figure(figname)
        r_top = 0.6 ## top of raster plotting area
        r_bot = 0.1 ## bottom of raster plotting area
        left = 0.1
        right = 0.9
        num_sess = len(sessions)
        a_width = (right-left)/num_sess
        
        A_stim = {}
        A_raster={}
        for c, (r, s) in enumerate(zip(rasters, stims)):

            A_stim[c] = F.add_axes([left+(a_width*c), r_top, a_width, 1-r_top])
            A_raster[c] = F.add_axes([left+(a_width*c), r_bot, a_width, r_top-r_bot])
            #A_raster[(c, 'twin')] = A_raster[c].twinx()
            
            A_stim[c].plot(s)
            A_raster[c].imshow(r, aspect = 'auto', vmin = f_min, vmax = f_max)
            
            A_raster[c].set_yticks([])
            A_raster[c].spines.bottom.set_visible(False)
            A_raster[c].set_xticks([])
            
        print(performance)
        return(rasters, stims)
        #align_thermo_rasters(sessions, Tseries = [(0,12), (40,45), (47,52)], plot=False)
        
        
    def update_ROI_view(self, ax = None, scale_bar = False):
        u_cells, selected_sessions = self.get_u_cells_and_sessions()
        if ax is None:
            ax = self.field_displays['union'].axes
            ax.cla()
        
        #neuron_color_series = np.array(rnd_color_series(len(u_cells)))
        neuron_color_series = []
        
        
        #cmap = cm.get_cmap('tab20')
        #cmap = cm.get_cmap('gist_rainbow')
        cmap = cm.get_cmap(self.params['ROI display']['Color map'])
        
        for i in range(len(u_cells)):
            neuron_color_series.append(cmap(i/len(u_cells)))
        
        
        
        #med_image = np.median(self.field_images, axis=0)
        
        
        show_field = True
        medField = np.median(self.field_images[selected_sessions], axis=0)
        field_for_disp = None
        if not self.params['ROI display']['Show background']:
            
            w = (medField*0) + 1
            field_for_disp = np.stack([w,w,w], axis=2)
        
       
        
        for c, session in enumerate(self.data.sessions):
            if len(self.selected_data[c]) == 0:
                h = session.ROIs.shape[0]
                w = session.ROIs.shape[1]
                RGBA = np.zeros([h,w,4])
                flatmap = np.zeros([h,w])
                self.ROI_overlays[c].setImage(RGBA)
                self.ROI_overlays[c].linkROImap(flatmap, self, session=c)
                continue
            
            OfloatMask = session.ROIs[:,:,self.selected_data[c]]
            floatMask = OfloatMask.copy()
            
            
            for ROI in range(floatMask.shape[-1]):
                floatMask[...,ROI] = floatMask[...,ROI]/np.amax(floatMask[...,ROI])
                
            
            ## Display contours overlaid
            
            numbers = {}
            for IX, i in enumerate(self.selected_data[c]):
                numbers[IX] = i
           
            color_mode = self.params['ROI display']['Contour labeling (session or neuron']
            caiman_ROIs = DYroi2CaimanROI(OfloatMask) 
 
            if color_mode == 'session':
                color = self.sess_color_series[c%8]
                plot_contours(caiman_ROIs, medField, ax=ax, numbers = numbers, color=color, show_field = show_field, field_for_disp=field_for_disp)
            elif color_mode == 'neuron':
                neuron_colors = []
                #pdb.set_trace()
                for u, u_cell in enumerate(u_cells):
                    if not np.isnan(self.data.assignments[u_cell, c]):
                        neuron_colors.append(neuron_color_series[u])
                plot_contours(caiman_ROIs, medField, ax=ax, numbers = numbers, color_series = neuron_colors, show_field = show_field, field_for_disp=field_for_disp)
            else:
                plot_contours(caiman_ROIs, medField, ax=ax, numbers = numbers, color='w', show_field = show_field)
                
            ax.set_xlim([0, medField.shape[1]])
            ax.set_ylim([0, medField.shape[0]])
            if not ax.yaxis_inverted:
                ax.invert_yaxis()
                
            if scale_bar:
                microns_per_pixel = 2.016
                bar_length = 100 # microns
                height = medField.shape[0]
                width = medField.shape[1]
                ax.plot([width*0.05, width*0.05 + (bar_length/microns_per_pixel)], [height*0.95 ,height*0.95 ], color = 'w', linewidth = 3)
            self.field_displays['union'].draw()
            
            
            ## Display ROIs in individual sessionss
            flatFloat = np.amax(floatMask, axis=2)
            flatFloat = flatFloat * 255
            flatFloat = flatFloat.astype(np.uint8)

            labelSelected = floatMask*0
            binaryMask = floatMask>0
           
            for label in range(0, binaryMask.shape[-1], 1):
               labelSelected[:,:,label] = binaryMask[:,:,label]*label+1
            
            flatLabel = np.max(labelSelected, axis=2)    
            truncatedLabel = (flatLabel % 255).astype(np.uint8)
            
            label_range = np.linspace(0,1,256)
            lut = np.uint8(plt.cm.prism(label_range)[:,2::-1]*256).reshape(256, 1, 3)
            RGB = cv2.LUT(cv2.merge((truncatedLabel, truncatedLabel, truncatedLabel)), lut)
            
            Alpha = np.expand_dims(flatFloat, 2)
            RGBA = np.concatenate((RGB,Alpha), axis = 2)
            
            bgMask = np.max(floatMask, axis=2)
            bgMaskBool = bgMask.astype(bool)
            flatLabel[~bgMaskBool]= 0
            
            self.ROI_overlays[c].setImage(RGBA)
            self.ROI_overlays[c].linkROImap(flatLabel, self, session = c)
            
    # def debug(self):
    #     pdb.set_trace()
    
    
    def sort_alignment(self):
        u_cells, sessions = self.get_u_cells_and_sessions() 
        self.data.sort_u_cells(key_session = sessions[0])
    
    def pick_session_color(self):
        u_cells, sessions = self.get_u_cells_and_sessions() 
        for session in sessions:
            C = QColorDialog.getColor()
            self.data.sess_color_series[session] = [C.red()/255, C.green()/255, C.blue()/255]
            self.sess_color_series[session] = [C.red()/255, C.green()/255, C.blue()/255]
        self.update_data_view()
        self.update_views()
            
    def update_plot_view(self):
        #pdb.set_trace()
        # for i in range(10):
        #     for n, axis in enumerate(self.trace_canvas.axes):
        #         #axis.cla()
        #         color = rnd_color_series(1)[0]
                
        #         axis.plot([0,randint(0,100),randint(0,100)],[0,randint(0,100),randint(0,100)], color=color)
        #         self.trace_canvas.draw()
        norm = self.params['Raster diplay']['Normalize'] 
        vmin = self.params['Raster diplay']['vmin'] 
        vmax = self.params['Raster diplay']['vmax']  
        cmap = self.params['Raster diplay']['cmap'] 
        
        u_cells, sessions = self.get_u_cells_and_sessions() 
        for session in self.selected_data.keys():
            
            
            #print(f'Trying to plot session {session}')
            AS = self.trace_canvas.axes[session]
            AR = self.trace_canvas.axes[session+self.data.n_sessions]
            AS.clear()
            AR.clear()
            #if len(self.selected_data[session]) > 0:
            self.data.sessions[session].show_raster(cells = self.selected_data[session], sort_raster=False,F=0, AR=AR, AS=AS, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
            self.trace_canvas.draw()
        ## get selected U_ROIs
        ##display selected U_ROI traces for each session
  
        
        
    
    
    def convert_to_unified(self):
        u_cells, sessions = self.get_u_cells_and_sessions()
        self.data = create_u_cells(self.data, selected_u_cells='All', selected_sessions='All')
        self.assignments = self.data.assignments
        self.update_data_view()
        self.update_views()
    
    def align_manually(self):
        
        o_assignments = self.data.assignments.copy()
        for session in self.selected_data.keys():
            if len(self.selected_data[session]) >1:
                print('Please do not select more than 1 ROI per session to align')
                return()
            
        u_cells, sessions = self.get_u_cells_and_sessions()
        new_alignment = np.zeros([1, self.data.n_sessions])*np.nan
        new_ROIs = []
        
        for uc, session in zip(u_cells, sessions):
            session_cell = int(self.data.assignments[uc,session])
            new_alignment[0, session] = session_cell
            o_assignments[uc,session] = np.nan
            new_ROIs.append(self.data.sessions[session].ROIs[:,:,session_cell])
        
        retain = []
        for c, row in enumerate(o_assignments):
            if not np.all(np.isnan(row)):
                retain.append(c)
        retained_assignments = o_assignments[retain,:]
        retained_spatial_unions = self.data.union_ROIs[:,retain]
        plt.figure('Retained spatial union')
        plot_contours(retained_spatial_unions, self.data.sessions[0].fieldImage)
        plt.figure('Original spatial union')
        plot_contours(self.data.union_ROIs, self.data.sessions[0].fieldImage)
        
        new_roistack = np.zeros([new_ROIs[0].shape[0], new_ROIs[0].shape[1], len(new_ROIs)])
        for c, ROI in enumerate(new_ROIs):
            new_roistack[:,:,c] = ROI
        
        cm = DYroi2CaimanROI(new_roistack)
        
        
        print(f'{cm.shape=}')
        union = np.amax(cm, axis = 1)
        print(f'{union.shape=}')
        union = np.expand_dims(union, 1)
        print(f'{union.shape=} after expansion')
        
        
        self.data.assignments = np.vstack([retained_assignments, new_alignment])
        self.data.union_ROIs = np.hstack([retained_spatial_unions, union])
        
        plt.figure('Assembled spatial union')
        plot_contours(self.data.union_ROIs, self.data.sessions[0].fieldImage)
        
        self.update_data_view()
        
    def sort_cells(self):
        self.data.sort_u_cells()
        self.update_data_view()
        
        
    def remove_u_cells(self):
        u_cells, sessions = self.get_u_cells_and_sessions()
        print(f'{u_cells=}')
        retain = []
        for c, row in enumerate(self.data.assignments):
            if not c in u_cells:
                retain.append(c)
        print(f'{retain=}')
        self.data.assignments = self.data.assignments[retain,:]
        if not self.data.union_ROIs is None:
            self.data.union_ROIs = self.data.union_ROIs[:,retain]
        if hasattr(self.data, 'trace_struct'):
            
            self.data.trace_struct
        #self.data
        self.update_data_view()
        print('removed?')
        
        
    @pyqtSlot()
    def reset_alignment(self):
        pass
    
    def save_alignment(self):
        self.data.pickle_path = self.data_path
        self.data.pickle(self.data_path)
        print(f'Data saved to {self.data_path}')
    
    def save_alignment_as(self):
        data_path = DY.selectFile(existing=False)
        
        self.data.pickle(data_path)
        self.data_path = data_path
        self.data.pickle_path = data_path
        print(f'Data saved to {self.data_path}')
        self.setWindowTitle(f'{self.data_path}')
        
    def open_multi_session(self):
        self.data_path = DY.selectFile()
        print(f'{self.data_path=}')
        self.original_data = DY.unpickle(self.data_path)
        self.data = copy.deepcopy(self.original_data)
        self.initialize_layout()
        #self.update_data_view()
        #self.update_views()
        
    def closeEvent(self, Event):
        self.save_alignment
        
    def save_params(self):
        config_path = DY.selectFile(existing=False, message = 'Save config...')
        file = open(config_path, 'wb')
        pickle.dump(self.params, file)
        file.close()
        print(f'Config saved to {config_path}')
    
    def load_params(self):
        config_path = DY.selectFile(existing=False, message = 'Load config...')
        try:
            self.params = DY.unpickle(config_path)
        except:
            print('That did  not work!')
        
    def set_params(self):
        params = self.params.keys()
        parameter_set, okPressed = QInputDialog.getItem(self,"Select parameter set:", "Param set:", params, 0, False)
        if okPressed == False:
            return
        p_set = self.params[parameter_set]
        for param in p_set:
            value, okPressed = QInputDialog.getText(self,"Set param value:", f'{param}:', QLineEdit.Normal, f'{p_set[param]}')
            if okPressed == False:
                return
            elif value[0] == '[' and value[-1] == ']':
                
                delisted = value.strip('][').split(',')
                value=[]
                print('Parsing list...')
                for item in delisted:
                    item = item.replace(' ','')
                    if item.isnumeric:
                        value.append(int(item))
                    else:
                        value.append(item)
                    print(f'Value {value[-1]}, dtype is {type(value[-1])}')
            elif value in ['False', 'false', 'F', 'f']:
                value = False
            elif value in ['True', 'true', 'T', 't']:
                value = True
            elif value.isnumeric():
                value = int(value)
            elif value == 'None':
                value = None
            elif value.split('.')[0].isnumeric():
                value = float(value)
                
            self.params[parameter_set][param] = value
            
            
        
        
        
        

    
def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=False, max_number=None,
                  cmap=None, swap_dim=False, color=None, colors='w', vmin=None, vmax=None, coordinates=None,
                  contour_args={}, color_series=None, numbers = None, number_args={}, ax=None, show_field = True, field_for_disp = None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)
    
         Cn:  np.ndarray (2D)
                   Background image (e.g. mean, correlation)
    
         thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy
    
         maxthr: [optional] scalar
                    Threshold of max value
    
         nrgthr: [optional] scalar
                    Threshold of energy
    
         thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)
                   Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr
    
         display_number:     Boolean
                   Display number of ROIs if checked (default True)
    
         max_number:    int
                   Display the number for only the first max_number components (default None, display all numbers)
    
         cmap:     string
                   User specifies the colormap (default None, default colormap)
         color_series: list of colors

     Returns:
          coordinates: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    if thr is None:
        try:
            thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
        except KeyError:
            thr = maxthr
    else:
        thr_method = 'nrg'


    for key in ['c', 'colors', 'line_color']:
        if key in kwargs.keys():
            color = kwargs[key]
            kwargs.pop(key)
            
    if ax is None:
        ax = plt.gca()
    
    if field_for_disp is None:
        field_for_disp = Cn
        
    if show_field:
        if vmax is None and vmin is None:
            ax.imshow(field_for_disp, interpolation=None, cmap='gist_gray',
                      vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
                      vmax=2*np.percentile(Cn[~np.isnan(Cn)], 99))
        else:
            ax.imshow(field_for_disp, interpolation=None, cmap='gist_gray', vmin=vmin, vmax=vmax)

    if coordinates is None:
        coordinates = visualization.get_contours(A, np.shape(Cn), thr, thr_method, swap_dim)
    for count, c in enumerate(coordinates):
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        
        if not color_series is None:
            color = color_series[count%len(color_series)]
        
        ax.plot(*v.T, c=color, **contour_args)

    if display_numbers:
        d1, d2 = np.shape(Cn)
        d, nr = np.shape(A)
        cm = visualization.com(A, d1, d2)
      #  if max_number is None:
       #     max_number = A.shape[1]
        for i in numbers.keys():

            ##
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(numbers[i]), color='w', **number_args)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(numbers[i]), color='w', **number_args)
    ax.axis('off')
    return coordinates


               
                
            

def rnd_color_series(n):
    color_series = []
    for i in range(n):
        color_series.append('#%06X' % randint(0, 0xFFFFFF))
    return(color_series)

if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = Alignment_GUI()
        sys.exit(app.exec_())
        
        
