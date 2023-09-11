
# -*- coding: utf-8 -*-

"""
Created on Fri Aug 28 14:06:34 2020

@author: USER
"""

#import caiman as cm
import inspect
import pyqtgraph as pg
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, \
    QLineEdit, QMainWindow, QSizePolicy, QLabel, QSlider, QMenu, QAction, \
    QComboBox, QListWidget, QListWidgetItem, QGridLayout, QPlainTextEdit, QDateTimeEdit, QTextEdit, QInputDialog, QColorDialog, QTableWidget
    
from pg_mpl import MplCanvas
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, QRectF, Qt
import h5py 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import pdb

import cv2
import DYpreProcessingLibraryV2 as DY
from DYpreProcessingLibraryV2 import getSubStack as gs
import DYroiLibrary as dyROI
import libAnalysis as LA
import thermalRingLib as TRG
import mech_stim_lib as mech
import behaviorLib as beh
import sys
import os
import imageio
import time
from scipy import stats
from scipy import ndimage as ndi
from scipy import signal
from skimage.segmentation import watershed
from skimage.draw import disk

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from Alignment_GUI import Alignment_GUI
from clickLib import clickImage



# class clickImage(pg.ImageItem):
    
    
    # def linkROImap(self, ROImap, parentApp):
    #     self.linkedROImap = ROImap
    #     self.parentApp = parentApp
    
    # def mouseClickEvent(self, event):
    #     X = round(event.pos()[0])
    #     Y = round(event.pos()[1])
        
    #     #print(f'X = {X}')
    #     #print(f'Y = {Y}')
    #     #print(f'Labelval = {self.linkedROImap[X,Y]}')
    #     #print(self.parentApp.curAnimal)
    #     self.parentApp.respondClick(X,Y, self.linkedROImap)
        
        

        
        
        
    

        
    
class DBgui(QMainWindow):
        
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        

        # Figure setup:
        self.setGeometry(0, 30, 1920, 1000)
        self.setWindowTitle('YarmoPain')
        p=self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
            
        #############
        ##
        ##      Graphical displays
        ##
        ##################
        
        
        #Container to display data:
        self.displayLayout = pg.GraphicsLayoutWidget(self)
        self.displayLayout.resize(1024,512)
        self.displayLayout.move(650,25)
        
        
        # PlotItem to plot ROI data
        self.ROIlayout = pg.GraphicsLayoutWidget(self)
        self.ROIlayout.resize(1024, 175)
        self.ROIlayout.move(650,540)
        self.ROIrasterView = self.ROIlayout.addViewBox(row = 0, col = 0)
        self.ROItracePlot = self.ROIlayout.addPlot(row=0, col = 1)
        

        # Graphics to navigate time in  available datasets
        self.timeLayout = pg.GraphicsLayoutWidget(self)
        self.timeLayout.resize(1024,200)
        self.timeLayout.move(650,750)
        #self.timeView = self.timeLayout.addViewBox()
        self.timePlotItem = pg.PlotItem(border='k')
        self.timePlotItem.hideAxis('left')
        self.timePlotItem.hideAxis('bottom')
        self.timePlotItem.enableAutoRange()
        #self.timeView.addItem(self.timePlotItem)
        self.timeLayout.addItem(self.timePlotItem)
     
        
        
        ######
        ##
        ##  Buttons for loading data
        ##
        ######
        
        #Button to create db
        self.DB = []
        self.makeDBbtn = QPushButton('Create database...',self)
        self.makeDBbtn.resize(140,40)
        self.makeDBbtn.move(10,0)
        self.makeDBbtn.clicked.connect(self.createDB)
        
        
        #Button to select db
        self.chooseDBbtn = QPushButton('Open database...',self)
        self.chooseDBbtn.resize(140,40)
        self.chooseDBbtn.move(10,50)
        self.chooseDBbtn.clicked.connect(self.openDB)
        
        #Button to select db
        self.chooseDBbtn = QPushButton('Open last',self)
        self.chooseDBbtn.resize(60,40)
        self.chooseDBbtn.move(150,50)
        self.chooseDBbtn.clicked.connect(self.openLast)
        
        #Button to lock db
        self.lockDBbtn = QPushButton('Lock database',self)
        self.lockDBbtn.resize(140,40)
        self.lockDBbtn.move(200,25)
        self.lockDBbtn.clicked.connect(self.closeDB)
        
        #Button to add animal
        self.addAnimalbtn = QPushButton('Add animal',self)
        self.addAnimalbtn.resize(90,40)
        self.addAnimalbtn.move(10,400)
        self.addAnimalbtn.clicked.connect(self.addAnimal)
        
        #Button to delete animal
        self.delAnimalbtn = QPushButton('Delete animal',self)
        self.delAnimalbtn.resize(90,40)
        self.delAnimalbtn.move(10,440)
        self.delAnimalbtn.clicked.connect(self.delAnimal)
        
        #Button to clear neurons
        self.clearNeuronBtn = QPushButton('Clear neurons',self)
        self.clearNeuronBtn.resize(90,40)
        self.clearNeuronBtn.move(10,520)
        self.clearNeuronBtn.clicked.connect(self.clearNeurons)
        
        #Button to deposit animal from folder
        #self.depositAnimalbtn = QPushButton('Deposit animal',self)
        #self.depositAnimalbtn.resize(100,40)
        #self.depositAnimalbtn.move(150,440)
        #self.depositAnimalbtn.clicked.connect(self.depositAnimal)
        
        
        
        
        
        self.depositMethods = {}
        self.depositMethods['Deposit...'] = print
        self.depositMethods['Deposit Sessions(s)'] = self.appendSessions
        self.depositMethods['Deposit miniscope'] = self.depositMiniscope
        #self.depositMethods['SLURM deposit'] = self.slurmDeposit
        #self.depositMethods['Local deposit'] = self.localDeposit
        #self.depositMethods['Deposit archive'] = self.archiveDeposit
        self.depositMethods['Close DB'] =self.closeDB
        self.depositMethods['Reopen DB'] = self.reOpenDB
        self.depositMethods['Import FOVs from DB'] = self.importFOVs
        self.depositMethods['Add FOV'] = self.addFOV
        self.depositMethods['Rename FOV'] = self.renameFOV
        self.depositMethods['Merge FOVs'] = self.mergeFOVs
        self.depositMethods['Set FOV flags...'] = self.setFOVflag
        self.depositMethods['Run analyses'] = lambda: DY.runAnalyses(self)
        self.depositMethods['Thermal tuning multi-animal'] = self.runThermoTuning
        self.depositMethods['Basal tuning...'] = lambda: DY.thermoTuning(FOVtag = 'Use for basal', split_genotypes = True)
        self.depositMethods['SNI tuning...'] = lambda: DY.thermoTuning(FOVtag = 'SNI 1 week', split_genotypes = True)
        self.depositMethods['CAPS tuning...'] = DY.CAPSfigs
        self.depositMethods['Plot transformations'] = DY.run_longitudinal
        #self.depositMethods['Track cells across sessions'] = lambda: DY.correspond_manual(self)
        self.depositMethods['Track cells across sessions'] = lambda: DY.correspond_manual(self)
        #self.depositMethods['Combine sessions...'] = self.combine_sessions
        #self.depositMethods['Temp vs Temp correlation'] = lambda: DY.TvsTcor(self)
        self.depositMethods['Send to new DB...'] = lambda: DY.isolateFOVtonewDB(self)
        self.depositMethods['Align chronic'] = lambda: DY.align_chronic_data(self)
        self.depositMethods['Export aligned chronic'] = self.export_aligned_data
        self.depositMethods['Remove aligned data'] = lambda: DY.remove_aligned_data(self)
        self.depositMethods['Move ca data flag to aligned'] = lambda: DY.remove_aligned_data(self)
        
        self.depositMethods['Fix file...'] = self.fix_file
        self.depositMethods['Flush'] = self.Flush
        self.depositMethods['Delete FOV'] = self.delFOV
        
        self.depositBox = QComboBox(self)
        self.depositBox.resize(100,40)
        self.depositBox.move(150,400)
        for key in self.depositMethods.keys():
            self.depositBox.addItem(key)
        self.depositBox.currentIndexChanged.connect(self.depositSelected)
       
        self.dataOperations = {}
        self.dataOperations['Change data...'] = print 
        #self.dataOperations['Annotate'] = self.annotate
        self.dataOperations['Rename data'] = self.renameData
        self.dataOperations['Merge data'] = self.concatenateData
        self.dataOperations['Glue data with ROIs'] = lambda: DY.glue(self)
        self.dataOperations['Move data'] = self.moveToNewFOV
        self.dataOperations['Split trials'] = self.splitTrials
        self.dataOperations['Stitch movies'] = self.stitchAndSplit
        self.dataOperations['Delete data'] = self.delData
        self.dataOperations['Set flag'] = self.setDataFlag
        self.dataOperations['Deposit manual'] = self.manualDepositData
        self.dataOperations['Print data info'] = lambda: DY.data_info(self)
        self.dataOperations['Stack 2D'] = lambda: DY.stack_2D(self)
        self.dataOperations['Transfer to new DB'] = lambda: DY.isolateDataToNewDB(self)
        
       

        self.dataOperationBox = QComboBox(self)
        self.dataOperationBox.resize(100,40)
        self.dataOperationBox.move(350,400)
        for key in self.dataOperations.keys():
            self.dataOperationBox.addItem(key)
        self.dataOperationBox.currentIndexChanged.connect(self.dataOperationSelected)
        
        self.FOVflags = {}
        self.FOVflags['Remove']={}
        self.FOVflags['In process'] = {}
        self.FOVflags['Completed'] = {}
        self.FOVflags['None'] = {}
        self.FOVflags['SNI baseline'] = {}
        self.FOVflags['SNI'] = {}
        self.FOVflags['SNI 1 week'] = {}
        self.FOVflags['SNI 2 week'] = {}
        self.FOVflags['SNI 3 week'] = {}
        self.FOVflags['SNI 4 week'] = {}
        self.FOVflags['Interesting'] = {}
        self.FOVflags['Interesting']['color'] = Qt.black
        self.FOVflags['Use for basal'] = {}
        self.FOVflags['CAPS'] = {}
        self.FOVflags['Remove']['color'] = Qt.red
        self.FOVflags['In process']['color'] = Qt.magenta
        self.FOVflags['Completed']['color'] =  Qt.green     
        self.FOVflags['None']['color'] = Qt.black
        self.FOVflags['SNI baseline'] = Qt.darkMagenta
        self.FOVflags['SNI']['color'] = Qt.darkCyan
        self.FOVflags['SNI 1 week']['color'] = Qt.magenta
        self.FOVflags['SNI 2 week']['color'] = Qt.magenta
        self.FOVflags['SNI 3 week']['color'] = Qt.magenta
        self.FOVflags['SNI 4 week']['color'] = Qt.magenta
        self.FOVflags['CAPS']['color'] = Qt.red
        self.FOVflags['Mech-thermo'] =  {}
        self.FOVflags['Mech-thermo']['color'] = Qt.darkCyan
        self.FOVflags['Chronic'] =  {}
        self.FOVflags['Chronic']['color'] = Qt.darkCyan
        self.FOVflags['VAS'] = {}
        self.FOVflags['VAS']['color'] = Qt.darkCyan
        
        
        
        #self.FOVflags['Interesting']['color'] = Qt.yellow
        self.FOVflags['Use for basal']['color'] = Qt.yellow
        
        self.DataFlags = {}
        self.DataFlags['Remove']={}
        self.DataFlags['In process'] = {}
        self.DataFlags['Process'] = {}
        self.DataFlags['Completed'] = {}
        self.DataFlags['None'] = {}
        self.DataFlags['Ca data'] = {}
        self.DataFlags['Ca data']['color'] = Qt.yellow
        self.DataFlags['Thermo stim'] = {}
        self.DataFlags['Thermo stim']['color'] = Qt.blue
        self.DataFlags['pre CAPS'] = {}
        self.DataFlags['pre CAPS']['color'] = Qt.darkCyan
        self.DataFlags['post CAPS'] = {}
        self.DataFlags['post CAPS']['color'] = Qt.red
        self.DataFlags['Remove']['color'] = Qt.red
        self.DataFlags['In process']['color'] = Qt.magenta
        self.DataFlags['Completed']['color'] =  Qt.green
        self.DataFlags['None']['color'] = Qt.black
        self.DataFlags['Process']['color'] = Qt.black
        self.DataFlags['Thermo-mech ca'] = {}
        self.DataFlags['Thermo-mech ca']['color'] = Qt.cyan
        self.DataFlags['Mech stim'] = {}
        self.DataFlags['Mech stim']['color'] = Qt.darkCyan
        self.DataFlags['Aligned ca'] = {}
        self.DataFlags['Aligned ca']['color'] = Qt.green
        self.DataFlags['TRG'] = {}
        self.DataFlags['TRG']['color'] = Qt.red
        
        
        
        self.exportFuncs = {}
        self.exportFuncs['Export types:'] = lambda: print('Hello there')
        self.exportFuncs['Export data to TIFF'] = self.expData
        self.exportFuncs['Export data as raw'] = lambda: DY.exp_data(self)
        self.exportFuncs['Export AVI'] = lambda: DY.write_movie(self)##self.expMovie
        self.exportFuncs['Save plots'] = lambda: self.generateSummaryTable(createFiles = True, trimStim = False)
        self.exportFuncs['Show plots'] = lambda: self.generateSummaryTable(createFiles = False, trimStim = False)
        self.exportFuncs['Export pivot DB'] = self.exportPivotDB
        self.exportFuncs['Export ROIs'] = self.expROImap
        self.exportFuncs['Export FOV to DB'] = self.exportFOV
        self.exportFuncs['Export Data to DB'] = self.exportDataStream
        self.exportFuncs['Show neurons'] = self.showNeurons
        self.exportFuncs['Show field'] = self.expField
        self.exportFuncs['Mechano map'] = lambda: DY.mechanoHeatMap(self)
        self.exportFuncs['Z-Mechano map'] = lambda: DY.z_mech_heat_map(self)
        self.exportFuncs['Show corr'] = self.expHybridCorr
        self.exportFuncs['Mechano plot'] = lambda: DY.mechThreshold(self)
        self.exportFuncs['Align ROIs'] = self.alignROIs
        self.exportFuncs['Align ROIs CAIMAN'] = self.alignROIsCaiman
        self.exportFuncs['Launch alignment'] = self.show_alignment_gui
        self.exportFuncs['Plot contours'] = lambda: DY.plot_contours(self)
        self.exportFuncs['Show session...'] = self.show_single_session
        self.exportFuncs['Show multi session...'] = self.show_multi_session
        self.exportFuncs['Plot transient correlation'] = lambda: DY.plot_transient_correlations(self)
        self.exportFuncs['Plot TRG'] = lambda: TRG.GUI_plot_TRG(self, make_movie=False)
        self.exportFuncs['Analyze TRGs'] = lambda: TRG.analyze_multi_trial(self)
        self.exportFuncs['Select export directory'] = self.updateSaveFolder
        self.exportFuncs['CPU count'] = DY.cpu_count
        self.exportFuncs['Sync figs to google drive'] = sync_figures_to_google_drive
        self.exportFuncs['segment mech trace'] = self.segment_force_trace
        self.exportFuncs['Analyze hot/cold plate'] = lambda: beh.analyze_cold_plate_data(self)
        self.exportFuncs['Merge channels..'] = lambda: DY.color_merge(self)
        self.exportFuncs['Plot eVF'] = lambda: DY.parse_eVF(obj=self)
        
        self.color_order = [[0,1,1], [1,1,0], [1,0,1]]
        
        #Combo box for data export functions 
   
        self.expDataCombo = QComboBox(self)
        self.expDataCombo.resize(100,40)
        self.expDataCombo.move(500,290)
        for key in self.exportFuncs.keys():
            self.expDataCombo.addItem(key)
        self.expDataCombo.currentIndexChanged.connect(self.exportSelected)
        
      
        self.inputDict = {}
        self.inputDict['CERNA'] = {}
        self.inputDict['CERNA']['data_string'] = 'CERNA.tif'
        self.inputDict['CERNA']['time_string'] = 'CERNAtime.txt'
        self.inputDict['CERNA']['data_name'] = 'CERNAraw'
        self.inputDict['CERNA']['read_method'] = DY.readCERNA
        self.inputDict['CERNA']['transform'] = DY.transformCERNA
        
        self.inputDict['CERNA processed'] = {}
        self.inputDict['CERNA processed']['data_string'] = 'CERNAex.tif'
        self.inputDict['CERNA processed']['time_string'] = 'CERNAex_time.txt'
        self.inputDict['CERNA processed']['data_name'] = 'CERNAex'
        self.inputDict['CERNA processed']['read_method'] = DY.readCERNA_proc
        self.inputDict['CERNA processed']['transform'] = None
        
        self.inputDict['TGR'] = {}
        self.inputDict['TGR']['data_string'] = 'FLIR.tif'
        self.inputDict['TGR']['time_string'] = 'PureTherma.txt'
        self.inputDict['TGR']['data_name'] = 'FLIR'
        self.inputDict['TGR']['read_method'] = DY.readTRG
        self.inputDict['TGR']['transform'] = None
        
        self.inputDict['FLIR'] = {}
        self.inputDict['FLIR']['data_string'] = 'FLIRmovie.tif'
        self.inputDict['FLIR']['time_string'] = 'FLIRtime.txt'
        self.inputDict['FLIR']['data_name'] = 'FLIR_degC'
        self.inputDict['FLIR']['read_method'] = DY.readFLIR
        self.inputDict['FLIR']['transform'] = None
        
        self.inputDict['eVF'] = {}
        self.inputDict['eVF']['data_string'] = 'VFdata.txt'
        self.inputDict['eVF']['time_string'] = 'VFtime.txt'
        self.inputDict['eVF']['data_name'] = 'eVF_mN'
        self.inputDict['eVF']['read_method'] = DY.readVF
        self.inputDict['eVF']['transform'] = None
        
        self.inputDict['NIR'] = {}
        self.inputDict['NIR']['data_string'] = ('NIRcamMovie.avi', 'NIRcamMovie.tif') ## endswith checks all memmbers of tuple
        self.inputDict['NIR']['time_string'] = 'NIRcamTime.txt'
        self.inputDict['NIR']['data_name'] = 'NIRcam'
        self.inputDict['NIR']['read_method'] = DY.readNIR
        self.inputDict['NIR']['transform'] = None
        
        self.inputDict['cam'] = {}
        self.inputDict['cam']['data_string'] = 'HD USB Cam.avi' ## endswith checks all memmbers of tuple
        self.inputDict['cam']['time_string'] = 'HD USB Cam.txt'
        self.inputDict['cam']['data_name'] = 'camera'
        self.inputDict['cam']['read_method'] = DY.read_linked_cam
        self.inputDict['cam']['link_method'] = DY.link_cam
        self.inputDict['cam']['transform'] = None
        
        self.inputDict['AuroraForce'] = {}
        self.inputDict['AuroraForce']['data_string'] = 'AuroraData.txt' ## endswith checks all emmbers of tuple
        self.inputDict['AuroraForce']['time_string'] = 'AuroraTime.txt'
        self.inputDict['AuroraForce']['data_name'] = 'AuroraForce_mN'
        self.inputDict['AuroraForce']['read_method'] = DY.readAurora
        self.inputDict['AuroraForce']['transform'] = None
        
        self.inputDict['AuroraX'] = {}
        self.inputDict['AuroraX']['data_string'] = 'XstageData.txt' ## endswith checks all emmbers of tuple
        self.inputDict['AuroraX']['time_string'] = 'AuroraTime.txt'
        self.inputDict['AuroraX']['data_name'] = 'Aurora_X'
        self.inputDict['AuroraX']['read_method'] = DY.readAurora
        self.inputDict['AuroraX']['transform'] = None
        
        self.inputDict['AuroraY'] = {}
        self.inputDict['AuroraY']['data_string'] = 'YstageData.txt' ## endswith checks all emmbers of tuple
        self.inputDict['AuroraY']['time_string'] = 'AuroraTime.txt'
        self.inputDict['AuroraY']['data_name'] = 'Aurora_Y'
        self.inputDict['AuroraY']['read_method'] = DY.readAurora
        self.inputDict['AuroraY']['transform'] = None
        
        self.inputDict['Notes'] = {}
        self.inputDict['Notes']['data_string'] = 'noteStream.txt'
        self.inputDict['Notes']['time_string'] = 'noteStreamTime.txt'
        self.inputDict['Notes']['data_name'] = 'notes'
        self.inputDict['Notes']['read_method'] = DY.read_annotation
        self.inputDict['Notes']['ROI_handler'] = DY.key_ROI_handler
        self.inputDict['Notes']['transform'] = None
        
        self.inputDict['Keys'] = {}
        self.inputDict['Keys']['data_string'] = 'keyStream.txt'
        self.inputDict['Keys']['time_string'] = 'keyStreamTime.txt'
        self.inputDict['Keys']['data_name'] = 'key_data'
        self.inputDict['Keys']['read_method'] = DY.read_annotation
        self.inputDict['Keys']['ROI_handler'] = DY.key_ROI_handler
        self.inputDict['Keys']['transform'] = None
        
        
        
        #Button to manually add ROI:
        self.drawROIbutton = QPushButton('Draw ROI',self)
        self.drawROIbutton.resize(100,40)
        self.drawROIbutton.move(450,540)
        self.drawROIbutton.clicked.connect(self.drawROI)
        
        #Button to delete  ROI:
        self.drawROIbutton = QPushButton('Remove ROI',self)
        self.drawROIbutton.resize(100,40)
        self.drawROIbutton.move(450,580)
        self.drawROIbutton.clicked.connect(self.delROI)
        
        #Button to sort  ROIs:
        self.sortROIbutton = QPushButton('Sort ROIs',self)
        self.sortROIbutton.resize(100,40)
        self.sortROIbutton.move(450,620)
        self.sortROIbutton.clicked.connect(self.KmeanSortTraceArray)
        
        
        
        ##########
        ##
        ##      DATASET NAVIGATION
        ##
        #########
        
        #Text to display current DB name
        self.DBidText = QLineEdit('DBpath', self)
     
        
        self.DBidText.setReadOnly(True)
        self.DBidText.setFrame(False)
        self.DBidText.move(20, 950)
        self.DBidText.resize(700, 25)
        
        
        self.infoText = QLineEdit('Info', self)
        self.infoText.setReadOnly(True)
        self.infoText.setFrame(False)
        self.infoText.move(20, 975)
        self.infoText.resize(700, 25)
        
        
        #list of experimental animals:
        self.animalsList = QListWidget(self)
        self.animalsList.move(10,100)
        self.animalsList.resize(90,300)
        self.animalsList.setSelectionMode(QtWidgets.QAbstractItemView.
                                          SingleSelection)
        self.animalsList.itemSelectionChanged.connect(self.updateActiveAnimal)
        
        #list of FOVs:
        self.FOVlist = QListWidget(self)
        self.FOVlist.move(100,100)
        self.FOVlist.resize(200,300)
        #self.FOVlist.setSelectionMode(QtWidgets.QAbstractItemView.
        #                                  SingleSelection)
        self.FOVlist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.FOVlist.itemSelectionChanged.connect(self.updateActiveFOV)
        
        #list of available data streams:
        self.DataList = QListWidget(self)
        self.DataList.move(300,100)
        self.DataList.resize(200,300)
        self.DataList.setSelectionMode(QtWidgets.QAbstractItemView.
                                          ExtendedSelection)
        self.DataList.itemSelectionChanged.connect(self.updateActiveData)
        
        #list of ROIs for datastream in focus
        self.ROIlist = QListWidget(self)
        self.ROIlist.move(550,540)
        self.ROIlist.resize(100,180)
        self.ROIlist.setSelectionMode(QtWidgets.QAbstractItemView.
                                          ExtendedSelection)
        self.ROIlist.itemSelectionChanged.connect(self.updateActiveROIs)
        
        #list of signals (1D data) for display/analysis:   
    #    self.sigList = QListWidget(self)
    #    self.sigList.move(1705,25)
    #    self.sigList.resize(200,300)
    #    self.sigList.setSelectionMode(QtWidgets.QAbstractItemView.
               #                          ExtendedSelection)
        
    #    self.sigList.itemSelectionChanged.connect(self.updateActiveSignals)
        
        #button to transfer 1D data to signal list
    #    self.transferSignalBtn = QPushButton('Add signal(s)',self)
    #    self.transferSignalBtn.resize(100,40)
    #    self.transferSignalBtn.move(1705,210)
    #    self.transferSignalBtn.clicked.connect(self.addSignal)
        
        #button to remove signals from signal list
     #   self.removeSigBtn = QPushButton('Remove signal(s)',self)
      #  self.removeSigBtn.resize(100,40)
        #self.removeSigBtn.move(1805,210)
       # self.removeSigBtn.clicked.connect(self.removeSignal)
        
        #button to run correlation analysis
       # self.corrBtn = QPushButton('Correlation map',self)
      #  self.corrBtn.resize(100,40)
     #   self.corrBtn.move(1705,250)
        #self.corrBtn.clicked.connect(self.corMap)
        
        
        ##### SEGMENTATION ###
        self.segmentationMethods = {}
        self.segmentationMethods['Adaptive threshold']  = {}
        self.segmentationMethods['Adaptive threshold']['Function'] = 'adaptiveThreshold'
        self.segmentationMethods['Adaptive threshold']['Params']={}
        self.segmentationMethods['Adaptive threshold']['Params']['Block size'] = [1,51,101] #min, default, max
        self.segmentationMethods['Adaptive threshold']['Params']['Erode cycles'] = [0,1,10]
        self.segmentationMethods['Adaptive threshold']['Params']['Erode area'] = [1,3,10]
        self.segmentationMethods['Adaptive threshold']['Params']['C'] = [-20,-5,20]
        self.segmentationMethods['Adaptive threshold']['Params']['Min area'] = [1,10,250]
        self.segmentationMethods['Adaptive threshold']['Params']['Max area'] = [1,50,250]
        
        
        self.segmentationMethods['Random params']  = {}
        self.segmentationMethods['Random params']['Function'] = 'doNothing'
        self.segmentationMethods['Random params']['Params']={}
        self.segmentationMethods['Random params']['Params']['nClusters'] = [1, 23, 128]
        self.segmentationMethods['Random params']['Params']['Components to split'] = [1,4,128]
        self.segmentationMethods['Random params']['Params']['Components to save'] = [0,1,128]
        self.segmentationMethods['Random params']['Params']['spatial filter'] = [1, 5 ,100] 
        self.segmentationMethods['Random params']['Params']['ROIalpha pct'] = [1, 50 ,100] 
        
    
        
        self.segmentationMethods['Stitching']  = {}
        self.segmentationMethods['Stitching']['Function'] = 'doNothing'
        self.segmentationMethods['Stitching']['Params']={}
        self.segmentationMethods['Stitching']['Params']['split alignment'] = [0, 1, 1]
        self.segmentationMethods['Stitching']['Params']['trim alignment'] = [0, 1, 1]
        self.segmentationMethods['Stitching']['Params']['Across FOVs'] = [0, 1, 1]
        self.segmentationMethods['Stitching']['Params']['Transform ROIs'] = [0, 1, 1]
        
        self.segmentationMethods['Transients']  = {}
        self.segmentationMethods['Transients']['Function'] = 'doNothing'
        self.segmentationMethods['Transients']['Params']={}
        self.segmentationMethods['Transients']['Params']['pad area'] = [0, 5, 512]
        self.segmentationMethods['Transients']['Params']['pad transients'] = [0, 0, 512]
        self.segmentationMethods['Transients']['Params']['SNR thresh'] = [0, 5, 32]
        self.segmentationMethods['Transients']['Params']['Min duration'] = [0, 10, 160]

        
        
        self.segmentationMethods['seeded CNMF']  = {}
        self.segmentationMethods['seeded CNMF']['Function'] = 'seededCNMF'
        self.segmentationMethods['seeded CNMF']['Params']={}
        self.segmentationMethods['seeded CNMF']['Params']['decay_time'] = [1, 14, 50]
        self.segmentationMethods['seeded CNMF']['Params']['p'] = [0, 1, 2]
        self.segmentationMethods['seeded CNMF']['Params']['gsig'] = [1, 11, 25]
        self.segmentationMethods['seeded CNMF']['Params']['ssub'] = [1, 1, 4]
        self.segmentationMethods['seeded CNMF']['Params']['tsub'] = [1, 1, 10]
        self.segmentationMethods['seeded CNMF']['Params']['merge_thr'] = [50, 85, 100]
        self.segmentationMethods['seeded CNMF']['Params']['Clusters'] = [1, 5, 25] #K means clusters for sorting
        self.segmentationMethods['seeded CNMF']['Params']['Append'] = [0, 1, 1] # if 0 process all ROIs and replace, if 1 process selected and append
        self.segmentationMethods['seeded CNMF']['Params']['User defined masks'] = [0,1,1] # run caiman with/without target ROIs
        self.segmentationMethods['seeded CNMF']['Params']['Separate input masks'] = [0,1,1] ## just for testing
        self.segmentationMethods['seeded CNMF']['Params']['Sort traces'] = [0,1,1] ## turn offfor consistency across FOVs
        self.segmentationMethods['seeded CNMF']['Params']['Detrend'] = [0,0,1]
        
        self.segmentationMethods['Extract traces']  = {}
        self.segmentationMethods['Extract traces']['Function'] = 'extractTracesVectorized'
        self.segmentationMethods['Extract traces']['Params']={}
        self.segmentationMethods['Extract traces']['Params']['Calculate DFF'] = [0, 0, 1]
        self.segmentationMethods['Extract traces']['Params']['Extract percentile'] = [0, 50, 100]
        self.segmentationMethods['Extract traces']['Params']['Z-score'] = [0, 0, 1]
        
        
        self.segmentationMethods['Calculate DFF']  = {}
        self.segmentationMethods['Calculate DFF']['Function'] = 'calcDFF'
        self.segmentationMethods['Calculate DFF']['Params']={}
        self.segmentationMethods['Calculate DFF']['Params']['F0 Quantile'] = [0, 1, 10]
        self.segmentationMethods['Calculate DFF']['Params']['Normalize'] = [0, 0, 1]
        self.segmentationMethods['Paint ROIs']  = {}
        self.segmentationMethods['Paint ROIs']['Function'] = 'paintROIs'
        self.segmentationMethods['Paint ROIs']['Params']={}
        self.segmentationMethods['Paint ROIs']['Params']['Painting on'] = [0, 0, 1]
        
        self.segmentationMethods['Reorder ROIs']  = {}
        self.segmentationMethods['Reorder ROIs']['Function'] = 'reorderROIs'
        self.segmentationMethods['Reorder ROIs']['Params']={}
        self.segmentationMethods['Reorder ROIs']['Params']['placeholder'] = [0, 0, 1]
        
        self.segmentationMethods['PCA']  = {}
        self.segmentationMethods['PCA']['Function'] = 'PCA'
        self.segmentationMethods['PCA']['Params']={}
        self.segmentationMethods['PCA']['Params']['n_PCs'] = [1, 3, 100]
        
        self.segmentationMethods['Merge']  = {}
        self.segmentationMethods['Merge']['Function'] = 'mergeROIs'
        self.segmentationMethods['Merge']['Params']={}
        self.segmentationMethods['Merge']['Params']['vmin'] = [0, 1, 100]
        self.segmentationMethods['Merge']['Params']['vmax'] = [0, 50, 100]
        
        self.segmentationMethods['Movie']  = {}
        self.segmentationMethods['Movie']['Function'] = 'mergeROIs'
        self.segmentationMethods['Movie']['Params']={}
        self.segmentationMethods['Movie']['Params']['vmin'] = [0, 5, 10000]
        self.segmentationMethods['Movie']['Params']['vmax'] = [0, 9995, 10000]
        self.segmentationMethods['Movie']['Params']['scale'] = [0, 3, 10]
        self.segmentationMethods['Movie']['Params']['acceleration']=[0, 5, 25]
        self.segmentationMethods['Movie']['Params']['margin'] = [0, 70, 200]
    
        self.segmentationMethods['Mark indentations']  = {}
        self.segmentationMethods['Mark indentations']['Function'] = 'autoMarkIndentations'
        self.segmentationMethods['Mark indentations']['Params']={}
        self.segmentationMethods['Mark indentations']['Params']['Min force '] = [0, 10, 300]
        self.segmentationMethods['Mark indentations']['Params']['Max baseline '] = [0, 5, 100]
        self.segmentationMethods['Mark indentations']['Params']['Min interval (s/10)'] = [0, 5, 100]
        self.segmentationMethods['Mark indentations']['Params']['Min prominence'] = [0, 10, 100]
        
        
        
        
        
        
        
        self.currentSegmentationMethod = 'Adaptive threshold'
       
        self.updateSegmentationBtn = QPushButton('Update ROIs...',self)
        self.updateSegmentationBtn.resize(100,40)
        self.updateSegmentationBtn.move(50,700)
        self.updateSegmentationBtn.clicked.connect(self.applySegmentation)
        
                               
        self.segmentaTionMethodBox = QComboBox(self)
        for key in self.segmentationMethods:
            self.segmentaTionMethodBox.addItem(key)
        self.segmentaTionMethodBox.resize(100,25)
        self.segmentaTionMethodBox.move(50,745)
        self.segmentaTionMethodBox.currentIndexChanged.connect(self.updateSegmentationMethod)
        
        
        self.segmentParamBox = QComboBox(self)
        for key in self.segmentationMethods[self.currentSegmentationMethod ]['Params']:
            self.segmentParamBox.addItem(key)
        self.segmentParamBox.resize(100,25)
        self.segmentParamBox.move(50,770)
        self.segmentParamBox.currentIndexChanged.connect(self.updateSegmentationParamSelection)
        
        self.segParamValText = QLineEdit('segParamVal', self)
        self.segParamValText.setReadOnly(True)
        self.segParamValText.setFrame(False)
        self.segParamValText.resize(200,20)
        self.segParamValText.move(200,745)
        
        self.updateSegmentationSlider = QSlider(self)
        self.updateSegmentationSlider.setOrientation(Qt.Horizontal)
        self.updateSegmentationSlider.setMinimum(0)
        self.updateSegmentationSlider.setMaximum(50)
        self.updateSegmentationSlider.setSingleStep(1)
        self.updateSegmentationSlider.setPageStep(10)
        self.updateSegmentationSlider.setValue(0)
        self.updateSegmentationSlider.move(150,725)
        self.updateSegmentationSlider.resize(200,20)
        self.updateSegmentationSlider.valueChanged.connect(self.updateSegParamValue)
        
        
        
        self.updateSegmentationMethod()
        
        
        self.clickAction  = 'None'
        
        self.clickActions = {}
        self.clickActions['None'] = self.doNothing
        self.clickActions['Select ROI'] = self.selectROI
        self.clickActions['Create ROI'] = self.createROI
        self.clickActions['Split ROI'] = self.splitROI
        #self.clickActions['Merge ROIs'] = self.mergeROIs
        
        self.clickActions['Add ROI to cell group'] = self.addNeuron
        self.clickActions['Mark event'] = self.markEvent
        self.clickActions['Mark event series'] = self.markAllEvents
        
        
        self.clickActionBox = QComboBox(self)
        for key in self.clickActions:
            self.clickActionBox.addItem(key)
        self.clickActionBox.resize(100,25)
        self.clickActionBox.move(500,180)
        self.clickActionBox.currentIndexChanged.connect(self.selectClickAction)
        
        
        
        
        
        self.transferROIBtn = QPushButton('Transfer masks...',self)
        self.transferROIBtn.resize(100,40)
        self.transferROIBtn.move(50,795)
        self.transferROIBtn.clicked.connect(self.transferMasks)
        
        self.combineROIBtn = QPushButton('Combine masks...',self)
        self.combineROIBtn.resize(100,40)
        self.combineROIBtn.move(150,795)
        self.combineROIBtn.clicked.connect(self.combineMasks)
        
        
        ####  CANVAS ###
        #Initialize canvas
        #self.initCanvasBtn = QPushButton('Init Canvas',self)
        #self.initCanvasBtn.resize(100,40)
        #self.initCanvasBtn.move(50,700)
        #self.initCanvasBtn.clicked.connect(self.initCanvas)
        
        
        #Add selected dataset to canvas
        #self.addtoCanvasBtn = QPushButton('Add data to canvas',self)
        #self.addtoCanvasBtn.resize(100,40)
        #self.addtoCanvasBtn.move(50,740)
        #self.addtoCanvasBtn.clicked.connect(self.addToCanvas)
        
        #Remove dataset from canvas
        
        #Export canvas image/data
        #self.dispCanvas = QPushButton('Show canvas',self)
        #self.dispCanvas.resize(100,40)
        #self.dispCanvas.move(50,780)
        #self.dispCanvas.clicked.connect(self.addToCanvas)
        
        
        

        #####################
        ##
        ##  Data interaction
        ##
        ####################
      
        
        
        
        # Slider to navigate selected time period
        self.timeSlider = QSlider(self)
        self.timeSlider.setOrientation(Qt.Horizontal)
        self.timeSlider.setMinimum(0)
        self.timeSlider.setMaximum(1)
        self.timeSlider.setSingleStep(1)
        self.timeSlider.setPageStep(10)
        self.timeSlider.setValue(0)
        self.timeSlider.move(650,725)
        self.timeSlider.resize(1024,20)
        self.timeSlider.valueChanged.connect(self.updateDisplay)
        #self.ledSlider.sliderReleased.connect(self.ledUpdateSliderRelease)
        
        
        self.TresSlider = QSlider(self)
        self.TresLUT = np.array([1/1000, 1/100, 1/50, 1/30,1/25,1/20,1/10,1/5,1/4,1/3,1/2,1,2,5,10])
        self.TresSlider.setValue(5)
        self.TresSlider.setOrientation(Qt.Vertical)
        self.TresSlider.setMinimum(0)
        self.TresSlider.setMaximum(len(self.TresLUT)-1)
        self.TresSlider.setSingleStep(1)
        self.TresSlider.setPageStep(3)
        self.TresSlider.move(630,745)
        self.TresSlider.resize(20,100)
        self.timeStep = 1
        self.TresSlider.valueChanged.connect(self.updateResSlider)
        self.TresSlider.sliderReleased.connect(self.updateSliderBounds)
        self.TresLabel = QLineEdit('T step: 1 s', self)
        self.TresLabel.setReadOnly(True)
        self.TresLabel.setFrame(False)
        self.TresLabel.move (540, 795)
        self.TresLabel.resize(75,25)
        
        self.timeLabel = QLineEdit('00:00', self)
        self.timeLabel.setReadOnly(True)
        self.timeLabel.setFrame(False)
        self.timeLabel.move(500, 725)
        self.timeLabel.resize(150,25)
        
        self.frameLabel = QTextEdit('', self)
        self.frameLabel.setReadOnly(True)
        #self.frameLabel.setFrame(False)
        self.frameLabel.move(1675, 755)
        self.frameLabel.resize(150,150)
        
        
        self.transformDict = {}
        self.suffixDict    = {}
        
        
        #dself.transformDict['FFT high pass']         = 'DY.FFTcorrectImageStack'
        self.transformDict['FFT high pass']         = DY.pFFThighPass
        self.suffixDict['FFT high pass']            = '_FFT_HP5'
        
        self.transformDict['Median filter 2D']      = DY.pMedFilt2
        self.suffixDict['Median filter 2D']         = '_medFilt2'
        
        self.transformDict['Median filter 3D']      = DY.medFilt3D
        self.suffixDict['Median filter 3D']         = '_medFilt3'
        
        self.transformDict['Reg and crop']   = DY.reg_and_crop
        self.suffixDict['Reg and crop']      = '_RBreg_cropz'
        
        self.transformDict['Rotate']   = DY.rotateStack
        self.suffixDict['Rotate']      = '_rot90'
        
        self.transformDict['Create annotation']   = DY.add_annotated_track
        self.suffixDict['Create annotation']      = '_annotation'
        #self.transformDict['Time code']   = DY.time_code
        #self.suffixDict['Time code']      = '_time_coded'
        
        
        self.transformDict['Crop with ROIs']   = DY.crop_with_ROIs
        self.suffixDict['Crop with ROIs']      = '_crop'
        
        #self.transformDict['Rigid Body Register']   = 'DY.RBregisterStack'
        self.transformDict['Rigid Body Register']   = DY.pReg
        self.suffixDict['Rigid Body Register']      = '_RBreg'
        
        self.transformDict['Affine Register']   = DY.affineReg
        self.suffixDict['Affine Register']      = '_Areg'
        
        self.transformDict['Bilinear Register']   = DY.bilinearReg
        self.suffixDict['Bilinear Register']      = '_RBreg'
        
        self.transformDict['Get transform-RB']   = DY.pGetTransform
        self.suffixDict['Get transform-RB']      = '_RBtrans'
        
        self.transformDict['Get transform-from template']   = DY.pGetTransformFromTemplate
        self.suffixDict['Get transform-from template']      = '_RBtransT'
        
        self.transformDict['Convert to 16-bit']   = DY.normalizeStack
        self.suffixDict['Convert to 16-bit']      = '_uint16'
        
        self.transformDict['Apply transform-RB']   = DY.applyTransform
        self.suffixDict['Apply transform-RB']      = '_RB_a_reg'
        
        self.transformDict['Max project']           = DY.stackMax
        self.suffixDict['Max project']              = '_MAX'
        
        self.transformDict['Median project']           = DY.stackMedian
        self.suffixDict['Median project']              = '_MED'
        
        self.transformDict['Invert']           = DY.invert_stack
        self.suffixDict['Invert']              = '_inv'
        
        self.transformDict['Standard deviation']           = DY.stackDev
        self.suffixDict['Standard deviation']              = 'std'
        
        self.transformDict['Neighbor correlate']           = DY.crossCorr
        self.suffixDict['Neighbor correlate']              = '_nCor'
        
        self.transformDict['DFF']                   = DY.dff
        self.suffixDict['DFF']                      = '_DFF'
        
        self.transformDict['Absolute value']   = DY.abs_value
        self.suffixDict['Absolute value']      = '_abs'
        
        self.transformDict['Erode']                   = DY.erode
        self.suffixDict['Erode'] =                      '_erode'
        
        self.transformDict['Simple Threshold']                   = DY.simpleThreshold
        self.suffixDict['Simple Threshold']                      = '_Sthresh'
        
        self.transformDict['Adaptive Threshold']                   = DY.adaptiveThresh
        self.suffixDict['Adaptive Threshold']                      = '_thresh'
        
        self.transformDict['Difference']                   = DY.diff
        self.suffixDict['Difference']                      = '_diff'
        
        
        self.transformDict['Recolor']                   = DY.recolor
        self.suffixDict['Recolor']                      = '_recol'
        
        self.transformDict['Track probe']                   = DY.track_probe
        self.suffixDict['Track probe']                      = '_probe'
        
        self.transformDict['Convert to deg C']                   = DY.convertTempToC
        self.suffixDict['Convert to deg C']                      = '_Cconvert'
        
        self.transformDict['Block register']    = DY.blockRegister
        self.suffixDict['Block register']       = '_bReg'
        
        self.transformDict['Register to template']    = DY.registerTemplate
        self.suffixDict['Register to template']       = '_tReg'
        
        self.transformDict['Register color movie']    = DY.regColor
        self.suffixDict['Register color movie']       = '_cReg'
        
        self.transformDict['Convert to grayscale']    = DY.makeGrayscale
        self.suffixDict['Convert to grayscale']       = '_gray'
        
        self.transformDict['Generate paw map']    = DY.createPawMap
        self.suffixDict['Generate paw map']       = '_paw'
        
        
      #  self.transformDict['Manual register']    = 'DY.manualRegister'
      #  self.suffixDict['Manual register']       = '_mReg'
        
        self.transformDict['Stitch']    = DY.Stitch
        self.suffixDict['Stitch']       = '_stitch'
        
        self.transformDict['Copy to new stream']    = DY.copyData
        self.suffixDict['Copy to new stream']       = '_copy'
        
        self.transformDict['Correlation map']    = DY.corMap
        self.suffixDict['Correlation map']       = '_corrS'
        
        self.transformDict['Inverse Correlation map']    = DY.inverseCorMap
        self.suffixDict['Inverse Correlation map']       = '_corrSI'
        
    
        self.transformDict['Split data']    = DY.removeSelected
        self.suffixDict['Split data']       = '_excised'
        
        
        self.transformDict['Correct RGB']    = DY.transposeRGB
        self.suffixDict['Correct RGB']       = '_RGBcor'
        
        self.transformDict['Grams to millinewtons']    = DY.grams_to_millinewtons
        self.suffixDict['Grams to millinewtons']       = '_mN'
        
        self.transformDict['Crop']    = DY.crop
        self.suffixDict['Crop']       = '_crop'
        
        self.transformDict['Crop out zeros']    = DY.cropToZeros
        self.suffixDict['Crop out zeros']       = '_cropZ'
        
        self.transformDict['Trim neg']    = DY.rectifyMech
        self.suffixDict['Trim neg']       = '_fixNegMech'
        
        self.transformDict['Adjust baseline force']    = DY.adjust_baseline_mech
        self.suffixDict['Adjust baseline force']       = '_fixBaseMech'
        
        self.transformDict['NORMCORE']    = DY.normCOR
        self.suffixDict['NORMCORE']       = '_normCOR'
        
        self.transformDict['Register from ROIs']    = DY.registerUsingROIs
        self.suffixDict['Register from ROIs']       = '_ROIreg'
        
        self.transformDict['Process TRG']    = TRG.GUI_plot_TRG
        self.suffixDict['Process TRG']       = '_TRG'
        
        
        self.transformDict['Mask ROIs']    = DY.mask_ROIs
        self.suffixDict['Mask ROIs']       = '_masked'
        
        self.transformDict['Mask inverse ROIs']    = DY.mask_inverse_ROIs
        self.suffixDict['Mask inverse ROIs']       = '_masked_i'
        
        self.transformDict['Flip horizontal']    = DY.flip_horizontal
        self.suffixDict['Flip horizontal']       = '_flphor'
        
        self.transformDict['Flip vertical']    = DY.flip_vertical
        self.suffixDict['Flip vertical']       = '_flpvert'
        
        self.transformDict['remove bad']    = DY.removeBadFrames
        self.suffixDict['remove bad']       = '_badFrameDummy'
        
        self.transformDict['downsample spatial']    = DY.downSampleSpatial
        self.suffixDict['downsample spatial']       = '_halfSize'
        
        self.transformDict['downsample temporal']    = DY.downSampleTemporal
        self.suffixDict['downsample temporal']       = '_dwnT'
        
        self.transformBox = QComboBox(self)
        self.transformBox.addItem('Data transformations...')
        for key in self.transformDict:
            self.transformBox.addItem(key)
        self.transformBox.resize(100,25)
        self.transformBox.move(500,100)
        self.transformBox.currentIndexChanged.connect(self.doTransform3)
        
        
        self.transformROIbtn = QPushButton('Draw ROI',self)
        self.transformROIbtn.resize(100,50)
        self.transformROIbtn.move(500,125)
        self.transformROIbtn.clicked.connect(self.drawTransformROI)
        
        self.transformROIlist = []
        
        self.reportSaveDir =  '/lab-share/Neuro-Woolf-e2/Public/Figure publishing/'

        self.play_direction = True
        
        
        #####################
        ##
        ##  Initialize variables
        ##
        ####################
        #self.timeDict = {}
        #self.timeDict['CERNAraw'] = 'CERNAtime'
        #self.timeDict['FLIRraw'] = 'FLIRtime'
        #self.timeDict['NIRraw'] = 'NIRtime'
        #self.timeDict['VFraw'] = 'VFtime'
        self.timeLine2_min = 0
        self.timeLine2_max = 1
        self.ROIsetDict = {}
        self.segmentationResult = {}
        self.exportPlots = False
        self.plotMode = 'full'
        self.curAnimal = ''
        self.curFOV = ''
        self.dataFocus = ''
        self.curDataStreams = []
        self.curTime = []
        self.dispTime = [0,1]
        self.curROIs = {}
        self.selectedROI = np.array([])
        self.sampleIndex = {}
        self.play_direction = True
        self.playTimer = pg.QtCore.QTimer()
        self.r_playTimer = pg.QtCore.QTimer()
        self.playTimer.timeout.connect(self.step_forward)
        self.r_playTimer.timeout.connect(self.step_back)
        self.show()
      
    def doNothing(self, X, Y, labelMap):
        pass
        
    def keyPressEvent(self, e):
        print (f'{e.key()=} {e.text()}')
        if e.key() == 16777234: ## left arrow
            if e.modifiers() & Qt.ShiftModifier:
                self.RstepDataFocus()
            else:
                self.step_back()
            #self.RstepDataFocus()
        elif e.key() == 16777236: ## right arrow
            if e.modifiers() & Qt.ShiftModifier:
                self.stepDataFocus()
            else:
                print('stepping forward')
                self.step_forward()
       
        
        elif e.key() == 72: ## h key
            self.toggleHists()
        elif e.key() == 65:  ## a key
            self.selectAllROIs()
        elif e.key() == 32: ## 
            if e.modifiers() and Qt.ControlModifier:
                spread = True
            else:
                spread = False
            if e.modifiers() & Qt.ShiftModifier:
                self.remove_behavior_mark(spread=spread)
            else:
                self.mark_behavior(spread=spread)
        elif e.key() == 82:  ## r key
            self.play_reverse()
        elif e.key() == 80:  ## p key
            self.play_data()
        elif e.key() == 78: ##n key
            self.name_roi()
        # elif e.key() == 83: # s key
        #     self.annotate(val = 1)
        # elif e.key() == 84: #t
        #     self.annotate(val = 0)
            
            
    def step_forward(self):
        curVal = self.timeSlider.value()
        if curVal + 1 > len(self.timeLUT)-1:
            self.timeSlider.setValue(0)
        else:
            self.timeSlider.setValue(curVal+1)
    
    def step_back(self):
        curVal = self.timeSlider.value()
        if curVal == 0:
            self.timeSlider.setValue(len(self.timeLUT)-1)
        else:
            self.timeSlider.setValue(curVal-1)
    
    def play_data(self):
        if self.playTimer.isActive() or self.r_playTimer.isActive():
            self.stop_playing_data()
        else:
            self.playTimer.start(10)
    
    def stop_playing_data(self):
        self.playTimer.stop()
        self.r_playTimer.stop()
    
    def play_reverse(self):
        if self.playTimer.isActive() or self.r_playTimer.isActive():
            self.stop_playing_data()
        else:
            self.r_playTimer.start(10)
        
        #self.play_data()
      
    def mark_behavior(self, spread = False):
    
        sample = self.sampleIndex[self.dataFocus]
        if spread:
            size = 10
            if sample + size > self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]:
                size = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]
            samples = np.arange(sample, sample+size, 1)
            
            if len(self.selectedROI)>0:
                for sample in samples:
                    for ROI in self.selectedROI:
                        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][ROI,sample] = 1
                self.updateROImask()
            if 'annotation' in self.dataFocus:
                for sample in samples:
                    self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][sample] = 1
                self.updateDisplay()
                
        else:
            if len(self.selectedROI)>0:
                for ROI in self.selectedROI:
                    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][ROI,sample] = 1
                self.updateROImask()
            if 'annotation' in self.dataFocus:
                self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][sample] = 1
                self.updateDisplay()
    
    def name_roi(self):
        if len(self.selectedROI) ==0:
            print('No ROI selected to name')
            return
        if len(self.selectedROI) > 1:
            print('Try naming your ROIs one at a time')
            return
        name, okPressed = QInputDialog.getText(self,"Name ROI:", "Name:", QLineEdit.Normal, "")
        if not okPressed:
            return
        if not ('names' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys()):
            dt = h5py.string_dtype(encoding = 'utf-8')
            nROIs= self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'].shape[2]
            print(f'{nROIs=}')
            self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].require_dataset('names',shape=(nROIs,1), dtype = dt)
            
        N = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['names']
        print(f'{N.shape=}')
        Rindex = self.selectedROI[0]
        N[Rindex,0] = name
        print(f'{N[...]=}')
        self.updateROIlist()
        
        
        
    def remove_behavior_mark(self, spread  = False):
        print(spread)
        sample = self.sampleIndex[self.dataFocus]
        if spread:
            size = 10
            if sample + size > self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]:
                size = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]
            samples = np.arange(sample, sample+size, 1)
            
            if len(self.selectedROI)>0:
                for sample in samples:
                    for ROI in self.selectedROI:
                        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][ROI,sample] = 0
                self.updateROImask()
            if 'annotation' in self.dataFocus:
                for sample in samples:
                    self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][sample] = 0
                self.updateDisplay()
                
        else:
            if len(self.selectedROI)>0:
                for ROI in self.selectedROI:
                    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][ROI,sample] = 0
                self.updateROImask()
            if 'annotation' in self.dataFocus:
                self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][sample] = 0
                self.updateDisplay()
        # elif e.key() == 67:
        #     if self.plotMode == 'compressed':
        #         self.plotMode = 'full'
        #     else:
        #         self.plotMode = 'compressed'
        #     print(self.plotMode)
        #     self.update1DdataDisplay()
    
    def getFile(self):
        #print(self.caiman)
        result = QFileDialog.getOpenFileName(self, "Choose file...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        return(os.path.normpath(result[0]))
    
    def saveFile(self):
        result = QFileDialog.getSaveFileName(self,  "Save file...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        return(os.path.normpath(result[0]))
               
    def getDir(self):
        result = QFileDialog.getExistingDirectory(self, "Choose directory...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        result = os.path.normpath(result)
        print(result)
        return(result)
    
    @pyqtSlot()
    def slurmDeposit(self):
        FOLDER = self.getDir()

        tT = time.localtime(time.time())
        mouseName = os.path.split(FOLDER)[1]
        IDstr = mouseName + '_proc_' + str(tT[2]) + str(tT[1]) + str(tT[0]) + str(tT[3]) + str(tT[4])
        os.mkdir(f'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Processed/{IDstr}')

        f = open('/home/ch184656/YarmoPain_GUI/targetPath.txt','w')
        f.write(FOLDER)
        f.close()

        g = open('/home/ch184656/YarmoPain_GUI/depositPath.txt','w')
        g.write(f'/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data/Processed/{IDstr}')
        g.close()

        #os.system('sbatch /home/ch184656/YarmoPain_GUI/slurmDepositCompute.sh')  ##run on compute node
        os.system('sbatch /home/ch184656/YarmoPain_GUI/slurmDepositBigMem.sh')     ##run on big memory node

    @pyqtSlot()
    def depositMiniscope(self):
        DY.depositMiniscopeSession(self)
        
    @pyqtSlot()
    def localDeposit(self):
        DY.localDepositMouse()
    
    def updateSaveFolder(self):
        self.reportsaveDir = DY.selectFolder()
        
        
    @pyqtSlot()
    def appendSessions(self):
        targetDir = DY.selectFolder()
       # transformations = self.transformDict.keys()  ## To do: select transforms when appending
       # preProcess = False
        
       # preProcessString, okPressed = QInputDialog.getItem(self,"Pre-process Ca Data:", "T/F:", ['True', 'False'], 0, False)
        
        #transformString, okPressed = QInputDialog.getItem(self,"Pre-procesing:", "T/F:", ['True', 'False'], 0, False)
        
        
       # if preProcessString == 'True':
       #     preProcess = True
       #     print('Will pre-process CERNA data')
       # elif preProcessString == 'False':
       #     preProcess = False
       #     print('Will not pre-process CERNA data')
        preProcess = False
        DY.appendSessions(targetDir, self, preProcess)
        self.Flush()
        self.closeDB()
        
        
        
        
        
    @pyqtSlot()
    def archiveDeposit(self):
        DY.depositArchive()
    
    
    @pyqtSlot()       
    def openDB(self, path = None):
        if self.DB:
            self.DB.close()
        if path is None:
            result = QFileDialog.getOpenFileName(self, "Choose Database...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
            path = os.path.normpath(result[0])
        
        self.DBpath = path
        with open('last_file.txt', 'w') as F:
            F.write(path)
            F.close()
        self.DBidText.setText(str(self.DBpath))
        print(self.DBpath)
        self.curAnimal = ''
        self.curFOV = ''
        self.curDataStreams = []
        self.curTime = []
        self.DB = h5py.File(self.DBpath,'a', rdcc_nbytes=1000000000*10, rdcc_nslots=1000000)
        self.updateAnimalList()
        self.printRegistry()
    
    def openLast(self):
        with open('last_file.txt', 'r') as F:
            path = F.read()
        self.openDB(path=path)
            
    def printRegistry(self):
        if 'Registry' in self.DB['Animals'].keys():  ## Each animal should have list of deposited experiment folders
            byteRegistry = self.DB['Animals']['Registry'][...]
            registry =[]
            for el in byteRegistry:
                registry.append(el.decode('utf-8'))   
            print(registry)
        
    @pyqtSlot()
    def createDB(self):
        result = QFileDialog.getSaveFileName(self)
        self.DBpath = os.path.normpath(result[0])
        self.DB = h5py.File(self.DBpath,'a', rdcc_nbytes=1000000000, rdcc_nslots=1000000) 
        self.DB.require_group('Animals')
        
    @pyqtSlot()       
    def addAnimal(self):
        print(self.sender())
        newKey, okPressed = QInputDialog.getText(self,"Add Animal:", "Animal #:", QLineEdit.Normal, "")
        if okPressed: 
            if self.DB:
                self.DB['Animals'].require_group(newKey)
        self.updateAnimalList()   
        self.updateFOVlist()
        self.updateDataList()
        self.updateLayout()
        self.updateDisplay()
    
    @pyqtSlot()       
    def addFOV(self):
        if self.DB:
            FOVcount = 0
            for key in self.DB['Animals'][self.curAnimal].keys():
                FOVcount = FOVcount + 1
            newFOV = self.DB['Animals'][self.curAnimal].require_group(f"{FOVcount+1}")
            newFOV.require_group('T')   ### Time data
            newFOV.require_group('R')    ## ROI data (masks and traces)
        self.updateFOVlist()   
        
    
    @pyqtSlot()
    def addSignal(self):
        FOV = self.DB['Animals'][self.curAnimal][self.curFOV]
        FOV.require_group('S')    ## signal data (selected 1D signals)
        FOV['S'].require_group('Signal')
        FOV['S'].require_group('Time')
        for item in self.DataList.selectedItems():
            sigKey = item.text()
            if self.DB['Animals'][self.curAnimal][self.curFOV][sigKey].ndim == 1:
                sigLink = self.DB['Animals'][self.curAnimal][self.curFOV][sigKey]
                timeLink = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][sigKey]
              #  self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'].require_dataset([sigKey]) 
               # self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'].require_dataset([sigKey])
                self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'][sigKey] = sigLink
                self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'][sigKey] = timeLink
                
             #   self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'][sigKey] = self.DB['Animals'][self.curAnimal][self.curFOV][sigKey]
             #   self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'][sigKey] = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][sigKey]
                
        for item in self.ROIlist.selectedItems():
            sigKey = self.dataFocus + '_ROI_' + item.text() 
            sigLink = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['Traces'][item.text()]   
            timeLink = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
           # self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'].require_dataset([sigKey]) 
           #self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'].require_dataset([sigKey])
            self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'][sigKey] = sigLink
            self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'][sigKey] = timeLink
           
            #self.DB['Animals'][self.curAnimal][self.curFOV]['S']['
            #][sigKey] = self.DB['Animals'][self.curAnimal][self.curFOV][sigKey]
            #self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'][sigKey] = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
                
        
        self.updateSignalList()   
    
 
    
    @pyqtSlot()       
    def depositAnimal(self):
        return 
    
    @pyqtSlot()       
    def depositSessionToFOV(self):
        #DY.depositSessionToFOV(self, DY.selectFolder()) 
        DY.depositSessionToFOV(self,self.getDir()) 
    
    
    
    
    @pyqtSlot()       
    def depositTrial(self):
        DY.depositTrial(self, DY.selectFolder())
      
    @pyqtSlot()       
    def depositData(self):
        return 
    
    @pyqtSlot()       
    def updateActiveAnimal(self):
        try:
            self.reOpenDB()
        except:
           print('DB already open') 
        selectedItem = self.animalsList.selectedItems()
        if selectedItem:
            self.curAnimal = str(selectedItem[0].text())
            if not self.curAnimal in self.curROIs:
                self.curROIs[self.curAnimal] = {}
            self.updateFOVlist()
        else:
            self.FOVlist.clear()
                
    @pyqtSlot()       
    def updateActiveFOV(self):
        try:
            self.reOpenDB()
        except:
           print('DB already open') 
        selectedItem = self.FOVlist.selectedItems()
        if selectedItem:
            self.curFOV = str(selectedItem[0].text())
            if not self.curFOV in self.curROIs[self.curAnimal]:
                self.curROIs[self.curAnimal][self.curFOV] = {}
            self.updateDataList()
            for FOVitem in selectedItem:
                FOV = FOVitem.text()
                print(f'FOV is {FOV}')
                self.infoText.setText(f'FOV {FOV} is not flagged')
                for flag in self.DB['Animals'][self.curAnimal][FOV].attrs:
                    if self.DB['Animals'][self.curAnimal][FOV].attrs[flag]:
                        self.infoText.setText(f'FOV {FOV} is flagged as {flag}')
                        print(f'FOV {FOV} is flagged as {flag}')
        else:
            self.DataList.clear()
        #self.drawTimeNavigator()   
 #       activeFOV = self.DB['Animals'][self.curAnimal][self.curFOV]
  #      activeFOV.require_group('S')    ## signal data (selected 1D signals)
  #      activeFOV['S'].require_group('Signal')
  #      activeFOV['S'].require_group('Time')
        
    @pyqtSlot()       
    def updateActiveData(self):
        try:
            self.reOpenDB()
        except:
           print('DB already open') 
        self.curDataStreams = []
        selectedItem = self.FOVlist.selectedItems()
        if selectedItem:
            for item in self.DataList.selectedItems():
                self.curDataStreams.append(item.text())
                self.dataFocus = item.text()
                if not item.text() in self.curROIs[self.curAnimal][self.curFOV]:
                    self.curROIs[self.curAnimal][self.curFOV][item.text()] = []
            for dataItem in self.DataList.selectedItems():
                DATA = dataItem.text()
                self.infoText.setText(f'Data {DATA} is not flagged')
                for flag in self.DB['Animals'][self.curAnimal][self.curFOV][DATA].attrs:
                    if self.DB['Animals'][self.curAnimal][self.curFOV][DATA].attrs[flag]:
                        self.infoText.setText(f'Data {DATA} is flagged as {flag}')
                        print(f'Data {DATA} is flagged as {flag}')
        self.stop_playing_data()
        self.updateLayout()
        self.drawTimeNavigator()
        self.updateDisplay()
        self.updateROIlist()
        self.ROIlist.selectAll()
        self.updateROImask()
        #self.updateSignalList()
        
    
    @pyqtSlot()       
    def updateActiveSignals(self):
        return
        
    @pyqtSlot()       
    def updateAnimalList(self):
        self.animalsList.clear()
        for key in self.DB['Animals'].keys():
            if key != 'Registry':
                self.animalsList.addItem(key)
        self.updateActiveAnimal()
        
    @pyqtSlot()       
    def delAnimal(self):
        del self.DB['Animals'][self.curAnimal]
        self.updateAnimalList()
    
    @pyqtSlot()       
    def delFOV(self):
        
        del self.DB['Animals'][self.curAnimal][self.curFOV]
        self.updateFOVlist()
    
    @pyqtSlot()       
    def delData(self):
         for item in self.DataList.selectedItems():
            key = item.text()
            del self.DB['Animals'][self.curAnimal][self.curFOV][key]
            del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][key]
            del self.DB['Animals'][self.curAnimal][self.curFOV]['T'][key]
            
         self.updateDataList()
    
    @pyqtSlot()       
    def removeSignal(self):
        for item in self.sigList.selectedItems():
            del self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'][item.text()]
            del self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Time'][item.text()]
        self.updateSignalList()
    
    
    def clearLayout(self):
        self.displayLayout.clear()
        self.ROIlayout.clear()
        self.clearTranformROIs() # stopgap - need better manager for these trasform rois
        self.viewsList = {}
        self.displayList = {}
        self.histList = {}
        self.frozenList = {}
        self.oneDdisplayList = {}
        #self.ROIoverlay = pg.ImageItem(self)
        self.ROIoverlay = clickImage()
        self.dLevels = {}
        self.readers={}
        self.rasterImage = pg.ImageItem()
        self.ROIrasterView = self.ROIlayout.addViewBox(row = 0, col = 0)
        self.ROIrasterView.addItem(self.rasterImage)
        self.ROItracePlot = self.ROIlayout.addPlot(row=0, col = 1)
        
    @pyqtSlot()       
    def updateLayout(self):
        self.clearLayout()
        
        
        
        if len(self.DataList.selectedItems())==0:
           return

        
        

        for item in self.DataList.selectedItems():
            key = item.text()
            if key != 'T' and key != 'R'and key !='S':      
                
                if  len(self.DB['Animals'][self.curAnimal][self.curFOV][key].shape) <2 and not 'Link' in self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs:  #if 1d data
                    self.viewsList[key] = self.displayLayout.addViewBox(lockAspect = 1)
                    #self.oneDdisplayMode[key] = 'Both'

                    self.viewsList[key].invertY(True)
                    self.viewsList[key].autoRange(padding=None)
                    self.displayList[key] = pg.PlotItem()
                    self.displayList[key].hideAxis('left')
                    self.displayList[key].hideAxis('bottom')
                       
         
                else:
                    self.frozenList[key] = False
                    if 'Link' in self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs:
                        self.readers[key] = imageio.get_reader(self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs['Link'])
                        im = self.readers[key].get_data(0)
                        dataShape = im.shape
                    else:
                        dataShape = self.DB['Animals'][self.curAnimal][self.curFOV][key].shape
                    aspect = dataShape[-1]/dataShape[-2]
                    #if aspect<1:
                    #    self.transposeList[key] = False
                    #else:
                    #    self.transposeList[key] = False
                    self.viewsList[key] = self.displayLayout.addViewBox(lockAspect = aspect, invertY = True)
                    self.displayList[key] = pg.ImageItem(autoLevels = False)
                    self.histList[key] = pg.HistogramLUTItem(image = self.displayList[key])
                    self.displayLayout.addItem(self.histList[key])
                    
                self.viewsList[key].addItem(self.displayList[key])
        self.dataFocus = self.DataList.selectedItems()[0].text()
        self.viewsList[self.dataFocus].addItem(self.ROIoverlay)
        try:
            self.displayList[self.dataFocus].setBorder('r') 
        except:
            a=1
        try:
            self.viewsList[self.dataFocus].setBorder('r') 
        except:
            a=1
        self.updateROIlist()
        
        
    @pyqtSlot()       
    def stepDataFocus(self):
        keyList = []
        counter = 0
        curItemNum = 0
        for item in self.DataList.selectedItems():
            try:
                self.displayList[item.text()].setBorder('k')
            except:
                a=1
            try:
                self.viewsList[item.text()].setBorder('k')
            except:
                a=1
            keyList.append(item.text())
            if item.text() == self.dataFocus:
                curItemNum = counter
            counter = counter + 1
            
        if curItemNum < len(keyList)-1:
            newNum = curItemNum + 1
        else:
            newNum = 0
            
        self.viewsList[self.dataFocus].removeItem(self.ROIoverlay)    
        self.dataFocus = keyList[newNum]
        try:
            self.displayList[item.text()].setBorder('r')
        except:
            self.viewsList[item.text()].setBorder('r')    
        self.updateROIlist()

    @pyqtSlot()       
    def RstepDataFocus(self):
        keyList = []
        counter = 0
        curItemNum = 0
        for item in self.DataList.selectedItems():
            try:
                self.displayList[item.text()].setBorder('k')
            except:
                self.viewsList[item.text()].setBorder('k')
            keyList.append(item.text())
            if item.text() == self.dataFocus:
                curItemNum = counter
            counter = counter + 1
            
        if curItemNum == 0:
            newNum = len(keyList)-1
        else:
            newNum = curItemNum - 1
        
        self.viewsList[self.dataFocus].removeItem(self.ROIoverlay)
        self.dataFocus = keyList[newNum]
        try:
            self.displayList[item.text()].setBorder('r')
        except:
            self.viewsList[item.text()].setBorder('r')  
        self.updateROIlist()
                    
        
            
    
            
     
     
    @pyqtSlot()       
    def toggleHists(self):
        #self.showHists = not self.showHists
        if self.dataFocus in self.histList:
            if self.histList[self.dataFocus].isVisible():
                self.histList[self.dataFocus].hide()
            else:
                self.histList[self.dataFocus].show()
        
    @pyqtSlot()       
    def updateFOVlist(self):
        self.FOVlist.clear()
        
        for key in self.DB['Animals'][self.curAnimal].keys():
            if key != 'T' and key != 'R' and key != 'S' and key !='Neurons':
                item  = QListWidgetItem(key)
                self.FOVlist.addItem(item)
                color = Qt.black
                for attr in self.DB['Animals'][self.curAnimal][key].attrs:
                    if self.DB['Animals'][self.curAnimal][key].attrs[attr]==True:
                        color = self.FOVflags[attr]['color']
                item.setForeground(color)
    
    @pyqtSlot()       
    def updateDataList(self):
        self.DataList.clear()
        
        for key in self.DB['Animals'][self.curAnimal][self.curFOV].keys():
            if key != 'T' and key != 'R' and key != 'S' and key !='Neurons':
                item  = QListWidgetItem(key)
                self.DataList.addItem(item)
                color = Qt.black
                for attr in self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs:
                    if self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs[attr]==True:
                        color = self.DataFlags[attr]['color']
                item.setForeground(color)
                
    # @pyqtSlot()       
    # def updateDataListOriginal(self):
    #     self.DataList.clear()
    #     color = Qt.black
    #     for key in self.DB['Animals'][self.curAnimal][self.curFOV].keys():
    #         if key != 'T' and key != 'R' and key != 'S':
    #             if 'Process' in self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs:
    #                 if self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs['Process'][...]:
    #                     color = Qt.red
    #                 else:
    #                     color = Qt.black
    #             else:
    #                 color = Qt.black
    #                 self.DB['Animals'][self.curAnimal][self.curFOV][key].attrs['Process'] = False
    #             item = QListWidgetItem(key)
    #             self.DataList.addItem(item) 
    #             item.setForeground(color)
                
    @pyqtSlot()       
    def updateSignalList(self):
        return
        self.sigList.clear()
        for key in self.DB['Animals'][self.curAnimal][self.curFOV]['S']['Signal'].keys():
            self.sigList.addItem(key) 
                
        
             
    @pyqtSlot()       
    def updateDisplay(self):
        #self.curTime = self.timeSlider.value()
        self.curTime = self.timeLUT[self.timeSlider.value()]
        self.timeLabel.setText(time.ctime(self.curTime))
        curFrames = ''
        #print(self.curTime)
        #self.drawTimeNavigator()
        
        for item in self.DataList.selectedItems():
            #timescale =  self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].dims[0][0]
            dataname = item.text()
            dataDims = len(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].shape)
            #data_dtype = len(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].dtype)
            #print(data_dtype)
            
           
            
                
            if dataDims == 2:
                self.displayList[item.text()].setImage(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][...])
                self.sampleIndex[dataname] = 0
                self.dLevels[item.text()] = self.histList[item.text()].getLevels()
                curFrames = curFrames + str(self.sampleIndex[dataname]) + ' of ' + str(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].shape[0]) + '\r'
                continue
            
            timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][item.text()]
            self.sampleIndex[dataname] = np.searchsorted(timescale,self.curTime)
            curFrames = curFrames + str(self.sampleIndex[dataname]) + ' of ' + str(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].shape[0]) + '\r'
            
            
            if 'Link' in self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].attrs:
                
                im = self.readers[dataname].get_data(self.sampleIndex[dataname])
                self.displayList[item.text()].setImage(im)
                
                
            if self.sampleIndex[dataname] > len(timescale)-1  or np.abs(timescale[self.sampleIndex[dataname]]-self.curTime)>1:
                self.displayList[item.text()].clear()
                continue 
            
           
                
            if dataDims == 3 or dataDims == 4:
                if self.frozenList[item.text()]:
                    self.dLevels[item.text()] = self.histList[item.text()].getLevels()
                    
                    self.displayList[item.text()].setImage(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][self.sampleIndex[dataname],:,:], levels = [self.dLevels[item.text()][0],self.dLevels[item.text()][1]])
                
                
                else:
                    #data =  self.DB['Animals'][self.curAnimal][self.curFOV][item.text()]
                    
        
                    im = self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][self.sampleIndex[dataname],:,:]
                    
                    imMin=np.min(np.min(im))
                    imMax=np.max(np.max(im))
                    self.dLevels[item.text()] = [imMin, imMax];
                    self.displayList[item.text()].setImage(self.DB
                                                           ['Animals'][self.curAnimal][self.curFOV][item.text()][self.sampleIndex[dataname],:,:])#, levels = [imMin,imMax])
                    self.frozenList[item.text()] = True
                    self.histList[item.text()].setLevels(self.dLevels[item.text()][0], self.dLevels[item.text()][1])
                

                 
            
            elif dataDims == 1 and not 'Link' in self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].attrs:
                trace = self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][...]
                self.displayList[item.text()].clear()
                plotStart = np.max([self.sampleIndex[dataname]-300,0])                                 
                plotEnd   = np.min([plotStart+600,timescale.shape[0]])  
                if plotStart < 300:
                    X = self.sampleIndex[dataname]
                else:
                    X = 300
                if np.amax(trace) < 30:
                    marker_height = np.amax(trace)
                else:
                    marker_height = 200
               # self.displayList[item.text()].plot(timescale[plotStart:plotEnd], self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][plotStart:plotEnd])
                
                self.displayList[item.text()].plot(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][plotStart:plotEnd],pen='m')

              #  self.displayList[item.text()].plot([self.curTime,self.curTime],[0,30], pen = 'r') #Mark current time in plot
                
                self.displayList[item.text()].plot([X,X],[0,marker_height], pen = 'r') #Mark current time in plot
                
                #self.viewsList[item.text()].setXRange(plotStart,plotEnd, padding = 0)
                #self.viewsList[item.text()].setYRange(np.min(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][plotStart:plotEnd]),np.max(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][plotStart:plotEnd]), padding = 0)
                self.viewsList[item.text()].autoRange()
                self.displayList[item.text()].autoRange() 
        self.timeLine.setData(np.array([self.curTime,self.curTime]),np.array([-1000,1000]))
        self.timeLine2.setData(np.array([self.curTime,self.curTime]),np.array([self.timeLine2_min, self.timeLine2_max]))
        
        
        self.frameLabel.setText(curFrames)
        
    def display1d(self): 
        pass
        
        
    @pyqtSlot()      
    def exportSelected(self):
        item = self.expDataCombo.currentText()
        func = self.exportFuncs[item]
        func()
        self.expDataCombo.setCurrentIndex(0)
        
    @pyqtSlot()      
    def depositSelected(self):
        item = self.depositBox.currentText()
        func = self.depositMethods[item]
        func()
        self.depositBox.setCurrentIndex(0)  
        #self.closeDB()
        
    @pyqtSlot()      
    def dataOperationSelected(self):
        item = self.dataOperationBox.currentText()
        func = self.dataOperations[item]
        func()
        self.dataOperationBox.setCurrentIndex(0)  
        #self.closeDB()
        
    @pyqtSlot()      
    def expPlots(self):
        result = QFileDialog.getSaveFileName(self)
        saveName = os.path.normpath(result[0])
        #DY.exportPlots(self.XtraceData,self.YtraceData,saveName)
        traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        DY.exportPlots(traceArray,traceArray, saveName)
        
    @pyqtSlot()
    def expField(self, FOV = None, DATA = None):
        if FOV is None:
            FOV = self.curFOV
        if DATA is None:
            DATA = self.dataFocus
            
        dlevels = self.dLevels[self.dataFocus]
        
        label_ROIs = True
        
        if len(self.DB['Animals'][self.curAnimal][FOV][self.dataFocus].shape)==2:
            data = self.DB['Animals'][self.curAnimal][FOV][self.dataFocus][...]
            RGBA = np.array([])
        else:
            RGBA = self.updateROImask()
            timescale =  self.DB['Animals'][self.curAnimal][FOV]['T'][self.dataFocus]
            sampleIndex = np.searchsorted(timescale,self.curTime)
            data = self.DB['Animals'][self.curAnimal][FOV][self.dataFocus][sampleIndex,...]
        F = plt.figure(f'{self.curAnimal} {self.curFOV} {self.dataFocus}')
        A = F.add_axes([0, 0, 1, 1])
        A.imshow(data.T, aspect='auto', interpolation='none', cmap='gist_gray', vmin = dlevels[0], vmax = dlevels[1])
        A.xaxis.set_visible(False)
        A.yaxis.set_visible(False)
        A.set_frame_on(False)
        print(RGBA)
        print(RGBA.shape)
        print(RGBA.size)
        if RGBA != np.array([]):
            R = np.swapaxes(RGBA,0,1)
            A.imshow(R)
        if label_ROIs:
            if 'floatMask' in self.DB['Animals'][self.curAnimal][FOV]['R'][self.dataFocus].keys():
                num_ROIs = self.DB['Animals'][self.curAnimal][FOV]['R'][self.dataFocus]['floatMask'].shape[2]
                for Rnum in range(num_ROIs):  
                    Y, X =  DY.mask_centroid(self.DB['Animals'][self.curAnimal][FOV]['R'][self.dataFocus]['floatMask'][:,:,Rnum])
                    A.text(X,Y, str(Rnum), fontsize=16, color = 'w')
  
        plt.show()
        saveName = os.path.join(self.reportSaveDir, self.curAnimal+FOV+self.dataFocus + '.png')
        F.savefig(saveName, transparent = True)
        return(F)
    
    #def align_chronic_from_gui():
     #   DY.align_chronic_data(DBpath = self.DBpath)
        
    @pyqtSlot()
    def export_aligned_data(self):
        ### 
        DY.interface_for_export_to_multi_session(self, default_flags = True, Animal = None, FOV_list = None, FOV_flags = None, stim_flags = None)
        
    @pyqtSlot()
    def expHybridCorr(self, rect=None):
        dlevels = self.dLevels[self.dataFocus]

        timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
        sampleIndex = np.searchsorted(timescale,self.curTime)
        snap = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][sampleIndex,...]
        

        stack = self.getSubStack(rect=rect)
        
        nCor = DY.crossCorr(stack, self)[0,...]
        sCor = DY.corMap(stack, self)[0,...]
        sCorRGB = ind2RGB(sCor, plt.cm.cool, minV = -1, maxV = 1)
        nCorAlpha = normalizeImage(nCor)
        sCorAlpha = normalizeImage(np.absolute(sCor))
        RGBnA = np.swapaxes(np.concatenate((sCorRGB,nCorAlpha), axis = 2),0,1)
        RGBsA = np.swapaxes(np.concatenate((sCorRGB,sCorAlpha), axis = 2),0,1)

        F = plt.figure()
        A = F.add_axes([0, 0, 1, 1])
        A.imshow(snap.T, aspect='auto', interpolation='none', cmap='gist_gray', vmin = dlevels[0], vmax = dlevels[1])
        A.xaxis.set_visible(False)
        A.yaxis.set_visible(False)
        A.set_frame_on(False)
        plt.imshow(RGBnA)
        
        G = plt.figure()
        A = G.add_axes([0, 0, 1, 1])
        A.imshow(snap.T, aspect='auto', interpolation='none', cmap='gist_gray', vmin = dlevels[0], vmax = dlevels[1])
        A.xaxis.set_visible(False)
        A.yaxis.set_visible(False)
        A.set_frame_on(False)
        plt.imshow(RGBsA)
        
        
        
        
        
        
    
    
    @pyqtSlot()    
    def update1DdataDisplay(self):
        print('update1DdataDisplay')

        sourceCount = 0
        self.YtraceData = {}
        self.XtraceData = {}
        #self.ROIlayout.clear()
        for item in self.DataList.selectedItems():
            dataDims = len(self.DB['Animals'][self.curAnimal][self.curFOV][item.text()].shape)
            if dataDims ==1:                
                self.YtraceData[sourceCount] = self.DB['Animals'][self.curAnimal][self.curFOV][item.text()][:]
                self.XtraceData[sourceCount] = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][item.text()][:]
                dataPlot = pg.PlotItem()
                if self.plotMode == 'full':
                    print('full')
                    Xdata = self.XtraceData[sourceCount][:]
                    Ydata = self.YtraceData[sourceCount][:]  
                    XRANGE = [self.dispTime]
                    pSymbol = 'o'
                    pPen = None
                elif self.plotMode == 'compressed':
                    print('compressed')
                    Xdata = np.arange(0,self.XtraceData[sourceCount].shape[0],1) # convert time scale to list matching number of datapoints
                    Ydata = self.YtraceData[sourceCount][:]                    
                    XRANGE = [0,len(Xdata)]
                    pSymbol = None
                    pPen = 'b'
                dataPlot.plot(Xdata,Ydata , pen = pPen, symbol = pSymbol, symbolSize = 1)
                dataPlot.setXRange(XRANGE[0],XRANGE[1], padding = 0)
                dataPlot.setYRange(np.min(Ydata), np.max(Ydata), padding = 0)
                self.ROIlayout.addItem(dataPlot, col = 0, row = sourceCount)
                dataPlot.hideAxis('bottom')
                if self.exportPlots:
                    a=1
                sourceCount = sourceCount + 1
            for ROIlist in self.curROIs[self.curAnimal][self.curFOV][item.text()]:
                #for ROI in ROIlist:
                ROI = ROIlist
                self.YtraceData[sourceCount] = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][item.text()]['Traces'][ROI][:]
                self.XtraceData[sourceCount] = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][item.text()][:]
                dataPlot = pg.PlotItem()
                if self.plotMode == 'full':
                    Xdata = self.XtraceData[sourceCount][:]
                    Ydata = self.YtraceData[sourceCount][:]
                    XRANGE = [self.dispTime[0], self.dispTime[1]]
                    pSymbol = 'o'
                    pPen = None
                elif self.plotMode == 'compressed':
                    Xdata = np.arange(0,self.YtraceData[sourceCount].shape[0],1)
                    Ydata = self.YtraceData[sourceCount][:]                     #TODO: decimate time and data
                    XRANGE = [0,len(Xdata)]
                    pSymbol = None
                    pPen = 'r'
                dataPlot.plot(Xdata,Ydata , pen = pPen, symbol = pSymbol, symbolSize = 1)
                dataPlot.setXRange(XRANGE[0],XRANGE[1], padding = 0)
                self.ROIlayout.addItem(dataPlot, col = 0, row = sourceCount)
                dataPlot.hideAxis('bottom')
                sourceCount = sourceCount + 1
        
        
  
        
        
    def drawTimeNavigator(self):
        # for each datastream get sample collection times, plot for selection
        
        self.timePlotItem.clear()
        
        #print('# selected:' + str(len(self.DataList.selectedItems())))
        if len(self.DataList.selectedItems()) < 1:
            return
        
        
        TIME = np.array([])
        num = np.array([])
        counter = 0
       # for d in self.DB['Animals'][self.curAnimal][self.curFOV]['T'].keys():   
       #    thisTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][d][:]
       #    time = np.append(time, thisTime, 0)
       #    num = np.append(num, np.ones(thisTime.shape)*counter, 0)
       #    counter = counter + 1
       
       # self.timePlotItem.plot(time,num, pen = None, symbol = 'o')
        
        for d in self.curDataStreams:   
            addTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][d][...]
            #thisTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][d][:]
            print(f'TIME shape: {TIME.shape}')
            print(f'addTime shape {addTime.shape}')
            
            if np.ndim(addTime) == 0:
                addTime = np.expand_dims(addTime,0)
            TIME = np.concatenate([TIME, addTime], axis=0)
            
            num = np.ones(addTime.shape)*counter
            symPen = 'b'
            if d == self.dataFocus:
                symPen = 'r'
            self.timePlotItem.plot(addTime, num , pen = None, symbol = 'o', symbolPen = symPen)
            counter = counter + 1
       
        
       
        self.timePlotItem.setXRange(np.min(TIME), np.max(TIME), padding = 0)
        self.timePlotItem.setYRange(-1, counter, padding = 0)
        self.curTime = TIME[0]
        
        self.timeLine = self.timePlotItem.plot(np.array([self.curTime,self.curTime]),np.array([-1000,1000]), pen='r')
        self.timeLine2 = self.ROItracePlot.plot(np.array([self.curTime,self.curTime]),np.array([self.timeLine2_min, self.timeLine2_max]), pen='r')
    
        
        self.timeROI = pg.ROI([np.amin(TIME),-1],[np.amax(TIME)-np.amin(TIME),counter], pen = 'y', )
        self.timeROI.addScaleHandle([0,0.5], [1,0.5])
        self.timeROI.addScaleHandle([1,0.5], [0,0.5])
        self.timeROI.sigRegionChangeFinished.connect(self.updateSliderBounds)
        self.timePlotItem.addItem(self.timeROI)
        self.updateSliderBounds()

    def updateTimeNavigator(self):
        self.timeLine.clear()
        self.timeLine2.clear()
        
        
        
        self.timeLine = self.timePlotItem.plot(np.array([self.curTime,self.curTime]),np.array([-1000,1000]), pen='r')
        
        
        raster = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        self.timeLine2_min = (np.amin(raster))
        self.timeLine2_max = (np.amax(raster))
        self.timeLine2 = self.ROItracePlot.plot(np.array([self.curTime,self.curTime]),np.array([self.timeLine2_min, self.timeLine2_max]), pen='r')
    
    
      
    
        
        
    def getSubStack(self, rect = None, FOV=None, datakey = None, Animal = None, getROIs = False):
        return(gs(self, rect = rect, FOV=FOV, datakey = datakey, Animal = Animal, getROIs = getROIs))
    # def getSubStack(self, rect = None):
    #     if len(self.transformROIList)>0:
    #         ROI = self.transformROIlist[-1]
    #         rect = ROI.parentBounds()
    #     if rect is None:
    #         timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
    #         Istart = np.searchsorted(timescale,self.timeLUT[0])
    #         Iend = np.searchsorted(timescale,self.timeLUT[-1])
    #         print('Retrieving data...')
    #         return(self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][Istart:Iend,...])
    #     else:
    #         top = round(rect.y())
    #         bot = round(rect.y()+rect.height())
    #         left = round(rect.x())
    #         right = round(rect.x()+rect.width())
    #         return(self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][Istart:Iend,left:right,top:bot,...])
    
    @pyqtSlot()
    def doTransform2(self):
        start = time.time()
        Tkey  = self.transformBox.currentText()
        if Tkey == 'Data transformations...':
            return
        print(Tkey)
        #print(self.transformDict[Tkey])
        

        DATA, TIME = self.transformDict[Tkey](self)
        if DATA is None:
            self.updateDataList()
            self.transformBox.setCurrentIndex(0)
            print(f'Transform took {time.time()-start} seconds')
            print('Locking database...')
            self.closeDB()
            return
        
        
        
        print(f'Data shape: {DATA.shape}, Time shape: {TIME.shape}')
        timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
        Istart = np.searchsorted(timescale,self.timeLUT[0])
        Iend = np.searchsorted(timescale,self.timeLUT[-1])
        
        dName = self.dataFocus + self.suffixDict[Tkey] + self.suffixDict[Tkey] + '_' + str(Istart) + '_' + str(Iend)
        DY.genericDepositTrial(self, DATA, TIME, dName)  
        
        self.updateDataList()

        self.transformBox.setCurrentIndex(0)
        print(f'Transform took {time.time()-start} seconds')
        print('Locking database...')
        self.closeDB()
        
        
    @pyqtSlot()
    def doTransform3(self):
        global_start = time.time()
        Tkey  = self.transformBox.currentText()
        if Tkey == 'Data transformations...':
            return
        print(Tkey)
        for item in self.DataList.selectedItems():
            sub_start = time.time()
            dataset = item.text()
            self.dataFocus = dataset
            DATA, TIME = self.transformDict[Tkey](self)
            if DATA is None:
                pass
            elif DATA == 'STOP':
                break
            else:
                print(f'Data shape: {DATA.shape}, Time shape: {TIME.shape}')
                
                timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
                if len(timescale.shape)>0:
                    Istart = np.searchsorted(timescale,self.timeLUT[0])
                    Iend = np.searchsorted(timescale,self.timeLUT[-1])                   
                    dName = self.dataFocus + self.suffixDict[Tkey] + '_' + str(Istart) + '_' + str(Iend)
                else:
                    dName = self.dataFocus + self.suffixDict[Tkey] + '_' + str(timescale)
                DY.genericDepositTrial(self, DATA, TIME, dName)  
                print(f'Data {dataset} transformed in {time.time()-sub_start} seconds')
                
        self.updateDataList()
        self.transformBox.setCurrentIndex(0) 
        print(f'Transforms took {time.time()-global_start} seconds')
        print('Locking database...')
            
        self.closeDB()
    
    @pyqtSlot()
    def drawTransformROI(self):
        stack = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus]
        self.transformROIlist.append(pg.ROI([0,0],[stack.shape[1],stack.shape[2]], pen = 'y', removable=True))
        self.transformROIlist[-1].sigRegionChanged.connect(self.transformROIcallBack)
        self.transformROIlist[-1].addScaleHandle([0,0],[0.5,0.5])
        self.viewsList[self.dataFocus].addItem(self.transformROIlist[-1])    
                                
    @pyqtSlot()
    def transformROIcallBack(self):
        return
        rect = self.transformROIlist[-1].parentBounds()
        print(rect.left())
        
    @pyqtSlot()
    def clearTranformROIs(self):
        for ROI in self.transformROIlist:
            del(ROI)
        self.transformROIlist=[]
        
    
    
        
    # @pyqtSlot()
    # def toggleProcess(self):
    #     if 'Process' in self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].attrs:
    #         self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].attrs['Process'] = ~self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].attrs['Process']
    #     else:
    #         self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].attrs['Process'] = False
    #     print(self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].attrs['Process'])
    #     self.updateDataList()
    
    
    @pyqtSlot()
    def setDataFlag(self):
        flags =  self.DataFlags.keys()
        selectedFlag, okpressed = QInputDialog.getItem(self,"Select flag:", "Flag", flags, 0, False)
        print(f'selectedFlag: {selectedFlag}')
        if okpressed != True:
            return
            
        for item in self.DataList.selectedItems():
            data = item.text()
            print(data)
            for flag in flags:
                self.DB['Animals'][self.curAnimal][self.curFOV][data].attrs[flag] = flag == selectedFlag
                print(f'Setting {flag} to {flag==selectedFlag}')
        self.updateDataList()
        
        
    @pyqtSlot()
    def setFOVflag(self):
        flags =  self.FOVflags.keys()
        selectedFlag, okpressed = QInputDialog.getItem(self,"Select flag:", "Flag", flags, 0, False)
        print(f'selectedFlag: {selectedFlag}')
        if okpressed != True:
            return
            
        for item in self.FOVlist.selectedItems():
            FOV = item.text()
            print(FOV)
            for flag in flags:
                self.DB['Animals'][self.curAnimal][FOV].attrs[flag] = flag == selectedFlag
                print(f'Setting {flag} to {flag==selectedFlag}')
        self.updateFOVlist()
                    
    def show_alignment_gui(self):
        
        multi_sessions = []
        
        if 'multi session files' in self.DB['Animals'][self.curAnimal].attrs:
            for file_name in self.DB['Animals'][self.curAnimal].attrs['multi session files']:
                multi_sessions.append(file_name)
        else:
            print('No multi session objects stored for this animal')
            return
        if len(multi_sessions) == 0:
            print('No multi session objects stored for this animal')
            return
        multi_session_path, okPressed = QInputDialog.getItem(self,"Select multi session:", "File:", multi_sessions, 0, False)
        if okPressed == False:
            return
        
        input_data = {}
        input_data['pickle file path'] = multi_session_path
        self.alignment_gui = Alignment_GUI(input_data)
        self.alignment_gui.show()
        
    
 
    
    def closeEvent(self, Event):
        if self.DB:
            self.DB.close()
            print('Closing DB...')
        print('Bye')
            
    def closeDB(self):
       self.DB.close()
       self.infoText.setText(f'DB {self.DBpath} is now closed, select "Reopen DB" to edit')
       self.lockDBbtn.clicked.disconnect(self.closeDB)
       self.lockDBbtn.clicked.connect(self.reOpenDB)
       self.lockDBbtn.setText('Unlock DB')
       print(f'{self.DBpath}')
       
      
    def reOpenDB(self):
        start = time.time()
        self.DB = h5py.File(self.DBpath,'a', rdcc_nbytes=1000000000, rdcc_nslots=1000000)
        #self.infoText.setText(f'DB {self.DBpath} is now open, opening took {time.time()-start} seconds')
        self.lockDBbtn.clicked.disconnect(self.reOpenDB)
        self.lockDBbtn.clicked.connect(self.closeDB)
        self.lockDBbtn.setText('Lock DB')


    
        
        
    @pyqtSlot()
    def expData(self):
        saveParent = DY.selectFolder()
        for dataKey in self.curDataStreams:
            pPath = os.path.normpath(f'{saveParent}/{self.curAnimal}/{self.curFOV}')
            os.makedirs(pPath, exist_ok=True)
            
            if len(self.DB['Animals'][self.curAnimal][self.curFOV][dataKey].shape)==2:
                data = self.DB['Animals'][self.curAnimal][self.curFOV][dataKey]
                path = os.path.normpath(f'{saveParent}/{self.curAnimal}/{self.curFOV}/{dataKey}.tif')
                imageio.imwrite(path, data, bigtiff = True)
            else:
                timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][dataKey]
                Ibegin = np.searchsorted(timescale,self.dispTime[0])
                Iend = np.searchsorted(timescale,self.dispTime[1])
                data = self.DB['Animals'][self.curAnimal][self.curFOV][dataKey][Ibegin:Iend,...]
                path = os.path.normpath(f'{saveParent}/{self.curAnimal}/{self.curFOV}/{dataKey}{Ibegin}_{Iend}.tif')
                if len(data.shape) >2:
                    imageio.volwrite(path, data, bigtiff = True)
                else:
                    path = os.path.normpath(f'{saveParent}/{self.curAnimal}/{self.curFOV}/{dataKey}{Ibegin}_{Iend}.txt')
                    np.savetxt(path,data)
            #Time = timescale[Ibegin:Iend]
            #tpath = os.path.normpath(f'{saveParent}/{self.curAnimal}/{self.curFOV}/{dataKey}_time_{Ibegin}_{Iend}.txt')
    

    @pyqtSlot()
    def stitchAndSplit(self):
        print('stitch and split')
        stacks = []
        TIMES = []
        original_names = []
        for item in self.DataList.selectedItems():
            datakey = item.text()
            original_names.append(datakey)
            stacks.append(self.DB['Animals'][self.curAnimal][self.curFOV][datakey][...])
            TIMES.append(self.DB['Animals'][self.curAnimal][self.curFOV]['T'][datakey][...])
        transformedStacks, tfs, glob_transform, transformedROIstacks = DY.stitch.Stitch(stacks, split_output=True)
        for tStack, TIME, oName in zip(transformedStacks, TIMES, original_names):
            dName = oName + '_stitched'
            print(dName)
            DY.genericDepositTrial(self, tStack, TIME, dName)
        self.updateDataList()
    
    
        
    @pyqtSlot()
    def alignROIs(self):
        ROIs = []
        traceArrays = []
        for item in self.DataList.selectedItems():
            datakey = item.text()
            masks = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datakey]['floatMask'][...]
            traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datakey]['traceArray'][...]
            ROIs.append(masks)
            traceArrays.append(traceArray)
            DY.alignROIs(ROIs, traceArrays)
    
    @pyqtSlot()
    def alignROIsCaiman(self): 
        ROIs = []
        templates = []
        for item in self.DataList.selectedItems():
            datakey = item.text()
            masks = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datakey]['floatMask'][...]
            #traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datakey]['traceArray'][...]
            template = np.mean(self.DB['Animals'][self.curAnimal][self.curFOV][datakey][...], axis =0)
            ROIs.append(masks)
            templates.append(template)
            
        DY.alignROIsCAIMAN(ROIs, templates)
            
        
        
    def expROImap(self):
        result = QFileDialog.getSaveFileName(self)
        filename = os.path.normpath(result[0])
        data = self.segmentationResult[self.dataFocus]
        imageio.mimwrite(filename, data)

    @pyqtSlot()
    def expMovie(self):
        path = DY.selectFolder()
        sgrab = self.displayLayout.grab()
        aviPath = os.path.normpath(os.path.join(path,'export.avi'))
        tifPath = os.path.normpath(os.path.join(path,'movie.tif'))
        tempPath = os.path.normpath(os.path.join(path,'temp.tif'))
        #V = cv2.VideoWriter(aviPath,cv2.VideoWriter_fourcc('M','J','P','G' ),25,(512, 1024))
        #V.open()
        try:
            self.displayList[self.dataFocus].setBorder('k')
        except:
            self.viewsList[self.dataFocus].setBorder('k')
        writer = imageio.get_writer(aviPath)
        output = np.zeros([self.timeLUT.shape[0], sgrab.height(), sgrab.width(), 4])
        for t in range(0, self.timeLUT.shape[0]):
            self.timeSlider.setValue(t)
            sgrab = self.displayLayout.grab()

            
            sgrab.save(tempPath)
            img = imageio.imread(tempPath)
            writer.append_data(img[:,:,:])
            print(f'Frame {t} of {self.timeLUT.shape[0]} written to disk')
            
        
        writer.close()
        os.remove(tempPath)
        print('Movie exported')
        try:
            self.displayList[self.dataFocus].setBorder('r')
        except:
            self.viewsList[self.dataFocus].setBorder('r')
             
    @pyqtSlot()      
    def updateSliderBounds(self):
        self.timeLUT = []
        allTimes = []
        for d in self.DB['Animals'][self.curAnimal][self.curFOV]['T'].keys():  
            #allTimes = list(allTimes) + list(self.DB['Animals'][self.curAnimal][self.curFOV]['T'][d][:])
            #print(f'allTimes: {allTimes}')
            toAdd = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][d][...]
            if toAdd.size == 1:
                #print(f'toAdd: {toAdd}, size is 1')
                allTimes.append(toAdd)  
            else:
                #print(f'toAdd shape: {toAdd.shape}')
                allTimes = allTimes + list(toAdd)
            #print(f'Key added: {d}')
            #print(f'allTimes length: {len(allTimes)}')
        
        Tnum = np.array(allTimes)
        Thig = Tnum[Tnum>self.timeROI.parentBounds().left()]
        Tlow = Thig[Thig<self.timeROI.parentBounds().right()]
        Tbig = Tlow/self.timeStep
        Trou = np.round(Tbig)
        Tcor = Trou*self.timeStep
        Tset = set(Tcor)
        Tlist = list(Tset)
        Tsort = sorted(Tlist)
     
        self.timeLUT = np.array(Tsort)
        self.timeSlider.setMinimum(0)
        self.timeSlider.setMaximum(len(self.timeLUT)-1)
        self.dispTime[0] = self.timeLUT[0]
        self.dispTime[1] = self.timeLUT[-1]
        #self.updateDisplay()
        #self.update1DdataDisplay()
        
    @pyqtSlot()      
    def updateResSlider(self):
        self.timeStep = self.TresLUT[self.TresSlider.value()]
        self.TresLabel.setText(f'{np.around(self.timeStep,2)} s')
                 
    
   
         


    
   
    
    @pyqtSlot()      
    def drawROI(self):    
        if hasattr(self, 'newROI'):
            self.viewsList[self.dataFocus].removeItem(self.newROI)
        self.newROI = pg.EllipseROI([50,50], [40,32], removable=True)
        #self.newROI.sigRegionChanged.connect(self.spotlightROI)
        self.newROI.sigRemoveRequested.connect(self.addROImanually)
        self.viewsList[self.dataFocus].addItem(self.newROI)
    
    
   
    
    
    
    @pyqtSlot()      
    def addROImanually(self): 
        
        newROIname = str(self.ROIlist.count()+1)
        imshape = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape
        print(f'{imshape=}')
        if 'Link' in self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].attrs:
            im = self.readers[self.dataFocus].get_data(0)
            imshape = im.shape
            maskTemplate = np.ones([imshape[0],imshape[1]])
            emptyTemplate = np.zeros([imshape[0],imshape[1]])
            imshape = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape
        elif len(imshape) == 3:
            maskTemplate = np.ones([imshape[-2],imshape[-1]])
            emptyTemplate = np.zeros([imshape[-2],imshape[-1]])
        elif len(imshape) == 4:
            maskTemplate = np.ones([imshape[1],imshape[2]])
            emptyTemplate = np.zeros([imshape[1],imshape[2]])
        elif len(imshape) == 2:
            maskTemplate = np.ones([imshape[0],imshape[1]])
            emptyTemplate = np.zeros([imshape[0],imshape[1]])
        mask = self.newROI.getArrayRegion(maskTemplate, self.displayList[self.dataFocus])
   
        
        left = round(self.newROI.pos()[0])
        top =  round(self.newROI.pos()[1])
        right = left + mask.shape[0]
        bot   = top + mask.shape[1]
        
       
        
        emptyTemplate[left:right,top:bot] = mask
  

        floatMask = np.array(emptyTemplate > 0)
        floatMask = floatMask.astype(np.float64)
        floatMask = np.expand_dims(floatMask, 2)
        trace = np.zeros(imshape[0])
        traces = np.expand_dims(trace, 0)
        print(f'Traces shape:{traces.shape}')
        if 'floatMask' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            existingFloatMask  = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...]
            existingTraceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
            print(f'Existing: {existingTraceArray.shape}')
            masks = np.concatenate((existingFloatMask, floatMask), axis = -1)
            traces = np.concatenate((existingTraceArray, traces), axis = 0)
        else:
            masks = floatMask
                                   
        self.updateROIdata(masks, traces)
        self.viewsList[self.dataFocus].removeItem(self.newROI)
  

        
        
        
    
        
   
    @pyqtSlot()     
    def getMask(self): ##Converts ROI selection to binary mask
        
        imshape = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape
        maskTemplate = np.ones([imshape[-2],imshape[-1]])
        emptyTemplate = np.zeros([imshape[-2],imshape[-1]])
        mask = self.newROI.getArrayRegion(maskTemplate, self.displayList[self.dataFocus])
        left = round(self.newROI.pos()[0])
        top =  round(self.newROI.pos()[1])
        right = left + mask.shape[1]
        bot   = top + mask.shape[0]
        emptyTemplate[left:right,top:bot] = mask
        #maskSum = np.sum(emptyTemplate, axis = (0,1))
        return(emptyTemplate)
        

    def Flush(self):
        start = time.time()
        print('Flushing...')
        self.DB.flush()
        print(f'Flushing took {time.time()-start} seconds')
    
    def moveToNewFOV(self):
        
        FOVs =  self.DB['Animals'][self.curAnimal].keys()
        destFOV, okPressed = QInputDialog.getItem(self,"Select destination:", "FOV:", FOVs, 0, False)
        if okPressed == False:
            return
        if destFOV == self.curFOV:
            return
        for item in self.DataList.selectedItems():
            key = item.text()
            
            startTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][key][0]
            t=time.localtime(startTime)
            new_key = f'{key} {t[0]}-{t[1]}-{t[2]}'
            self.DB['Animals'][self.curAnimal][destFOV][new_key] = self.DB['Animals'][self.curAnimal][self.curFOV][key]
            self.DB['Animals'][self.curAnimal][destFOV]['T'][new_key] = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][key]
            self.DB['Animals'][self.curAnimal][destFOV]['R'][new_key] = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][key]
        
    def mergeFOVs(self):
        destFOV, okPressed = QInputDialog.getText(self,"Name of merged FOV name:", "New name:", QLineEdit.Normal, "Merged FOVs")
        if okPressed:
            print(f'Creating merged FOV {destFOV}')
            self.DB['Animals'][self.curAnimal].require_group(destFOV)
            self.DB['Animals'][self.curAnimal][destFOV].require_group('T')
            self.DB['Animals'][self.curAnimal][destFOV].require_group('R')
        else:
            return
        
        FOVs = []
        for item in self.FOVlist.selectedItems():
            FOVs.append(item.text())
        
        for FOV in FOVs:
            for datakey in self.DB['Animals'][self.curAnimal][FOV].keys():
                if len(datakey)>1: ## Don't include keys like 'R', 'T', etc.
                
                    startT =  self.DB['Animals'][self.curAnimal][FOV]['T'][datakey][0]
                    dname = datakey + time.strftime('%d-%m-%y %H:%M', time.localtime(startT)) #tag data with start date to differentiate
                    print(f'Adding data {dname}')
                    self.DB['Animals'][self.curAnimal][destFOV][dname] = self.DB['Animals'][self.curAnimal][FOV][datakey]
                    self.DB['Animals'][self.curAnimal][destFOV]['T'][dname] = self.DB['Animals'][self.curAnimal][FOV]['T'][datakey]
                    self.DB['Animals'][self.curAnimal][destFOV]['R'][dname] = self.DB['Animals'][self.curAnimal][FOV]['R'][datakey]
                
    
    
  
    def show_multi_session(self):
        multi_sessions = []
        
        
        if 'multi session files' in self.DB['Animals'][self.curAnimal].attrs:
            for file_name in self.DB['Animals'][self.curAnimal].attrs['multi session files']:
                multi_sessions.append(file_name)
        else:
            print('No multi session objects stored for this animal')
            return
        if len(multi_sessions) == 0:
            print('No multi session objects stored for this animal')
            return
        multi_session_path, okPressed = QInputDialog.getItem(self,"Select multi session:", "File:", multi_sessions, 0, False)
        if okPressed == False:
            return
        multi_session = DY.unpickle(multi_session_path)
        for session in multi_session.sessions:
            session.show_raster()

    def show_single_session(self):

         if 'session file' in self.DB['Animals'][self.curAnimal][self.curFOV].attrs:
             session_path = self.DB['Animals'][self.curAnimal][self.curFOV].attrs['session file']
         else:
             print('No  session objects stored for this FOV')
             return

         session = DY.unpickle(session_path)
         session.show_raster()      
        
        
    @pyqtSlot()  
    def importFOVs(self):
        result = QFileDialog.getOpenFileName(self, "Choose Database...", "/lab-share/Neuro-Woolf-e2/Public/DavidY/CERNA data")
        source = os.path.normpath(result[0])
        print(f'Source: {source}')
        sDB = h5py.File(source,'a')
        Animals = sDB['Animals'].keys()
        Animal, okPressed = QInputDialog.getItem(self,"Select source animal:", "Animal:", Animals, 0, False)
        if okPressed == False:
            return
        allSourceFOVs = sDB['Animals'][Animal].keys()
        FOV, okPressed = QInputDialog.getItem(self,"Select FOVs to import:", "FOVs:", allSourceFOVs, 0, False)
        if okPressed == False:
            return
        print(f'Importing FOV {FOV} from DB {sDB}')
        sourceGroup = sDB['Animals'][Animal][FOV]
        destGroup = self.DB['Animals'][self.curAnimal]
        print(f'Source group: {sourceGroup}')
        print(f'Dest group: {destGroup}')
        sDB.copy(sourceGroup, destGroup, name = FOV)
        print('Done')
        sDB.close()
        self.updateActiveAnimal()
        

    @pyqtSlot()  
    def renameFOV(self):
        newKey, okPressed = QInputDialog.getText(self,"Change FOV name:", "New name:", QLineEdit.Normal, f"{self.curFOV}")
        if okPressed and newKey != '' and newKey !=self.curFOV:
            self.DB['Animals'][self.curAnimal][newKey] = self.DB['Animals'][self.curAnimal][self.curFOV]
            del(self.DB['Animals'][self.curAnimal][self.curFOV])
            self.curFOV = newKey
            self.updateFOVlist()
            
    @pyqtSlot() 
    def manualDepositData(self):
        sessionFolder = DY.selectFolder()
        DY.depositSessionToFOV(self, sessionFolder, select_data=True)
        self.updateDataList()

       
    @pyqtSlot()  
    def renameData(self):
        oldKey = self.dataFocus
        newKey, okPressed = QInputDialog.getText(self,"Change Data name:", "New name:", QLineEdit.Normal, f"{self.dataFocus}")
        if okPressed and newKey != '' and newKey !=self.curFOV:
            
            #change name of main data
            self.DB['Animals'][self.curAnimal][self.curFOV][newKey] = self.DB['Animals'][self.curAnimal][self.curFOV][oldKey]
            del(self.DB['Animals'][self.curAnimal][self.curFOV][oldKey])
            
            #change tname of Time data
            self.DB['Animals'][self.curAnimal][self.curFOV]['T'][newKey] = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][oldKey]
            del(self.DB['Animals'][self.curAnimal][self.curFOV]['T'][oldKey])
            
            #change name of ROI data
            self.DB['Animals'][self.curAnimal][self.curFOV]['R'][newKey] = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][oldKey]
            del(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][oldKey])
            
            
            self.dataFocus = newKey
            self.updateDataList()
            
            
    @pyqtSlot() 
    def splitTrials(self):
        T = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][...]
        separation_gap = 10 ## separate wherever there is 10s gap or more between data points
        stopIXs = np.where(np.diff(T)>separation_gap)[0]
        startIXs = stopIXs+1
        startIXs = np.insert(startIXs, 0, 0)
        stopIXs = np.append(stopIXs, T.shape[0]-1)
        
        print(f'Starts: {startIXs}')
        print(f'Stops: {stopIXs}')
        for c, start in enumerate(startIXs):
            DATA = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][start:stopIXs[c],...]
            TIME = T[start:stopIXs[c]]
            dName = self.dataFocus + '_' + str(c)
            DY.genericDepositTrial(self, DATA, TIME, dName)   
        self.updateDataList()
        self.closeDB()
            
    @pyqtSlot()  
    def concatenateData(self):
        #Collect selected datastreams, check if non-time dimensions are consistent, sort by time, concatenate, create new stream, timebase, roiset, signalset
        startList = []
        keyList = []
        dimList = []
        #numSamples = 0
        newName = ''
        for item in self.DataList.selectedItems(): #check selected data for consistency, get dimensions
            key = item.text()
            newName = newName+key
            startTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][key][0]
            startList.append(startTime)
            keyList.append(key)
           
            dimList.append(self.DB['Animals'][self.curAnimal][self.curFOV][key].shape[1:]) ##get non-time dimensions
            if dimList[-1] != dimList[0]:
                print('Inconsistent dimensnions, cancelling concatenation')
                return
            #pdb.set_trace()
            #numSamples = numSamples + dimList[-1][0]
        
        #sort by time
        zippedList = zip(startList, keyList)
        sortedZipped = sorted(zippedList)
        sortedKeys = [element for _, element in sortedZipped]
        
        counter = 0
        for key in sortedKeys:
            if counter == 0:
                
                newData = self.DB['Animals'][self.curAnimal][self.curFOV][key][:]
                timeOut = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][key][:]
            else:
                newData = np.concatenate((newData, self.DB['Animals'][self.curAnimal][self.curFOV][key][:]))
                timeOut = np.concatenate((timeOut, self.DB['Animals'][self.curAnimal][self.curFOV]['T'][key][:]))
            counter = counter+1
        
                     
        self.DB['Animals'][self.curAnimal][self.curFOV].create_dataset(newName, data = newData, track_order = True) 
        self.DB['Animals'][self.curAnimal][self.curFOV]['T'].create_dataset(newName,  data = timeOut, dtype = timeOut.dtype, shape = timeOut.shape, maxshape = (None,) )
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'].require_group(newName)
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][newName].require_group('Masks')
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][newName].require_group('Traces')
        print('Finished concatenating')
        self.updateDataList()
    
    @pyqtSlot() 
    def updateSegmentationMethod(self):
        self.currentSegmentationMethod = self.segmentaTionMethodBox.currentText()
        self.segmentParamBox.clear()
        for key in self.segmentationMethods[self.currentSegmentationMethod ]['Params']:
            self.segmentParamBox.addItem(key)
        self.segmentParamBox.setCurrentIndex(0)
        #self.updateSegParamValue()
        
                          
    @pyqtSlot()    
    def updateSegmentationParamSelection(self): #update slider and param textbox to refulect value of selected parameter when combo box changes value
        print('updateSegmentationParamSelection')
        selectedParam = self.segmentParamBox.currentText()
        print(f'selected param: {selectedParam}')
        paramValues = self.segmentationMethods[self.currentSegmentationMethod]['Params'][self.segmentParamBox.currentText()][:]
        print(f'paramValues: {paramValues}')                                         
        self.updateSegmentationSlider.setMinimum(paramValues[0])
        self.updateSegmentationSlider.setMaximum(paramValues[2])
        self.updateSegmentationSlider.setSingleStep(1)
        self.updateSegmentationSlider.setPageStep(10)
        #self.updateSegmentationSlider.valueChanged.disconnect(self.updateSegmentationParamSelection)
        self.updateSegmentationSlider.setValue(paramValues[1])
        #self.updateSegmentationSlider.valueChanged.connect(self.updateSegmentationParamSelection)
        
    @pyqtSlot()
    def updateSegParamValue(self): #triggered when segmentation parameter slider changes value
        paramValues = self.segmentationMethods[self.currentSegmentationMethod]['Params'][self.segmentParamBox.currentText()][:]
        paramValues[1] = self.updateSegmentationSlider.value()
        self.segmentationMethods[self.currentSegmentationMethod]['Params'][self.segmentParamBox.currentText()][:] = paramValues
        self.segParamValText.setText(str(paramValues[1]))
        
        
    @pyqtSlot()
    def adaptiveThreshold(self, img = None, appendMode = False):
        
        blocksize = self.segmentationMethods['Adaptive threshold']['Params']['Block size'][1]
        erodeRep  = self.segmentationMethods['Adaptive threshold']['Params']['Erode cycles'][1]
        erodeArea = self.segmentationMethods['Adaptive threshold']['Params']['Erode area'][1]
        C         = self.segmentationMethods['Adaptive threshold']['Params']['C'][1]
        minArea   = self.segmentationMethods['Adaptive threshold']['Params']['Min area'][1]
        maxArea = self.segmentationMethods['Adaptive threshold']['Params']['Max area'][1]
        
        
        if blocksize %2 == 0:
            blocksize = blocksize + 1
        print('Thresholding...')
        if img is None:
            img = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][0,...]
        img = img - np.min(np.min(img))
        img = img/np.max(np.max(img))
        img = img * 255
        img = img.astype('uint8')
        
        newim = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, C)
        kernel = np.ones((erodeArea,erodeArea), dtype=np.uint8)
        for i in range(erodeRep):
            newim = cv2.erode(newim, kernel)
        numCells, roimap, stats, centroids = cv2.connectedComponentsWithStats(newim)
        tooBig = stats[:,3]>maxArea
        tooSmall = stats[:,3]<minArea
        rightSize = np.logical_not(np.logical_or(tooBig,tooSmall))
        IX = np.where(rightSize)
        #filteredMap = np.zeros(roimap.shape)   #for creating 2D labelmap
        labelMask = np.zeros([roimap.shape[0], roimap.shape[1], len(IX[0])])  #for creating 3D labelmap
        counter = 1
        for ix in IX[0]:
            pix = np.where(roimap==ix)
            labelMask[:,:,counter-1][pix] = counter
            counter = counter + 1
        #print(f'IX: {IX[0].shape}')
       # print(f'labelMask shape: {labelMask.shape}')
        floatMask = labelMask>0
        floatMask = floatMask.astype('float64')
    #    boolMask = labelMask.astype('bool')
        traces = np.zeros([labelMask.shape[-1],self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]])
        self.updateROIdata(floatMask, traces, appendMode = appendMode)
    #    for key in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]:
    #        del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus][key]
    #    #return(filteredMap, np.zeros(counter, 100))
    #    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].require_dataset('labelMask', shape = labelMask.shape, dtype = labelMask.dtype, data =labelMask) 
    #    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].require_dataset('floatMask', shape = floatMask.shape, dtype = floatMask.dtype, data =floatMask) 
    #    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].require_dataset('boolMask', shape = boolMask.shape, dtype = boolMask.dtype, data =boolMask)     
    #    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].require_dataset('traceArray', shape = traces.shape, dtype=traces.dtype, data = traces)
        
    
    @pyqtSlot()
    def applySegmentation(self):
        print('Segmenting...')
        func = self.segmentationMethods[self.currentSegmentationMethod]['Function']
        #self.segmentationResult[self.dataFocus] = eval(f'self.{func}()')
        eval(f'self.{func}()')  ## This is super dumb - I am currently storing all function refs as strings
        print('Segmentation applied')
        self.selectedROI = np.array([], dtype = np.uint16)
        self.closeDB()
        #self.updateROImask()
        
    @pyqtSlot()
    def KmeanSortTraceArray(self):
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...]
        traces = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        newIX = DY.sortByKmean(traces, self.segmentationMethods['seeded CNMF']['Params']['Clusters'][1]) 
        if 'names' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]:
            names = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['names'][...]
            new_names = names[newIX, :]
            self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['names'] = new_names
        newTraces = traces[newIX, :]
        newMask   = floatMask[:,:,newIX] 
        
        self.updateROIdata(newMask, newTraces)
    
        
    @pyqtSlot()      
    def extractTracesVectorized(self):
        #extractMode = 
        print('Loading movie...')
        stack = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][...]
        print('Movie loaded')
        print('Loading mask...')
        print('Mask loaded')
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...]    
        vStack = np.reshape(stack, [stack.shape[0], stack.shape[1]*stack.shape[2]], order = 'F')
        mask = floatMask>0
        if len(mask.shape) <3:
            mask = np.expand_dims(mask, axis = 2)
        oMask = np.moveaxis(mask,[0,1,2],[1,2,0])
        vMask = np.reshape(oMask, [oMask.shape[0], oMask.shape[1]*oMask.shape[2]], order = 'F')

        if 'traceArray' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        else:
            traceArray = np.zeros([vMask.shape[0], vStack.shape[0]], dtype = np.float64)
        if 'traceArray_DFF' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            traceArray_DFF = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        else:
            traceArray_DFF = np.zeros([vMask.shape[0], vStack.shape[0]], dtype = np.float64)
        
        
        for cell, ROI in enumerate(vMask):
            if cell in self.selectedROI:
                q = self.segmentationMethods['Extract traces']['Params']['Extract percentile'][1]
                DFF_TRACE = np.percentile(vStack[:,ROI], q, axis = 1)
                if self.segmentationMethods['Extract traces']['Params']['Z-score'][1]>0:
                    print('calculating z-score')
                    DFF_TRACE = LA.modified_z_score(DFF_TRACE)
                traceArray[cell,:] = DFF_TRACE
                traceArray_DFF[cell, :] = DFF_TRACE
                  #  traceArray[cell,:] = np.amin(vStack[:,ROI], axis = 1)
                
                   # traceArray[cell,:] = np.mean(vStack[:,ROI], axis = 1)
                   
                print(f'Extracted cell {cell} of {vMask.shape[0]}')
                
                    
        self.updateROIdata(floatMask, traceArray, trace_type = 'dff')
        if self.segmentationMethods['Extract traces']['Params']['Calculate DFF'][1]:
            self.calcDFF()
        #plt.imshow(traceArray, aspect = 2*traceArray.shape[1]/traceArray.shape[0])
    
    @pyqtSlot()
    def paintROIs(self):
        if 'paintedROImap' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            pMap = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['paintedROImap'][...]
            del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['paintedROImap']
        else:
            S = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape
            pMap = np.zeros([S[1],S[2],3], dtype = np.uint8)
        C = QColorDialog.getColor()
        print(C.red()) 
        print(C.green())   
        print(C.blue())             
        S = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape
        if 'boolMask' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            boolMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['boolMask'][...]
        else:
            boolMask = np.zeros([S[1],S[2]], dtype = np.bool)
        #outMap = np.zeros([S[1],S[2],3], dtype = np.uint16) 
        boolSelected = boolMask[:,:,self.selectedROI]
        flatMask = np.max(boolSelected, axis=2)
        pMap[flatMask,0] = C.red()
        pMap[flatMask,1] = C.green()
        pMap[flatMask,2] = C.blue()
     
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['paintedROImap']  = pMap
        self.updateROImask()
        
    @pyqtSlot()  
    def calcDFF(self):
        traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...] 
        Q = self.segmentationMethods['Calculate DFF']['Params']['F0 Quantile'][1]/10
        Normalize = bool(self.segmentationMethods['Calculate DFF']['Params']['Normalize'][1])
        for cell, trace in enumerate(traceArray):
            Fnought = np.quantile(trace, Q)
            while Fnought == 0:
                Q=Q+0.1
                Fnought = np.quantile(trace, Q)
                if Q>0.9:
                    Fnought = 1
                    break
            DF= trace-Fnought
            if Normalize:
                rawTrace = np.divide(DF, Fnought)
                traceArray[cell, :] = rawTrace/np.max(rawTrace)
            else:
                traceArray[cell, :] = np.divide(DF, Fnought)
        self.updateROIdata(floatMask, traceArray)
        
    @pyqtSlot()      
    def updateROIlist(self):  
        #print('updateROIlist')
        #print(inspect.stack()[1].function)
        
        
       
        
        
        
        self.ROIlist.clear()
        #numROIs = self.DB['Animals'][self.curAnimal][self.curFOV]['R']['Masks'][self.dataFocus].shape
        if not self.DB['Animals'][self.curAnimal][self.curFOV]['R'].__contains__(self.dataFocus):
            print('no mask found')
            return
        if not self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].__contains__('traceArray'):
            print('no traces found')
            return
        counter = 0
        
        
        self.ROIlist.itemSelectionChanged.disconnect(self.updateActiveROIs)
        try:
            print(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'].shape)
        except:
            print('No label mask found')
            self.ROIlist.itemSelectionChanged.connect(self.updateActiveROIs) 
            return
        for cell in range(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'].shape[2]):
            name = str(cell)
            if 'names' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
                strname = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['names'][cell,0]
                if len(strname) >0:
                    name = strname
            
            self.ROIlist.addItem(name)
        #    it = self.ROIlist.item(cell)
        #    it.setSelected(True)
        #for key in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['Masks']:
        #    self.ROIlist.addItem(key)
        #    it = self.ROIlist.item(counter)
        #    it.setSelected(True)
        #    counter = counter + 
        self.ROIlist.itemSelectionChanged.connect(self.updateActiveROIs) #TODO why is this here ant not in initialization?
        self.updateActiveROIs()
         
    def getDataDialog(self, prompt = 'Select data:'):
        datastreams = self.DB['Animals'][self.curAnimal][self.curFOV].keys()
        data, okPressed = QInputDialog.getItem(self,prompt, "Datastream:", datastreams, 0, False)
        if okPressed == False:
            return(None)
        else:
            return(data)
        
    def updateActiveROIs(self):  
        try:
            self.reOpenDB()
        except:
            print('DB already open')
        print('updateActiveROIs')
        self.selectedROI = []
        for item in self.ROIlist.selectedIndexes():
            #self.selectedROI.append(int(item.text()))
            self.selectedROI.append(item.row())

        self.selectedROI.sort()
        self.selectedROI = np.array(self.selectedROI, dtype=np.uint16)
        if self.selectedROI.size == 0:
            print('No ROI selected')
            return
        self.updateROImask()
        
        #print(selectedROI)
        #return
        
    def selectAllROIs(self):
        self.ROIlist.selectAll()
        self.updateROImask()
        
        
    
    def updateActiveROIsProgramatically(self, IXs):
        ##Idea is to toggle the state of ROIs passed to function in IXs
        self.ROIlist.itemSelectionChanged.disconnect(self.updateActiveROIs)
        self.ROIlist.clearSelection()
        if -1 in IXs:
            IXs = []    
        
            
        for IX in IXs:
            print(IX)
            items = self.ROIlist.findItems(str(IX), Qt.MatchExactly)
            for item in items:
                if item.text() == str(IX):
                    item.setSelected(~item.isSelected())
                
            
        self.ROIlist.itemSelectionChanged.connect(self.updateActiveROIs)
        self.selectedROI = np.array(IXs)
        self.selectedROI.sort()
        self.updateROImask()
                
                
                
        
    @pyqtSlot()                              
    def updateROImask(self, FOV=None, animal = None,  datakey = None): #Formats ROI information for display and sets image
        #print('updateROImask')
        if FOV is None:
            FOV = self.curFOV
        if datakey is None:
            datakey = self.dataFocus
        if animal is None:
            animal = self.curAnimal
        
        
        
        if 'labelMask' in self.DB['Animals'][animal][FOV]['R'][datakey].keys():
            labelMask = self.DB['Animals'][animal][FOV]['R'][datakey]['labelMask'][...]
        else:
            #S = labelMask = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape() ## WTF is this? TODO: find out what this is about
            S = self.DB['Animals'][animal][FOV][datakey].shape ##
            if len(S)>2:
                labelMask = np.zeros([S[1],S[2]], dtype = np.uint16)
            elif len(S) == 2:
                labelMask = np.zeros([S[0],S[1]], dtype = np.uint16)
        if self.selectedROI.size == 0:
            RGBA = np.zeros([labelMask.shape[0], labelMask.shape[1], 4])
            self.ROIoverlay.setImage(RGBA)
            
            self.ROIoverlay.linkROImap(np.zeros(labelMask.shape), self)
            return(RGBA)
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datakey]['floatMask'][...]
        
        for cell in range(floatMask.shape[-1]):
            floatMask[...,cell] = floatMask[...,cell]/np.max(np.max(floatMask[...,cell]))
        
        labelSelected = labelMask[:,:,self.selectedROI]
        floatSelected = floatMask[:,:,self.selectedROI]
        flatLabel = np.max(labelSelected, axis=2)
        flatFloat = np.max(floatSelected, axis=2)
        #flatLabel  = np.max(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['labelMask'][:,:,self.selectedROI], axis=2)
        #flatFloat  = np.max(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][:,:,self.selectedROI], axis=2)
        
        #flatFloat = 0.9*flatFloat/np.max(np.max(flatFloat))*255 #Normalize to max value for display
        #flatFloat = (flatFloat/floatMax)*255 #Normalize to max value for display
        flatFloat = flatFloat * 255
        flatFloat = flatFloat.astype(np.uint8)
        
   #     floatMax8 = np.max(np.max(np.max(flatFloat)))  #Unused - see if commenting breaks anything

        truncatedLabel = (flatLabel % 255).astype(np.uint8) # remove values above 255, cycle back to 0 for display
        
        #print(f"Pon: {self.segmentationMethods['Paint ROIs']['Params']['Painting on']}")
        if self.segmentationMethods['Paint ROIs']['Params']['Painting on'][1]==0:
            label_range = np.linspace(0,1,256)
            lut = np.uint8(plt.cm.prism(label_range)[:,2::-1]*256).reshape(256, 1, 3)
            RGB = cv2.LUT(cv2.merge((truncatedLabel, truncatedLabel, truncatedLabel)), lut)
        else:
            try:
                RGB = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datakey]['paintedROImap']
            except:
                print('No painted ROI map')
                
        ROIalpha = self.segmentationMethods['Random params']['Params']['ROIalpha pct'][1]
        Alpha = np.expand_dims(flatFloat*(ROIalpha/100), 2)
        
        
        RGBA = np.concatenate((RGB,Alpha), axis = 2)
        #self.ROIoverlay.setImage(Alpha)
        self.RGBA = RGBA
        self.ROIoverlay.setImage(RGBA)
        
        
        bgMask = np.max(floatSelected, axis=2)
        bgMaskBool = bgMask.astype(bool)
        flatLabel[~bgMaskBool]= 0
        self.ROIoverlay.linkROImap(flatLabel, self)
        
        
        self.selectedROI = np.sort(self.selectedROI)
        
        print(f'selected: {self.selectedROI}')
        
        
        
        
        raster = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][self.selectedROI,:]
        traceTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][...]
        self.ROItracePlot.clear()
        
        self.rasterImage.clear()
        for trace in raster:
            #trace = np.squeeze(trace) #to plot single ROI selections
            self.ROItracePlot.plot(traceTime, trace)
        img = raster.T
        self.rasterImage.setImage(img)
        s = self.sampleIndex[self.dataFocus]
        self.timeLine2_min = (np.amin(raster))
        self.timeLine2_max = (np.amax(raster))
        self.timeLine2 = self.ROItracePlot.plot(np.array([self.curTime,self.curTime]), np.array([self.timeLine2_min, self.timeLine2_max]), pen='r')
        
        #self.ROItracePlot.plot([traceTime[s],traceTime[s]],[0, np.amax(raster)], 'r', linewidth=5)
        #self.rasterImage.plot([s,s],[0,raster.shape[0]], 'r')
        #plt.imshow(raster, aspect = 2*raster.shape[1]/raster.shape[0])
        #plt.show()
        
        return(RGBA)
       
    def store_multisession_link(self, multi_session):
        pass
        
    @pyqtSlot()      
    def delROI(self, rois_to_delete = None): 
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...]
        traces =    self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]

        if rois_to_delete is None:
            self.selectedROI = []
            for item in self.ROIlist.selectedIndexes():
                #self.selectedROI.append(int(item.text()))
                self.selectedROI.append(item.row())
            self.selectedROI.sort()
            self.selectedROI = np.array(self.selectedROI)
            self.selectedROI = self.selectedROI.astype('uint16')
            rois_to_delete = self.selectedROI
  
        
        
        newMask = np.delete(floatMask, rois_to_delete, axis = -1)
        newTraces = np.delete(traces, rois_to_delete, axis = 0)
        
        #newMask = floatMask
        #newTraces = traces
        
        newMask = newMask.astype('float64')
        self.updateROIdata(newMask, newTraces)
        
        if 'names' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            names = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['names'][...]
            new_names = np.delete(names, rois_to_delete, axis = 0)
            self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['names'] = new_names
        
        if 'paintedROImap' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['paintedROImap']       ## Fix later -gets rid of all painted rois, should make it only delete selected
        

    
    
    @pyqtSlot()      
    def seededCNMF(self):
        fname = self.DBpath
        stack = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][...]
  #      if 'mov' in self.DB.keys():
  #          del self.DB['mov']
  #      self.DB.require_dataset('mov', shape = stack.shape, dtype=stack.dtype, data = stack)
        
        mask = None
        A = None
        Ain = None
        
        var_name_hdf5 = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].name
        
        print(f'Segmenting movie: {var_name_hdf5} in file {fname}')
       
        
        decay_time = self.segmentationMethods['seeded CNMF']['Params']['decay_time'][1]/100
        p = self.segmentationMethods['seeded CNMF']['Params']['p'][1] 
        gsigOneD = self.segmentationMethods['seeded CNMF']['Params']['gsig'][1] 
        ssub = self.segmentationMethods['seeded CNMF']['Params']['ssub'][1]
        tsub = self.segmentationMethods['seeded CNMF']['Params']['tsub'][1]
        merge_thr = self.segmentationMethods['seeded CNMF']['Params']['merge_thr'][1]/100
        userDefinedMasks = self.segmentationMethods['seeded CNMF']['Params']['User defined masks'][1]
        inputFootprintSeparately = self.segmentationMethods['seeded CNMF']['Params']['Separate input masks'][1]
        detrend = self.segmentationMethods['seeded CNMF']['Params']['Detrend'][1]
        
        
        try:
   
          #  mask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['labelMask']
            mask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...]
            
         
        except:
            print('No input mask available')
            #return([np.array([]), np.array([])])
            userDefinedMasks= False
        
        appendMode = bool(self.segmentationMethods['seeded CNMF']['Params']['Append'][1])
        if userDefinedMasks:
            if appendMode:
                self.selectedROI.sort()
                selected = self.selectedROI
                print(selected)
                mask = mask[:,:,selected]
                
                
                              
            
          
    
            
            h = mask.shape[0]
            w = mask.shape[1]
            n = mask.shape[2]
            A = np.zeros([h*w,n], dtype=bool)
            for i in range(mask.shape[-1]):
                ROI = mask[...,i]
                ROI = ROI.astype(bool)
                A[:,i] = ROI.flatten('F')
                
            mask = np.max(mask, axis = 2)
            inputMask= mask
        
        
        
            
        
            if inputFootprintSeparately:
                    Ain = A
            else:
                    Ain = cm.base.rois.extract_binary_masks_from_structural_channel(mask, gSig=gsigOneD, expand_method='dilation')[0]
                
        
           
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False, ignore_preexisting=True)
        # dataset dependent parameters

        rf = None                   # half-size of the patches in pixels. Should be `None` when seeded CNMF is used.
        only_init = False           # has to be `False` when seeded CNMF is used
        gSig = (gsigOneD, gsigOneD)               # expected half size of neurons in pixels, very important for proper component detection
        method_init = 'corr_pnr' # 'greedy_roi'
        use_cnn = False # True
    # params object

        opts_dict = {'fnames': fname,
                     'var_name_hdf5': var_name_hdf5,
                     'decay_time': decay_time,  # 0.4,
                     'p': p,
                     'nb': 2,
                     'rf': rf,
                     'only_init': only_init,
                     'gSig': gSig,
                     'ssub': ssub,
                     'tsub': tsub,
                     'use_cnn' : use_cnn,
                     'method_init' : method_init,
                     'merge_thr': merge_thr}
                    
                     #'K' : K}
        opts = params.CNMFParams(params_dict=opts_dict)
        
      
        if not userDefinedMasks:
            pass
            #opts_dict['only_init'] = True
        print('starting cnmf')
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)

        print(cnm.params)
  
        cnm.fit(stack)
        #e=cnm.estimates
       # pdb.set_trace()
   #     cnm.estimates.
        masks=np.reshape(np.array(cnm.estimates.A.todense()),[cnm.dims[0],cnm.dims[1],cnm.estimates.C.shape[0]],'F')
        
        try:
            outputMask = (np.max(masks, axis = 2))
        except:
            pdb.set_trace()
        plt.imshow(np.concatenate((inputMask, outputMask), axis = 1))
        #plt.imshow(cnm.estimates.C, aspect = 3)
        

        #(masks, traces) = self.seededCNMF()
        if detrend:
            cnm.estimates.detrend_df_f(detrend_only=True)
            traces = cnm.estimates.F_dff
        else:
            traces    = cnm.estimates.C
        if self.segmentationMethods['seeded CNMF']['Params']['Sort traces'][1]:
            newIX     = DY.sortByKmean(traces, self.segmentationMethods['seeded CNMF']['Params']['Clusters'][1]) 
            traces = traces[newIX, :]
            masks   = masks[:,:,newIX] 

        
        self.updateROIdata(masks, traces, appendMode=appendMode)



    def reorderROIs(self):
        oMasks = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask']
        oTraces = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'] 
        
        ordering = []
        numEl = oMasks.shape[-1]
        n =0
        while len(ordering)<numEl:
            ele, pressed = QInputDialog.getInt(self, f'Selected ROI to put number {n}', 'ROI number', n,0,numEl,1 )
            print(ele)
            print(pressed)
            if not pressed:
                return()
            if ele not in ordering and ele < numEl:
                ordering.append(ele)
                n = n+1
                print('Good choice')
            else:
                print('Try again...')
        
        o = np.array(ordering)
        nMasks = np.zeros(oMasks.shape)
        nTraces = np.zeros(oTraces.shape)
        for c, IX in enumerate(o):
            nMasks[:,:,c] = oMasks[:,:,IX]
            nTraces[c,:]  = oTraces[IX,:]
        
        self.updateROIdata(nMasks, nTraces, appendMode = False)
            
        
            
            
        
    def updateROIdata(self, masks, traces, names = None, appendMode = False, animal=None, FOV = None, datakey = None, trace_type = None, updateGUI=True):
        if animal is None:
            animal = self.curAnimal
        if FOV is None:
            FOV = self.curFOV
        if datakey is None:
            datakey = self.dataFocus
            
        self.selectedROI = np.array([], dtype = np.uint16)
        if appendMode:
            oMasks = self.DB['Animals'][animal][FOV]['R'][datakey]['floatMask']
            oTraces = self.DB['Animals'][animal][FOV]['R'][datakey]['traceArray'] 
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
            
        for key in self.DB['Animals'][animal][FOV]['R'][datakey]:
            del self.DB['Animals'][animal][FOV]['R'][datakey][key]
        self.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('labelMask', shape = labelMask.shape, dtype = labelMask.dtype, data =labelMask) 
        self.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('floatMask', shape = floatMask.shape, dtype = floatMask.dtype, data =floatMask) 
        self.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('boolMask', shape = boolMask.shape, dtype = boolMask.dtype, data =boolMask)     
        self.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('traceArray', shape = traces.shape, dtype=traces.dtype, data = traces)
        
        if not (names is None):
            dt = h5py.string_dtype(encoding = 'utf-8')
            nROIs= floatMask.shape[2]
            if not ('names' in self.DB['Animals'][animal][FOV]['R'][datakey].keys()):
                N = self.DB['Animals'][animal][FOV]['R'][datakey].require_dataset('names',shape=(nROIs,1), dtype = dt)
              
            for r, n in enumerate(names):
                N[r,0] = n
        
        if updateGUI:
            self.updateROIlist()
            self.updateROImask()
            self.ROIlist.selectAll()
        
        
        
    @pyqtSlot()      
    def transferMasks(self):
        datastreams =  self.DB['Animals'][self.curAnimal][self.curFOV].keys()
        dest, okPressed = QInputDialog.getItem(self,"Select destination:", "FOV:", datastreams, 0, False)
        if okPressed == False:
            return
        if dest == self.dataFocus:
            return
        
        print(f'dest is: {dest}')
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][...]
        boolMask  = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['boolMask'][...]
        labelMask  = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['labelMask'][...]
        #traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][...]
        traceArray = np.zeros([labelMask.shape[-1],self.DB['Animals'][self.curAnimal][self.curFOV][dest].shape[0]])
        
        
        
        for key in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest]:
            print(f'Deleting: {key}')
            del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest][key]
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest].require_dataset('floatMask', shape = floatMask.shape, dtype = floatMask.dtype, data =floatMask) 
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest].require_dataset('boolMask', shape = boolMask.shape, dtype = boolMask.dtype, data =boolMask) 
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest].require_dataset('labelMask', shape = labelMask.shape, dtype = labelMask.dtype, data =labelMask) 
        self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest].require_dataset('traceArray', shape = traceArray.shape, dtype = traceArray.dtype, data =traceArray) 

        for key in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest]:
           print(key)
           print(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest][key].shape)
           
           
    def combineMasks(self):
        datastreams =  self.DB['Animals'][self.curAnimal][self.curFOV].keys()
        dest, okPressed = QInputDialog.getItem(self,"Select destination:", "Datastream:", datastreams, 0, False)
        if okPressed == False:
            return
        if dest == self.dataFocus:
            return
        
        print(f'dest is: {dest}')
        
        sourceData = {}
        destData = {}
        combinedData = {}
        targets = ['floatMask', 'boolMask', 'labelMask']
        for target in targets:
           
            try:
                destData[target] = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest][target][...]            
            except:
                print(f'Did not find {target}  in destination: {dest}')
                return
            try:
                sourceData[target] = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus][target][...]            
            except:
                print(f'Did not find {target}  in source: {self.dataFocus}')
                return
            if len(sourceData[target].shape) == 2:
                AX = 0
            elif len(sourceData[target].shape) == 3:
                AX = 2
            else:
                print('Source data is wrong dimension')
                return
            combinedData[target] = np.concatenate((destData[target], sourceData[target]), axis = AX)
             
        for key in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest]:
            print(f'Deleting: {key}')
            del self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest][key]
        
        traceArray = np.zeros([combinedData['labelMask'].shape[-1],self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]])
        for target in targets:
            self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest].require_dataset(target, shape = combinedData[target].shape, dtype = combinedData[target].dtype, data = combinedData[target])
            self.DB['Animals'][self.curAnimal][self.curFOV]['R'][dest].require_dataset('traceArray', shape = traceArray.shape, dtype = traceArray.dtype, data = traceArray)
        print('Masks concatetnated')
           
    @pyqtSlot() 
    def generateSummaryTable(self, createFiles = True, selectedDataStreams = None, activityStream = None, selectedFOV = None, trimStim = True, timeStep = 0.1): 
        ## for each FOV generate data table with stimuli and responses in common time frame
        if createFiles:
            saveDir = self.getDir()
            os.mkdir(os.path.join(saveDir,'Masks'))
        
        OriginalFOV = self.curFOV
        OriginalFocus = self.dataFocus
        if selectedFOV != None: ## If passing FOV to analyze programatically
            self.curFOV = selectedFOV
            self.dataFocus = activityStream
           
            
        ## First create paired dictionaries of Xdata(timestamps) and Ydata(measurements)
        FOV = self.DB['Animals'][self.curAnimal][self.curFOV]
        if selectedDataStreams == None: ## If running from within interface use selected datastreams
            selectedDataStreams = self.DataList.selectedItems()
            selectedROI = []
            for item in self.ROIlist.selectedIndexes():
                selectedROI.append(item.row())
        else:
            traceArray = self.DB['Animals'][self.curAnimal][self.curFOV]['R']
            selectedROI = np.arange(traceArray.shape[0])
            
            
        Xdata = {}
        Ydata = {}
        Labels = {}
        
        

        
        selectedROI.sort()
        selectedROI = np.array(selectedROI, dtype=np.uint16)
        print(f'selected for report: {selectedROI}')
        
        
        
        for item in selectedDataStreams:
            datastream = item.text()
            for att in FOV[datastream].attrs:
                print(f'Attribute {att} is set for data {datastream}')
            if len(FOV[datastream].shape) ==1 and not 'Link' in FOV[datastream].attrs:
                trace = FOV[datastream][...]   
                Ydata[datastream] = np.expand_dims(trace, axis = 0)
                #plt.plot(Ydata[datastream])
                #plt.show()
            elif len(FOV[datastream].shape) >= 3 or 'Link' in FOV[datastream].attrs:
                if datastream == self.dataFocus:
                    print('selecting cells:')
                    Ydata[datastream] = FOV['R'][datastream]['traceArray'][selectedROI,...]
                    floatMask = FOV['R'][datastream]['floatMask'][:,:,selectedROI]
                    flatLabel = np.max(FOV['R'][datastream]['labelMask'][:,:,selectedROI], axis=2)
                else:
                    Ydata[datastream] = FOV['R'][datastream]['traceArray'][...]
                    floatMask = FOV['R'][datastream]['floatMask'][...]
                    flatLabel = np.max(FOV['R'][datastream]['labelMask'][...], axis=2)
               
         
                meanImage = np.median(FOV[datastream][0:10,...], axis = 0)
                meanImage = meanImage - np.amin(meanImage)
                meanImage = meanImage/np.amax(meanImage)
                meanImage = meanImage*255
                meanImage = meanImage.astype(np.uint8)
                if createFiles:
                    imName = datastream + '_mean.tif'
                    imPath = os.path.join(saveDir,imName)
                    imageio.imwrite(imPath, meanImage)
                if datastream == self.dataFocus:
                    outMeanImage = meanImage
                
                
                    
                
        
                for cell in range(floatMask.shape[-1]):
                    floatMask[...,cell] = floatMask[...,cell]/np.max(np.max(floatMask[...,cell]))
                
                if createFiles:
                    for cell in range(floatMask.shape[-1]): ## Write masks to disk individually (may not be really nec)
                        flatFloat = floatMask[...,cell]
                
                        flatFloat = flatFloat * 255
                        flatFloat = flatFloat.astype(np.uint8)
                        truncatedLabel = (flatLabel % 255).astype(np.uint8) # 
                        label_range = np.linspace(0,1,256)
                        lut = np.uint8(plt.cm.prism(label_range)[:,2::-1]*256).reshape(256, 1, 3)
                        RGB = cv2.LUT(cv2.merge((truncatedLabel, truncatedLabel, truncatedLabel)), lut)
                        Alpha = np.expand_dims(flatFloat, 2)
                        RGBA = np.concatenate((RGB,Alpha), axis = 2)
                        imName = datastream + '_masks_' + str(cell) +'.tif'
                        imPath = os.path.join(saveDir,'Masks',imName)
                        imageio.imwrite(imPath, RGBA)
                    
                    
                    
                    
                    
                flatFloat = np.max(floatMask, axis=2)
                if datastream == self.dataFocus: #If processing main stream (Ca++ data), send ROI array to output
                    outROIs = floatMask
                    
                    
                flatFloat = flatFloat * 255
                flatFloat = flatFloat.astype(np.uint8)
                truncatedLabel = (flatLabel % 255).astype(np.uint8) # 
                print(f'Datastream: {datastream}')
                print(f'Data focus: {self.dataFocus}')
                paintOn = self.segmentationMethods['Paint ROIs']['Params']['Painting on'][1]
                print(f'Paint on {paintOn}')
                if paintOn == 1 and datastream == self.dataFocus:
                    try:
                        RGB = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['paintedROImap']
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
                if createFiles:
                    imName = datastream + '_masks.tif'
                    imPath = os.path.join(saveDir,imName)
                    imageio.imwrite(imPath, RGBA)
                
            Xdata[datastream] = FOV['T'][datastream][...]
            Labels[datastream]  = datastream
            
            
            
        ## Next get list of all time points and create new regular time base spanning experiment:   
        allTimes = []
        for Time in Xdata:  
            allTimes = list(allTimes) + list(Xdata[Time][:])
        
        if timeStep is None:
            timeStep = self.timeStep
        
   
        allTimes.sort()
        
    #    start = allTimes[0] - 1
    #    end = allTimes[-1] + 1
        
        
        ##Set time span used from time navigator rectangle:
            
        start = self.timeROI.parentBounds().left()
        end   = self.timeROI.parentBounds().right()
        if trimStim: ## Option to clip time span to first and last frames of CA data
            print(f'{activityStream=}')
            #print(f"T keys: {self.DB['Animals'][self.curAnimal][self.curFOV]['T'].keys()}")
            
            start = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][0]
            end  =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][-1]
        
        length = end-start
        print(f'Length: {length}')
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
            ## check if ROIs have names
            if 'names' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datastream].keys():
                print('Names found')
                names = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][datastream]['names'][...]
                for c, name in enumerate(names):
                    print(f'{name[0]=}')
                    if len(name[0]) >0:
                        stimIX.append(rowCounter + c)
                        stimLabels.append(name[0])
            
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
        #plt.imshow(compressedArray, aspect = output.shape[1]/output.shape[0]*0.33)
        
        if not createFiles:
            #saveDir = None
            saveDir = self.reportSaveDir + self.curAnimal+self.curFOV+self.dataFocus+'plots' + str(time.time())[-5:]
        
        print(stimIX)
        print(stimLabels)
        #TODO DY.GUI_plot_traces(ob=self, raster=compressedArray, timestep = timeStep, stimIX = stimIX, stimLabels = stimLabels, savePath = saveDir, nullValue=nullValue)
        
        
        print(f'{stimIX=}')
        print(f'{stimLabels=}')
      
        
        plotRaster(compressedArray, timestep = timeStep, stimIX = stimIX, stimLabels = stimLabels, savePath = saveDir, nullValue=nullValue)
        
        
        
        
        timeStep = np.array(timeStep)
        nullValue = np.array(nullValue)
        if selectedFOV != None:
            self.curFOV = OriginalFOV
            self.dataFocus = OriginalFocus
        return(output, compressedArray, timeStep, stimIX, stimLabels, outMeanImage, outROIs, nullValue, start)
        
    
    
    
    
    
    @pyqtSlot()
    def exportFOV(self):
        DBpath = self.saveFile()
        DB = h5py.File(DBpath,'a')
        DB.require_group('Animals')           
        DB['Animals'].require_group(self.curAnimal)
        print(f'Copying to {DBpath}...')
  
        self.DB['Animals'][self.curAnimal].copy(self.curFOV, DB['Animals'][self.curAnimal], name = self.curFOV)
                  
    @pyqtSlot()
    def exportDataStream(self):
        repairMode = True
        DBpath = self.saveFile()
        DB = h5py.File(DBpath,'a')
        DB.require_group('Animals')           
        DB['Animals'].require_group(self.curAnimal)
        DB['Animals'][self.curAnimal].require_group(self.curFOV)
        DB['Animals'][self.curAnimal][self.curFOV].require_group('R')
        DB['Animals'][self.curAnimal][self.curFOV].require_group('T')
        print(self.DB['Animals'][self.curAnimal][self.curFOV].keys())
        for item in self.DataList.selectedItems():
                dataKey = item.text()
                print(dataKey)
                self.DB['Animals'][self.curAnimal][self.curFOV].copy(dataKey, DB['Animals'][self.curAnimal][self.curFOV], name = dataKey)
                if not repairMode:
                    self.DB['Animals'][self.curAnimal][self.curFOV]['R'].copy(dataKey, DB['Animals'][self.curAnimal][self.curFOV]['R'], name = dataKey)
                self.DB['Animals'][self.curAnimal][self.curFOV]['T'].copy(dataKey, DB['Animals'][self.curAnimal][self.curFOV]['T'], name  = dataKey)
        DB.close()
                  
            
    @pyqtSlot()
    def exportPivotDB(self):
        DBpath = self.saveFile()
        fullArr, compactArray, timestep, stimIX, stimLabels, meanImage, ROIs, nullValue = self.generateSummaryTable(createFiles = False)
        DB = h5py.File(DBpath,'a')
        DB.require_group('stims')
        for IX, label in zip(stimIX, stimLabels):
            stim = compactArray[IX,:]
            DB['stims'].require_dataset(label, data = stim, shape = stim.shape, dtype = stim.dtype)
        
        cellRaster = np.delete(compactArray, stimIX, axis = 0)
        source = h5py.ExternalLink(self.DBpath, f'/Animals/{self.curAnimal}/{self.curFOV}')
        DB['source'] = source
        DB.require_dataset('timestep', data = timestep, shape = timestep.shape, dtype = timestep.dtype)
        DB.require_dataset('nullValue', data = nullValue, shape = nullValue.shape, dtype = nullValue.dtype)
        DB.require_dataset('raster', data = cellRaster, shape = cellRaster.shape, dtype = cellRaster.dtype)
        DB.require_dataset('ROIs', data = ROIs, shape = ROIs.shape, dtype = ROIs.dtype)
        DB.require_dataset('meanImage', data = meanImage, shape = meanImage.shape, dtype = meanImage.dtype)
        DB.close()
        print('DB closed')
        
        
            
    @pyqtSlot()
    def respondClick(self, X, Y, labelMap, **kwargs):
        self.clickActions[self.clickAction](X,Y, labelMap)
            
    @pyqtSlot()
    def selectROI(self, X, Y, labelMap):
        IX = [int(labelMap[X,Y])-1]
        self.updateActiveROIsProgramatically(IX)
    

        
        
    
    @pyqtSlot()
    def markEvent(self, X, Y, labelMap):
        radius = 5
        dataShape = (labelMap.shape[0], labelMap.shape[1])
       
        xx, yy = disk((X,Y), radius, shape = dataShape)
        mask = np.zeros(dataShape)
        mask[xx,yy] = 1
        
        forceData = None
        forceTime = None
        for item in self.DataList.selectedItems():
            dataname = item.text()
            if '_mN' in dataname or 'VFraw' in dataname or 'Force' in dataname:
                forceData = self.DB['Animals'][self.curAnimal][self.curFOV][dataname][...]
                forceTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][dataname][...]
                break
        
        print(f'forceData shape: {forceData.shape}')
        print(f'forceTime shape: {forceTime.shape}')
        
        cameraTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][...]
        cameraForce = np.zeros(cameraTime.shape)
        IXs = np.searchsorted(forceTime, cameraTime)
        IXs[IXs >= forceTime.shape[0]] = forceTime.shape[0]-1
        cameraForce = forceData[IXs]
        
        forceData = cameraForce   
        
        trace = np.zeros(forceData.shape)
        F = plt.figure('Force segmentation')
        A = F.add_subplot(1,1,1)
        A.plot(forceData,'k')
        
        
        
        ##Calculate local baseline:
        Twindow = 10 # 10 seconds in each direction
        Tstart = self.curTime - Twindow
        Tend = self.curTime + Twindow
        #Istart = np.where(forceTime>Tstart)[0][0]
        Istart = np.where(cameraTime>Tstart)[0][0]
        #Iend = np.where(forceTime<Tend)[0][-1]
        Iend = np.where(cameraTime<Tend)[0][-1]
        print(f'Tstart: {Tstart}')
        print(f'Tend: {Tend}')
        print(f'Istart: {Istart}')
        print(f'Iend: {Iend}')
        
        
        #w_trace = forceData[Istart:Iend] #Windowed trace of force data Twindow from current timepoint
     
        
        baseline = np.quantile(forceData[Istart:Iend], 0.2)
        baselineoff = 10
        print(f'Baseline 0.2 = {baseline}, using 2 mN instead')
        baseline = 10#[2]
        #maxVal = np.quantile(forceData, 0.99)
        
        previousdata = forceData[0:self.sampleIndex[self.dataFocus]]
        afterdata = np.full(trace.shape, np.inf)
        afterdata[self.sampleIndex[self.dataFocus]:] = forceData[self.sampleIndex[self.dataFocus]:]
        #A.plot(previousdata, 'b')
        #A.plot(afterdata, 'r')
        stimStart = np.where(previousdata<baseline)[0][-1]
        #A.scatter(stimStart, previousdata[stimStart])
        try:
            if np.where(afterdata<baselineoff)[0].shape[0] > 0:
                stimEnd = np.where(afterdata<baselineoff)[0][0]
            else:
                stimEnd = np.where(afterdata==np.min(afterdata))[0][0]
        except:
            #print(f'{stimEnd=}')
            LA.dump(locals())
            print('Dumped log to pickle')
                               
    
        print(f'stimStart: {stimStart}')
        print(f'stimEnd: {stimEnd}')
        
        trace[stimStart:stimEnd] = forceData[stimStart:stimEnd]
        A.plot(trace,'r')
        appendMode = False
        if 'traceArray' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            print('trace array found')
            if self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'].size>1:
                print(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'].size)
                print('trace array size is more than 0')
                appendMode = True
        mask = np.expand_dims(mask, 2)
        trace = np.expand_dims(trace, 0)
        print(f'Mask shape: {mask.shape}')
        self.updateROIdata(mask, trace, appendMode = appendMode)
        
        
    @pyqtSlot()
    def autoMarkEvents(self):
        
        thresh =  self.segmentationMethods['Mark indentations']['Params']['Min force '][1]
        interval = self.segmentationMethods['Mark indentations']['Params']['Min interval (s/10)'][1]
        prom  = self.segmentationMethods['Mark indentations']['Params']['Min prominence'][1]
        base =  self.segmentationMethods['Mark indentations']['Params']['Max baseline '][1]
        
        forceData = None
        forceTime = None
        for item in self.DataList.selectedItems():
            dataname = item.text()
            if '_mN' in dataname or 'VFraw' in dataname or 'Force' in dataname:
                forceData = self.DB['Animals'][self.curAnimal][self.curFOV][dataname][...]
                forceTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][dataname][...]
                break
        
        cameraTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][...]
        cameraForce = np.zeros(cameraTime.shape)
        
        ## Translate force time coordinates from force time base to NIR camera timebase
        IXs = np.searchsorted(forceTime, cameraTime)
        IXs[IXs >= forceTime.shape[0]] = forceTime.shape[0]-1
        cameraForce = forceData[IXs]
        
        forceData = cameraForce   
        traces = DY.segmentMechTrace(forceData, thresh, interval, prom, base)
    
        
    @pyqtSlot()
    def markAllEvents(self, X, Y, labelMap):
        radius = 5
        dataShape = (labelMap.shape[0], labelMap.shape[1])
       
        xx, yy = disk((X,Y), radius, shape = dataShape)
        mask = np.zeros(dataShape)
        mask[xx,yy] = 1
        mask = np.expand_dims(mask, 2)
        
        forceData = None
        forceTime = None
        for item in self.DataList.selectedItems():
            dataname = item.text()
            if '_mN' in dataname or 'VFraw' in dataname or 'Force' in dataname:
                forceData = self.DB['Animals'][self.curAnimal][self.curFOV][dataname][...]
                forceTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][dataname][...]
                break
        
        cameraTime = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][...]
        cameraForce = np.zeros(cameraTime.shape)
        
        ## Translate force time coordinates from force time base to NIR camera timebase
        IXs = np.searchsorted(forceTime, cameraTime)
        IXs[IXs >= forceTime.shape[0]] = forceTime.shape[0]-1
        cameraForce = forceData[IXs]
        
        forceData = cameraForce   
        
        
        plt.plot(forceData)
        plt.show()
        
        ##Calculate  baseline:

        Tstart = self.timeLUT[0]  ## constrain to selected time
        Tend = self.timeLUT[-1]
        Istart = np.where(cameraTime>Tstart)[0][0]
        Iend = np.where(cameraTime<Tend)[0][-1]
   
        baseline = np.quantile(forceData[Istart:Iend], 0.5)
        print(baseline)
        baseline = 2
        maxVal = np.quantile(forceData, 0.99)
        
        selectedData = np.zeros(forceData.shape)
        selectedData[Istart:Iend] = forceData[Istart:Iend]
        plt.plot(selectedData)
        
        diff = np.diff(selectedData, prepend=0)
        startIXs = np.where(diff>5)[0]
        endpoint = -1
        print(startIXs)
        for c, startIX in enumerate(startIXs):
            if startIX <= endpoint:
                continue
            trace = np.zeros(forceData.shape)
            after = selectedData[startIX:]
            if np.where(after<baseline)[0].size > 0:
                endpoint = startIX+np.where(after<baseline)[0][0]
            else:
                endpoint = startIX + after.shape[0]-1
            trace[startIX:endpoint] = forceData[startIX:endpoint]
            trace = np.expand_dims(trace, 0)
            if c == 0:
                traces = trace
                masks = mask
            else:
                traces = np.concatenate((traces, trace), axis = 0)
                masks = np.concatenate((masks, mask), axis = -1)
        
       
            
        
        #Add identfied traces and ROI masks to paw map
        appendMode = False
        if 'traceArray' in self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus].keys():
            print('trace array found')
            if self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'].size>1:
                print(self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'].size)
                print('trace array size is more than 0')
                appendMode = True
                
        print(f'Append mode: {appendMode}')
        print(f'Mask shape: {masks.shape}')
        print(f'Trace shape: {traces.shape}')
        self.updateROIdata(masks, traces, appendMode = appendMode)
        
        
    @pyqtSlot()
    def localStimCorr(self, X, Y, labelMap):
        ##Calculate and plot correlation to stim and/or local correlation around cell:
        pass
            
    def mergeROIs(self):
        stack = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][:,:,self.selectedROI]
        flattened = np.amax(stack, axis=2)
        flattened = np.expand_dims(flattened, axis = 2)
        traces = np.zeros([1, self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]])
        self.updateROIdata(flattened, traces, appendMode = True)
            
    def splitROI(self, X, Y, labelMap):
        self.selectROI(X, Y, labelMap)
        print(f'{self.selectedROI=}')
        if len(self.selectedROI) ==0:
            print('None selected to split')
            return
        mask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][:,:,self.selectedROI[0]]
        mask = mask*255
        mask = mask.astype(np.uint8)
        #plt.figure('Mask')
        #plt.imshow(mask)
        y,x,h,w = cv2.boundingRect(mask)
        print(f'{x=},{y=},{w=},{h=}')
       # movie = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][:,x:x+w,y:y+h]
        movie, time = DY.getSubStack(self, rect=[x,y,w,h])
        print(f'{movie.shape=}')
        #plt.figure('movie first frame')
        #plt.imshow(movie[0,:,:])
        
        v_movie = np.reshape(movie, [movie.shape[0], movie.shape[1]*movie.shape[2]], order = 'F')
        
        
        traces = np.zeros([movie.shape[0], movie.shape[1]*movie.shape[2]])
        for ii in range(w):
            for jj in range(h):
                 f = movie[:,ii,jj]
                 f_nought = np.mean(f)
                 f_delta  = f-f_nought
                 traces[:,(ii*h)+jj] = f_delta/f_nought
       # plt.figure('Traces')
        #plt.imshow(traces, aspect = traces.shape[1]/traces.shape[0] )
        print(f'{traces.shape}')
        nClusters = self.segmentationMethods['Random params']['Params']['Components to split'][1]
        num_components = self.segmentationMethods['Random params']['Params']['Components to save'][1]
        kmeans = KMeans(n_clusters = nClusters).fit(traces.T)
        
        ## sort by variance of temporal component to get indices
        t_k = []
        for c in range(kmeans.n_clusters):
            t_k.append(stats.kurtosis(kmeans.cluster_centers_[c,:]))
        IX = np.argsort(t_k)
        IX = np.flip(IX)
        
        
        
        
        F = plt.figure('Components k means')
        spatial_components = {}
        grid = np.zeros([movie.shape[1],movie.shape[2]])
        for cluster in range(kmeans.n_clusters):
            spatial_components[cluster] = np.zeros([movie.shape[1],movie.shape[2]])
        for c, label in enumerate(kmeans.labels_):
            grid[np.unravel_index(c, grid.shape, 'C')] = label
            spatial_components[label][np.unravel_index(c, grid.shape, 'C')] = 1
          
              
        for c in range(kmeans.n_clusters):
            F.add_subplot(2, kmeans.n_clusters+1, c+1)
            plt.imshow(spatial_components[c])
        
       
            
            F.add_subplot(2, kmeans.n_clusters+1, kmeans.n_clusters+c+2)
            plt.plot(kmeans.cluster_centers_[c,:])
            ax = plt.gca()
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
        
        F.add_subplot(2, kmeans.n_clusters+1, kmeans.n_clusters+1)
        plt.imshow(grid)
        
        print(f'{t_k=}')
        print(f'{IX=}')
        print(f'Adding {num_components} ROIs...')
        masks = np.zeros([mask.shape[0], mask.shape[1], num_components])
        for n in range(num_components):
            print(f'Adding component {IX[n]} with kurtosis {t_k[IX[n]]}')
            masks[x:x+w,y:y+h,n]  = spatial_components[IX[n]]
            
            
        traces = np.zeros([masks.shape[-1], self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]])
        self.updateROIdata(masks, traces, appendMode = True)
        # newIX = DY.sortByKmean(traces.T, 3) 
        # newTraces = traces[:, newIX]
        # plt.figure('Sorted traces')
        # plt.imshow(newTraces, aspect = newTraces.shape[1]/newTraces.shape[0] )
        # kmeans = KMeans(n_clusters = nClusters).fit(traces.T)
       ## for value in set(newIX):
         #   sub_cell = np.zeros(movie.shape[1],movie.shape[2])
            
        #newMask   = floatMask[:,:,newIX] 
        
        
    @pyqtSlot()
    def splitROI2(self, X, Y, labelMap):
        self.selectROI(X, Y, labelMap)
        if len(self.selectedROI) ==0:
            print('None selected to split')
            return
        mask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][:,:,self.selectedROI[0]]
        distances = ndi.distance_transform_edt(mask)
  
        gsig = self.segmentationMethods['seeded CNMF']['Params']['gsig'][1]
        localmax = distances.copy()
        localmax[localmax>=gsig/2] = gsig
        localmax[localmax<gsig/2]  = 0
        localmax = localmax.astype(bool)
        #plt.imshow(localmax)
        
        markers = ndi.label(localmax)[0]

        labels = watershed(-distances, markers, mask=mask)
        plt.imshow(labels)
        numNewLabels = np.amax(labels)
        
        
        splitMasks = np.zeros([labels.shape[0], labels.shape[1], numNewLabels])  #for creating 3D labelmap
        counter = 1
        for ix in range(numNewLabels):
            splitMasks[:,:,ix] = labels == ix+1
        #print(f'IX: {IX[0].shape}')
       # print(f'labelMask shape: {labelMask.shape}')
        floatMask = splitMasks>0
        floatMask = floatMask.astype('float64')
    #    boolMask = labelMask.astype('bool')
        traces = np.zeros([floatMask.shape[-1], self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus].shape[0]])
        
        
        
        
        self.delROI()        
        self.updateROIdata(floatMask, traces, appendMode = True)
        
        
       
           
        
    
    @pyqtSlot()
    def createROI(self, X, Y, labelMask):
        print('Local correlation starting...')
        radius  = self.segmentationMethods['seeded CNMF']['Params']['gsig'][1] 
        data = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus]
        left = X-radius
        bot = Y - radius
        right = X+radius
        top = Y + radius
        if left <0:
            left = 0
        if bot < 0:
            bot = 0
        if right >= data.shape[1]:
            right = data.shape[1]-1
        if top >= data.shape[2]:
            top = data.shape[2]-1
        #print(f'right: {right}')
        #print(f'Y: {Y}')
        #print(f'Data shape: {data.shape}')
        
        timescale =  self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus]
        Istart = np.searchsorted(timescale,self.timeLUT[0])
        Iend = np.searchsorted(timescale,self.timeLUT[-1])
        
        print('Isolating signal and neighborhood...')
        signal = data[Istart:Iend,X,Y]

        area = data[Istart:Iend,left:right,bot:top]
        #plt.plot(signal)
        #plt.show()
        print('Correlating signal to neighborhood...')
        cormap = np.zeros([area.shape[1],area.shape[2]])
        for ii in range(0,area.shape[1]):
            for jj in range(0,area.shape[2]):
                cormap[ii,jj] = np.corrcoef(area[:,ii,jj],signal)[0,1]
        
        fullim = np.zeros([data.shape[1], data.shape[2]])
        fullim[left:right, bot:top] = cormap
        plt.figure('Correlation map')
        plt.imshow(fullim)
        plt.show()
        self.adaptiveThreshold(img = fullim, appendMode = True)
       
    
    @pyqtSlot()
    def selectClickAction(self):
        self.clickAction = self.clickActionBox.currentText()
    
    def runThermoTuning(self):
        TF = ['True', 'False']
        separate, okpressed = QInputDialog.getItem(self,"Separate genotypes:", "T/F", TF, 0, False)
        if okpressed != True:
            return
        if separate == 'True':
            split_genotypes = True
        else:
            split_genotypes = False
            
        
        
        flags =  self.FOVflags.keys()
        FOVflag, okpressed = QInputDialog.getItem(self,"Select FOV flag:", "Flag", flags, 0, False)
      
        if okpressed != True:
            return
        
        
        flags =  self.DataFlags.keys()
        activeTag, okpressed = QInputDialog.getItem(self,"Select data flag:", "Flag", flags, 0, False)
      
        if okpressed != True:
            return
        
        Mice, populations = DY.mouseSetup()
        populations.append('SNI split')
        populations.append('Default')
        population, okpressed = QInputDialog.getItem(self,"Select genotype or group:", "Group", populations, 0, False)
      
        if okpressed != True:
            return
        if population == 'Default':
            genotypes=None
        elif population == 'SNI split':
            genotypes = ['SNIgpr','SNItac','SNIpbn']
        else:
            genotypes = [population]
        
        
        DY.thermal_tuning(FOVtag = FOVflag, activeTag = activeTag, genotypes = genotypes, split_genotypes = split_genotypes)
        
    @pyqtSlot()
    def createNeuron(self):
        nameMode = 'auto'
        if not 'Neurons' in self.DB['Animals'][self.curAnimal]:
            self.DB['Animals'][self.curAnimal].require_group('Neurons')
        if nameMode == 'manual':
            pass
        else:
            Tag = 1 ## add manual naming option dialog - auto or manual naming mode
        while str(Tag) in self.DB['Animals'][self.curAnimal]['Neurons'].keys():
            Tag = Tag + 1
        self.DB['Animals'][self.curAnimal]['Neurons'].create_group(str(Tag))
        self.updateNeuronList()
        print(self.DB['Animals'][self.curAnimal]['Neurons'].keys())
        return(str(Tag))
    
    def clearNeurons(self):
        for neuron in self.DB['Animals'][self.curAnimal]['Neurons'].keys():
            del self.DB['Animals'][self.curAnimal]['Neurons'][neuron]
        
    @pyqtSlot()
    def addNeuron(self, X, Y, labelMap):
        self.selectROI(X, Y, labelMap)
        if len(self.selectedROI) ==0:
            print('None selected to add')
            return
        neuronList = ['Create New Neuron...']
        if not('Neurons') in self.DB['Animals'][self.curAnimal].keys():
            self.createNeuron()
        for neuron in self.DB['Animals'][self.curAnimal]['Neurons'].keys():
            neuronList.append(neuron)
        neuronID, okPressed = QInputDialog.getItem(self,"Select cell group:", "Cell:", neuronList, 0, False)
        print(okPressed)
        if not okPressed: 
            print('Returning')
            return()
        if neuronID == 'Create New Neuron...':
            neuronID = self.createNeuron()
        print(f'NeuronID {neuronID}')
        #source = np.mean(self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][...], axis = 0)
        source = self.DB['Animals'][self.curAnimal][self.curFOV][self.dataFocus][0,...]
        trace = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][self.selectedROI[0],:]
        floatMask = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['floatMask'][:,:,self.selectedROI[0]]
        timedata = self.DB['Animals'][self.curAnimal][self.curFOV]['T'][self.dataFocus][...]
        entry  = self.DB['Animals'][self.curAnimal]['Neurons'][neuronID].require_group(self.curFOV + '_' + self.dataFocus)
        entry['source'] = source
        entry['trace'] = trace
        entry['time'] = timedata
        entry['mask'] = floatMask
        
        print(f'added observation from ROI #{self.selectedROI[0]} in FOV {self.curFOV}, datastream {self.dataFocus} to neuron {neuronID}')
        
    
    
    def PCA(self):
        n_PCs = self.segmentationMethods['PCA']['Params']['n_PCs'][1]
        raster = self.DB['Animals'][self.curAnimal][self.curFOV]['R'][self.dataFocus]['traceArray'][self.selectedROI,:]
        LA.show_raster_PCs(raster, n_PCs=n_PCs)
        
    def fix_file(self):
        File = self.getFile()
        os.system('source /programs/biogrids.shrc')
        os.system('"export HDF5_X=1.12.1"')
        os.system(f'h5clear --increment "{File}"')
        
    @pyqtSlot()
    def showNeurons(self):
        neuronList = []
        for neuron in self.DB['Animals'][self.curAnimal]['Neurons'].keys():
            neuronList.append(neuron)
        #neuronIDs, okPressed = QInputDialog.getItem(self,"Select neurons:", "Neuron:", neuronList, 0, False)
        #if not okPressed: 
        #    return()
        
        #numNeurons = len(neuronList)
        print(neuronList)
        for count, neuron in enumerate(neuronList):
            F = plt.figure(figsize =[3, 3])
            print(neuron)
            ROIlist = self.DB['Animals'][self.curAnimal]['Neurons'][neuron].keys()
            numROIs = len(ROIlist)
            maxValue = np.zeros(numROIs)
            minValue = np.zeros(numROIs)
            startTime = np.zeros(numROIs)
            for count, ROI in enumerate(ROIlist):
                maxValue[count] = max(maxValue[count], max(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['trace'][...]))
                minValue[count] = min(maxValue[count], min(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['trace'][...]))
                startTime[count] = self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['time'][0]
            
            zipped = zip(startTime, ROIlist)
            sz = sorted(zipped)
            ROIlist = [element for _, element in sz]
            
            
            #colors = matplotlib.cm.linspace
            
            t = F.add_subplot(3, numROIs, ((2*numROIs)+1, 3*numROIs))
            for count, ROI in enumerate(ROIlist):
                
                p = F.add_subplot(3, numROIs, count+1 )
                plt.plot(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['time'][...], self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['trace'][...])
                p.set_ylim(min(minValue), max(maxValue))
                p2 = F.add_subplot(3, numROIs, count+1+numROIs )
                p2.imshow(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['source'][...].T, 'gray')
                p2.imshow(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['mask'][...].T, 'inferno', alpha = 0.5)
                t.scatter(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['time'][...], np.ones(self.DB['Animals'][self.curAnimal]['Neurons'][neuron][ROI]['time'].shape), s=12)
                
            
    def create_manual_correspond_window(self, input_data):
        for session in input_data['sessions']:
            pass
            
    
    @pyqtSlot()
    def updateNeuronList(self):        
        pass
       
    @pyqtSlot()      
    def generateReport(self):    
        #Dialog for where to save
        for mouse in self.DB['Animals'].keys():
            for FOV in self.DB['Animals'][mouse]:               
                for data in 1:
                    pass
                    F = plt.Figure()
    
    
    def uploadCorr(self):  ##deprecated manual multisession alignment
         corrFile = self.getFile()           
         corr = np.genfromtxt(corrFile, delimiter = ',')
         ## remove any cell unions with no entries (typically disqualified because some sessionsn don't include in field of view)
         corr = corr[~np.all(np.isnan(corr), axis=1)]    
         self.DB['Animals'][self.curAnimal].require_dataset('Corr chart', data = corr, shape = corr.shape, dtype = corr.dtype)
    
    
    
    
    def segment_force_trace(self):
        trace, time = self.getSubStack()
        self.force_trace = mech.construct_segmented_mech_trace(trace)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(self.force_trace.T)
        plt.subplot(1,2,2)
        plt.imshow(self.force_trace)
    #@pyqtSlot()  
    #def initCanvas(self):
    #    self.DB['Animals'][self.curAnimal].require_group('Canvas')
    #    DBlist = self.DB['Animals'][self.curAnimal]['Canvas'].require_group('Frames')
    #    CoordList = self.DB['Animals'][self.curAnimal]['Canvas'].require_group('Coords')
    #    imTemp  = np.zeros((4800,2700))
    #    DBlist = self.DB['Animals'][self.curAnimal]['Canvas'].require_dataset('Image', shape = imTemp.shape, dtype = imTemp.dtype, data = imTemp)
    #    self.canvasWindow = pg.ImageView()
    #    #self.canvasWindow.setImage(self.DB['Animals'][self.curAnimal]['Canvas']['Image'])
    #    self.canvasWindow.show()
        
    #@pyqtSlot()  
    #def addToCanvas(self):
    #    for item in self.DataList.selectedItems():
    #        stack = self.DB['Animals'][self.curAnimal][self.curFOV][item.text()]
    #        frame = np.median(stack, axis = 0)
    #        coords = np.array([0,0])
    #        self.DB['Animals'][self.curAnimal]['Canvas']['Frames'].create_dataset(item.text(), data = frame)
    #        self.DB['Animals'][self.curAnimal]['Canvas']['coords'].create_dataset(item.text(), data = coords)
            
    #@pyqtSlot()  
    #def updateCanvasDisplay(self):

    #    self.canvasROIlist = {}
        
    #    for key in self.DB['Animals'][self.curAnimal]['Canvas']['Frames']:
    #        location = self.DB['Animals'][self.curAnimal]['Canvas']['coords'][key]
    #        frame = self.DB['Animals'][self.curAnimal]['Canvas']['Frames'][key]
    #        width = frame.shape[0]
    #        height = frame.shape[1]
    #        left = location[0]
    #        right = location[0]+width
    #        top = location[1]
    #        bottom = location[1]+height
    #        self.canvasROIlist[key] = pg.ROI(location, size = frame.shape)
    #        self.canvasROIlist[key].data = key
    #        self.canvasROIlist[key].sigRegionChanged.connect(self.updateCanvasFramePos)
    #        self.canvasROIlist[key].sigRegionChangeStarted.connect(self.selectActiveCanvas)
    #        self.canvasWindow.addItem(self.canvasROIlist[key])
    #        self.DB['Animals'][self.curAnimal]['Canvas'][top:bottom,left:right] = frame
    #        self.canvasWindow.setImage(self.DB['Animals'][self.curAnimal]['Canvas'])
        





def analyzeRaster(raster, stimIX, stimLabels):

    stims = raster[stimIX,:]
    raster = np.delete(raster, stimIX, axis = 0)
    
    F = plt.figure(figsize =[3, 3])
    T = F.add_axes([0, 0, 1, 1])
    ## Analyze thermal trace
    for s, label in enumerate(stimLabels):
        if 'Temp' in label:
            
            temp  = stims[s,:]
            baseTemp = stats.mode(temp.astype(np.uint16))[0][0]
            print(f'Base temp: {baseTemp}')
            deltaTemp = temp-baseTemp

            abstemp = np.absolute(deltaTemp)
            dTemp = np.ediff1d(abstemp, to_end=abstemp[-1])
        
        rollingMax = np.zeros(temp.shape)
        window = 100
        for f, value in enumerate(temp):

            if f==0:
                continue
            start = np.max([0,f-window])
            #plt.plot(abstemp)
            IX = np.argmax(abstemp[start:f])
            rollingMax[f] = temp[IX + start]
            
            
        plt.plot(rollingMax)
        

    
   # for trace in raster:
    #    print(trace.shape)
     #   print(temp.shape)
      #  T.scatter(rollingMax, trace, alpha = 0.25)
            
            


                            
    

def plotRaster(raster, items='all', timepoints = 'all', stimIX = [], stimLabels = [], timestep = 0.05, savePath = '/home/ch184656/testSave.pdf', nullValue = 0, uniform_trace_display = True, normalize_raster_display=False):
    if items != 'all':
        raster = raster[items,:]
    if timepoints != 'all':
        raster = raster[:,timepoints]
    if savePath is None:
        doSave = False
    else:
        doSave = True
        
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
    
    if normalize_raster_display:
        for count, data in enumerate(displayRaster):
            minV = np.amin(data)
            maxV = np.amax(data-minV)
            
            displayRaster[count,:] = (data-minV)/maxV
    RA.imshow(displayRaster, aspect='auto', interpolation='none', cmap=rasterMap, vmin = 2, vmax= 20)
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
    
    stimcount=0
    for c, data in enumerate(raster):
        d=c+extra
        
        
        if c in stimIX:
            traceAxes[c] = F.add_axes([X,d*traceSpan,W,traceSpan*stimSpace])
            traceAxes[c].yaxis.set_visible(True)                       
            traceAxes[c].plot(data, color='m', linewidth = 1)
            traceAxes[c].set_ylabel(stimLabels[stimcount], rotation=0, labelpad=25)
           
            
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
            if not ('Force' in stimLabels[stimcount]) and not ('Temp' in stimLabels[stimcount]):
                TA.yaxis.set_visible(False)  
                DY.LA.box_off(TA, All=True)
            else:
                TA.yaxis.set_visible(True)
            stimcount = stimcount+1
            ## try this
            TA.set_ylim([0, 55])
            TA.set_yticks([5, 20, 35, 50])
    
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
        if np.amax(raster) >10:
            traceAxes[c].set_ylim([-2,np.amax(raster)])
        #print(f'Max z = {np.amax(raster)}')
        

    G  = plt.figure('All traces')
    nStim = len(stimIX)
    stim_count  = 0
    AT = G.add_subplot(nStim+1,1,nStim+1)
    for c, trace in enumerate(raster):
        if c in stimIX:
            AS = G.add_subplot(nStim+1, 1, stim_count+1)
            AS.set_ylabel(stimLabels[stim_count])
            stim_count = stim_count+1
            AS.plot(trace, color = 'k')
            DY.LA.box_off(AS, left_only=True)
        else:
            AT.plot(trace)
            AT.set_ylabel('Fz')
            DY.LA.box_off(AT, left_only=True)
            
        
    
    
    
    if doSave:
        F.savefig(os.path.join(savePath,'traces.pdf'),transparent = True)
        F.savefig(os.path.join(savePath,'traces.png'),transparent = True)
        G.savefig(os.path.join(savePath, 'all_traces.pdf'), transparent=True)
    F.show()
    #saveName = os.path.join('/home/ch184656/Default reports', self.curAnimal+self.curFOV+self.dataFocus + '_plots.png')
    #F.savefig(saveName, transparent = True)
    
    
    #analyzeRaster(raster, stimIX, stimLabels)
    #Correlate to stimuli:
        

    
    
    
    
def ind2RGB(im, colormap, minV = 0, maxV = None):
    ##truncate to stay within bounds
    im = normalizeImage(im, minV=minV, maxV=maxV)
    label_range = np.linspace(0,1,256)
    lut = np.uint8(colormap(label_range)[:,2::-1]*256).reshape(256, 1, 3)
    RGB = cv2.LUT(cv2.merge((im, im, im)), lut)
    return(RGB)

def normalizeImage(im, minV = 0, maxV = None):
    im[im<minV] = minV
    if maxV == None:
        maxV = np.amax(im)
    im[im>maxV] = maxV
    im = im - minV
    im = im/np.amax(im)
    im = im*255
    im  = im.astype(np.uint8)
    return(im)

def sync_figures_to_google_drive():
    os.system('rclone sync "/lab-share/Neuro-Woolf-e2/Public/Figure publishing/" "DY-e2-sync:rclone_test"  --bwlimit 8650k --tpslimit 8')




        
        
                
            
    
    
    
    
if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = DBgui()
        sys.exit(app.exec_())
        
        

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        