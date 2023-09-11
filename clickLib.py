#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:35:58 2022

@author: ch184656
"""

from pyqtgraph import ImageItem
from PyQt5.QtCore import Qt

class clickImage(ImageItem):
    
    
    def linkROImap(self, ROImap, parentApp, session = None):
        self.linkedROImap = ROImap
        self.parentApp = parentApp
        self.session = session
    
    def mouseClickEvent(self, event):
        X = round(event.pos()[0])
        Y = round(event.pos()[1])
        
        mod = event.modifiers()
        mods = []
        if mod & Qt.ControlModifier:
            mods.append('ctrl')
        if mod & Qt.ShiftModifier:
            mods.append('shift')
        if mod & Qt.AltModifier:
            mods.append('alt')

      
        self.parentApp.respondClick(X,Y, self.linkedROImap, mod=mods, session = self.session)