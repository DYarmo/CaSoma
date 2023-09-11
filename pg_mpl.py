#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:19:42 2022

@author: ch184656
"""

import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        
        
        

class Mpl_multi_canvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100,  n_rows = 1, n_columns = 1, num_items=None):
        if num_items is None:
            num_items = n_rows * n_columns
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = []
        for i in range(num_items):
            self.axes.append( fig.add_subplot(n_rows, n_columns, i+1))
        super(Mpl_multi_canvas, self).__init__(fig)
