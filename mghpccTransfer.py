#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:53:25 2021

@author: ch184656
"""

import os


def sendToMGHPCC(packagePath):
    command = 'rsync -avzh ' + packagePath + ' mghpcc:/project/RC_Neuro-Woolf-e2/inbox'           
    os.system(command)