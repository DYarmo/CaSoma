#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:14:16 2021

@author: ch184656
"""

import DYpreProcessingLibrary as DY
import os


NP = os.environ.get('SLURM_NPROCS')
print(NP)
DY.slurmDepositMouse()
