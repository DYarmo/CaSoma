#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:35:38 2023

@author: ch184656
"""

import subprocess

def squeue():
    p = subprocess.run(['squeue', '-u', 'ch184656', '-o', "%.20A %.20C %.20P %.20m %.20T %.20L %.N"], stdout = subprocess.PIPE, text=True)
    #p.wait()
    return(p.stdout)