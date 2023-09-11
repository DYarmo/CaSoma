#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:55:56 2021

@author: ch184656
"""

import sys
import DYpreProcessingLibrary as DY
import psutil

print('Total RAM: %', psutil.virtual_memory()[0])
print('RAM availabe: %', psutil.virtual_memory()[1])
print('RAM used: %', psutil.virtual_memory()[2])


FOLDER = DY.selectFolder()

f= open('/home/ch184656/YarmoPain GUI/depositPath.txt','w')
f.write(FOLDER)
f.close()

sys.os(f'sbatch /home/ch184656/YarmoPain GUI/autoDepostMousebigMem.sh')

#DY.slurmDepositMouse()
#DY.autoDepositMouse()