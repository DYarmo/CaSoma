#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:55:56 2021

@author: ch184656
"""

import os
import time
import DYpreProcessingLibrary as DY

FOLDER = DY.selectFolder()


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


os.system('sbatch /home/ch184656/YarmoPain_GUI/slurmDepositBigMem.sh')

#os.system('sbatch /home/ch184656/YarmoPain_GUI/slurmDepositCompute.sh')
