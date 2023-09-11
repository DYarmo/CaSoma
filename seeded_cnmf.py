#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:14:15 2022

@author: ch184656
"""
import h5py
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

def seeded_CNMF(stack, masks=None, opts_dict = None, detrend=False):
      
      fname = '/lab-share/Neuro-Woolf-e2/Public/DavidY/tempCNMF.h5'
      varName = 'mov'
      F = h5py.File(fname,'a')
      if varName in F.keys():
          del F[varName]
      F.require_dataset(varName, shape = stack.shape, dtype = stack.dtype, data = stack)
     
      var_name_hdf5 = F[varName].name
      
      if opts_dict is None:
          opts_dict = {'fnames': fname,
                       'var_name_hdf5': var_name_hdf5,
                       'decay_time':  0.14,
                       'p': 1,
                       'nb': 2,
                       'rf': None,
                       'only_init': False,
                       'gSig': (11, 11),
                       'ssub': 1,
                       'tsub': 1,
                       'use_cnn' : False,
                       'method_init' : 'corr_pnr',
                       'merge_thr': 85}
                      
                       
          opts = params.CNMFParams(params_dict=opts_dict)
      
      h = masks.shape[0]
      w = masks.shape[1]
      n = masks.shape[2]
      Ain = np.zeros([h*w,n], dtype=bool)
      for i in range(masks.shape[-1]):
            ROI = masks[...,i]
            ROI = ROI.astype(bool)
            Ain[:,i] = ROI.flatten('F')
            
      
      c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False, ignore_preexisting=True)
      cnm = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
      
      cnm.fit(stack)
      masks_out=np.reshape(np.array(cnm.estimates.A.todense()),[cnm.dims[0],cnm.dims[1],cnm.estimates.C.shape[0]],'F')
      if detrend:
          cnm.estimates.detrend_df_f(detrend_only=True)
          traces_out = cnm.estimates.F_dff
      else:
          traces_out    = cnm.estimates.C
      output = {}
      output['masks'] = masks_out
      output['traces_out'] = traces_out
      return(output)
    
    
     