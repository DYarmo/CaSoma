#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:59:02 2023

@author: ch184656
"""

def printc(txt_msg, mode = None, fore_tupple=[255,255,255],back_tupple=[0,0,0]):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    if mode == 'g':
        fore_tupple = [0,255,0]
    if mode == 'r':
        fore_tupple = [255,0,0]
    if mode == 'y':
        fore_tupple = [255,255,0]    
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
    print(msg .format(mat))
    print('\33[0m') # returns default print color to back to black
    
def y(text):
    printc(text, mode='y')

def g(text):
    printc(text, mode = 'g')
    
def r(text):
    printc(text, mode = 'r')