# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:46:58 2024

@author: ali
"""



import numpy as np

N=10
T=10
U=10
F=1000
mem=[]
for i in range(F):    
    a=np.random.randint(1,N,(U*T))
    set_a=set(a)
    len_set_a=len(set_a)
    mem.append(len_set_a)