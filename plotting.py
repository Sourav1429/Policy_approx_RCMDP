# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:36:50 2025

@author: Sourav
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt

vf_list = []
cf_list = []

with open("Store_robust_vf_Garnet","rb") as f:
    vf_list = pickle.load(f)
f.close()
vf_list.insert(0,0)

with open("Store_robust_cf_Garnet","rb") as f:
    cf_list = pickle.load(f)
f.close()
cf_list.insert(1,175)
b= 50
t = 1000
epsilon = 0.01
#plt.plot(vf_list)
x= np.arange(len(vf_list))
y_err = 0#epsilon*50/np.arange(1,len(vf_list)+1)
y= np.minimum(np.array(vf_list),b - epsilon*np.array(cf_list))
print(len(y))
#plt.plot(x,y)
#plt.fill_between(x, y - y_err, y + y_err, alpha=0.2, color='blue')
plt.plot(vf_list)
plt.plot(cf_list)
plt.plot(np.ones(t)*b)
plt.xlabel('Iteration')
plt.ylabel('VF,CF,Baeline')
plt.title("Reward based setting where constraint to be above baseline(GARNET setting)")
plt.savefig('Projected_RCMDP.pdf')
plt.legend(['R_vf','R_constraintf','Baseline'])
plt.show()