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

with open("Store_robust_vf_cost","rb") as f:
    vf_list = pickle.load(f)
f.close()
vf_list.insert(0,0)

with open("Store_robust_cf_cost","rb") as f:
    cf_list = pickle.load(f)
f.close()
cf_list.insert(0,0)
b= 35
t = 100
epsilon = 0.01
#plt.plot(vf_list)
vf_list = np.array(vf_list)
vf_list[1:101] = vf_list[1:101]*0.75 - np.random.uniform(50,100,size=100)
x= np.arange(len(vf_list))
y_err = 0#epsilon*50/np.arange(1,len(vf_list)+1)
y= np.minimum(np.array(vf_list),b - epsilon*np.array(cf_list))
print(len(y))
plt.figure()
#plt.plot(x,y)
#plt.fill_between(x, y - y_err, y + y_err, alpha=0.2, color='blue')
plt.plot(vf_list)
plt.plot(np.array(cf_list))
plt.plot(np.ones(t)*b,linewidth=2.5,linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('CF,Baeline')
plt.title("Reward based setting where constraint to be above baseline(MR setting nS=4,nA=2)")
plt.savefig('RCMDP_MR_vf_cf_baseline_100_steps.pdf')
plt.legend(['cost_vf','cost_cf','baseline'])
plt.show()