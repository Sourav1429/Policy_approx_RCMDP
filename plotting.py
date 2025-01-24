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

with open("Store_robust_vf","rb") as f:
    vf_list = pickle.load(f)
f.close()
vf_list.insert(0,0)

with open("Store_robust_cf","rb") as f:
    cf_list = pickle.load(f)
f.close()
cf_list.insert(1,175)
b= 3
epsilon = 0.01
#plt.plot(vf_list)
x= np.arange(len(vf_list))
y_err = epsilon*50/np.arange(1,len(vf_list)+1)
y= np.minimum(np.array(vf_list),b - epsilon*np.array(cf_list))
print(len(y))
plt.plot(x,y)
plt.fill_between(x, y - y_err, y + y_err, alpha=0.2, color='blue')
plt.xlabel('Iteration')
plt.ylabel('min(V,(b-epsilon*Vc))')
plt.title("Plot showing the change in min(V,(b-epsilon*Vc))")
plt.savefig('Projected_RCMDP.pdf')
plt.show()