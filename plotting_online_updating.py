# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:59:17 2025

@author: Sourav
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle

file1 = "Store_robust_vf_RS_new_way"
file2 = "Store_robust_cf_RS_new_way"
#file3 = "Epi_RC_objective_MR_4s_2a"
#file4 = "Epi_RC_constrainte_MR_4s_2a"

with open(file1,"rb") as f:
    vf_list = pickle.load(f)
f.close()

with open(file2,"rb") as f:
    cf_list = pickle.load(f)
f.close()

'''with open(file3,"rb") as f:
    evf_list = pickle.load(f)
f.close()

with open(file4,"rb") as f:
    ecf_list = pickle.load(f)
f.close()'''

vf_list = np.array(vf_list)
cf_list = np.array(cf_list)

lambda_ = 10
b = 40
x = np.arange(1,len(vf_list)+1)
y = np.max([vf_list/lambda_,(b-cf_list)],axis=0)
plt.figure()
plt.plot(x,y)
plt.plot(vf_list,alpha=0.5)
plt.plot(cf_list,alpha=0.6)
#plt.plot(evf_list,alpha=0.55)
#plt.plot(ecf_list,alpha=0.56)
plt.plot(np.ones(1000)*b)
plt.legend(['max(Vr/lambda,(b-Vc)','vf','cf','baseline'])
plt.title('For lambda=10, RS(6,2)')
plt.savefig('Online_update_RPPG_RS(6,2).pdf')
plt.show()

