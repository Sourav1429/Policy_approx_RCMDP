# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:53:04 2025

@author: Sourav
"""

import numpy as np
from Machine_Rep import Machine_Replacement
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity

mr_obj = Machine_Replacement()
ch = 2
P,R,C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(ch),mr_obj.gen_expected_cost()
nS,nA = mr_obj.nS,mr_obj.nA
cost_list = [R,C]
init_dist = np.array([0.8,0.04,0.05,0.11])
pol_eval = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist)
C_KL = 0.5 # what will be this parameter
store=[]

policy = np.zeros((nS,nA))*1/nA
T = 1000
for t in range(T):
    Vr,Vc = pol_eval.evaluate_policy(policy, P, C_KL, 0),pol_eval.evaluate_policy(policy, P, C_KL, 1)
    if(Vr >= b - eps):
        policy = Proj(policy,Vr) ##define this
    else:
        policy = Proj(policy,Vc)
    store.append(np.min(Vr,-np.clip((b-Vc),0,np.inf)))
with open("Store_robust_vf","wb") as f:
    pickle.dump(store,f)
f.close()
