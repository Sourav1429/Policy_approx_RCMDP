# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:53:04 2025

@author: Sourav
"""

import numpy as np
from Machine_Rep import Machine_Replacement
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
import torch
import pickle

def Proj(policy,V,Pi):
    policy = policy + torch.gradient(V)
    smallest_distance = np.argmin([np.norm(policy-pi) for pi in Pi])
    return Pi[smallest_distance]
    
mr_obj = Machine_Replacement()
ch = 2
P,R,C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(ch),mr_obj.gen_expected_cost()
nS,nA = mr_obj.nS,mr_obj.nA
cost_list = [R,C]
init_dist = np.array([0.8,0.04,0.05,0.11])
pol_eval = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist)
C_KL = 0.5 # what will be this parameter
store=[]
eps = 0.01
b = 3

policy = np.zeros((nS,nA))*1/nA
policy_space= []
n_pol = np.power(nA,nS)
vf_store = []
for i in range(n_pol):
    policy_space.append(list(map(int,format(i, '04b'))))
policy_space = np.array(policy_space)
T = 1000
for t in range(T):
    Vr,Vc = pol_eval.evaluate_policy(policy, P, C_KL, 0),pol_eval.evaluate_policy(policy, P, C_KL, 1)
    if(Vc >= b - eps):
        policy = Proj(policy,Vr,policy_space) ##define this
    else:
        policy = Proj(policy,Vc,policy_space)
    store.append(np.min(Vr,-np.clip((b-Vc),0,np.inf)))
    vf_store.append(Vr)
print(np.argmax(store))
with open("Store_robust_output","wb") as f:
    pickle.dump(store,f)
f.close()

with open("Store_robust_vf","wb") as f:
    pickle.dump(vf_store,f)
f.close()
