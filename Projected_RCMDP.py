# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:53:04 2025

@author: Sourav
"""

import numpy as np
from Machine_Rep import Machine_Replacement
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
#import torch
import pickle

def onehot(policy_space,nS,nA):
    ret_pol = []
    for pol in policy_space:
        policy = np.zeros((nS,nA))
        for s,j in enumerate(pol):
            policy[s,j]=1
        ret_pol.append(policy)
    return np.array(ret_pol,dtype=np.int16)

def Proj(policy,V,Pi,grad):
    alpha = 0.001
    policy = policy + alpha* grad
    smallest_distance = np.argmin([np.linalg.norm(policy-pi) for pi in Pi])
    return Pi[smallest_distance]
    
mr_obj = Machine_Replacement()
ch = 2
alpha = 0.000001
P,R,C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(ch),mr_obj.gen_expected_cost()
nS,nA = mr_obj.nS,mr_obj.nA
cost_list = [R,C]
init_dist = np.array([0.8,0.04,0.05,0.11])
pol_eval = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist,alpha)
C_KL = 0.5 # what will be this parameter
store=[]
eps = 0.01
b = 3

#####Remember to convert policy to one_hot encoding
policy_space= []
n_pol = np.power(nA,nS)
vf_store = []
cf_store = []
for i in range(1,n_pol):
    policy_space.append(list(map(int,format(i, '04b'))))
policy_space = np.array(policy_space)
policy_space = onehot(policy_space,nS,nA)
choice_of_policy = np.random.choice(len(policy_space))
policy = policy_space[choice_of_policy]
#print(policy_space)
with open("nominal_model","rb") as f:
    P_nominal = pickle.load(f)
f.close()
T = 1000
for t in range(T):
    Vr,gradr = pol_eval.evaluate_policy(policy, P_nominal, C_KL, 0,t)
    Vc,gradc = pol_eval.evaluate_policy(policy, P_nominal, C_KL, 1,t)
    if(Vc >= b - eps):
        policy = Proj(policy,Vr,policy_space,gradr) ##define this
    else:
        policy = Proj(policy,Vc,policy_space,gradc)
    #print("New policy:",policy)
    #store.append(np.min(Vr,-np.clip((b-Vc),0,np.inf)))
    vf_store.append(Vr)
    cf_store.append(Vc)
    #print("One step done")
#print(np.argmax(store))
with open("Store_robust_output","wb") as f:
    pickle.dump(store,f)
f.close()

with open("Store_robust_vf","wb") as f:
    pickle.dump(vf_store,f)
f.close()

with open("Store_robust_cf","wb") as f:
    pickle.dump(cf_store,f)
f.close()
