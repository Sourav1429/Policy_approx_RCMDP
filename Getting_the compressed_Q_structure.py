# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:33:40 2025

@author: Sourav
"""

import torch
from Machine_Rep import Machine_Replacement
mr_obj = Machine_Replacement()
nS,nA = mr_obj.nA,mr_obj.nA
P = mr_obj.gen_probability()
cn = torch.tensor(mr_obj.gen_expected_reward(ch=2),dtype=torch.float32)
Q = torch.zeros((nS,nA))

T = torch.zeros((nS,nS))

policy = torch.ones((4,2))*0.5

for s in range(nS):
    for s_next in range(nS):
        T[s,s_next] = torch.sum(torch.tensor([policy[s,a]*P[a,s,s_next] for a in range(nA)]))

I = torch.eye(nS)
Q = torch.mm(torch.linalg.inv(I-T),cn)

print(Q)