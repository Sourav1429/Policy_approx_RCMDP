# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:00:27 2025

@author: Sourav
"""

import numpy as np
import torch

class Robust_pol_Kl_uncertainity:
    def __init__(self,nS,nA,cost_list,init_dist):
        self.nS = nS
        self.nA = nA
        self.cost_list = cost_list
        self.init_dist = init_dist
        self.gamma = 0.995
    def calculate_infinite_Q(self,n,policy,P,C_KL):
        C = self.cost_list[n]
        Q = torch.zeros((self.nS,self.nA),requires_grad=True)
        V = torch.zeros(self.nS,requires_grad=True)
        tau = 1000
        s = np.random.choice(self.nS,p=self.init_dist)
        for t in range(tau):
            a = np.random.choice(self.nA,p = policy[s])
            next_state = np.random.choice(self.nS,p=P[a,s,:])
            P_star = torch.tensor([P[a,s,i]*np.exp(V[i]/C_KL) for i in range(self.nS)],requires_grad = True)
            Q[s,a] = C[s,a] + self.gamma * torch.mm(P_star,V)
            V = torch.tensor([torch.dot(policy[s],Q[s,a]) for s in range(self.nS)],requires_grad=True)
            s = next_state
        return Q,V
    def evaluate_policy(self,policy,P,C_KL,n):
        Q,V = self.calculate_infinite_Q(n, policy, P, C_KL)
        P_star = torch.zeros((self.nS,self.nA,self.nS))
        #Pi_pi = torch.zeros((self.nS,self.nS,self.nA))
        Q_ = torch.zeros((self.nS,self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                P_star[s,a,:] = torch.tensor([P(a,s,i)*np.exp(V[i]/C_KL) for i in range(self.nS)],requires_grad=True)
                Q_[s,a] = self.cost_list[n][s,a] + self.gamma*torch.sum([P_star[s,a,s_next]*torch.sum([policy[s_next,a_next]*Q_[s_next,a_next] for a_next in range(self.nA)]) for s_next in range(self.nS)])#not correct
        #inv = np.linalg.inv(np.eye(self.nS) - self.gamma*np.matmul(P_star,Pi_pi))
        #Q_ = np.matmul(inv,self.cost_list[n])
        J = torch.sum([self.init_dist[s]*torch.sum([policy[s,a]*Q_[s,a] for a in range(self.nA)]) for s in range(self.nS)])
        return J
        