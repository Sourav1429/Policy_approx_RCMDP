# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:00:27 2025

@author: Sourav
"""

import numpy as np

class Robust_pol_Kl_uncertainity:
    def __init__(self,nS,nA,cost_list,init_dist):
        self.nS = nS
        self.nA = nA
        self.cost_list = cost_list
        self.init_dist = init_dist
        self.gamma = 0.995
    def calculate_infinite_Q(self,n,policy,P,C_KL):
        C = self.cost_list[n]
        Q = np.zeros(self.nS,self.nA)
        V = np.zeros(self.nS)
        tau = 1000
        s = np.random.choice(self.nS,p=self.init_dist)
        for t in range(tau):
            a = np.random.choice(self.nA,p = policy[s])
            next_state = np.random.choice(self.nS,p=P[a,s,:])
            P_star = np.aray([P[a,s,i]*np.exp(V[i]/C_KL) for i in range(self.nS)])
            Q[s,a] = C[s,a] + self.gamma * np.dot(P_star,V)
            V = np.array([np.dot(policy[s],Q[s,a]) for s in range(self.nS)])
            s = next_state
        return Q,V
    def evaluate_policy(self,policy,P,C_KL,n):
        Q,V = self.calculate_infinite_Q(n, policy, P, C_KL)
        P_star = np.zeros((self.nS,self.nA,self.nS))
        Pi_pi = np.zeros((self.nS,self.nS,self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                P_star[s,a,:] = np.array([P(a,s,i)*np.exp(V[i]/C_KL) for i in range(self.nS)])
            Pi_pi[s,s,:] = policy[s,:] 
        inv = np.linalg.inv(np.eye(self.nS) - self.gamma*np.matmul(P_star,Pi_pi))
        Q_ = np.matmul(inv,self.cost_list[n])
        J = np.sum([self.init_dist[s]*np.sum([policy[s,a]*Q_[s,a] for a in range(self.nA)]) for s in range(self.nS)])
        return J
        