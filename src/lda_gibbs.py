'''
Python Implementation of LDA Using Gibbs Sampling.
Created on Apr 16, 2014
@author: Jie Yang [j.yang-3@tudelft.nl]
Todo: alpha, beta != 1
'''

import numpy as np
import sys

def gibbs_init(docs, K, V):
    m, n = np.array(docs).shape
    NDZ = np.zeros((m,K))
    NZW = np.zeros((K,V))
    NZ = np.zeros(K)
    tp_assign = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            this_word = docs[i][j]
            init_topic = list(np.random.multinomial(1, [float(1)/K]*K))
            this_topic = init_topic.index(1)
            # initialization
            tp_assign[i,j] = this_topic
            NDZ[i,this_topic] += 1
            NZW[this_topic, this_word] += 1
            NZ[this_topic] += 1
            
    return NDZ, NZW, NZ, tp_assign

def gibbs_update(NDZ, NZW, NZ, docs, tp_assign, alpha, beta, MaxIter):
    K,V = NZW.shape
    m, n = np.array(docs).shape
    
    OldPhi = np.zeros((K,V))
    for i in range(K):
        for j in range(V):
            OldPhi[i,j] = float(NZW[i,j]+beta)/(NZ[i]+V)
    
    NoIteration = 0
    while True:
        NoIteration += 1
        if NoIteration%10 == 0:
            print '..'+str(NoIteration)+'th iteration.'
        if NoIteration>MaxIter:
            break
        NewPhi = np.zeros((K,V))
        
        for i in range(m):
            for j in range(n):
                this_word = docs[i][j]
                this_topic = tp_assign[i,j]
                # decrease counts
                NDZ[i,this_topic] -= 1
                NZW[this_topic, this_word] -= 1
                NZ[this_topic] -= 1
                # update topic
                new_topic = -1
                this_new_multi = [None]*K
                for k in range(K):
                    this_new_multi[k] = (NZW[k, this_word]+beta)*(NDZ[i,k]+alpha)/(NZ[k]+V)
                this_new_multi= this_new_multi/np.linalg.norm(this_new_multi,1)
                if np.linalg.norm(this_new_multi,1)==0:
                        print this_new_multi
                try:
                    new_topic = (list(np.random.multinomial(1, this_new_multi))).index(1)
                except:
                    print '...[error]: posterior probabilities do not sum to 1'
                    sys.exit(1)
                # increase counts
                NDZ[i,new_topic] += 1
                NZW[new_topic, this_word] += 1
                NZ[new_topic] += 1
                tp_assign[i,j] = new_topic
                
        for i in range(K):
            for j in range(V):
                NewPhi[i,j] = float(NZW[i,j]+beta)/(NZ[i]+V)
        diff = np.max(np.fabs(OldPhi-NewPhi))
        if diff<0.0001:
            break
        else:
            OldPhi = NewPhi
            print '..gap: '+str(diff)
            
    return NewPhi, NDZ, NZW, NZ

def gibbs_learn(docs, K, MaxIter, alpha = 1, beta = 1):
    flat_docs = set([item for sublist in docs for item in sublist])
    V = len(flat_docs)
    
    NDZ, NZW, NZ, tp_assign = gibbs_init(docs, K, V)
    Phi, NDZ, NZW, NZ = gibbs_update(NDZ, NZW, NZ, docs, tp_assign, alpha, beta, MaxIter)
    
    return Phi