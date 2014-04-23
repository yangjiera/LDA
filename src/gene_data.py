'''
Generate synthetic data set specified in:
    Finding scientific topics
    Thomas L. Griffiths and Mark Steyvers
    PNAS, 2004
    
Created on Apr 15, 2014
@author: jyang3
'''

import numpy as np
import math

def constructPhi(K, V):
    Phi = []
    noRow = int(math.sqrt(V))
    for i in range(K/2):
        multi = np.zeros(V)
        for j in range(noRow):
            multi[i+j*noRow] = float(1)/noRow
        Phi.append(list(multi))
    for i in range(K/2):
        multi = np.zeros(V)
        multi[i*noRow:i*noRow+noRow] = float(1)/noRow
        Phi.append(list(multi))
    return Phi

def gene_synthetic(alpha, K, d, V, noDocs):
    #draw samples from dirichlet distribution
    thetas = np.random.dirichlet(np.ones(K), noDocs)
    #construct the multinomial distributions
    Phi = constructPhi(K, V)
    #construct vocabulary
    vocab = []
    for i in range(V):
        vocab.append(i)
    
    #draw sample documents
    Docs = []
    for i in range(noDocs):
        doc = []
        # STEP1: get the topic distribution, i.e., theta
        theta = thetas[i]
        # STEP2: get the topic assignment, i.e., z
        tp_multi = np.random.multinomial(d, theta)
        # STEP3: draw topic and word
        for topic_index in range(len(tp_multi)):
            topic_occurence = tp_multi[topic_index]
            wd_multi = np.random.multinomial(topic_occurence, Phi[topic_index])
            #print wd_multi
            for word_index in range(len(wd_multi)):
                if wd_multi[word_index] != 0:
                    for j in range(wd_multi[word_index]):
                        doc.append(word_index)
                        
        Docs.append(doc)
        
    return Docs

if __name__ == '__main__':
    K=10
    V=25
    d=100 #number of words in each document
    alpha=1
    noDocs = 2000
    gene_synthetic(alpha, K, d, V, noDocs)