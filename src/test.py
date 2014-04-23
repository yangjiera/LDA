'''
Created on Apr 17, 2014

@author: jyang3
'''
from gene_data import *
from lda_gibbs import *

if __name__ == '__main__':
    docs = gene_synthetic(1, 10, 100, 25, 1000)
    phi = gibbs_learn(docs, 10, 200)
    for i in range(10):
        print phi[i, :]