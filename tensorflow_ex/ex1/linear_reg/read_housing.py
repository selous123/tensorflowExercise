#coding:utf-8

from __future__ import division


import sklearn
import numpy as np

from sklearn.model_selection import train_test_split

def read_data_housing():
#    file = open("housing.data.txt",'r');

#    for line in file:
#        print line;
    a = np.loadtxt("housing.data.txt");

    data = a[:,0:-1];
    target = a[:,-1];


    keys = ['data','labels']
    train = {}.fromkeys(keys);
    test = {}.fromkeys(keys);

    train['data'],test['data'],train['labels'],test['labels'] = train_test_split(data,target,
        test_size = 0.4,random_state = 0);
    return train,test
    

if __name__=='__main__':

    read_housing_data();

