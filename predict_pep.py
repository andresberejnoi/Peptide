#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:55:39 2017

@author: Andres Berejnoi
"""
import NetBuilder as nb
import numpy as np
import sys
from sklearn.preprocessing import normalize as norm

def extract_data(filename):
    with open(filename) as f:
        data = np.loadtxt(f,delimiter=',',skiprows=1)
        
        #The samples should be mixed more randomly to improve convergence
        np.random.shuffle(data)
        
        #normalize the data
        data = norm(data,axis=0)  #with axis=0, each column in normalized independently of each other
        
        #split inputs from target outputs (assume last column corresponds to labels or targets)
        inputs = data[:,:-1]
        targets = data[:,-1] #
        targets = np.reshape(a=targets,newshape=(len(targets),1))           #targets should have the shape of [number of sumples x number of output features per sample]
        
        #array_change_zeros = np.vectorize(change_zeros)
        #targets = array_change_zeros(targets)           #Here I am changing 0s to -1s because I intend to use the hyperbolic tangent function for the final layer of the network and it works better this way
        
        
    return inputs, targets



def change_zeros(num):
    if num==0:
        return -1.
    else:
        return 1
    
def get_one_sample(data_inputs,data_targets,idx=None):
    if idx==None:
        idx = np.random.randint(0,len(data_inputs))
    sample_in = data_inputs[idx:idx+1]
    sample_tar = data_targets[idx:idx+1]
    
    return sample_in,sample_tar
        
#First read the training data:
#data_file = sys.argv[1]
if __name__=='__main__':
    array_change_zeros = np.vectorize(change_zeros)         
    
    data_file = sys.argv[1]
    #data_file = 'curated_train_set.csv'
    inputs,targets = extract_data('curated_train_set.csv')
    
    #define the network structure:
    input_features = len(inputs[0])         #the number of columns in the data matrix (this will be the number of input nodes to the network's first layer)
    output_features = len(targets[0])       #this should be 1 (this will be the number of output nodes to the last layer)
    
    #create the model
    topology = [input_features,5,output_features]
    try:
        max_epochs = int(sys.argv[4])
    except:
        max_epochs = 1000    
    batch_size = int(sys.argv[2])
    print_rate = int(sys.argv[3])
    net = nb.Network(topology=topology,learningRate=0.01)
    net.set_hiddenactivation_fun('sigmoid')
    net.set_outActivation_fun('sigmoid')
    net.train(input_set=inputs,
              target_set=targets,
              epochs=max_epochs,
              batch_size=batch_size,
              print_rate=print_rate)
    
    #Saved the state of the network:
    net.save('peptideNet.csv')
    print('='*80,'\nTEST\n:')
    test_in,test_tar = get_one_sample(inputs,targets)
    test_out = net.feedforward(test_in)
    print('Input:',test_in,'\nExpected:\tActual Output\n',test_tar,test_out)
     
