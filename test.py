# -*- coding: utf-8 -*-

import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T

import numpy as np


def shared_dataset(data_xy,borrow=True):
    data_x,data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=True)
    return shared_x,T.cast(shared_y,'int32')

print "...... loading dataset ......"

dataset = 'mnist.pkl.gz'
with gzip.open(dataset, 'rb') as f:
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = cPickle.load(f)

train_set_x1,train_set_y1 = train_set
print type(train_set_x1)#<type 'numpy.ndarray'>
print type(train_set_y1)#<type 'numpy.ndarray'>
print train_set_x1.shape#(50000,784)
print train_set_y1.shape#(50000,)

test_set_x1,test_set_y1 = test_set
print type(test_set_x1)#<type 'numpy.ndarray'>
print type(test_set_y1)#<type 'numpy.ndarray'>
        
test_set_x,test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)  
train_set_x, train_set_y = shared_dataset(train_set)  
rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]  

print type(train_set_x)#<class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>
print type(train_set_y)#<class 'theano.tensor.var.TensorVariable'>
print type(test_set_x)#<class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>
print type(test_set_y)#<class 'theano.tensor.var.TensorVariable'>