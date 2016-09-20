# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy

import os
import sys
import timeit

from lr import LogisticRegression,load_dataset

import PIL
from PIL import Image

from utils import tile_raster_images

class dA(object):
    """
    numpy_rng:numpy.random.RandomStreams
    theano_rng:theano.tensor.shared_randomstreams.RandomStreams
    W:(n_visible*n_hidden) W'=W
    """
    def __init__(self,numpy_rng,theano_rng=None,input=None,
                 n_visible=784,n_hidden=500,
                 W=None,bhid=None,bvis=None):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4*numpy.sqrt(6./(n_visible + n_hidden)),
                    high=4*numpy.sqrt(6./(n_visible + n_hidden)),
                    size=(n_visible,n_hidden)                
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W,borrow=True,name='W')
        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                borrow=True            
            )
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.W = W
        # b 和隐藏层的bias关联
        self.b = bhid
        # b_prime和输入层的bias关联
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # DA可以在不给定输入的情况下训练            
        if not input:
            self.x = T.dmatrix("x")
        else:
            self.x = input
        self.params = [self.W,self.b,self.b_prime]
        
    """
    计算隐藏层输出
    """
    def get_hidden_values(self,input):
        return T.nnet.sigmoid(T.dot(input,self.W) + self.b)
    """
    计算重构层输出
    """
    def get_reconstructed_input(self,hidden):
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime) + self.b_prime)

    """
     The binomial function return int64 data type by default.
     int64 multiplicated by the input type(floatX) always return float64.  
     To keep all data in floatX when floatX is float32, we set the dtype of
     the binomial to floatX. As in our case the value of the binomial is always
     0 or 1, this don't change the result. This is needed to allow the gpu to work
     correctly as it only support float32 for now.
    """
    def get_corrupted_input(self,input,corruption_level):
        return self.theano_rng.binomial(size=input.shape,n=1,
            p=1-corruption_level,
            dtype=theano.config.floatX)*input

    def get_cost_update(self,corruption_level,learning_rate):
        tiled_x = self.get_corrupted_input(self.x,corruption_level)
        y = self.get_hidden_values(tiled_x)
        z = self.get_reconstructed_input(y)
        #批量数据minibatch下L（交叉熵损失）是一个向量        
        L = -T.sum(self.x * T.log(z) + (1. - self.x)* T.log(1-z),axis=1)
        cost = T.mean(L)
        
        #计算梯度
        gparams = T.grad(cost,self.params)
        updates = [
            (param,param - learning_rate * gparam)
            for param,gparam in zip(self.params,gparams)
        ]
        return cost,updates
    
def test_DA(learning_rate=0.1,training_epochs=15,
            dataset='mnist.pkl.gz',batch_size=20,output_folder='DA_plots'):
    datasets = load_dataset(dataset)
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    index = T.lscalar()
    x = T.matrix('x')
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

#==============================================================================
#================================Autoencoders =================================  
    print '...... building corruption = 0.0 model ......'  
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))
    
    da = dA(numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_hidden=500,
            n_visible=28*28
            )
    cost,updates = da.get_cost_update(corruption_level=0.,
                                      learning_rate=learning_rate)
    train_da = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x:train_set_x[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    print '...... training corruption = 0.0 ......'
    
    start_time = timeit.default_timer()
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print 'Training epoch %d,cost = '%epoch,numpy.mean(c)
        
    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print "###################"
    print da.W.shape
    
    print >> sys.stderr, ('The 0% corruption code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % (training_time))  
           
    image = Image.fromarray(
        tile_raster_images(
            X=da.W.get_value(borrow=True).T,
            img_shape=(28, 28), 
            tile_shape=(20, 20),
            tile_spacing=(1, 1)
        )
    )
    image.save('filters_corruption_0.png')
    
#==============================================================================
#========================Denoising Autoencoder=================================    
    
    print '...... building corruption = 0.3 model ......'  
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_update(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    print '...... training corruption = 0.0 ......'
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print('Training epoch %d, cost ' % epoch, numpy.mean(c))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)


    print >> sys.stderr, ('The 30% corruption code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % (training_time))  
    image = Image.fromarray(
        tile_raster_images(
            X=da.W.get_value(borrow=True).T,
            img_shape=(28, 28), 
            tile_shape=(20, 20),
            tile_spacing=(1, 1)
        )
    )
    image.save('filters_corruption_30.png')

    os.chdir('../')
    
if __name__== '__main__':
    test_DA(training_epochs=5)