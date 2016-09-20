# -*- coding: utf-8 -*-
"""
Stacked Denoising AutoEncoders
"""
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from lr import LogisticRegression, load_dataset
from mlp import HiddenLayer
from denoising_autoencoders import dA

class sdA(object):
    """
    n_ins:输入层的维度（第一层为数据输入层）
    hidden_layer_sizes:中间层大小
    n_out:网络的输出大小
    corruption_levels:每层的corruption程度
    """
    def __init__(self,
                numpy_rng,
                theano_rng=None,
                n_ins=784,
                hidden_layer_sizes=[500,500],
                n_out=10,
                corruption_levels=[0.1,0.1]
                ):
        self.sigmoid_layers =[]
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)#网络层数
        
        assert self.n_layers > 0                    
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.x = T.matrix("x")#图像序列
        self.y = T.ivector("y")#标签

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i-1]
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            """HIddenLayer
            def __init__(self,rng,input,n_in,n_out,
                 W=None,b=None,activation=T.tanh):
            """
            sigmoid_layer = HiddenLayer(
                        rng=numpy_rng,
                        input=layer_input,
                        n_in=input_size,
                        n_out=hidden_layer_sizes[i],
                        activation=T.nnet.sigmoid
                        )
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            #创建一个dA与HiddenLayer共享权值
            """dA
            def __init__(self,numpy_rng,theano_rng=None,input=None,
                 n_visible=784,n_hidden=500,
                 W=None,bhid=None,bvis=None):
            """
            dA_layer = dA(
                        numpy_rng=numpy_rng,
                        theano_rng=theano_rng,
                        input=layer_input,
                        n_visible=input_size,
                        n_hidden=hidden_layer_sizes[i-1],
                        W=sigmoid_layer.W,
                        bhid=sigmoid_layer.b
            )
            self.dA_layers.append(dA_layer)
        """LogisticRegression
        def __init__(self,input,n_in,n_out):
        """
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_out
        )
        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining
        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.error(self.y)
        
    """
    下面这个函数是为!!!每i层!!!生成一个Theano function类型的预训练函数，
    返回值是一个list，list中就是每层的预训练函
    为了能在训练中更改 the corruption level or the learning rate，
    我们定义Theano variables。
    """    
    def pretraining_functions(self,train_set_x,batch_size):
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        for dA in self.dA_layers:
            cost,updates = dA.get_cost_update(corruption_level,learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level,value=0.2),
                    theano.In(learning_rate,value=0.1)                        
                    ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x:train_set_x[batch_begin:batch_end]                
                }
            )
            pretrain_fns.append(fn)
        return pretrain_fns            
    
    """
    创建微调函数
    Once all layers are pre-trained, 
    the network goes through a second stage of training called fine-tuning.
    降噪自编码完成之后，网络进行监督学习得到误差再微调参数
    """
    def build_finetune_functions(self,datasets,batch_size,learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]//batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]//batch_size
        index = T.lscalar('index')
        gparams = T.grad(self.finetune_cost,self.params)
        updates=[
            (param,param - gparam * learning_rate)
            for param,gparam in zip(self.params,gparams)
        ]
        
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]            
            },
            name='train'
        )
        
        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]            
            },
            name='test'
        )
        
        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            },
            name='valid'
        )
        
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]
            
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]
            
        return train_fn,valid_score,test_score
        
def test_sdA(finetune_lr=0.1,pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):
    datasets = load_dataset(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    numpy_rng = numpy.random.RandomState(89677)
    
    print '...... building model ......'
    """sdA
     def __init__(self,
                numpy_rng,theano_rng=None,n_ins=784,hidden_layer_sizes=[500,500],
                n_out=10,corruption_levels=[0.1,0.1]):
    """
    sda = sdA(numpy_rng=numpy_rng,n_ins=28*28,
        hidden_layer_sizes=[1000,1000,1000],
        corruption_levels=[0.1,0.1,0.1],
        n_out=10            
    )
    
#==============================================================================
#============首先进行无监督学习，即根据给定的隐藏层节点个数构建自编码网络================
#==============================================================================
    print('...... getting the pretraining functions ......')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    print('...... pre-training the model......')
    start_time = timeit.default_timer()
    corruption_levels = [.1, .2, .3]
    
    for i in range(sda.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](
                         index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()    
    
    
    print >> sys.stderr, ('The pretraining code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % (end_time - start_time))  

#==============================================================================
#========进行监督学习，根据创建好的自编码网络，结合MLP和Softmax回归训练网络参数==========
#==============================================================================    
    print('...... getting the finetuning functions .......')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('...... finetunning the model.......')
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()                      

#==============================================================================
#======================All Done================================================
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % (end_time-start_time))  

if __name__ == '__main__':
    test_sdA()