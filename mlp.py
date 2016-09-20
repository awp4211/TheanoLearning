# -*- coding: utf-8 -*-
"""
Multilayer Perceptron
"""
import os  
import sys  
import timeit

import numpy as np 
      
import theano  
import theano.tensor as T  

import lr

class HiddenLayer(object):
    """
    rng:随机数生成器:numpy.random.RandomState，用于初始化W    
    input:隐藏层输入,隐藏层神经元输入(n_example*n_in,每一行一个样本)
    n_in:输入的维
    n_out:隐藏层神经元个数
    W:全连接的权值矩阵 n_in*n_out
    activiation:激活函数T.tanh和T.nnet.sigmoid
    """
    def __init__(self,rng,input,n_in,n_out,
                 W=None,b=None,activation=T.tanh):
        self.input = input
        #init W
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6./(n_in+n_out)),
                    high=np.sqrt(6./(n_in+n_out)),
                    size=(n_in,n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(
                value=W_values,
                name='W',
                borrow=True
            )
        #init b
        if b is None:
            b_values = np.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=b_values,name='b',borrow=True)
        self.W = W
        self.b = b
        #隐藏层输出
        lin_output = T.dot(input,self.W) + b
        self.output =(lin_output if activation is None
                else activation(lin_output))
        self.params = [self.W,self.b]
"""
三层的MLP
"""             
class MLP(object):
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.logRegressionLayer = lr.LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = (np.abs(self.hiddenLayer.W).sum() +
                    np.abs(self.logRegressionLayer.W.sum()))
        self.L2_sqr = ((self.hiddenLayer.W ** 2).sum()+ 
                       (self.logRegressionLayer.W ** 2).sum())
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.error
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


"""
learning_rate:学习率
L1_reg,L2_reg:L1,L2正则化洗漱，权衡正则化项与Nll项的比重
n_epochs:迭代的最大次数
dataset:数据集路径
n_hidden:隐藏层神经元个数
batch_size:批量大小
"""
def test_mlp(learning_rate=0.01,L1_reg=0.00,L2_reg=0.0001,n_epochs=10,
             dataset='mnist.pkl.gz',batch_size=20,n_hidden=500):
    datasets = lr.load_dataset(dataset)
    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    print '...... building model ......'    
    
    index = T.iscalar()
    x = T.matrix("x")
    y = T.ivector("y")
    
    rng = np.random.RandomState(1234)
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out=10
    )
    
    cost = (classifier.negative_log_likelihood(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr)
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x:test_set_x[index*batch_size:(index+1)*batch_size],
            y:test_set_y[index*batch_size:(index+1)*batch_size]
        }    
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x:valid_set_x[index*batch_size:(index+1)*batch_size],
            y:valid_set_y[index*batch_size:(index+1)*batch_size]
        }    
    )
    gparams = [T.grad(cost,param) for param in classifier.params]
    #zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表
    updates = [(param,param - learning_rate*gparam)
                for param,gparam in zip(classifier.params,gparams)]
    train_model=theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x:train_set_x[index*batch_size:(index+1)*batch_size],
            y:train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )
    
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    print '...... training model ......'    
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    test_mlp()