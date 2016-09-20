# -*- coding: utf-8 -*-
"""
Softmax回归with Theano
"""
import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T

import numpy as np

""" 
input，输入的一个batch，假设一个batch有n个样本(n_example)，
      则input大小就是(n_example,n_in)  
n_in,每一个样本的大小，MNIST每个样本是一张28*28的图片，故n_in=784  
n_out,输出的类别数，MNIST有0～9共10个类别，n_out=10   
"""
class LogisticRegression():
    def __init__(self,input,n_in,n_out):
        #W:n_in * n_out
        self.W = theano.shared(
            value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
            name='W',
            borrow=True#深度拷贝
        )
        #b:n_out * 1
        #每个输出对应W一列以及b的一个元素 WX+b
        self.b = theano.shared(
            value=np.zeros((n_out,),dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        #input:(n_example*n_in)*W ==> (n_example*n_out)+b
        self.p_y_given_x = T.nnet.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
        self.params = [self.W,self.b]
        self.input = input
    
    """     
    代价函数NLL  
    因为我们是MSGD，每次训练一个batch，一个batch有n_example个样本，则y大小是(n_example,),  
    y.shape[0]得出行数即样本数，将T.log(self.p_y_given_x)简记为LP，  
    则LP[T.arange(y.shape[0]),y]得到
        [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,LP[n-1,y[n-1]]]  
    最后求均值mean，也就是说，minibatch的SGD，是计算出batch里所有样本的NLL的平均值，作为它的cost 
    """
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    def error(self,y):
        # 首先检查y与y_pred的维度是否一样，即是否含有相等的样本数  
    
        if y.ndim != self.y_pred.ndim:  
            raise TypeError(  
                'y should have the same shape as self.y_pred',  
                ('y', y.type, 'y_pred', self.y_pred.type)  
            )  
        # 再检查是不是int类型，是的话计算T.neq(self.y_pred, y)的均值，作为误差率  
        # 举个例子，假如self.y_pred=[3,2,3,2,3,2],而实际上y=[3,4,3,4,3,4]  
        # 则T.neq(self.y_pred, y)=[0,1,0,1,0,1],1表示不等，0表示相等  
        # 故T.mean(T.neq(self.y_pred, y))=T.mean([0,1,0,1,0,1])=0.5，即错误率50%  
        if y.dtype.startswith('int'):  
            return T.mean(T.neq(self.y_pred, y))  
        else:  
            raise NotImplementedError()         
        

"""
载入数据，返回训练集、测试集、验证集的样本和标签
"""
def load_dataset(dataset):
    print "...... loading dataset ......"
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = cPickle.load(f)
    
    def shared_dataset(data_xy,borrow=True):
        data_x,data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=True)
        return shared_x,T.cast(shared_y,'int32')
        
    test_set_x,test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)  
    train_set_x, train_set_y = shared_dataset(train_set)  
  
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),  
            (test_set_x, test_set_y)]  
    return rval

def sgd_optimization_mnist(learning_rate=0.13,n_epochs=1000,
                           dataset='mnist.pkl.gz',batch_size=600):
    datasets = load_dataset(dataset)
    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]

#====TODO===================================
    #print type(train_set_x) #<class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>
    #print type(train_set_y) #<class 'theano.tensor.var.TensorVariable'>
    print type(test_set_x)
    print type(test_set_y)
#===========================================    
    
    #MSGD,计算minibatch个数，一个batch计算一次cost
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
#=====TODO=================
    #print "train set size = {0},f = {1}".format(train_set_x.get_value(borrow=True).shape[0],train_set_x.get_value(borrow=True).shape[1])
    #print "test set size = {0}".format(test_set_x.get_value(borrow=True).shape[0])
    #print "validate set size = {0}".format(valid_set_x.get_value(borrow=True).shape[0])
    #print "n_train_batches = {0}".format(n_train_batches)
    #print "n_valid_batches = {0}".format(n_valid_batches)
    #print "n_test_batches = {0}".format(n_test_batches)
#======================    
    
    print "...... building models ......"
    #index表示minibatch下标
    #x表示训练样本，y表示标签
    index = T.lscalar()
    x = T.matrix("x")
    y = T.ivector("y")
    
    #定义分类器
    classifier = LogisticRegression(x,n_in=28*28,n_out=10)
    cost = classifier.negative_log_likelihood(y)
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.error(y),
        givens={
            x:test_set_x[index * batch_size:(index+1) * batch_size],
            y:test_set_y[index * batch_size:(index+1) * batch_size]             
        }
    )    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.error(y),
        givens={
            x:valid_set_x[index * batch_size:(index+1) * batch_size],
            y:valid_set_y[index * batch_size:(index+1) * batch_size]
        }
    )
     #计算梯度
    g_W = T.grad(cost=cost,wrt=classifier.W)
    g_b = T.grad(cost=cost,wrt=classifier.b)
    #梯度下降法更新
    updates = [(classifier.W,classifier.W - learning_rate * g_W),
               (classifier.b,classifier.b - learning_rate * g_b)]   
    
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )    
   
     
    
    print "...... training model ......."
    
    patience = 5000
    patience_increase = 2
    #提高的阈值，在验证误差减小到之前的0.995倍时，会更新best_validation_loss     
    improvement_threshold = 0.995    
    #这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。  
    validation_frequency = min(n_train_batches, patience / 2)  
                                  
    #最好的验证集上的loss，最好即最小。初始化为无穷大  
    best_validation_loss = np.inf   
    test_score = 0.  
    start_time = time.clock()#执行时间
  
    done_looping = False  
    epoch = 0  

    """
    下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。  
    for循环是遍历一个个batch，一次一个batch地训练。
    for循环体里会用train_model(minibatch_index)去训练模型，  
    train_model里面的updatas会更新各个参数。  
    for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试，  
    如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，  
    则更新best_validation_loss和best_iter，同时在testset上测试。  
    如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。  
    当达到最大步数n_epoch时，或者patience<iter时，结束训练  
    """
    while(epoch<n_epochs) and (not done_looping):
        epoch = epoch + 1
        #minibatch_index表示当前batche的索引，一个epoch有n_train_batches个batch
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch -1) * n_train_batches + minibatch_index
#====================================================
#            print ('epoch =%i,minibatch= %i,iter = %i')%(epoch,minibatch_index,iter)  
#====================================================            
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
                #损失小于之前的最佳的损失best_validation_loss,更新最小损失,并去测试集测试
                if this_validation_loss < best_validation_loss:  
                    save_param(classifier.W,classifier.b)
                    #improve patience if loss improvement is good enough  
                    if this_validation_loss < best_validation_loss * improvement_threshold:  
                        patience = max(patience, iter * patience_increase)  
                    best_validation_loss = this_validation_loss  
                    # test it on the test set  
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]  
                    test_score = np.mean(test_losses)  
                    print(  
                        (  
                            '     epoch %i, minibatch %i/%i, test error of'  
                            ' best model %f %%'  
                        ) %  
                        (  
                            epoch,  
                            minibatch_index + 1,  
                            n_train_batches,  
                            test_score * 100.  
                        )  
                    )  
  
            if patience <= iter:  
                done_looping = True  
                break  
  
    #while循环结束  
    end_time = time.clock()  
    print(  
        (  
            'Optimization complete with best validation score of %f %%,'  
            'with test performance %f %%'  
        )  
        % (best_validation_loss * 100., test_score * 100.)  
    )  
    print 'The code run for %d epochs, with %f epochs/sec' % (  
        epoch, 1. * epoch / (end_time - start_time))  
    print >> sys.stderr, ('The code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % ((end_time - start_time)))  

def save_param(param1,param2):
    import cPickle
    param_file = open('logistic_sgd_params','wb')
    cPickle.dump(param1.get_value(borrow=True),param_file,-1)
    cPickle.dump(param2.get_value(borrow=True),param_file,-1)
    param_file.close()
 
def set_default_value():
     if os.path.exists('logistic_sgd_params'):  
        f=open('logistic_sgd_params')  
        W = cPickle.load(f)  
        b = cPickle.load(f)
        return W,b
        

if __name__ == '__main__':
    """
    dataset='mnist.pkl.gz'
    load_dataset(dataset)
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = cPickle.load(f)
    img = np.asarray(train_set[0][0:1])
    print img.sum()
    """
    sgd_optimization_mnist()