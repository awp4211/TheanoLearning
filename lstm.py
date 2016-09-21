# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

datasets = {'imdb':(imdb.load_data,imdb.prepare_data)}


SEED = 123
numpy.random.seed(SEED)

# 将数据转换为theano.config.floatX类型
def numpy_floatX(data):
    return numpy.asarray(data,dtype=config.floatX)
    
# 打乱数据
def get_minibatches_idx(n,minibatch_size,shuffle=False):
    idx_list = numpy.arange(n,dtype="int32")
    if shuffle:
        numpy.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
def get_dataset(name):
    return datasets[name][0],datasets[name][1]

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params
    
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def _p(pp, name):
    return '%s_%s' % (pp, name)
    
"""
初始化全部参数，不仅是LSTM的参数,还有用于embeding和classifer的参数
option传递的是train_lstm中的参数 encoder='lstm'
"""
def init_params(options):
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    # Wemb为(n_words,dim_proj)每行为一个词的词向量
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    # get_layer(options['encoder'])[0]返回param_init_lstm函数
    # options['encoder']=lstm
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder']
                                              )
    # 输出后用于softmax分类器回归
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(
                                                config.floatX
                                                )
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
    return params
    
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

"""
将参数转化为GPU变量(shared variables)
"""
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

"""
返回layer下的函数，name
"""
def get_layer(name):
    # layers = {'lstm':(param_init_lstm,lstm_layer)}
    fns = layers[name]
    return fns
    
#随机生成 ndim * ndim 的二维矩阵之后进行奇异值分解，得到的u为每个词对应的向量
def ortho_weight(ndim):
    W = numpy.random.randn(ndim,ndim)
    u,s,v = numpy.linalg.svd(W)
    return u.astype(config.floatX)
    
    
"""
Init the LSTM parameter:
初始化LSTM的参数，W有四个，U有四个，b有四个
"""    
def param_init_lstm(options,params,prefix='lstm'):
    # numpy 的 concatenate函数用于按照某个轴连接矩阵
    # W对应LSTM的四个W矩阵，分别为W_i,W_c,W_f,W_o
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    # U对应LSTM的四个U矩阵,分别为U_i,U_c,U_f,U_o
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params    

"""
LSTM层的参数 tparams为初始化的shared variables
    state_below 为 Word Embeding 的向量
"""
def lstm_layer(tparams,state_below,options,prefix='lstm',mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    
    assert mask is not None

    # 用于切分lstm初始化的四个W和四个U
    def _slice(_x,n,dim):
        if _x.ndim == 3:
            return _x[:,:,n * dim:(n+1)*dim] 
        return _x[:,n*dim:(n+1)*dim]
    
    def _step(m_,x_,h_,c_):
        # 由于LSTM计算公式中均为 U_i*h(t-1),U_c*h(t-1),U_f * h(t-1),U_o*h(t-1)
        # 所以可以并行化计算这四个值
        preact = tensor.dot(h_,tparams[_p(prefix,'U')])
        preact += x_
        
        # i(t) = sigmoid(W_i * x(t) + U_i * h(t-1) +b_i)
        i = tensor.nnet.sigmoid(_slice(preact,0,options['dim_proj']))
        # f(t) = sigmoid(W_f * x(t) + U_f * h(t-1) + b_f)
        f = tensor.nnet.sigmoid(_slice(preact,1,options['dim_proj']))
        # o(t) = sigmoid(W_o * x(t) + U_o * h(t-1) + b_o)
        o = tensor.nnet.sigmoid(_slice(preact,2,options['dim_proj']))
        # C^ = tanh(W_c*x(t)+U_c * h(t-1) + b_c)
        c = tensor.tanh(_slice(preact,3,options['dim_proj']))
        # C(t) = i(t) * C^ + f(t) * C(t-1)
        c = i * c + f * c_
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]    

layers = {'lstm': (param_init_lstm, lstm_layer)}

def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update 
    
def build_model(tparams,options):
    trng = RandomStreams(SEED)
    
    # Dropout
    use_noise = theano.shared(numpy_floatX(0.))
    
    x = tensor.matrix('x',dtype='int64')
    mask = tensor.matrix('mask',dtype=config.floatX)
    y = tensor.vector('y',dtype='int64')
    
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    # get_layer('lstm')[1]=lstm_layer
    proj = get_layer(options['encoder'])[1](tparams,emb,options,
                                            prefix=options['encoder'],
                                            mask=mask)
    
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:,:,None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()
    return use_noise, x, mask, y, f_pred_prob, f_pred, cost
    
    
def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err
    

def train_lstm(
    dim_proj=128,# word embeding dimension and LSTM number of hidden units(词向量维度)
    patience=10,# NUmber of epoch to wait before early stop if no progress(cost不减少则提前停止训练)
    max_epochs=5000,
    dispFreq=10,# 显示信息的间隔
    decay_c=0.,# weight decay
    lrate=0.0001,# Learning rate for sgd(NOT for adadelta and rmsprop)
    n_words=10000,# Vocabulary size 词库单词数
    optimizer=adadelta,
    encoder='lstm',
    saveto='lstm_model.npz',# best model will be saved there
    validFreq=370,# Compute the validation error after this number of update
    saveFreq=1110,# Save the parameters after every  saveFreq updates
    maxlen=100,# Sequence longer then this get ignored(最大句子长度)
    batch_size=16,
    valid_batch_size=64,
    dataset='imdb',
    
    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,
    reload_model=None,
    test_size=-1,
):
    # locals记录函数传递的参数和函数中的变量，返回一个字典
    model_options = locals().copy()
    print("model options",model_options)    
    load_data,prepare_data = get_dataset(dataset)
    
    print('...... loading data .......')
    train,valid,test = load_data(n_words=n_words,valid_portion=0.05,maxlen=maxlen)
    
    if test_size>0:
        # The test set is sorted by size, but we want to keep random size example.  
        # So we must select a random selection of the examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    
    ydim = numpy.max(train[1])+1 # 总共的句子个数
    model_options['ydim'] = ydim
    
    print('...... building model ......')
    # This create the initial parameters as numpy ndarrays
    # Dict name (string)->numpy ndarray
    params = init_params(model_options)
    
    if reload_model:
        load_params('lstm_model.npz',params)
    
    # This create Theano Shared Variable from the parameters.
    # Dict name(String)-> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights
    tparams = init_tparams(params)
    
    # use_noise is for dropout
    (use_noise,x,mask,
     y,f_pred_prob,f_pred,cost) = build_model(tparams,model_options)
     
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('...... Optimization ......')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break
            print('Seen %d samples' % n_samples)
            if estop:
                break
    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took {0}s'.format(end_time - start_time)))
    return train_err, valid_err, test_err

if __name__ == '__main__':
    train_lstm(
        max_epochs=100,
        test_size=500
    )