import theano
import theano.tensor as T 
from theano.tensor.nnet import conv
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample

import numpy as np


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, activation = Tanh, W=None, b=None,
                 use_bias=True):

        self.input = input
        self.activation = activation

        if W is None:            
            if activation.func_name == "ReLU":
                W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
            else:                
                W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)),
                                                     size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


class LogisticRegression(object):
    

    def __init__(self, input, n_in, n_out, W=None, b=None):
        
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.y_score = self.p_y_given_x

        self.params = [self.W, self.b]
        self.test = 0


      
    def negative_log_likelihood(self, y):
                
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    
    def L1_kernel_loss(self, y, weight, l):
        
        self.test = T.log(self.p_y_given_x)

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + l * abs(weight).sum()

    def errors(self, y):
       
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
           
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



class CNNLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize = (1, 1)):
        
        self.input = input  

        fan_in = np.prod(filter_shape[1:]) 
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /  
                   np.prod(poolsize))  


        W_bound = np.sqrt(6. / (fan_in + fan_out)) 
       
        self.W = theano.shared(  
            np.asarray(  
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),  
                dtype=theano.config.floatX  
            ),  
            borrow=True  
        )  

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)  
        self.b = theano.shared(value=b_values, borrow=True)  

        conv_out = conv.conv2d(  
            input=input,  
            filters=self.W,  
            filter_shape=filter_shape,  
            image_shape=image_shape, 
            # border_mode='full' 
        )  

    
        pooled_out = downsample.max_pool_2d(  
            input=conv_out,  
            ds=poolsize,  
            ignore_border=True  
        )  
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))  

        self.params = [self.W, self.b]  


class Density(object):
    """
    This class constructs density matrix representation for a batch sentences 
    The density is symmetric, positive semidefinite, and of trace 1.
    This class ensures the first two properties, and the third is limited by penalty term in the loss function
    The density matrices are obtained by the sum of the outer product of embedding vectors
    W is the weights of each postion in the sentence
    smooth_term is to avoid 0 in the denominator
    """
    def __init__(self, rng, input, in_len, max_len):
       
        smooth_term = theano.shared(value= np.asarray(0.0001, dtype=theano.config.floatX))
        # smooth_term = theano.shared(0.0001)
        self.W = theano.shared(
                value= np.ones(max_len, dtype=theano.config.floatX),
                name='W')
        W_effection = self.W

        projectors, _ = theano.scan(fn = lambda projector, coef, somooth: coef * \
            T.outer(projector, projector) / (T.dot(projector, projector) + somooth) , \
            outputs_info = None, sequences = [input.reshape((input.shape[0] * \
                input.shape[1], input.shape[2])), W_effection.repeat(input.shape[0])\
            .reshape((input.shape[1], input.shape[0])).T.flatten()], \
            non_sequences = [smooth_term])

        self.output = T.tanh(T.sum(projectors.reshape((input.shape[0], input.shape[1], \
            input.shape[2], input.shape[2])), axis = 1))

        self.params = [self.W]

class Density1(object):
    
    def __init__(self, rng, input, in_len, max_len):
       
        smooth_term = theano.shared(value= np.asarray(0.0001, dtype=theano.config.floatX))
        # smooth_term = theano.shared(0.0001)
        self.W = theano.shared(
                value= np.ones(max_len, dtype=theano.config.floatX),
                name='W')
        L = T.cast(1. / input.shape[1], theano.config.floatX)
        # W_effection = T.cast(self.W / input.shape[1], theano.config.floatX)
        W_effection = self.W * L

        projectors, _ = theano.scan(fn = lambda projector, coef, somooth: coef * \
            T.outer(projector, projector) / (T.dot(projector, projector) + somooth) , \
            outputs_info = None, sequences = [input.reshape((input.shape[0] * \
                input.shape[1], input.shape[2])), W_effection.repeat(input.shape[0])\
            .reshape((input.shape[1], input.shape[0])).T.flatten()], \
            non_sequences = [smooth_term])

        
        self.output = T.tanh(T.cast(input.shape[1], theano.config.floatX) * T.sum(projectors.reshape((input.shape[0], input.shape[1], \
            input.shape[2], input.shape[2])), axis = 1))

        self.params = [self.W]


class Density_Dot(object):
    """
    The multiplication of two density matrices for a batch
    """

    def __init__(self, input1, input2):

        densities, _ = theano.scan(fn = lambda density1, density2 : \
            T.dot(density1, density2), outputs_info = None, \
            sequences = [input1, input2])

        # densities = T.mul(input1, input2)

        self.output = densities


class Trace_Inner(object):
    """
    Obtaining the trace of a matrix for a batch
    """
    def __init__(self, input):

        densities, _ = theano.scan(fn = lambda density : \
            T.nlinalg.trace(density), outputs_info = None, \
            sequences = [input])
        
        self.output = densities

class Get_Diag(object):
    """
    Obtaining the diagonal elements of a matrix for a batch
    """
    def __init__(self, input):
       

        densities, _ = theano.scan(fn = lambda density : \
            T.nlinalg.diag(density), outputs_info = None, \
            sequences = [input])
        
        self.output = densities



    


