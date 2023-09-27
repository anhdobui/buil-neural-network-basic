from .layer import Layer
import numpy as np

class FCLayer(Layer):
  def __init__(self,input_shape,output_shape):
    """"
    :param input_shape tuple (1,3)
    :param output_shape tuple (1,4)
    """
    
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.weights = np.random.rand(input_shape[1],output_shape[1]) - 0.5
    # weights (3,4)
    self.bias = np.random.rand(1,output_shape[1]) - 0.5
    # bias (1,4)
  
  def forward_propagation(self,input):
    self.input = input
    self.output = np.dot(self.input,self.weights) + self.bias
    return self.output
  
  def backward_propagation(self,output_error,learning_rate):
    #output_error (1,4) weight.T (4,3)
    current_layer_err = np.dot(output_error,self.weights.T)
    # dweight shape = weights_shape = (3,4)
    #input (1,3)
    # (3,4) = (3,1)*(1,4)
    dweight = np.dot(self.input.T,output_error)
    
    self.weights -= dweight*learning_rate
    self.bias -= learning_rate*output_error
    return current_layer_err
