from network.network import Network
from layers.FCLayer import FCLayer
from layers.activation_layer import ActivationLayer
import numpy as np
def relu(z):
  """_summary_

  Args:
      z (_type_): numpy array
      return 0 nếu z <= 0
              z nếu z > 0
  """
  return np.maximum(0,z)
def relu_prime(z):
  z[z<0]=0
  z[z>0]=1
  return z

def loss(y_true,y_pred):
  return 0.5*(y_true-y_pred)**2

def loss_prime(y_true,y_pred):
  return y_pred-y_true

x_train = np.array([ [[0,0]],[[0,1]],[[1,0]],[[1,1]] ])

y_train = np.array([ [[0]], [[1]] , [[1]], [[0]] ])

net = Network()

net.add(FCLayer(input_shape=(1,2), output_shape=(1,3)))
net.add(ActivationLayer(output_shape=(1,3),input_shape=(1,3),activation=relu,activation_prime=relu_prime))
net.add(FCLayer((1,3),(1,1)))
net.add(ActivationLayer((1,1),(1,1)),relu,relu_prime)
net.setup_loss(loss,loss_prime)
net.fit(x_train,y_train,epochs=1000,learning_rate=0.01)