import dennis2
from dennis2 import Network
from dennis2 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
#import network3

import sample_loader

#expanded_training_data, validation_data, test_data = dennis2.load_data_shared("data/mnist_expanded.pkl.gz")
#training_data, validation_data, test_data = dennis2.load_data_shared("data/mnist.pkl.gz")
#sample_loader.regenerate_data()
training_data, validation_data, test_data = sample_loader.load_data_shared()
'''
import numpy as np
print expanded_training_data
print validation_data
print test_data
'''

#sample_loader.regenerate_data()
#training_data, validation_data, test_data = sample_loader.load_data_shared()

#expanded_training_data, validation_data, test_data = dennis2.load_data_shared("../data/mnist_expanded.pkl.gz")
#training_data, validation_data, test_data = dennis2.load_data_shared()

mini_batch_size = 5

'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
        filter_shape=(20, 1, 3, 3), 
        poolsize=(2, 2), 
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 13, 13), 
        filter_shape=(40, 20, 4, 4), 
        poolsize=(2, 2), 
        activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*5*5, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
'''
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 290, 290),
        filter_shape=(20, 1, 36, 36),
        subsample=(2,2),
        poolsize=(2,2),
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 64, 64),
        filter_shape=(20, 1, 5, 5),
        subsample=(1,1),
        poolsize=(2,2),
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 30, 30),
        filter_shape=(20, 1, 5, 5),
        subsample=(1,1),
        poolsize=(2,2),
        activation_fn=ReLU),
    FullyConnectedLayer(n_in=20*13*13, n_out=100, activation_fn=ReLU),
    FullyConnectedLayer(n_in=100, n_out=30, activation_fn=ReLU),
    SoftmaxLayer(n_in=30, n_out=2)], mini_batch_size)
'''
net = Network([ 
    FullyConnectedLayer(n_in=290*290, n_out=1000, activation_fn=sigmoid),
    FullyConnectedLayer(n_in=1000, n_out=100, activation_fn=sigmoid),
    FullyConnectedLayer(n_in=100, n_out=30, activation_fn=sigmoid),
    #FullyConnectedLayer(n_in=20*8*8, n_out=256, activation_fn=ReLU),
    #FullyConnectedLayer(n_in=256, n_out=16, activation_fn=ReLU),
    SoftmaxLayer(n_in=30, n_out=2)], mini_batch_size)

'''
ConvPoolLayer(image_shape=(mini_batch_size, 1, 128, 128),
    filter_shape=(40, 20, 29, 29),
    poolsize=(2, 2),
    activation_fn=ReLU),
'''
'''
ConvPoolLayer(image_shape=(mini_batch_size, 20, 38, 38),
    filter_shape=(20, 20, 13, 13),
    subsample=(5,5),#stride
    poolsize=(2, 2),
    activation_fn=ReLU),

'''
#print len(training_data), len(validation_data), len(test_data)
#net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
'''
training_data = zip(training_data[0], training_data[1])
test_data = zip(test_data[0], test_data[1])
validation_data = test_data
'''
net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
