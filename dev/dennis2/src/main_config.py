import dennis2
from dennis2 import Network
from dennis2 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

#import sample_loader

#training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/mnist.pkl.gz")
training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/samples.pkl.gz", normalize_x=True)
#training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/expanded_samples.pkl.gz", normalize_x=True)

mini_batch_size = 10

#I don't understand why it won't let me add extra on the end with 290*290.
#So we're going to shorten it ever so slightly to 289*289.
#That is the source of the weird problem we were having, as far as I can gather.

'''
'''
'''
    FullyConnectedLayer(n_in=28*28, n_out=100, activation_fn=ReLU),
    FullyConnectedLayer(n_in=100, n_out=30, activation_fn=ReLU),
    SoftmaxLayer(n_in=30, n_out=10)], mini_batch_size)
'''
'''
ConvPoolLayer(image_shape=(mini_batch_size, 1, 289, 289), 
    filter_shape=(20, 1, 35, 35),#289-35 = 254
    subsample=(2, 2),#254/2 = 127+1 = 128
    poolsize=(2, 2),#128/2 = 64
    activation_fn=ReLU),
#FullyConnectedLayer(n_in=20*64*64, n_out=100),
'''
net = Network([
    FullyConnectedLayer(n_in=289*289, n_out=100),
    FullyConnectedLayer(n_in=100, n_out=30),
    FullyConnectedLayer(n_in=30, n_out=10),
    SoftmaxLayer(n_in=10, n_out=2)], mini_batch_size)
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 289, 289), 
        filter_shape=(20, 1, 35, 35),#289-35 = 254
        subsample=(2, 2),#254/2 = 127+1 = 128
        poolsize=(2, 2),#128/2 = 64
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 64, 64), 
        filter_shape=(20, 20, 5, 5),#64-5+1=60
        poolsize=(2, 2),#60/2 = 30
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 30, 30),
        filter_shape=(20, 20, 5, 5),#30-5 = 25
        poolsize=(2,2),#(25+1)/2 = 13
        activation_fn=ReLU),
    FullyConnectedLayer(n_in=20*13*13, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=2)], mini_batch_size)
'''
#net.SGD(training_data, 60, mini_batch_size, 1000.03, validation_data, test_data, lmbda=2.1)
#net.SGD(training_data, 99999, mini_batch_size, .1, validation_data, test_data)
net.SGD(training_data, 100, mini_batch_size, .1, validation_data, test_data, momentum_coefficient=0.0)
'''
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 289, 289), 
        filter_shape=(20, 1, 35, 35),#289-35 = 254
        subsample=(2, 2),#254/2 = 127+1 = 128
        poolsize=(2, 2),#128/2 = 64
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 64, 64), 
        filter_shape=(20, 20, 5, 5),#64-5+1=60
        poolsize=(2, 2),#60/2 = 30
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 30, 30),
        filter_shape=(20, 20, 5, 5),#30-5 = 25
        poolsize=(2,2),#(25+1)/2 = 13
        activation_fn=ReLU),
    FullyConnectedLayer(n_in=20*13*13, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=2)], mini_batch_size)
'''
'''
'''
'''
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 289, 289),
        filter_shape=(10, 1, 34, 34),#289-34 = 255
        poolsize=(2,2),#(255+1)/2 = 128
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 10, 128, 128),
        filter_shape=(10, 10, 35, 35),#128-33 = 95
        poolsize=(2,2),#(95+1)/2 = 96/2 = 48
        activation_fn=ReLU),
    FullyConnectedLayer(n_in=10*48*48, n_out=100, activation_fn=ReLU),
    FullyConnectedLayer(n_in=100, n_out=30, activation_fn=ReLU),
    SoftmaxLayer(n_in=30, n_out=2)], mini_batch_size)
'''
'''
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 290, 290),
        filter_shape=(20, 1, 36, 36),#290-36 = 254
        subsample=(2,2),#254/2 = 127
        poolsize=(2,2),#(127+1)/2 = 64
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 64, 64),
        filter_shape=(20, 1, 5, 5),#64-5 = 59
        poolsize=(2,2),#(59+1)/2 = 30
        activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 30, 30),
        filter_shape=(20, 1, 5, 5),#30-5 = 25
        poolsize=(2,2),#(25+1)/2 = 13
        activation_fn=ReLU),
'''
