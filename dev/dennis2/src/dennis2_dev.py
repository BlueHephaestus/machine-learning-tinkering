"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."

def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        our layers will be convolutional/pooling, softmax, etc.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        #for each layer, get all params, and add that to our array so that params becomes all the params in all layers into one list
        self.params = [param for layer in self.layers for param in layer.params]
        #initialized as symbolic variables with theano, this makes it easier and faster to do sgd and backpropagation
        #symbolic variables mean they become a variable of name ("name") and of the type specified with T, in this case
        #self.x becomes a symbolic variable referencing a matrix,
        #and self.y becomes a symbolic variable referencing an integer vector, since our outputs are 1s or 0s
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        #first specified layer
        init_layer = self.layers[0]
        #define symbolic outputs from our network, we set our inputs one mini batch at a time
        #set_inpt is like input, input_dropout, and the mini batch size. We pass the same inputs to inpt and inpt_dropout because this is where we will first apply dropout and make them different, if we do choose to use dropout.
        #of course, if it's a convolutional layer we're dealing with, we don't do dropout at all because they take care of that themselves in the way that the shared weights force the neurons to adjust to the entire image instead of overfitting
        #to their specific regions
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            #we then propogate our self.x symbolic variable through the network layers, by assigning our layer and previous layer(initially our init_layer) to variables, then setting the input of our 
            #new layer equal to the output of our previous layer, thus propogating our initial self.x symbolic variable through.
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        #now that we've propogated our x symbolic variable through the network we can symbolically represent our Network output with the following
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        #guess this is how the data is formatted this time
        training_x, training_y = training_data
        #print training_x.shape()
        training_x, training_y = training_data
        #training_x = training_data[0]
        #training_y = training_data[1]
        validation_x, validation_y = validation_data
        #validation_x = validation_data[0]
        #validation_y = validation_data[1]
        test_x, test_y = test_data
        #test_x = test_data[0]
        #test_y = test_data[1]

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+0.5*lmbda*l2_norm_squared/num_training_batches
        #Get our gradients and updates for each parameter for us to update mini batch with
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.

        #i is our value in the number of validation batches we have, and if we multiply that by our number of mini batches((n/m) * m = n), we get the index of the beginning of our mini batch;
        #then when we do the same with i+1 we get the beginning of our next mini batch so for i = 2 and m = 10, we could have training_x[20:30]
        #so that we get the right values for our mini batch for both self.x and self.y
        #we use our givens for the output part of our function
        i = T.lscalar() # mini-batch index

        #We have our givens so that the storage gets put on the GPU - I still don't fully understand this.
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                #i is our value in the number of validation batches we have, and if we multiply that by our number of mini batches((n/m) * m = n), we get the index of the beginning of our mini batch;
                #then when we do the same with i+1 we get the beginning of our next mini batch so for i = 2 and m = 10, we could have training_x[20:30]
                #so that we get the right values for our mini batch for both self.x and self.y
                #we use our givens for the output part of our function
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_training_accuracy = 0.0
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    training_accuracy = np.mean([train_mb_accuracy(j) for j in xrange(num_training_batches)])
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print "Epoch %i" % (epoch)
                    print "\tTraining Accuracy: %f" % (training_accuracy)
                    print "\tValidation Accuracy: %f" % (validation_accuracy)

                    if training_accuracy > best_training_accuracy:
                        print "\tThis is the best training accuracy to date."
                        best_training_accuracy = training_accuracy
                    if validation_accuracy > best_validation_accuracy:
                        print("\tThis is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('\tThe corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, subsample=(1, 1), poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.subsample = subsample
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        #shared vars let us access anywhere, borrow=True makes it so we don't make copies of it but keep one on the CPU(not GPU, right?)
        self.w = theano.shared(
            #converts to array of dtype float(theano.config.floatX, default is float64)
            np.asarray(
                #loc = mean, scale=stddev, size=# samples, If the given shape is, e.g., (m, n, k), then m * n * k samples returned/drawn
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        #we reshape our inputs accordingly,
        self.inpt = inpt.reshape(self.image_shape)
        #we get our filters from our weights and the shape(height & width) from when we initialize the layer)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape, subsample=self.subsample)
        #We get our pooling output from the poolsize we specify, and I guess theano does the rest
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        #dimshuffle, in this case, casts each dimension in the order specified, i.e. 0, 1, 2
        #and makes the 'x' equal to 1, so that if you had a tensor(symbolic array in this case i believe) of shape
        #20, 30, 40, and dimshuffled ('x', 'x', 2, 1, 0, 'x'), you would get a new tensor with shape
        #(1, 1, 40, 30, 20, 1)
        #so with our one dimensional self.b, we dimshuffle to get a shape (1, len(self.b), 1, 1) from our original 1d
        #self.b of len len(self.b)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(
               np.random.normal(
                   loc=0.0, scale=1.0, size=(n_out,)),
               dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        #we only use these two layers for evaluating accuracy on the validation and test data
        #This is a LAYER, not an entire network. So it's just one hidden layer we are accounting for
        #We make the input values of width mini_batch_size, and of height n_in the number of them there are.
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        #We make the output of our network equal to the following:
        '''if we had 55% dropout, then we drop 45% of these. We do that by doing 1-55, then we multiply our dropout perc(.45) to our transposed matrices/vectors 
        by the matrices/vectors of our output activations from the previous layer times our weights, plus our bias. Then we plug that into our activation function and
        we've got o(wa+b)'''

        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        #Then the y output is just the transposed argmax of our output
        self.y_out = T.argmax(self.output, axis=1)

        #we use the following two for our actual training(if we choose dropout, of course)
        #removes p_dropout percentage of the neurons in our layer
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        #Gets the output from our crippled and emotionally damaged (input dropout) layer
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases to 0 because it works better for softmax
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        print net.y, net.y.shape[0]
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    '''
            return -T.mean(
                        T.log(
                            self.output_dropout
                        )
                        [
                            T.arange(
                                net.y.shape[0]
                            ), net.y
                        ]
                    )
    '''

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
