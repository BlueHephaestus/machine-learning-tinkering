"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #so that we loop through like 0,1 1,2, 2,3... and so on with our connections being made just right.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        #So that we continually update it, thanks to our previous init function for weights and biases we can make this very small.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            '''
            so n is the length of our training data and k becomes each of the points where n % mini_batch_size == 0, so if we have n = 50 and m = 10, 
            k would be 0, 10, 20, 30, 40, and 50.
            so we are making mini_batches using k, so that the mini_batches become all the values in our shuffled training data [0:10], [10:20], ... , [40:50]
            '''
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #Do our backpropogation
                self.update_mini_batch(mini_batch, eta)


            #Mostly irrelevant, no actual computation performed here.
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta`` is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """nabla is our inverted triangle, or gradient vector.
        At the beginning of this, we initialize each with zeroes so that at first it has 0 for each respective weights/biases change over time / partial derivative
        """
        for x, y in mini_batch:
            #Where x is our input and y is our desired output
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #Now that we've got the changes that need to be made to weights and biases, we get the sum of our new changes and our nablas, which are being updated each iteration of the loop. So this way, we continually get the changes that our training data is telling us to make to each weight and bias in the network
            #So that we end up with a sum change to do for each weight and bias in the network
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #We put these changes into our weights and biases here, updating them by subtracting the change we know we need to do, that we obtained previously as nabla_b/w
        #Since we already summed it, we don't need to do that in the following steps, and instead can just plug in our "nw" value
        #Then since we've got that, we just multiply nw by our eta(learning rate). We can plainly see here the reasoning behind it, in that it affects the amount we travel in the direction the change indicates. We divide by our mini batch, since this isn't the total resulting change and we want it to hold as much power in affecting the total change as the portion of the total training data that our mini batch represents.
        #So we get all that, and we subtract our mini batches' net change from our weight/bias for each weight/bias.
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        #Where x is our input and y is our desired output
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """yet again, we initialize each with zeroes so that at first it has 0 for each respective weights/biases change over time / partial derivative"""
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            #Starting with our basic x for input, we keep going through the network, doing our sigmoid(wa+b)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #At this point we have completed our forward pass through the network, and obtained all the activations (a) and weighted inputs (z) from our network. 
        # backward pass

        #This is essentially our equation 1b, where 
        #error of last layer = output activations of last layer - y(desired output of last layer) hadamard product of the sigmoid's first derivative of the weighted inputs for the output layer
        #Apparently python treats * as the same thing as the hadamard product
        #Delta is a horrible name for this, we always refer to it as error so why suddenly call it the variable name (lowercase delta). I don't like it, just like with nabla
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        #Our rate of change of our cost C with respect to any bias/neuron in the network is simply the delta for that neuron
        #See: Equation 3a
        nabla_b[-1] = delta
        #Whereas it's a bit different for weights of any layer in the network
        #See: Equation 4
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())#and yea we've also gotta transpose
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            #Get zs for this layer
            z = zs[-l]
            #Get our sigmoid prime for each z in this layer
            sp = sigmoid_prime(z)
            #Not as simple as equation 1b, this time we've got equation 2 to deal with for any layer
            #See: Equation 2
            #At this point it's also pretty self explanatory
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #See: Equation 3a
            nabla_b[-l] = delta
            #About the same as our previous encounter with Eq4.
            #See: Equation 4
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever neuron in the final layer has the highest activation."""
        #We do this if test data is supplied when we go through each epoch
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x / \partial a for the output activations."""
        #Remember that this is computing our nabla (gradient vector) of C with respect to our output activations. This means it's the vector of partial derivatives 6C/6a for each neuron j in our output layer, which we can compute by taking the output activations - desired output activations
        #It's still a-y in this function if we were doing cross_entropy, but without the sigmoid_prime multiplication we do where this is referenced, for the reasons specified in the notes.
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
