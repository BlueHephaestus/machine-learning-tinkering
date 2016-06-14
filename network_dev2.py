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
import math
import json

class default_weight_initializer(object):

    @staticmethod
    def initialize(sizes):
        weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        return weights

class large_weight_initializer(object):

    @staticmethod
    def initialize(sizes):
        weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        return weights

class Network(object):

    def __init__(self, sizes, weight_init=default_weight_initializer):
        """The list ``sizes`` contains the number of neurons in the respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #so that we loop through like 0,1 1,2, 2,3... and so on with our connections being made just right.
        #self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = weight_init.initialize(self.sizes)
        self.output_dict = {}

        #For momentum sgd
        #Should be all good with initial v of 0, makes sense heuristically from our analogy
        
        #Which of these is the ideal one? The difference is not noticeable but it would seem that doing it according to our shape would be the only way that makese sense, so we're doing that.

        #self.w_velocities = np.zeros(len(self.weights))
        #self.b_velocities = np.zeros(len(self.biases))
        self.w_velocities = [np.zeros(w.shape) for w in self.weights]
        self.b_velocities = [np.zeros(b.shape) for b in self.biases]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        #So that we continually update it, thanks to our previous init function for weights and biases we can make this very small.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_accuracy_check_interval, eta_decrease_rate, momentum_coefficient, training_data_subsections=None, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)

            #Here is where we add our automatic learning scheduling and early stopping, if test_data is supplied
            self.test_accuracy_check_interval = test_accuracy_check_interval
            a_test_accuracy = []
            a_test_x = list(range(1, self.test_accuracy_check_interval+1))
            self.eta_decrease_rate = eta_decrease_rate#factor we decrease by
            #So if we have 3.0 initially with self.test_accuracy_check_interval == 10, we stop at .000003(3e-6)
            eta_stop_threshold = eta * math.pow(self.eta_decrease_rate, -6)

        self.momentum_coefficient = momentum_coefficient
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)

            '''
            so n is the length of our training data and k becomes each of the points where n % mini_batch_size == 0, so if we have n = 50 and m = 10, 
            k would be 0, 10, 20, 30, 40, and 50.
            so we are making mini_batches using k, so that the mini_batches become all the values in our shuffled training data [0:10], [10:20], ... , [40:50]
            '''
            if training_data_subsections:
                self.output_dict[j] = {}
                #Now we seperate our training_data into subsections for us to get output quicker if specified
                training_data_subsection_size = n/training_data_subsections
                training_data_subsection_num = 0#For display
                for training_data_subsection_index in range(0, len(training_data), training_data_subsection_size):
                    training_data_subsection = training_data[training_data_subsection_index:training_data_subsection_index+training_data_subsection_size]

                    mini_batches = [training_data_subsection[k:k+mini_batch_size] for k in xrange(0, training_data_subsection_size, mini_batch_size)]
                    for mini_batch in mini_batches:
                        #Do our backpropogation
                        self.update_mini_batch(mini_batch, eta)

                    '''
                    if test_data:
                    '''
                    test_data_accuracy = self.evaluate(test_data)
                    #Give us our percentage
                    test_data_accuracy = ((100.0*test_data_accuracy)/n_test)
                    print "Epoch {0}, Training Data Subsection {1}: {2}%".format(j, training_data_subsection_num, test_data_accuracy)
                    self.output_dict[j][training_data_subsection_num] = test_data_accuracy

                    '''
                    else:
                        print "Epoch {0} complete".format(j)
                    '''
                    training_data_subsection_num+=1
            else:
                #otherwise we just chill with normal epochs
                mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    #Do our backpropogation
                    self.update_mini_batch(mini_batch, eta)

                '''
                if test_data:
                '''
                test_data_accuracy = self.evaluate(test_data)
                test_data_accuracy = ((100.0*test_data_accuracy)/n_test)
                print "Epoch {0}: {1}%".format(j, test_data_accuracy)
                self.output_dict[j] = test_data_accuracy

                '''
                else:
                    print "Epoch {0} complete".format(j)
                '''


            #Since we only want to check this every epoch, not every training data subsection
            a_test_accuracy.append(test_data_accuracy)

            if len(a_test_accuracy) >= self.test_accuracy_check_interval:
                #Do our checks on the last check_interval number of accuracy results(which is the size of the array)
                #Checks = 
                    #get average slope of interval
                        #(interval * sigma(x*y) - sigma(x)*sigma(y)) / (interval * sigma(x**2) - (sigma(x))**2)
                        #where x is each # in our interval 1, 2, 3... interval
                        # and y is each of our accuracies
                test_accuracy_avg_slope = (self.test_accuracy_check_interval*sigma(list_product(a_test_x, a_test_accuracy)) - sigma(a_test_x) * sigma(a_test_accuracy))/(self.test_accuracy_check_interval*sigma([x**2 for x in a_test_x])*1.0 - (sigma(a_test_x))**2)
                if test_accuracy_avg_slope < 1:
                    eta /= self.eta_decrease_rate
                    print "Reducing eta by factor of {0} to {1}".format(self.eta_decrease_rate, eta)
                    if eta <= eta_stop_threshold:
                        print "Early stopped with low threshold"
                        break
                    #If we decrease the learning rate, we reset the interval by clearing our a_test_accuracy
                    a_test_accuracy = []
                else:
                    #remove the first element
                    a_test_accuracy.pop(0)
        #After training is complete
        write_dict(self.output_dict)

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

        #Sorry to interrupt, but first we do our velocity update rule for our momentum based stochastic gradient descent
        #we have our w_ and b_ so that we don't loop through our self.velocities differently for each, and they both can be added at the same time without problems. This should be the same as the example described with our formulas

        self.w_velocities = [self.momentum_coefficient*v - (eta/len(mini_batch))*nw
                for v, nw in zip(self.w_velocities, nabla_w)]
        self.b_velocities = [self.momentum_coefficient*v - (eta/len(mini_batch))*nb
                for v, nb in zip(self.b_velocities, nabla_b)]

        self.weights = [w + v 
                for w, v, in zip(self.weights, self.w_velocities)]
        self.biases = [b + v 
                for b, v, in zip(self.biases, self.b_velocities)]
        '''
        boring, sgd-without-momentum method
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        '''

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

def write_dict(d):
    #Write our dictionary as a json
    f = open('test_output.txt', 'a')
    f.write(json.dumps(d))
    f.close()
'''
Save this for our own implementation
def tanh(z):
    #tanh, obviously
    return np.tanh(z)

def tanh_prime(z):
    return np.sech(z)**2

def relu_max(z):
    return 
'''
def sigma(a):
    sum = 0
    for val in a:
        sum += val
    return sum

def list_product(a_x, a_y):
    return [x*y for x, y in zip(a_x, a_y)]

