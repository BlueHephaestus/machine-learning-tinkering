"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.


I added a lot more,
ayylmao -DE
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import math
import json
import sys#for exiting quickly during debugging

class default_weight_initializer(object):

    @staticmethod
    def initialize(sizes):
        weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        return weights

class large_weight_initializer(object):

    @staticmethod
    def initialize(sizes):
        #so that we loop through like 0,1 1,2, 2,3... and so on with our connections being made just right.
        weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        return weights

class quadratic_cost(object):

    @staticmethod
    def total_cost(a, y):
        return .5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(a, y, z):
        #Where the given vectors are one dimensional so we don't have to [-1]
        return (a-y) * sigmoid_prime(z)

class cross_entropy_cost(object):
    #this can actually be written two ways and he programs it in a different way than it seems he explains it, you can do it as - in front of everything or do 
    #-y*ln(a) - (1-y)*ln(1-a)
    #I'm doing it the way it's explained.
    
    @staticmethod
    def total_cost(a, y):
        return -np.sum(np.nan_to_num((y*np.log(a)+(1-y)*np.log(1-a))))

    @staticmethod
    def delta(a, y, z):
        #Where the given vectors are one dimensional so we don't have to [-1]
        #print np.array(a).shape
        #print a
        #print np.array(y).shape
        return (a-y)

class log_likelihood_cost(object):

    @staticmethod
    def total_cost(a, y):
        return -np.sum(np.nan_to_num(np.log(a-y)**2))

    @staticmethod
    def delta(a, y, z):
        #Where the given vectors are one dimensional so we don't have to [-1]
        return (a-y)

class log_likelihood_cost2(object):

    @staticmethod
    def total_cost(a, y):
        return -np.nan_to_num(np.log(np.linalg.norm(a-y))**2)

    @staticmethod
    def delta(a, y, z):
        #Where the given vectors are one dimensional so we don't have to [-1]
        return (a-y)
    

class l2_regularization(object):

    @staticmethod
    def cost_term(lmbda, n, weights):
        return (lmbda/(2.0*n))*sum(np.linalg.norm(w)**2 for w in weights)

class l1_regularization(object):

    @staticmethod
    def cost_term(lmbda, n, weights):
        return (lmbda/n)*sum(np.linalg.norm(weights))

class Network(object):

    def __init__(self, sizes, output_filename, weight_init=default_weight_initializer, softmax=True, cost=cross_entropy_cost, regularization=l2_regularization):
        """The list ``sizes`` contains the number of neurons in the respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weight_init = weight_init
        self.weights = self.weight_init.initialize(self.sizes)

        self.output_dict = {}
        self.output_filename = output_filename

        self.regularization=regularization
        self.cost=cost
        self.softmax=softmax
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
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            if self.softmax:
                if i == len(self.biases)-1:
                    #If output layer, we do our softmax
                    z = np.dot(w, a)+b
                    a = np.exp(z)/sum(np.exp(z))
                else:
                    a = sigmoid(np.dot(w, a)+b)
            else:
                a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_accuracy_check_interval, eta_decrease_rate, momentum_coefficient, lmbda, 
            training_data_subsections=None, validation_data=None, test_data=None,
            early_stopping=True,
            output_training_cost=False, output_training_accuracy=False, output_validation_cost=False, output_validation_accuracy=False, output_test_cost=False, output_test_accuracy=True, print_results=True,
            configs=1, config_num=1, output_types=1, run_count=1):
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
        self.lmbda = lmbda
        n = len(training_data)
        for r in range(run_count):
            #Re initialize everything
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = self.weight_init.initialize(self.sizes)
            self.output_dict[r] = {}
            a_test_accuracy = []
            for j in xrange(epochs):
                random.shuffle(training_data)

                '''
                so n is the length of our training data and k becomes each of the points where n % mini_batch_size == 0, so if we have n = 50 and m = 10, 
                k would be 0, 10, 20, 30, 40, and 50.
                so we are making mini_batches using k, so that the mini_batches become all the values in our shuffled training data [0:10], [10:20], ... , [40:50]
                '''
                if training_data_subsections:
                    self.output_dict[r][j] = {}
                    #Now we seperate our training_data into subsections for us to get output quicker if specified
                    training_data_subsection_size = n/training_data_subsections
                    training_data_subsection_num = 0#For display
                    for training_data_subsection_index in range(0, len(training_data), training_data_subsection_size):
                        training_data_subsection = training_data[training_data_subsection_index:training_data_subsection_index+training_data_subsection_size]

                        mini_batches = [training_data_subsection[k:k+mini_batch_size] for k in xrange(0, training_data_subsection_size, mini_batch_size)]
                        for mini_batch in mini_batches:
                            #Do our backpropogation
                            self.update_mini_batch(mini_batch, eta, n)

                        #Since we'd have tons of these to accomodate the various scenarios, we just omit the data for the graphs to show.
                        #Not perfect % calc yet
                        #perc_complete = 100.0*(config_num/float(configs-1) + r/float((configs-1)*(run_count-1)) + j/float((configs-1)*(run_count-1)*(epochs-1)) + training_data_subsection_num/float((configs-1)*(run_count-1)*(epochs-1)*(len(training_data)/training_data_subsection_size-1)))
                        #print "Config %i/%i, Run %i/%i, Epoch %i/%i, Subsection %i/%i, %f%% Complete. " % (config_num, configs-1, r, run_count-1, j, epochs-1, training_data_subsection_num, len(training_data)/training_data_subsection_size-1, perc_complete)
                        print "Config %i/%i, Run %i/%i, Epoch %i/%i, Subsection %i/%i" % (config_num, configs-1, r, run_count-1, j, epochs-1, training_data_subsection_num, len(training_data)/training_data_subsection_size-1)

                        #our various output monitoring choices
                        if test_data or validation_data or output_training_cost or output_training_accuracy:
                            self.output_dict[r][j][training_data_subsection_num] = []
                            if output_training_cost:
                                training_cost = self.total_cost(training_data, lmbda, convert=True)
                                self.output_dict[r][j][training_data_subsection_num].append(training_cost)
                                if print_results: print "\tTraining Cost: {0}".format(training_cost)
                            if output_training_accuracy:
                                training_accuracy_perc = self.accuracy(training_data, convert=True)
                                self.output_dict[r][j][training_data_subsection_num].append(training_accuracy_perc)
                                if print_results: print "\tTraining Accuracy: {0}%".format(training_accuracy_perc)

                            if validation_data:
                                if output_validation_cost:
                                    validation_cost = self.total_cost(validation_data, lmbda)
                                    self.output_dict[r][j][training_data_subsection_num].append(validation_cost)
                                    if print_results: print "\tValidation Cost: {0}".format(validation_cost)
                                if output_validation_accuracy:
                                    validation_accuracy_perc = self.accuracy(validation_data)
                                    self.output_dict[r][j][training_data_subsection_num].append(validation_accuracy_perc)
                                    if print_results: print "\tValidation Accuracy: {0}%".format(validation_accuracy_perc)

                            if test_data:
                                if output_test_cost:
                                    test_cost = self.total_cost(test_data, lmbda)
                                    self.output_dict[r][j][training_data_subsection_num].append(test_cost)
                                    if print_results: print "\tTest Cost: {0}".format(test_cost)

                                if output_test_accuracy:
                                    test_accuracy_perc = self.accuracy(test_data)
                                    #Give us our raw number by reversing our percentage equation, we need this for our early stopping. Then again, we could also do this according to validation data if we want. We can make a functio nto check it with if we so wish.
                                    test_accuracy = (test_accuracy_perc*n_test)/100.0
                                    #print "Config {0}, Run {1}, Epoch {2}, Training Data Subsection {3}: {4}%".format(config_num, r, j, training_data_subsection_num, test_data_accuracy_perc)
                                    self.output_dict[r][j][training_data_subsection_num].append(test_accuracy_perc)
                                    if print_results: print "\tTest Accuracy: {0}%".format(test_accuracy_perc)
                        '''
                        else:
                            print "Epoch {0} complete".format(j)
                        '''

                        training_data_subsection_num+=1
                else:
                    #otherwise we just chill with normal epochs, still doing our normal monitoring checks
                    mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
                    for mini_batch in mini_batches:
                        #Do our backpropogation
                        self.update_mini_batch(mini_batch, eta, n)

                    print "Config %i/%i, Run %i/%i, Epoch %i/%i" % (config_num, configs-1, r, run_count-1, j, epochs-1)
                    if test_data or validation_data or output_training_cost or output_training_accuracy:
                        self.output_dict[r][j] = []
                        if output_training_cost:
                            training_cost = self.total_cost(training_data, lmbda, convert=True)
                            #test_accuracy = (test_accuracy_peri*n_test)/100.0
                            test_accuracy = training_cost
                            self.output_dict[r][j].append(training_cost)
                            if print_results: print "\tTraining Cost: {0}".format(training_cost)
                        if output_training_accuracy:
                            training_accuracy_perc = self.accuracy(training_data, convert=True)
                            self.output_dict[r][j].append(training_accuracy_perc)
                            if print_results: print "\tTraining Accuracy: {0}%".format(training_accuracy_perc)

                        if validation_data:
                            if output_validation_cost:
                                validation_cost = self.total_cost(validation_data, lmbda)
                                self.output_dict[r][j].append(validation_cost)
                                if print_results: print "\tValidation Cost: {0}".format(validation_cost)
                            if output_validation_accuracy:
                                validation_accuracy_perc = self.accuracy(validation_data)
                                self.output_dict[r][j].append(validation_accuracy_perc)
                                if print_results: print "\tValidation Accuracy: {0}%".format(validation_accuracy_perc)

                        if test_data:
                            if output_test_cost:
                                test_cost = self.total_cost(test_data, lmbda)
                                self.output_dict[r][j].append(test_cost)
                                if print_results: print "\tTest Cost: {0}".format(test_cost)

                            if output_test_accuracy:
                                test_accuracy_perc = self.accuracy(test_data, convert=True)
                                #Give us our raw number by reversing our percentage equation, we need this for our early stopping. Then again, we could also do this according to validation data if we want. We can make a functio nto check it with if we so wish.
                                #print "Config {0}, Run {1}, Epoch {2}, Training Data Subsection {3}: {4}%".format(config_num, r, j, training_data_subsection_num, test_data_accuracy_perc)
                                self.output_dict[r][j].append(test_accuracy_perc)
                                if print_results: print "\tTest Accuracy: {0}%".format(test_accuracy_perc)
                    '''
                    if test_data:
                        test_data_accuracy = self.evaluate(test_data)
                        test_data_perc = ((100.0*test_data_accuracy)/n_test)
                        print "Config {0}, Run {0}, Epoch {1}: {2}%".format(config_num, r, j, test_data_perc)
                        self.output_dict[r][j] = test_data_perc
                    else:
                        print "Epoch {0} complete".format(j)
                    '''


                if early_stopping:
                    #Since we only want to check this every epoch, not every training data subsection
                    a_test_accuracy.append(test_accuracy)

                    if len(a_test_accuracy) >= self.test_accuracy_check_interval:
                        #Do our checks on the last check_interval number of accuracy results(which is the size of the array)
                        #Checks = 
                            #get average slope of interval
                                #(interval * sigma(x*y) - sigma(x)*sigma(y)) / (interval * sigma(x**2) - (sigma(x))**2)
                                #where x is each # in our interval 1, 2, 3... interval
                                # and y is each of our accuracies
                        test_accuracy_avg_slope = (self.test_accuracy_check_interval*1.0*sigma(list_product(a_test_x, a_test_accuracy)) - sigma(a_test_x) * 1.0 * sigma(a_test_accuracy))/(self.test_accuracy_check_interval*sigma([x**2 for x in a_test_x])*1.0 - (sigma(a_test_x))**2)
                        if test_accuracy_avg_slope > 1.0:
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

        #After all runs have executed

        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch/subsection, and doing our usual mean calculations
        if run_count > 1:
            self.output_dict[run_count+1] = {}#For our new average entry
            for j in range(epochs):
                self.output_dict[run_count+1][j] = {}#For our new average entry
                if training_data_subsections:
                    for s in range(training_data_subsections):
                        self.output_dict[run_count+1][j][s] = []#For our new average entry
                        for o in range(output_types):
                            avg = sum([self.output_dict[r][j][s][o] for r in range(run_count)]) / run_count
                            self.output_dict[run_count+1][j][s].append(avg)
                else:
                    self.output_dict[run_count+1][j] = []#For our new average entry
                    for o in range(output_types):
                        '''
                        print r, j, o
                        print self.output_dict[r]
                        print self.output_dict[r][j]
                        print self.output_dict[r][j][o]
                        '''
                        avg = sum([self.output_dict[r][j][o] for r in range(run_count)]) / run_count
                        self.output_dict[run_count+1][j].append(avg)

        #Write our dictionary as a json
        f = open('{0}_output.txt'.format(self.output_filename), 'a')
        f.write(json.dumps(self.output_dict))
        #add a newline to seperate our configs
        f.write("\n")
        #wrap up by closing our file behind us.
        f.close()

    def update_mini_batch(self, mini_batch, eta, n):
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

        #Where we add our l2 regularization to our weights
        #Should we pass in the len of our training data in a different way? If we end up using the entire data in here for another purpose we'll do that.
        self.weights = [(1-((eta*self.lmbda)/n))*w + v 
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
            #Should this be softmax? Why would it not?
            activation.shape = (len(activation), 1)#maybe some other mistake I made earlier necessitates this, unsure
            #print np.array(activation).shape
            #print np.array(b).shape
            #print np.array(w).shape
            z = np.dot(w, activation)+b
            #print np.array(z).shape
            zs.append(z)
            activation = sigmoid(z)

            #print activation
            activations.append(activation)
        y.shape = (len(y), 1)#see above comment
        #At this point we have completed our forward pass through the network, and obtained all the activations (a) and weighted inputs (z) from our network. 
        # backward pass

        #This is essentially our equation 1b, where 
        #error of last layer = output activations of last layer - y(desired output of last layer) hadamard product of the sigmoid's first derivative of the weighted inputs for the output layer
        #Apparently python treats * as the same thing as the hadamard product(<3 u python)
        #Delta is a horrible name for this, we always refer to it as error so why suddenly call it the variable name (lowercase delta). I don't like it, just like with nabla
        #this is specific to the quad cost, we will fix this.
        #delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        #print np.array(activations[-1]).shape, np.array(y).shape, np.array(zs[-1]).shape
        delta = self.cost.delta(activations[-1], y, zs[-1])
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

    '''Pertaining to the convert parameter in the following functions:
    we only convert if it's for the training data, where the y values are vectors of 0s and a 1 for the correct value,
    We have convert=False if it's for the validation or test data where we already have the argmax of our vector,
        And we actually need to get a vector for this in our total cost function to get back to this vector.
        So maybe it would be better to switch the decisions for convert, but it would be confusing to compare the two.
    '''
    def total_cost(self, data, lmbda, convert=False):
        #return the total cost on our data with our regularization term
        cost = 0
        n = len(data)
        for x, y in data:
            a = self.feedforward(x)
            if not convert:
                y = vectorized_result(y)
            cost += self.cost.total_cost(a, y)
        #Add our term, we will have an extra thing for dropout when we add it.
        
        cost += self.regularization.cost_term(lmbda, n, self.weights)
        return cost

    def accuracy(self, data, convert=False):
        #return a percent accuracy score on the data
        n = len(data)
        #relatively straightforward, we do argmax so we have a final decision for the data from our distribution of outputs to compare against desired result
        #straightforward? more like feedforward if you know what i'm sayin hahaha fucking kill me XD
        if convert:
            #For training data
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            #if validation or test data
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        #then we get the number of ones correct, where x == y in our results. We do this by summing the array in a clever way that I learned from him here
        results = sum(int(x == y) for x, y in results)
        #then we get our percentage
        results_perc = 100.0*(results/float(n))
        '''
        print results, results_perc, float(n)
        print "{0}%".format(results_perc)
        '''
        return results_perc

'''
    def evaluate(self, test_data):

        """Return the number of test inputs for which the neural network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever neuron in the final layer has the highest activation."""
        #We do this if test data is supplied when we go through each epoch
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

'''
'''
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x / \partial a for the output activations."""
        #Remember that this is computing our nabla (gradient vector) of C with respect to our output activations. This means it's the vector of partial derivatives 6C/6a for each neuron j in our output layer, which we can compute by taking the output activations - desired output activations
        #It's still a-y in this function if we were doing cross_entropy, but without the sigmoid_prime multiplication we do where this is referenced, for the reasons specified in the notes.
        return (output_activations-y)
'''

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

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

'''
def vectorized_result(y):
    #return a 10 unit vector with the yth value as 1 and every other value as 0
    a = np.zeros((10, 1))
    a[y] = 1.0
    return a
'''
def vectorized_result(y):
    #return a 2 unit vector with the yth value as 1 and every other value as 0
    a = np.zeros((2, 1))
    a[y] = 1.0
    return a
