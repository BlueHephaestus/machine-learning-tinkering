import scipy.io.wavfile as wav
import numpy as np
import os  #for directories
import cPickle, gzip #for storing our data
import pickle

a = [0, 1, 2, 3]
b = [4, 5]

training_data = zip(np.zeros(shape=(80, 83968)), np.zeros(shape=(80, 2)))
test_data = zip(np.zeros(shape=(40, 83968)), np.zeros(shape=(40, 2)))

#print np.array(training_data[0]).shape
print list(training_data[0])
training_data[0] = list(training_data[0]) 
training_data[0] = a, b
training_data[0] = tuple(training_data[0])
print training_data[0]
print np.array(training_data[0]).shape
#print np.array(training_data[0]).shape
#print np.array(zip(a, b)).shape
#print np.array(zip(a, b))
'''
print np.array(training_data).shape, np.array(training_data).shape

training_data[0] = zip(a, b)
print np.array(training_data).shape, np.array(training_data).shape
print training_data[0]
'''
