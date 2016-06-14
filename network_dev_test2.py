import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network_dev2

f = open('test_output.txt', 'w').close()
#Subplot 1
net = network_dev2.Network([784, 30, 10])
net.SGD(training_data, 
        2, #epochs
        10,#m
        3.0,#eta
        5,#test_accuracy_check_rate
        2,#eta_decrease_factor
        .45,#u
        training_data_subsections=50, 
        test_data=test_data)

#Subplot 2
'''
net = network_dev2.Network([784, 30, 10], weight_init=network_dev2.large_weight_initializer)
net.SGD(training_data, 
        2, #epochs
        10,#m
        3.0,#eta
        5,#test_accuracy_check_rate
        2,#eta_decrease_factor
        .45,#u
        training_data_subsections=50, 
        test_data=test_data)
'''
