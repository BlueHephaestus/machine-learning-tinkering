import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2_2
net = network2_2.Network([784, 30, 10], cost=network2_2.CrossEntropyCost)
net.SGD(training_data, 10, 8.0,
  lmbda = 5.0,
  evaluation_data=validation_data,
  monitor_evaluation_accuracy=True,
  monitor_evaluation_cost=True,
  monitor_training_accuracy=True,
  monitor_training_cost=True)
