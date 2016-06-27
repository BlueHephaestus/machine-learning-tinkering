import network_dev3

mini_batch_size = 10

net = network_dev3.Network([
  network_dev3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
    filter_shape=(20, 1, 5, 5), 
    poolsize=(2, 2), 
    activation_fn=network_dev3.ReLU),
  network_dev3.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
    filter_shape=(40, 20, 5, 5), 
    poolsize=(2, 2), 
    activation_fn=network_dev3.ReLU),
  network_dev3.FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=network_dev3.ReLU),
  network_dev3.FullyConnectedLayer(n_in=100, n_out=100, activation_fn=network_dev3.ReLU),
  network_dev3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
