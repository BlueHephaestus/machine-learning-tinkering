
'''
run_count = 3
epochs = 1
training_data_subsections=10
'''
#First, we run our configurations
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network_dev2

f = open('test_output.txt', 'w').close()

#So we have similarities to use for our graphing / these need to be the same for it to be more or less reasonable
run_count = 3
epochs = 1
training_data_subsections=100

#Subplot 1
net = network_dev2.Network([784, 30, 10])
net.SGD(training_data, 
        epochs, #epochs
        10,#m
        3.0,#eta
        5,#test_accuracy_check_rate
        2,#eta_decrease_factor
        .45,#u
        training_data_subsections=training_data_subsections, 
        test_data=test_data,
        run_count=run_count)

#Subplot 2
net = network_dev2.Network([784, 30, 10], weight_init=network_dev2.large_weight_initializer)
net.SGD(training_data, 
        epochs, #epochs
        10,#m
        3.0,#eta
        5,#test_accuracy_check_rate
        2,#eta_decrease_factor
        .45,#u
        training_data_subsections=training_data_subsections, 
        test_data=test_data,
        run_count=run_count)

#Then we graph the results
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import json

f = open('test_output.txt', 'r')
traces = []

for config in f.readlines():
    config_results = json.loads(config)
    if training_data_subsections:
        N = epochs*training_data_subsections
        x = np.linspace(0, N, N)
        for r in config_results:
            y = []
            for j in config_results[r]:
                for s in config_results[r][j]:
                    y.append(config_results[r][j][s])
            trace = go.Scatter(
                x = x,
                y = y
            )
            traces.append(trace)

    else:
        N = epochs
        x = np.linspace(0, N, N)
        for r in config_results:
            y = []
            for j in config_results[r]:
                y.append(config_results[r][j])
            trace = go.Scatter(
                x = x,
                y = y
            )
            traces.append(trace)
    
fig = tools.make_subplots(rows=2, cols=4, shared_xaxes=True, shared_yaxes=True) 
fig.append_trace(traces[0], 1, 1)
fig.append_trace(traces[1], 1, 2)
fig.append_trace(traces[2], 1, 3)
fig.append_trace(traces[3], 1, 4)
fig.append_trace(traces[4], 2, 1)
fig.append_trace(traces[5], 2, 2)
fig.append_trace(traces[6], 2, 3)
fig.append_trace(traces[7], 2, 4)
#fig.append_trace(trace2, 2, 1)
#fig.append_trace(trace3, 2, 2)
# Plot and embed in ipython notebook!
#py.iplot(data, filename='basic-line')
# Open new tab in browser
plot_url = py.plot(fig, filename='basic-line')

