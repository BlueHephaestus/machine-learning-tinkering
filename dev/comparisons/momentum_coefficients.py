#So we have similarities to use for our graphing / these need to be the same for it to be more or less reasonable
#Basically these are our global things
configs = 10
run_count = 16
epochs = 1
training_data_subsections=40
comparison_title="Momentum Coefficient comparisons"
comparison_file="momentum_comparison"
xaxis_title="Epoch Subsection"
yaxis_title="MNIST % Accuracy"
update_output = True
subplots=False

if update_output:
    #First, we run our configurations
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = training_data[:1000]

    import network_dev2

    f = open('{0}_output.txt'.format(comparison_file), 'w').close()
    
    config_count = 0
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .9,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)

    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .8,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .7,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)

    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .6,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .5,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .4,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .3,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .2,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .1,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)
    config_count +=1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            3.0,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            0,#u
            training_data_subsections=training_data_subsections, 
            test_data=test_data,
            config_num=config_count,
            run_count=run_count)

#Then we graph the results
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import json
from collections import OrderedDict

f = open('{0}_output.txt'.format(comparison_file), 'r')
traces = []

for config in f.readlines():
    config_results = json.loads(config, object_pairs_hook=OrderedDict)#So we preserve the order of our stored json
    if training_data_subsections:
        N = epochs*training_data_subsections
        x = np.linspace(0, N, N)
        for r in config_results:
            y = []
            for j in config_results[r]:
                for s in config_results[r][j]:
                    y.append(config_results[r][j][s])
            if int(r) >= run_count:
                #Our final, average run
                trace = go.Scatter(
                    x = x,
                    y = y,
                    line = dict(
                        width = 4
                    ),
                    name="Average Run",
                    mode="line"
                )
            else:
                trace = go.Scatter(
                    x = x,
                    y = y,
                    line = dict(
                        width = 1,
                        dash = "dot"
                    ),
                    name="Test Run #{0}".format(r)
                )
            traces.append(trace)

    else:
        N = epochs
        x = np.linspace(0, N, N)
        for r in config_results:
            y = []
            for j in config_results[r]:
                y.append(config_results[r][j])
            if int(r) >= run_count:
                #Our final, average run
                trace = go.Scatter(
                    x = x,
                    y = y,
                    line = dict(
                        width = 4
                    ),
                    name="Average Run",
                    mode="line"
                )
            else:
                trace = go.Scatter(
                    x = x,
                    y = y,
                    line = dict(
                        width = 1,
                        dash = "dot"
                    ),
                    name="Test Run #{0}".format(r)
                )
            traces.append(trace)
    
if subplots:
    fig = tools.make_subplots(rows=1, cols=configs, shared_xaxes=True, shared_yaxes=True) 
    col_num = 1
    for trace_index, trace in enumerate(traces):
        if np.mod(trace_index, (len(traces)/configs)) == 0 and trace_index > 0:
            col_num += 1
        fig.append_trace(trace, 1, col_num)

    fig["layout"].update(title=comparison_title)
    for xaxis_num in range(1,configs+1):
        fig["layout"]["xaxis{0}".format(xaxis_num)].update(title=xaxis_title)
    fig["layout"]["yaxis1"].update(title=yaxis_title)
else:
    layout = dict(title=comparison_title, xaxis=dict(title=xaxis_title), yaxis=dict(title=yaxis_title),)
    fig = dict(data=traces, layout=layout)

plot_url = py.plot(fig, filename='ayy-lmao')
#fig.append_trace(trace2, 2, 1)
#fig.append_trace(trace3, 2, 2)
# Plot and embed in ipython notebook!
#py.iplot(data, filename='basic-line')
# Open new tab in browser

