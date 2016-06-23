import time#For unix timestamp generation of our filenames
#So we have similarities to use for our graphing / these need to be the same for it to be more or less reasonable
#Basically these are our global things
configs = 2
output_types = 2
run_count = 4 
epochs = 400
training_data_subsections=None
early_stopping=False

output_training_cost=False
output_training_accuracy=False
output_validation_cost=False
output_validation_accuracy=True
output_test_cost=False
output_test_accuracy=True

comparison_title="Regularization vs no regularization"
comparison_file="regularization_comparison-{0}".format(str(time.time()))
xaxis_title="Epoch Subsection"
yaxis_title="MNIST % Accuracy"
update_output = True
graph_output = True
#Will by default subplot the output types, will make config*outputs if that option is specified as well.
subplot_seperate_configs = True

if update_output:
    #First, we run our configurations
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #If we want to speed up our training or magnify differences for comparisons
    #Note: will lower accuracy and learning speed.
    training_data = training_data[:1000]

    import network_dev2

    f = open('{0}_output.txt'.format(comparison_file), 'w').close()
    
    config_count = 0
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file, softmax=False, cost=network_dev2.quadratic_cost, weight_init=network_dev2.large_weight_initializer)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            0.5,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            0,#u
            2.0,#lmbda / regularization rate
            training_data_subsections=training_data_subsections, 
            validation_data=validation_data,
            test_data=test_data,
            early_stopping=early_stopping,
            output_training_cost=output_training_cost,
            output_training_accuracy=output_training_accuracy,
            output_validation_cost=output_validation_cost,
            output_validation_accuracy=output_validation_accuracy,
            output_test_cost=output_test_cost,
            output_test_accuracy=output_test_accuracy,
            config_num=config_count,
            output_types=output_types,
            run_count=run_count)

    config_count += 1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file, softmax=False, cost=network_dev2.quadratic_cost, weight_init=network_dev2.large_weight_initializer)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            0.5,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            0,#u
            0,#lmbda / regularization rate
            training_data_subsections=training_data_subsections, 
            validation_data=validation_data,
            test_data=test_data,
            early_stopping=early_stopping,
            output_training_cost=output_training_cost,
            output_training_accuracy=output_training_accuracy,
            output_validation_cost=output_validation_cost,
            output_validation_accuracy=output_validation_accuracy,
            output_test_cost=output_test_cost,
            output_test_accuracy=output_test_accuracy,
            config_num=config_count,
            output_types=output_types,
            run_count=run_count)

if graph_output:
    #Then we graph the results
    from plotly import tools
    import plotly.plotly as py
    import plotly.graph_objs as go

    import numpy as np
    import json
    from collections import OrderedDict

    f = open('{0}_output.txt'.format(comparison_file), 'r')
    traces = [[]]

    config_num = 0
    for config in f.readlines():
        config_results = json.loads(config, object_pairs_hook=OrderedDict)#So we preserve the order of our stored json
        traces.append([])
        #traces[config_num] = []
        #print config_results['3']['0']['0'][0]
        if training_data_subsections:
            N = epochs*training_data_subsections
            x = np.linspace(0, N, N)
            for output_type in range(output_types):
                traces[config_num].append([])
                for r in config_results:
                    y = []
                    for j in config_results[r]:
                        for s in config_results[r][j]:
                            #print r, j, s, output_type
                            y.append(config_results[r][j][s][output_type])
                        #y.append(config_results[r][j][s])
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
                    
                    #print config_num, output_type
                    traces[config_num][output_type].append(trace)

        else:
            N = epochs
            x = np.linspace(0, N, N)
            for output_type in range(output_types):
                traces[config_num].append([])
                for r in config_results:
                    y = []
                    for j in config_results[r]:
                        y.append(config_results[r][j][output_type])
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
                    traces[config_num][output_type].append(trace)
        config_num+=1
        
    #print traces
    if subplot_seperate_configs:
        #Don't combine the configs into one, so we go to new column for each one, making the rows our output types
        fig = tools.make_subplots(rows=configs, cols=output_types, shared_xaxes=False, shared_yaxes=False) 
        row_num = 1
        col_num = 1
        for trace_index, trace in enumerate(traces):
            if np.mod(trace_index, (len(traces)/configs)) == 0 and trace_index > 0:
                row_num += 1
                col_num = 1
            for output_type in traces[trace_index]:
                for run in output_type:
                    fig.append_trace(run, row_num, col_num)
                col_num+=1

        #fig["layout"].update(title=comparison_title, yaxis=dict(autotick=False, dtick=1.0, range=[75, 87]))
        #fig["layout"]["yaxis1"].update(yaxis=dict(dtick=1.0, range=[75,87]))
        #fig["layout"].update(title=comparison_title, yaxis=dict(autotick=False, range=[85.8, 87.2))
        '''
        for xaxis_num in range(1,configs+1):
            fig["layout"]["xaxis{0}".format(xaxis_num)].update(title=xaxis_title)
        fig["layout"]["yaxis1"].update(title=yaxis_title)
        '''
    else:
        layout = dict(title=comparison_title, xaxis=dict(title=xaxis_title), yaxis=dict(title=yaxis_title),)
        fig = dict(data=traces, layout=layout)

    plot_url = py.plot(fig, filename=str(np.random.rand(1)))
    #fig.append_trace(trace2, 2, 1)
    #fig.append_trace(trace3, 2, 2)
    # Plot and embed in ipython notebook!
    #py.iplot(data, filename='basic-line')
    # Open new tab in browser

