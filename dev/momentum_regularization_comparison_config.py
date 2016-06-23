#So we have similarities to use for our graphing / these need to be the same for it to be more or less reasonable
#Basically these are our global things
configs = 4
output_types = 6#DON'T FORGET TO UPDATE THIS WITH THE OTHERS
run_count = 5 
epochs = 50
training_data_subsections=10
early_stopping=False

output_training_cost=True
output_training_accuracy=True
output_validation_cost=True
output_validation_accuracy=True
output_test_cost=True
output_test_accuracy=True

comparison_title="Momentum & Regularization Comparison"
comparison_file="momentum_regularization_comparison"
print_results = False
update_output = True
graph_output = True
#Will by default subplot the output types, will make config*outputs if that option is specified as well.
subplot_seperate_configs = False

if update_output:
    #First, we run our configurations
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    #If we want to speed up our training or magnify differences for comparisons:
    #Note: will lower accuracy and learning speed.
    #training_data = training_data[:1000]

    import network_dev2

    f = open('{0}_output.txt'.format(comparison_file), 'w').close()
    
    config_count = 0
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file, softmax=False, cost=network_dev2.cross_entropy_cost, weight_init=network_dev2.large_weight_initializer)
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
            print_results=print_results,
            config_num=config_count,
            configs=configs,
            output_types=output_types,
            run_count=run_count)

    config_count += 1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file, softmax=False, cost=network_dev2.cross_entropy_cost, weight_init=network_dev2.large_weight_initializer)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            0.5,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .5,#u
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
            print_results=print_results,
            config_num=config_count,
            configs=configs,
            output_types=output_types,
            run_count=run_count)
    config_count += 1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file, softmax=False, cost=network_dev2.cross_entropy_cost, weight_init=network_dev2.large_weight_initializer)
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
            print_results=print_results,
            config_num=config_count,
            configs=configs,
            output_types=output_types,
            run_count=run_count)
    config_count += 1
    net = network_dev2.Network([784, 30, 10], output_filename=comparison_file, softmax=False, cost=network_dev2.cross_entropy_cost, weight_init=network_dev2.large_weight_initializer)
    net.SGD(training_data, 
            epochs, #epochs
            10,#m
            0.5,#eta
            5,#test_accuracy_check_interval
            2,#eta_decrease_factor
            .5,#u
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
            print_results=print_results,
            config_num=config_count,
            configs=configs,
            output_types=output_types,
            run_count=run_count)


if graph_output:
    #Then we graph the results
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    from collections import OrderedDict

    f = open('{0}_output.txt'.format(comparison_file), 'r')

    config_num = 0
    plt.figure(1)
    for config in f.readlines():
        config_results = json.loads(config, object_pairs_hook=OrderedDict)#So we preserve the order of our stored json
        if training_data_subsections:
            N = epochs*training_data_subsections
        else:
            N = epochs
        x = np.linspace(0, N, N)
        for output_type in range(output_types):
            if subplot_seperate_configs:
                plt.subplot(configs, output_types, config_num*output_types+output_type+1)#number of rows, number of cols, number of subplot
            else:
                plt.subplot(1, output_types, output_type+1)#number of rows, number of cols, number of subplot
                
            for r in config_results:
                y = []
                for j in config_results[r]:
                    if training_data_subsections:
                        for s in config_results[r][j]:
                            #print r, j, s, output_type
                            y.append(config_results[r][j][s][output_type])
                    else:
                        y.append(config_results[r][j][output_type])
                    #y.append(config_results[r][j][s])
                if int(r) >= run_count:
                    #Our final, average run
                    #print str(config_num*1.0/(configs))
                    #The brighter the line, the later the config(argh)
                    plt.plot(x, y, c=str(config_num*1.0/configs), lw=4.0)
                    #plt.plot(x, y, lw=4.0)
                else:
                    #plt.plot(x, y, c=np.random.randn(3,1), ls='--')
                    plt.plot(x, y, ls='--')
                    #insert plt.title here for our config name metadata
        config_num+=1
        
        
    plt.show()

    #need to test with different special cases
