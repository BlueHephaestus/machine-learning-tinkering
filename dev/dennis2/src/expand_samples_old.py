import pysox
import numpy as np
import os
import cPickle, gzip #for storing our data
import pickle

data_dir = "../data/expanded_samples.pkl.gz"
training_data_dir = "../data/training_data/"
validation_data_dir = None
test_data_dir = "../data/test_data/"

#For each sample, make two augmented versions:
  #One with speed .9,
  #One with speed 1.1 .

#speed(1.1), speed(.9)
max_audio_len = 83521#289*289
expansion_factor = 3#amount data will be increased
decrease_speed_effect = pysox.CEffect("speed", ['.9'])
increase_speed_effect = pysox.CEffect("speed", ['1.1'])
#effect = pysox.CEffect("vol", [ b'18dB' ])

def get_expanded_data(data_dir, data):
    label_num = 0
    sample_num = 0
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        label = f.split("/")[-1]

        if os.path.isdir(f):
            local_sample_total = len(os.listdir(f))#for use with new increments of filenames
            local_sample_num = local_sample_total
            infiles = []
            for sample in os.listdir(f):
                infiles.append(f+"/"+sample)
            for infile in infiles:
                #Do our sox stuff
                infile = pysox.CSoxStream(infile)

                output_fname = f + "/%i.wav" % (local_sample_num)
                outfile = pysox.CSoxStream(output_fname, 'w', infile.get_signal())

                chain = pysox.CEffectsChain(infile, outfile)
                chain.add_effect(decrease_speed_effect)
                chain.flow_effects()
                infile.close()
                outfile.close()

                local_sample_num+=1
            '''
            for infile in infiles:

                infile = pysox.CSoxStream(infile)
                output_fname = f + "/%i.wav" % (local_sample_num)
                outfile = pysox.CSoxStream(output_fname, 'w', infile.get_signal())

                chain = pysox.CEffectsChain(infile, outfile)
                chain.add_effect(increase_speed_effect)
                chain.flow_effects()
                infile.close()
                outfile.close()
                
                local_sample_num+=1
            '''
            sample_num+=1
        label_num += 1

def regenerate_expanded_data():
    expanded_training_data = [np.zeros(shape=(80*expansion_factor, max_audio_len)), np.zeros(shape=(80*expansion_factor,))]
    expanded_test_data = [np.zeros(shape=(40*expansion_factor, max_audio_len)), np.zeros(shape=(40*expansion_factor,))]

    expanded_training_data = get_expanded_data(training_data_dir, expanded_training_data)
    expanded_test_data = get_expanded_data(test_data_dir, expanded_test_data)
    expanded_validation_data = expanded_test_data
    
    f = gzip.open(data_dir, "wb")
    pickle.dump((expanded_training_data, expanded_validation_data, expanded_test_data), f)
    f.close()

regenerate_expanded_data()
