import scipy.io.wavfile as wav
import numpy as np
import os  #for directories
import cPickle, gzip #for storing our data
import pickle

import theano
import theano.tensor as T
#They recommended just doing this so we get a spectrogram instead of using mfccs
data_dir = "../data/samples.pkl.gz"
training_data_dir = "../data/training_data/"
validation_data_dir = None
test_data_dir = "../data/test_data/"
#max_audio_len = 83968
max_audio_len = 84100

max = 0.0
min = 0.0

#Get maximum length
'''
for f in os.listdir(os.path.abspath(training_data_dir)):
    f = os.path.abspath(training_data_dir + f)

    if os.path.isdir(f):
        for sample_index, sample in enumerate(os.listdir(f)):
            #get our spectrogram for the sample
            samplerate, audio_raw = wav.read(f + "/" + sample)
            if len(audio_raw) > max:
                max = len(audio_raw)
print max
'''



def get_data_old(data_dir, data_x, data_y):
    #if we ened up needing the not-data-wrapper version
    label_num = 0
    labels = []#so we actually know what each index value associated represents, str reps
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        label = f.split("/")[-1]
        labels.append(label)

        if os.path.isdir(f):
            for sample_index, sample in enumerate(os.listdir(f)):
                #get our spectrogram for the sample
                samplerate, audio_raw = wav.read(f + "/" + sample)

                #start with zeros so we have our blank trailing space
                audio = np.zeros(max_audio_len)

                #convert our signed integer to an unsigned one by adding 32768, then divide by max of unsigned integer to get percentage and use that as our float value
                #fill as far as we've got
                for a_index, a in enumerate(audio_raw):
                    audio[a_index] = float((a+32768)/65536.0)
                data_x[sample_index] = audio
                data_y[sample_index][label_num] = 1.0
                #audio = [float((a+32768)/65536.0) for a in audio]
        label_num += 1
    return zip(data_x, data_y)

def get_data(data_dir, data):
    label_num = 0
    sample_num = 0
    labels = []#so we actually know what each index value associated represents, str reps
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        label = f.split("/")[-1]
        labels.append(label)

        if os.path.isdir(f):
            for sample in os.listdir(f):
                #get our spectrogram for the sample
                samplerate, audio_raw = wav.read(f + "/" + sample)

                #start with zeros so we have our blank trailing space
                audio = np.zeros(max_audio_len)
                label = np.zeros(2)

                #convert our signed integer to an unsigned one by adding 32768, then divide by max of unsigned integer to get percentage and use that as our float value
                #fill as far as we've got
                for a_index, a in enumerate(audio_raw):
                    audio[a_index] = float((a+32768)/65536.0)
                label[label_num] = 1.0
                #have to convert to list before we can do assignment correctly, then we have to convert back after.
                '''
                data[sample_num] = list(data[sample_num])
                data[sample_num] = audio, label
                data[sample_num] = tuple(data[sample_num])
                '''
                data[0][sample_num] = audio
                data[1].append(label_num)
                
                #data[sample_num][0] = audio
                #data[sample_num][1] = label
                sample_num+=1
        label_num += 1
    return data

'''
def regenerate_data_old():
    training_data = zip(np.zeros(shape=(80, max_audio_len)), np.zeros(shape=(80, 2)))
    test_data = zip(np.zeros(shape=(40, max_audio_len)), np.zeros(shape=(40, 2)))

    training_data = get_data(training_data_dir, training_data)
    test_data = get_data(test_data_dir, test_data)
    validation_data = test_data
    
    f = gzip.open(data_dir, "wb")
    pickle.dump((training_data, validation_data, test_data), f)
    f.close()
'''

def regenerate_data():
    #training_data = zip(np.zeros(shape=(max_audio_len, 80)), np.zeros(shape=(2, 80)))
    #test_data = zip(np.zeros(shape=(max_audio_len, 40)), np.zeros(shape=(2, 40)))
    training_data = [np.zeros(shape=(80, max_audio_len)), []]
    test_data = [np.zeros(shape=(40, max_audio_len)), []]

    training_data = get_data(training_data_dir, training_data)
    #training_data = zip(training_data[0], training_data[1])
    test_data = get_data(test_data_dir, test_data)
    #test_data = zip(test_data[0], test_data[1])
    validation_data = test_data
    
    f = gzip.open(data_dir, "wb")
    pickle.dump((training_data, validation_data, test_data), f)
    f.close()

def load_data_wrapper():
    f = gzip.open(data_dir, "rb")
    training_data, validation_data, test_data = pickle.load(f)
    f.close()

    return (training_data, validation_data, test_data)

def load_data_shared():
    f = gzip.open(data_dir, "rb")
    training_data, validation_data, test_data = pickle.load(f)
    #print np.asarray(training_data[0]).shape()#validation_data.shape(), test_data.shape()
    '''
    for x in training_data:
        print len(x)
    '''

    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]
