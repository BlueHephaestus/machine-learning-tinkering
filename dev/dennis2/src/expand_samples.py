import librosa
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle, gzip #for storing our data
import pickle

expanded_data_dir = "../data/expanded_samples.pkl.gz"
training_data_dir = "../data/training_data/"
validation_data_dir = None
test_data_dir = "../data/test_data/"

#Training, validation, and test
initial_sizes = [80, 40, 40]

#For each sample, make two augmented versions:
  #One with speed .9,
  #One with speed 1.1 .

#speed(1.1), speed(.9)
max_audio_len = 83521#289*289
expansion_factor = 3#amount data will be increased, used to get size of output array

def get_expanded_data(data_dir, data):

    #Expand the data
    print "Expanding Data..."
    f_index = 0
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        if os.path.isdir(f):
            f_index += 1
            local_sample_total = len(os.listdir(f))#for use with new increments of filenames
            local_sample_num = local_sample_total
            if local_sample_total <= initial_sizes[f_index]:
                #So we only expand if we haven't already expanded
                for sample in os.listdir(f):
                    #Do our librosa stuff
                    input_fname = f + "/" + sample
                    print "\tAugmenting %s..." % sample
                    output_slow_fname = f + "/" + "%i.wav" % (local_sample_num)#I realize two strings weren't necessary but clarity (as remixed by Zedd)
                    output_fast_fname = f + "/" + "%i.wav" % (local_sample_num+1)

                    y, sr = librosa.load(input_fname)
                    y_slow = librosa.effects.time_stretch(y, 0.9)
                    y_fast = librosa.effects.time_stretch(y, 1.1)

                    librosa.output.write_wav(output_slow_fname, y_slow, sr)
                    librosa.output.write_wav(output_fast_fname, y_fast, sr)
                    local_sample_num+=2

    #Loop through again and get the new samples as well
    print "Archiving Expanded Data..."
    label_num = 0
    sample_num = 0
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        label = f.split("/")[-1]

        if os.path.isdir(f):
            for sample in os.listdir(f):
                print "\tArchiving %s..." % sample
                #get our spectrogram for the sample
                samplerate, audio_raw = wav.read(f + "/" + sample)

                #start with zeros so we have our blank trailing space
                audio = np.zeros(max_audio_len)

                #convert our signed integer to an unsigned one by adding 32768, then divide by max of unsigned integer to get percentage and use that as our float value
                #fill as far as we've got
                for a_index, a in enumerate(audio_raw):
                    audio[a_index] = float((a+32768)/65536.0)
                    if a_index == max_audio_len-1:
                        break

                data[0][sample_num] = audio
                data[1][sample_num] = label_num
                sample_num+=1
        label_num += 1
    return data

def regenerate_expanded_data():
    #training_data = [np.zeros(shape=(80, max_audio_len), dtype=np.float32), np.zeros(shape=(80), dtype=np.int)]
    expanded_training_data = [np.zeros(shape=(80*expansion_factor, max_audio_len), dtype=np.float32), np.zeros(shape=(80*expansion_factor), dtype=np.int)]
    expanded_test_data = [np.zeros(shape=(40*expansion_factor, max_audio_len), dtype=np.float32), np.zeros(shape=(40*expansion_factor), dtype=np.int)]

    expanded_training_data = get_expanded_data(training_data_dir, expanded_training_data)
    expanded_test_data = get_expanded_data(test_data_dir, expanded_test_data)
    expanded_validation_data = expanded_test_data
    
    f = gzip.open(expanded_data_dir, "wb")
    pickle.dump((expanded_training_data, expanded_validation_data, expanded_test_data), f)
    f.close()

#regenerate_expanded_data()
