# this is the client it uses the MQTT protocol
 
# It publish the preprocessed audio and
# Subscribes for the output of the N neural network

from DoSomething import DoSomething
import time
import json
import datetime
import base64
import numpy as np
import os
import tensorflow as tf

# SignalGenerator class
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

       
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)
        self.preprocess = self.preprocess_with_mfcc
        

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        
        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio


    def get_spectrogram(self, audio):        
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        
        return ds

def majority_voting(predictions):

    val, count = np.unique(predictions, return_counts=True)
    top_idx = np.argmax(count)
    return val[top_idx]



class RpiClient(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)

        
    def notify(self, topic, msg):

        # Here what it receives   
        # i.e. a Senml with a prediction 

        input_json = json.loads(msg)
        events = input_json['e']

        for event in events:
            if event['n'] == 'pred':
                # means that there is a predicted label

                y_pred = int(event['v'])
                index = int(input_json['idx'])

                #print(f'INDEX: {index}, LABEL: {y_pred}')

                # add the element in the list
                pred_dictionary[index].append(y_pred)
                

                
if __name__ == '__main__':

    options = { # for MFCCS
        'frame_length':400,
        'frame_step':200,
        'mfcc':True,
        'lower_frequency':20,
        'upper_frequency':4000,
        'num_mel_bins':20,
        'num_coefficients':10
    }

    # dictionary where to store the prediction
    # k=index, v=prediction
    pred_dictionary = {}

    # instantiate publisher and subscriber
    test = DoSomething('Rasp-publisher')
    test.run()

    sub = RpiClient('Prediction-client')    
    sub.run()
    sub.myMqttClient.mySubscribe('/277757/predictions', qos=0)

    # dictionary where to store true labels
    # k=index, v=true label
    true_lbs_dict = {}

    # download and import the dataset
    zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    # retrieve the splits
    test_files = np.loadtxt('kws_test_split.txt', dtype=str)

    # retrieve the labels
    labels_file = open("Lab6/labels.txt", "r")
    LABELS = labels_file.read()
    LABELS = np.array(LABELS.split(" "))
    labels_file.close()

    # build the dataset
    generator = SignalGenerator(LABELS, 16000, **options)
    test_ds = generator.make_dataset(test_files)

    start_tot = time.time()
    
    i = 0 # the index

    for audio, y_true in test_ds:
        #initialize the index
        
        # initialize the dictionary
        pred_dictionary[i] = []
        
        # keep track of the true label
        true_lbs_dict[i] = y_true

        # build the senml with the preprocessed audio
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())

        # set the audio in base64
        audio_b64bytes = base64.b64encode(audio)
        audio_string = audio_b64bytes.decode()        

        senml = {
            'bn':'http://169.254.100.190/',
            'bt':timestamp,
            'e':[
                {'n':'audio', 'u':'/', 't':0, 'vd':audio_string}
            ],
            'idx':i
        }

        json_senml = json.dumps(senml)

        # Publish the senml with topic "preprocessed_audio"
        test.myMqttClient.myPublish('/277757/preprocessed_audio', json_senml, qos=2)      

        i += 1

                
    # wait for the responses
    while not pred_dictionary[len(test_files)-1]:
        time.sleep(0.5) 

    # stop the subscriber
    sub.end()   
    

    accuracy = 0
    tot = len(test_files)
    

    for idx in pred_dictionary.keys():
        y_true = true_lbs_dict[idx] # get the true label
        y_pred = majority_voting(pred_dictionary[idx])
        #print(f'IDX = {idx}')

        if y_pred == y_true:
            accuracy += 1

    print(f'Accuracy: {accuracy/tot*100 :.3f} %')

    end_tot = time.time()
    #print(f'Total execution time: {end_tot-start_tot} s')




                   
