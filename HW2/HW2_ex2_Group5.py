import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zlib
from tensorflow import lite as tflite
import time

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

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

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

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds    

def use_dscnn(width_multiplier, summary=False, learning_rate=0.001, epochs=20):

    dscnn = keras.Sequential([
        keras.layers.Conv2D(filters=256*width_multiplier, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256*width_multiplier, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256*width_multiplier, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=8)
    ])

    checkpoint_filepath = './HW2/checkpoints/kws_HW2/weights'
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
        save_weights_only=True, monitor='val_sparse_categorical_accuracy', mode=max,
        save_best_only=True)

    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.makedirs(os.path.dirname(checkpoint_filepath))

    dscnn.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )    

    dscnn.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=cp)

    if summary:
        dscnn.summary()

    # evaluate the model
    dscnn.load_weights(checkpoint_filepath)
    _, accuracy = dscnn.evaluate(test_ds)
    n_param = dscnn.count_params()
    print("The {} has {} parameters, Accuracy = {}".format('dscnn', n_param, accuracy))

    return dscnn

def save_as_keras(model, version):
    saved_model_dir = f'./HW2/models/hw2_ex2_{version}'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tensor_spec)
    model.save(saved_model_dir, signatures=concrete_func)
    return saved_model_dir

def convert_as_tflite(path, ptq=True, print_size=True, compression=True):

    filename = f'Group5_kws_{VERSION}'
    # conversion in TFlite model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    if ptq:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    tflite_models_path = './HW2/tflite_models/'
    if not os.path.exists(tflite_models_path):
        os.makedirs(tflite_models_path)

    if compression:
        with open(tflite_models_path+filename+'.zlib', 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_model)
            fp.write(tflite_compressed)

    with open(tflite_models_path+filename+'.tflite', 'wb') as fp:
        fp.write(tflite_model)

    path = tflite_models_path+filename+'.zlib'
    print(f'New size is: {os.path.getsize(path)/1024:.2f} KB')


parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True, help='model version')
args = parser.parse_args()

VERSION = args.version.lower()

# setting the seed for replicability
seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)

# download and import the dataset
zip_path = tf.keras.utils.get_file(
  origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
  fname='mini_speech_commands.zip',
  extract=True,
  cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

# the labels are the different folders
LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != 'README.md']

# retrieve the splits
train_files = np.loadtxt('HW2/kws_train_split.txt', dtype=str) 
val_files = np.loadtxt('HW2/kws_val_split.txt', dtype=str) 
test_files = np.loadtxt('HW2/kws_test_split.txt', dtype=str)

if VERSION == 'c':
    # define the parameters
    STFT_OPTIONS = {
        'frame_length':256,
        'frame_step':128,
        'mfcc':False
    }

    # STFT OPTION
    options = STFT_OPTIONS
    strides = [2, 2]
    tensor_spec = (tf.TensorSpec([None, 32, 32, 1], tf.float32))

else:
    # define the parameters
    MFCCS_OPTION = {
        'frame_length':640,
        'frame_step':320,
        'mfcc':True,
        'lower_frequency':20,
        'upper_frequency':4000,
        'num_mel_bins':40,
        'num_coefficients':10
    }

    # MFCCS OPTIONS
    options = MFCCS_OPTION
    strides = [2, 1]
    tensor_spec = (tf.TensorSpec([None, 49, 10, 1], tf.float32))

# invoke the generator and create the datasets
generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

#tf.data.experimental.save(test_ds, './hw2_ex2_test_stft') # SAVE THE TEST DATASET

if VERSION =='a':
    print('***The version "a" has been chosen***')
    alpha = 0.31
    model = use_dscnn(width_multiplier=alpha)

    # save the model
    keras_path = save_as_keras(model, VERSION)
    # conversion in TFlite model
    convert_as_tflite(keras_path)

elif VERSION == 'b':
    print('***The version "b" has been chosen***')

    alpha = 0.22
    
    model = use_dscnn(width_multiplier=alpha, learning_rate=0.03)

    # save the model
    keras_path = save_as_keras(model, VERSION)
    # conversion in TFlite model
    convert_as_tflite(keras_path, ptq=False)

elif VERSION =='c':
    print('***The version "c" has been chosen***')

    alpha = 0.52
    
    model = use_dscnn(width_multiplier=alpha, learning_rate=0.01, epochs=30)

    # save the model
    keras_path = save_as_keras(model, VERSION)

    convert_as_tflite(keras_path)
    # conversion in TFlite model


    
    
