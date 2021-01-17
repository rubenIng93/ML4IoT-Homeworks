import numpy as np
import tensorflow as tf
import argparse
import os
import pandas as pd
import time
from tensorflow import keras
import zlib
import tensorflow_model_optimization as tfmot
from scipy import signal

def save_as_keras(model, version):
    saved_model_dir = f'./HW3_Group5/ex2/{version}'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tensor_spec)
    model.save(saved_model_dir, signatures=concrete_func)
    return saved_model_dir

def convert_as_tflite(path, ptq=True, print_size=True, compression=True):

    filename = f'{VERSION}'
    # conversion in TFlite model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    if ptq:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    tflite_models_path = './HW3_Group5/ex2/tflite_models/'
    if not os.path.exists(tflite_models_path):
        os.makedirs(tflite_models_path)

    if compression:
        with open(tflite_models_path+filename+'.zlib', 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_model)
            fp.write(tflite_compressed)
    else:
        with open(tflite_models_path+filename+'.tflite', 'wb') as fp:
            fp.write(tflite_model)
    
    path = tflite_models_path+filename+'.tflite'
    print(f'New size is: {os.path.getsize(path)/1024:.2f} KB')

def lr_scheduler(epoch, lr):
    if epoch % 6 == 0 and epoch != 0:
        if epoch < 10:
            return lr * 0.1
        else:
            return lr * tf.math.exp(-0.5)
    else:
        return lr


# instantiate the argument parser
parser = argparse.ArgumentParser()

# add the required arguments
parser.add_argument("--version", type=int, help="Choose 1, 2 or 3", 
                    required=True)

args = parser.parse_args()

VERSION = args.version

# SignalGenerator class
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False, resampling=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.resampling = resampling
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            if self.resampling is True:
                self.sampling_rate = 8000
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
        if self.resampling is True:
            audio = tf.numpy_function(self.resampling_poly, [audio], tf.float32)
            self.sampling_rate = 8000

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def resampling_poly(self, audio):
        audio = signal.resample_poly(audio, 1, 2) # resample 8kHz
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)

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

# setting the seed
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
labels_file = open("labels.txt", "r")
LABELS = labels_file.read()
LABELS = np.array(LABELS.split(" "))
labels_file.close()

# retrieve the splits
train_files = np.loadtxt('kws_train_split.txt', dtype=str) 
val_files = np.loadtxt('kws_val_split.txt', dtype=str) 
test_files = np.loadtxt('kws_test_split.txt', dtype=str)

options = {
        'frame_length':400,
        'frame_step':200,
        'mfcc':True,
        'lower_frequency':20,
        'upper_frequency':4000,
        'num_mel_bins':20,
        'num_coefficients':10
    }

  
strides = [2, 1]
num_frames = (16000 - options['frame_length']) // options['frame_step'] + 1
tensor_spec = (tf.TensorSpec([None, num_frames, 10, 1], tf.float32))

# invoke the generator and create the datasets
generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

if VERSION == 1:
    # as first the big model of the previous exercise

    big = keras.Sequential([
        keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),    
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),    
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.6),
        keras.layers.Dense(units=8)
    ])

    # checkpoint callback
    checkpoint_filepath = './checkpoints/kws_{}/weights'.format('1')
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
        save_weights_only=True, monitor='val_sparse_categorical_accuracy', mode='max',
        save_best_only=True)

    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.makedirs(os.path.dirname(checkpoint_filepath))

    # learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    big.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )

    big.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=[cp, lr_callback])

    # evaluate the big
    big.load_weights(checkpoint_filepath)
    _, accuracy = big.evaluate(test_ds)
    n_param = big.count_params()
    print("The {} has {} parameters, Accuracy = {}".format(str(VERSION) + ' model', n_param, accuracy))

    # save the keras model
    keras_dir = save_as_keras(big, '1')

    # convert in tflite
    convert_as_tflite(keras_dir, ptq=False, compression=False)

elif VERSION == 2:
    # second model

    big = keras.Sequential([
        keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),    
        keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),    
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.6),
        keras.layers.Dense(units=8)
    ])

    # checkpoint callback
    checkpoint_filepath = './checkpoints/kws_{}/weights'.format('2')
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
        save_weights_only=True, monitor='val_sparse_categorical_accuracy', mode=max,
        save_best_only=True)

    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.makedirs(os.path.dirname(checkpoint_filepath))

    # learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    big.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )

    big.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=[cp, lr_callback])

    # evaluate the big
    big.load_weights(checkpoint_filepath)
    _, accuracy = big.evaluate(test_ds)
    n_param = big.count_params()
    print("The {} has {} parameters, Accuracy = {}".format(str(VERSION) + ' model', n_param, accuracy))

    # save the keras model
    keras_dir = save_as_keras(big, '2')

    # convert in tflite
    convert_as_tflite(keras_dir, ptq=False, compression=False)



elif VERSION == 3:
    # third model

    big = keras.Sequential([
        keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=8)
    ])

    # checkpoint callback
    checkpoint_filepath = './checkpoints/kws_{}/weights'.format('3')
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
        save_weights_only=True, monitor='val_sparse_categorical_accuracy', mode=max,
        save_best_only=True)

    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.makedirs(os.path.dirname(checkpoint_filepath))

    # learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    big.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )

    big.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=[cp, lr_callback])

    # evaluate the big
    big.load_weights(checkpoint_filepath)
    _, accuracy = big.evaluate(test_ds)
    n_param = big.count_params()
    print("The {} has {} parameters, Accuracy = {}".format(str(VERSION) + ' model', n_param, accuracy))

    # save the keras model
    keras_dir = save_as_keras(big, '3')

    # convert in tflite
    convert_as_tflite(keras_dir, ptq=False, compression=False)


else:
    print('Error: Choose a model between 1,2,3')