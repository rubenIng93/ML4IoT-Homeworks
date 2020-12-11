import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#!pip install tensorflow-model-optimization
import tensorflow_model_optimization as tfmot
import zlib
from tensorflow import lite as tflite
import time

# define the class to handle the windows
class WindowGenerator:
    def __init__(self, input_width, mean, std):
        self.input_width = input_width 
        # dimension: [batch_size, 1st_dimension, 2nd, ...]
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        
        inputs = features[:, :6, :] #takes the 1st 6 values
        labels = features[:, -6:, :] #takes the last 6 values
        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.input_width, 2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train=False):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width * 2, # takes 6 inputs and 6 labels
                sequence_stride=1,
                batch_size=32)
        
        
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds.prefetch(tf.data.experimental.AUTOTUNE)

# define the custom metric (double output)
class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        #error = tf.reduce_mean(error, axis=0)
        error = tf.reduce_mean(error, axis=[0,1])
        self.total.assign_add(error)
        self.count.assign_add(1.)

        return

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result

# function for exploit the MLP
def use_mlp(units, width_multiplier, final_sparsity, summary=True):
    # Multi-layer perceptron
    mlp = keras.Sequential([
            keras.layers.Flatten(input_shape=shape),
            keras.layers.Dense(units=int(128*width_multiplier)),
            keras.layers.ReLU(),
            keras.layers.Dense(units=int(128*width_multiplier)),
            keras.layers.ReLU(),
            keras.layers.Dense(units=12),
            keras.layers.Reshape(shape, input_shape=(12,))
        ])

    # define the magnitude-based pruning
    pruning_params = {
        'pruning_schedule':tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=final_sparsity,
            begin_step=len(train_ds)*5,
            end_step=len(train_ds)*15
        )
    }
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    mlp = prune_low_magnitude(mlp, **pruning_params)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    mlp.compile(
            optimizer = 'adam',
            loss= keras.losses.MeanSquaredError(),
            metrics=[MultiOutputMAE()]
        )

    history = mlp.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)

    _, test_mae = mlp.evaluate(test_ds)

    if summary:
        mlp.summary()
    n_param = mlp.count_params()
    print("The mlp has {} parameters, MAE = {}".format(n_param, test_mae))
    print(f"Sparsity: {final_sparsity} - width multiplier: {width_multiplier}")

    return mlp

def use_cnn(units, width_multiplier, final_sparsity, summary=True):
    cnn = keras.Sequential([
        keras.layers.Conv1D(input_shape=shape, filters=int(64*width_multiplier), kernel_size=3),
        keras.layers.ReLU(),
        keras.layers.Flatten(),
        keras.layers.Dense(units=int(64*width_multiplier)),
        keras.layers.ReLU(),
        keras.layers.Dense(units=12),
        keras.layers.Reshape(shape, input_shape=(12,))
    ])

    # define the magnitude-based pruning
    pruning_params = {
        'pruning_schedule':tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=final_sparsity,
            begin_step=len(train_ds)*5,
            end_step=len(train_ds)*15
        )
    }
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    cnn = prune_low_magnitude(cnn, **pruning_params)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    cnn.compile(
            optimizer = 'adam',
            loss= keras.losses.MeanSquaredError(),
            metrics=[MultiOutputMAE()]
        )

    history = cnn.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)

    _, test_mae = cnn.evaluate(test_ds)

    if summary:
        cnn.summary()
    n_param = cnn.count_params()
    print("The cnn has {} parameters, MAE = {}".format(n_param, test_mae))
    print(f"Sparsity: {final_sparsity} - width multiplier: {width_multiplier}")

    return cnn

def save_as_keras(model, VERSION):
    keras_model_dir = f"./models/ht_version({VERSION})_keras"
    if not os.path.exists(keras_model_dir):
        os.makedirs(keras_model_dir)

    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
    model.save(keras_model_dir, signatures=concrete_func)
    return keras_model_dir

def convert_as_tflite(keras_path, print_size=True, ptq=True, compression=True):

    converter = tf.lite.TFLiteConverter.from_saved_model(keras_path)
    if ptq:# apply post training quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_models_path = './tflite_models/'
    if not os.path.exists(tflite_models_path):
        os.makedirs(tflite_models_path)

    if compression:
        filename = f'Group5_th_{VERSION}.zlib'
    else:
        filename = f'Group5_th_{VERSION}.tflite'

    with open(tflite_models_path+filename, 'wb') as fp:
        if compression:
            tflite_model = zlib.compress(tflite_model)
        fp.write(tflite_model)

    if print_size:
        path = tflite_models_path+filename
        print(f'New size is: {os.path.getsize(path)/1024:.2f} KB')


# define the argument
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, 
    help='Choose "a" or "b": \nversion a meets the requirement MAE<[0.5,1.8] size<2kB;\
    \nversion b meets the requirement MAE<[0.6,1.9] size<1.7kB')
args = parser.parse_args()

seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)

start = time.time()

print('***DATASET PREPARATION***')
# download and import the dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
# read the file as pandas dataset
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32) # cast in float 32

# split data in train/validation/test sets
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

# compute mean and std aimed to normalize the window
# (on the train data)
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

# define the input width
input_width = 6 

# invoke the generator and create the datasets
generator = WindowGenerator(input_width, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

VERSION = args.version

# define the output units
units = 2
shape = [6, 2]

if VERSION == 'a':
    print('\nThe version "a" has been chosen')
    print('***TRAIN THE MLP***')
    model = use_mlp(units=2, width_multiplier=0.25, final_sparsity=0.9)
    

if VERSION == 'b':
    print('\nThe version "b" has been chosen')
    print('***TRAIN THE CNN***')

    model = use_cnn(units=2, width_multiplier=0.12, final_sparsity=0.78)

# strip the model
model = tfmot.sparsity.keras.strip_pruning(model)

# save the model as keras model
print('***SAVE THE KERAS MODEL***')
keras_path = save_as_keras(model, VERSION)

# convert the model as tflite model
print('***CONVERT IN TFLITE MODEL***')
convert_as_tflite(keras_path)

end = time.time()
print(f'Tflite model version {VERSION} retrieved in {(end-start):.2f} s')