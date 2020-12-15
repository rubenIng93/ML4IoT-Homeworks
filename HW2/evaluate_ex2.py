import numpy as np
import argparse
import tensorflow as tf
import tensorflow.lite as tflite

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--prep', type=str, required=True, help='sftf or mfccs')
args = parser.parse_args()

PATH = args.model
PREPROCESSING = args.prep

if PREPROCESSING == 'stft':
    tensor_spec = (tf.TensorSpec([None, 32, 32, 1], dtype=tf.float32),
                            tf.TensorSpec([None], dtype=tf.int64))
    test_ds = tf.data.experimental.load('hw2_ex2_test_stft', tensor_spec) # load the test dataset

elif PREPROCESSING == 'mfcc':
    tensor_spec = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
                            tf.TensorSpec([None], dtype=tf.int64))
    test_ds = tf.data.experimental.load('hw2_ex2_test_mfccs', tensor_spec) # load the test dataset

test_ds = test_ds.unbatch().batch(1)

interpreter = tf.lite.Interpreter(model_path=PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = 0
counter = 0

for x, y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    if y_true == np.argmax(y_pred):
        accuracy += 1
    counter += 1

accuracy = accuracy / counter

print(f"MODEL IN PATH: {PATH} - ACCURACY = {accuracy*100:.2f} %")
