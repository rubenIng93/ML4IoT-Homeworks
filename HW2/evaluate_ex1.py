import numpy as np
import argparse
import tensorflow as tf
import tensorflow.lite as tflite

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model path')
args = parser.parse_args()

PATH = args.model

# load the datasets
tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32),
                tf.TensorSpec([None, 6, 2]))

test_ds = tf.data.experimental.load('HW2_EX2_test', tensor_specs)
test_ds = test_ds.unbatch().batch(1)


interpreter = tf.lite.Interpreter(model_path=PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MAE = [0, 0]
counter = 0

for x, y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    # y_pred = [ 0, 6, 2]
    # 
    #print(y_pred)
    #print(y_true)
    abs_err = np.abs(y_pred - y_true)
    # abs_err = [6,2]
    #print(abs_err)
    MAE[0] += np.mean(abs_err[: ,:, 0])
    MAE[1] += np.mean(abs_err[:, :, 1])
    counter += 1

MAE[0] = MAE[0] / counter
MAE[1] = MAE[1] / counter

print(f"Model: {PATH} - MAE = {MAE}")



