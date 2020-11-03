import numpy as np
import tensorflow as tf
import argparse
import os
import pandas as pd
import time

# Conversion functions for TFDataset
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


# instantiate the argument parser
parser = argparse.ArgumentParser()

# add the required arguments
parser.add_argument("--input", type=str, help="Input Folder (RAW DATA)", 
                    required=True)
parser.add_argument("--output", type=str, help="Output File",
                    required=True)
args = parser.parse_args()


# instantiate useful variables
input_folder = args.input
out_file = args.output

# load the csv file
feature_list = ["date", "time", "temperature", "humidity"]
start = time.time()
csv_df = pd.read_csv("Homework_1_IoT/example.csv", sep=",", names=feature_list) 
end = time.time()
print("Pandas execution time: {:.4f}s".format(end-start))

# create the TFDataset
start = time.time()

with tf.io.TFRecordWriter(out_file) as writer:
    for i in range(len(csv_df)):
        
        mapping = {
            "date":_bytestring_feature([u"".join(csv_df["date"][i]).encode("utf-8")]),
            "time":_bytestring_feature([u"".join(csv_df["time"][i]).encode("utf-8")]),
            "temperature":_int_feature([csv_df["temperature"][i]]),
            "humidity":_int_feature([csv_df["humidity"][i]])
        }

        example = tf.train.Example(features=tf.train.Features(feature=mapping))

        writer.write(example.SerializeToString())

end = time.time()

print("Conversion Done in {:.4f} s".format(end-start))

