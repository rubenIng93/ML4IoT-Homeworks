import numpy as np
import tensorflow as tf
import argparse
import os
import pandas as pd
import time
import datetime
from scipy.io import wavfile
from scipy import signal

def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _audio_feature(audio_path):
        # read the file audio
    rate, audio = wavfile.read(audio_path)

    # resampling with poly-phase filtering at the frequency 16000Hz
    sampling_ratio = int(rate / 16000)
    start = time.time()
    audio = signal.resample_poly(audio, 1, sampling_ratio)
    end = time.time()
    resampling_time = end - start
    print ("Resampling time: {:.4f} s".format(resampling_time))

    # cast to the original datatype (Int16)
    audio = audio.astype(np.int16)

    # produce the TFRecord
    return _float_feature(audio.tolist())


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

# loop aimed to explore the different files
# in the folder passed by the user
start = time.time()

for _file in os.listdir(input_folder):
    if _file.find(".csv") >= 0:
        csv_file = input_folder+"/"+_file

end = time.time()
#print("Folder exploration time: {:.4f}s".format(end-start))

# load the csv file
feature_list = ["date", "time", "temperature", "humidity", "audio"]
start = time.time()
csv_df = pd.read_csv(csv_file, sep=",", names=feature_list) 
end = time.time()
print("File reading execution time: {:.4f}s".format(end-start))

# create the TFDataset
start = time.time()

with tf.io.TFRecordWriter(out_file) as writer:
    for i in range(len(csv_df)):

        # 18/10/2020,09:45:34 must be converted as POSIX timestamp
        date = time.mktime(datetime.datetime.strptime(csv_df["date"][i]+","+csv_df["time"][i], 
            "%d/%m/%Y,%H:%M:%S").timetuple())
        #print(date)
        
        mapping = {
            "datetime":_int_feature([int(date)]), # as int since is in POSIX
            "temperature":_int_feature([csv_df["temperature"][i]]),
            "humidity":_int_feature([csv_df["humidity"][i]]),
            "audio":_audio_feature(input_folder+"/"+csv_df["audio"][i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=mapping))

        writer.write(example.SerializeToString())

end = time.time()

print("Conversion Done in {:.4f} s".format(end-start))
print("TFRecord size: {} KB".format(os.path.getsize(out_file)/1000))
