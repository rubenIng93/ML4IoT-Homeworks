import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import time
import os
from scipy import signal
import io
import wave
import base64
import datetime
import json
import requests

# the file from minispeech commands are 16Khz

def read_audio(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label_id = tf.argmax(label == LABELS)
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    audio = tf.numpy_function(resampling_poly, [audio], tf.float32)

    return audio, label_id, audio_binary

def pad(audio):
    zero_padding = tf.zeros([8000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([8000])

    return audio

def resampling_poly(audio):
        audio = signal.resample_poly(audio, 1, 2) # resample 8kHz
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

def apply_mfcc(audio):
    # stft part
    stft = tf.signal.stft(audio,
                        frame_length=400,
                        frame_step=200,
                        fft_length=400)

    spectrogram = tf.abs(stft)

    num_spectrogram_bins = 400 // 2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    40, num_spectrogram_bins, 8000,
                    20, 4000)

    mel_spectrogram = tf.tensordot(spectrogram,
                linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :10]

    mfccs = tf.expand_dims(mfccs, -1)

    return mfccs

# download and import the dataset
zip_path = tf.keras.utils.get_file(
  origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
  fname='mini_speech_commands.zip',
  extract=True,
  cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

# retrieve the splits
test_files = np.loadtxt('kws_test_split.txt', dtype=str)

LABELS = ['down', 'stop', 'right','left', 'up', 'yes', 'no', 'go']

# invoke the little
little_interpreter = tf.lite.Interpreter(model_path='little.tflite')
little_interpreter.allocate_tensors()
little_input_details = little_interpreter.get_input_details()
little_output_details = little_interpreter.get_output_details()

accuracy = 0
tot_instances = len(test_files)
big_inferences = 0
communication_cost = 0

for i in range(len(test_files)):

    # read the audio
    audio, label_id, audio_binary = read_audio(test_files[i])
    audio = pad(audio)    
        
    # call the little model that uses STFT
    mfccs = apply_mfcc(audio)

    num_frames = (8000 - 400) // 200 + 1
    tensor = np.zeros([1, num_frames, 10, 1], dtype=np.float32)
    tensor[0, :, :, :] = mfccs[:, :, :]
    
    little_interpreter.set_tensor(little_input_details[0]['index'], tensor)
    little_interpreter.invoke()
    y_pred = little_interpreter.get_tensor(little_output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_pred_temp = sorted(y_pred, reverse = True)
    label_idx = np.argmax(y_pred)

    # compute the score margin
    score_margin = y_pred_temp[0] - y_pred_temp[1]
    
    if score_margin <= 0.08: # score margin set for sending less data possible

        # send to the big model for a better prediction

        big_inferences += 1

        # Modify here with your own IP adress
        # (where the file 'big_service.py' is located)
        url = 'http://169.254.169.248:8080/'

        # create the senml
        audio_b64bytes = base64.b64encode(audio_binary.numpy())
        audio_string = audio_b64bytes.decode()

        now = datetime.datetime.now()
        timestamp = int(now.timestamp())        

        body = {
            "bn": "169.254.100.190",
		    "bt": timestamp,
	      	"e": [
		      	{"n": "audio", "u": "/", "t":0, "vd": audio_string}
		    ]
	    }
  
        jsonbody = json.dumps(body)
        communication_cost += len(jsonbody)

        # request to the big service
        r = requests.put(url, data=jsonbody)

        if r.status_code == 200:
            body_from_big = r.json()
            events = body_from_big['e']

            for event in events:
                if event['n'] == 'label_idx':
                    label_idx = int(event['v'])
                else:
                    print('Error')
        else:
            print(r.text)
    
    if label_idx == label_id:
        accuracy += 1


print(f'Accuracy: {accuracy/tot_instances * 100:.3f}%')
#print('Big inferences: ', big_inferences)
print(f'Communication Cost: {communication_cost / 2**20:.3f} MB')

