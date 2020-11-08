import numpy as np
import tensorflow as tf
import argparse
import time
import pyaudio
from scipy.io import wavfile
from scipy import signal
import io
from scipy.io.wavfile import read

# function that returns the recording sequence
# as concatenation of binary frames
def record_audio(samp_rate, chunk, record_sec, dev_index, resolution):
        
    # instantiate the pyaudio
    audio = pyaudio.PyAudio()

    # create audio stream and append audio chuncks to frame array
    stream = audio.open(format=resolution, rate=samp_rate, channels=1,
                        input_device_index=dev_index, input=True,
                        frames_per_buffer=chunk)

    print("Start Recording")
    frames = []

    # loop through stream and append audio chunks to frame array
    for ii in range(int((samp_rate / chunk) * record_sec)):
        data = stream.read(chunk)
        frames.append(data)

    print("Stop Recording")

    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.start_stream()
    stream.close()
    audio.terminate()
    binary_audio = b"".join(frames)
    
    return binary_audio

# function that applies the poly-phase filtering
# it returns the audio
def resample_audio(audio, frequency):
    sampling_ratio = int(samp_rate / frequency)
    start = time.time()
    audio = np.frombuffer(audio)
    audio = signal.resample_poly(audio, 1, sampling_ratio)
    # cast to the original datatype (Int16)
    #audio = audio.astype(np.int16)
    audio = audio.astype(np.float32, order='C') / 32768.0
    end = time.time()
    resampling_time = end - start
    print("Resampling time: {:.4f} s".format(resampling_time))
    return audio

# It returns the spectrogram
def retrieve_spectrogram(resampled_audio, frame_length, frame_step):
    # convert the signal
    resampled_audio = tf.io.parse_tensor(resampled_audio, out_type=tf.float32)
    tf_audio, _ = tf.audio.decode_wav(resampled_audio)
    tf_audio = tf.squeeze(resampled_audio, 1)

    # convert the waveform in a spectrogram applying the STFT

    stft = tf.signal.stft(tf_audio,
                            frame_length=frame_length,
                            frame_step=frame_step,
                            fft_length=frame_length)

    spectrogram = tf.abs(stft)

    return spectrogram

# it saves on disk the mfccs as string of bytes
def retrieve_mfccs(spectrogram, mel_bins, lower_frequency, upper_frequency, out_file):

    spectrogram = tf.io.parse_tensor(spectrogram, out_type=tf.float32)

    # compute the log scale mel spec

    num_spectrogram_bins = spectrogram.shape[-1]
    start = time.time()
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        mel_bins, 
        num_spectrogram_bins,
        16000, # frequency
        lower_frequency, 
        upper_frequency
    )

    mel_spectrogram = tf.tensordot(
        spectrogram,
        linear_to_mel_weight_matrix,
        1
    )
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]
    ))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrogram)[...,:10]
    end = time.time()

    print("MFCC tranformation time: {:.4f} s".format(end-start))
    mfccs_byte = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(out_file, mfccs_byte)

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add the required arguments
parser.add_argument("--num_samples", type=str, help="Number of samples", 
                    required=True)
parser.add_argument("--output", type=str, help="Output path",
                    required=True)
args = parser.parse_args()

# instantiate useful variables
num_samples = args.num_samples
output_path = args.output

# recording setting
samp_rate = 48000 # sampling rate 48kHz
chunk = 4800 # size of the chunk
record_sec = 1 # second to record
dev_index = 0 # device index found by p.get_device_info_by_index(ii)
resolution = pyaudio.paInt16 

# resampling setting
resampling_frequency = 16000 # 16 kHz

# spectrogram setting
# frame length = frequency=16000Hz * l(s) = 40ms
frame_length = int(16000 * 0.040) 
# frame step = frequency(Hz) * stride(s) = 20ms
frame_step = int(16000 * 0.020) 

#mfccs setting
mel_bins = 40
lower_frequency = 20
upper_frequency = 4000

print("***Start application***")

for i in range(int(num_samples)):
    # record the audio
    start = time.time()
    audio = record_audio(samp_rate, chunk, record_sec, dev_index, resolution)
    end = time.time()
    print("Recording time: {:.4f}".format(end-start))
    # apply resampling
    start = time.time()
    audio = resample_audio(audio, resampling_frequency)
    # apply stft to get the spectrogram
    spectrogram = retrieve_spectrogram(audio, frame_length, frame_step)
    # retrieve and save on disk mfccs
    string_path = output_path+"mfccs_sample_"+(i+1)
    retrieve_mfccs(spectrogram, mel_bins, lower_frequency, upper_frequency, string_path)
    end = time.time()
    #print(read(audio))
    #print(type(audio))
    print("Preprocessing time: {:.4f}".format(end-start))

print("***JOB DONE***")