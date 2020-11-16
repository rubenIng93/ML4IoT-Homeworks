import numpy as np
import tensorflow as tf
import argparse
import time
import pyaudio
from scipy import signal
import io
import wave
import subprocess
import os
import sys

# function that returns the recording sequence
# as concatenation of binary frames
def record_audio(stream, pyaudio, samp_rate, chunk, record_sec, dev_index, resolution, alternative_acquisition=False):
        
    stream.start_stream()
    # powersafe mode
    subprocess.Popen(
       ['sudo', '/bin/sh','-c','echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
    )
    #print("Start Recording")

    # loop through stream and append audio chunks to frame array
    # instantiate the buffer

    if alternative_acquisition:
        frames = []

    buffer = io.BytesIO()    

    for ii in range(int((samp_rate / chunk) * record_sec)): 
        if ii == int((samp_rate / chunk) * record_sec) - performance_mode_trigger:  
            # set the performance mode   
            subprocess.Popen(
                ['sudo', '/bin/sh','-c','echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
            )
        if alternative_acquisition:             
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        else:    
            buffer.write(stream.read(chunk, exception_on_overflow=False))
    #print("Stop Recording")
    
    # stop the stream
    stream.stop_stream()

    if alternative_acquisition:    
        # 2ND MODALITY
            temp_input = wave.open(buffer, 'wb')
            temp_input.setnchannels(1)
            temp_input.setsampwidth(pyaudio.get_sample_size(resolution))
            temp_input.setframerate(samp_rate)
            temp_input.writeframes(b"".join(frames))
            temp_input.close()
    
    # move the cursor back to the beginning of the "file"
    buffer.seek(0)
    
    if alternative_acquisition:
        # 2ND MODALITY
        sample, _ = tf.audio.decode_wav(buffer.read())
        # this is a tensor
        return sample
    else:    
        binary_audio = buffer.read()
        return binary_audio

# function that applies the poly-phase filtering
# it returns the audio
def resample_audio(audio, frequency):
    sampling_ratio = int(samp_rate / frequency)
    #start = time.time()
    # load the audio from the buffer 
    # COMMENT IT FOR 2ND MODALITY
    if type(audio) == bytes:
        audio = np.frombuffer(audio, dtype=np.int16)
    #print(audio.shape)
    audio = signal.resample_poly(audio, 1, sampling_ratio)
    # cast to float normalizing datatype as should do decode_wav()
    #audio = audio.astype(np.int16)
    audio = audio.astype(np.float32) / 32768.0
    #end = time.time()
    #resampling_time = end - start
    #print("Resampling time: {:.4f} s".format(resampling_time))
    #print(audio.shape)
    return audio

# It returns the spectrogram
def retrieve_spectrogram(resampled_audio, frame_length, frame_step):
    # convert the signal
    tf_audio = tf.convert_to_tensor(resampled_audio, dtype=tf.float32)
    
    if tf_audio.shape == (16000, 1):
        # 2ND MODALITY
        tf_audio = tf.squeeze(tf_audio, 1)
    
    # convert the waveform in a spectrogram applying the STFT
    #start = time.time()
    stft = tf.signal.stft(tf_audio,
                            frame_length=frame_length,
                            frame_step=frame_step,
                            fft_length=frame_length)
    #end = time.time()
    #print("STFT: {:.4f} ".format(end-start))
    spectrogram = tf.abs(stft)
    return spectrogram

# it saves on disk the mfccs as string of bytes
def retrieve_mfccs(linear_to_mel_weight_matrix, out_file):

    #start = time.time()
    
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
    #end = time.time()

    #print("MFCC tranformation time: {:.4f} s".format(end-start))
    mfccs_byte = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(out_file, mfccs_byte)


# avoid to print the warning due to the mic
#os.close(sys.stderr.fileno())

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
performance_mode_trigger = 2
# choose an higher value if the costraint 
# (pre-processing < 80ms) has not been reached; must be anyway
# less than int((samp_rate / chunk) * record_sec) in this case 10
# the suggestion is to leave it between 1 and 3
# otherwise starting the performance mode earlier, doesn't
# improve the result.
pai_audio = pyaudio.PyAudio() # instantiate the pyaudio
samp_rate = 48000 # sampling rate 48kHz
chunk = 4800 # size of the chunk
record_sec = 1 # second to record
dev_index = 0 # device index found by p.get_device_info_by_index(ii)
resolution = pyaudio.paInt16 
 # create audio stream and append audio chuncks to frame array
stream = pai_audio.open(format=resolution, rate=samp_rate, channels=1,
                        input_device_index=dev_index, input=True,
                        frames_per_buffer=chunk)
stream.stop_stream()

# resampling setting
resampling_frequency = 16000 # 16 kHz

# spectrogram setting
# frame length = frequency=16000Hz * l(s) = 40ms
frame_length = int(16000 * 0.040) 
# frame step = frequency(Hz) * stride(s) = 20ms
frame_step = int(16000 * 0.020) 

# mfccs setting all these parameters are constant
# than the matrix is constant -> compute it only once!
num_spectrogram_bins = 321 # computed as spectrogram.shape[-1]
mel_bins = 40
lower_frequency = 20
upper_frequency = 4000
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        mel_bins, 
        num_spectrogram_bins,
        16000, # frequency
        lower_frequency, 
        upper_frequency
    )

# reset the performance counter
subprocess.Popen(['sudo', '/bin/sh','-c','echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset']).wait()

for i in range(int(num_samples)):
    start_tot = time.time()
    # record the audio
    #start = time.time()
    audio = record_audio(stream, pai_audio, samp_rate, chunk, record_sec, dev_index, resolution, True)
    #end = time.time()
    #print("\n>>>>Recording time: {:.4f} s<<<<\n".format(end-start))
    # apply resampling
    #start = time.time()
    audio = resample_audio(audio, resampling_frequency)
    # apply stft to get the spectrogram
    spectrogram = retrieve_spectrogram(audio, frame_length, frame_step)
    # retrieve and save on disk mfccs
    string_path = output_path+"/mfccs"+str(i+1)+".bin"
    retrieve_mfccs(linear_to_mel_weight_matrix, string_path)
    #end = time.time()
    end_tot = time.time()
       
    #print("\n>>>>Preprocessing time: {:.4f} s<<<<".format(end-start))
    print("{:.3f}".format(end_tot-start_tot))
    

# retrieve the tiem spent for each clock freq
subprocess.call(['cat', '/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'])

# retrive the powersafe mode after the loop has been finished
subprocess.Popen(
    ['sudo', '/bin/sh','-c','echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
)   

stream.close() # close the stream
pai_audio.terminate() # close the pyaudio object
