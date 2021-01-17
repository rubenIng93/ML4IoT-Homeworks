import base64
import cherrypy
import tensorflow as tf
import json
import tensorflow.lite as tflite
import numpy as np
import wave


class BigService(object): 
    exposed = True 

    def __init__(self):
        # load the big model
        self.big_interpreter = tf.lite.Interpreter(model_path='big.tflite')
        self.big_interpreter.allocate_tensors()
        self.big_input_details = self.big_interpreter.get_input_details()
        self.big_output_details = self.big_interpreter.get_output_details()

        self.LABELS = ['down', 'stop', 'right','left', 'up', 'yes', 'no', 'go']
        
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(20, 400//2+1, 16000, 20, 4000)

    def pad(self, audio):
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([16000])

        return audio

    
    def preprocess(self, audio_bytes):
        # decode and normalize
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)
        
        # padding
        audio = self.pad(audio)

        # STFT
        tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        stft = tf.signal.stft(
            tf_audio,
            frame_length=400, 
            frame_step=200, 
            fft_length=400)
        spectrogram = tf.abs(stft)

        # MFCC
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :10]

        # add a channel dimension
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs


    def GET(self, *uri, **params):
        pass

    def POST(self, *uri, **params): 
        pass

    def PUT(self, *path, **query): 
        # retrieve the senml
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        events = input_body['e']

        timestamp = input_body['bt']

        audio_string = None

        for event in events:
            if event['n'] == 'audio':
                audio_string = event['vd']

        if audio_string is None:
            raise cherrypy.HTTPError(400, 'no audio event')        

        #print(audio_string)
        audio_bytes = base64.b64decode(audio_string)

        mfccs = self.preprocess(audio_bytes)

        # set the empty tensor 
        tensor = np.zeros([1, 79, 10, 1], dtype=np.float32)
        tensor[0, :, :, :] = mfccs[:, :, :]

        self.big_interpreter.set_tensor(self.big_input_details[0]['index'], tensor)
        self.big_interpreter.invoke()
        y_pred = self.big_interpreter.get_tensor(self.big_output_details[0]['index'])
        y_pred = y_pred.squeeze()
        #print(y_pred)
        idx_label = np.argmax(y_pred)

        body = {
            'bn': '/192.168.0.1/',
            'bt': 0,
            'e':[
                {'n':'label_idx', 'u':'/', 't':0, 'v':str(idx_label)}
            ]
        }

        senml = json.dumps(body)

        return senml

    
    def DELETE(self): 
        pass

		
if __name__ == '__main__':
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			'tools.sessions.on': True
		}
	}
	cherrypy.tree.mount(BigService(), '', conf)

	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()
