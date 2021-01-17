from DoSomething import DoSomething
import time
import json
import numpy as np
import argparse
import base64
import tensorflow as tf
import tensorflow.lite as tflite


class Subscriber(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)

        self.interpreter = tf.lite.Interpreter(model_path=str(VERSION) + '.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.LABELS = ['down', 'stop', 'right','left', 'up', 'yes', 'no', 'go']

    def notify(self, topic, msg):

        # Here waht it receives   
        # i.e. a Senml with a prediction 

        input_json = json.loads(msg)
        events = input_json['e']        

        for event in events:
            if event['n'] == 'audio':
                # means that there is a predicted label
                timestamp = input_json['bt']
                index = input_json['idx']   

                audio_string = event['vd']
                # manipulate the preprocessed audio
                # to get the correct shape
                preprocessed_audio = base64.b64decode(audio_string)
                preprocessed_audio = tf.io.decode_raw(preprocessed_audio, tf.float32)
                preprocessed_audio = tf.reshape(preprocessed_audio, [79,10,1])

                tensor = np.zeros([1,79,10,1], dtype=np.float32)
                tensor[0, :, :, :] = preprocessed_audio

                # get the prediction
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
                self.interpreter.invoke()
                y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
                y_pred = y_pred.squeeze()
                idx_label = np.argmax(y_pred)

                body = {
                    'bn': '/192.168.0.1/',
                    'bt': timestamp,
                    'e':[
                        {'n':'pred', 'u':'/', 't':0, 'v':str(idx_label)}
                    ],
                    'idx':index
                }

                senml = json.dumps(body)
          
                publisher.myMqttClient.myPublish('/277757/predictions', senml, qos=0)
                
parser = argparse.ArgumentParser()
# setting the input arguments
parser.add_argument("--model", type=int, help="1,2,3") 
args = parser.parse_args()

VERSION = args.model


if __name__ == '__main__':

    publisher = DoSomething('Pred-Publisher'+str(VERSION))
    publisher.run()

    test = Subscriber('Inference-subscriber'+str(VERSION))
    test.run()
    
    test.myMqttClient.mySubscribe("/277757/preprocessed_audio", qos=2)

    while True:
    	time.sleep(0.000001)

    test.end()
    publisher.end()


