#
# Copyright (C) 2023 Texas Instruments Incorporated - http://www.ti.com/
#
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
'''
This source file is a standalone application for running live inference for 
 keyword spotting on audio data from a microphone connected in linux 
'''

import os, time
import pyaudio
import numpy as np
import librosa
import onnxruntime as ort
import yaml

p = pyaudio.PyAudio()

class AudioInference(object):
    PROCESSING_RATE = 16000 #Hz
    NUM_MFCC_BINS = 101
    NUM_MFCC_PER_BIN = 64
    BIN_WINDOW_SIZE = int(PROCESSING_RATE * 0.025)
    BIN_WINDOW_STEP = int(PROCESSING_RATE * 0.01)
    LOGIT_THRESHOLD = 12 #arbitrary..
    def __init__(self, modeldir, modelname, rate=48000, data_format=pyaudio.paInt16, channels=1, device_index=1, labels_file='labels.yaml'):
        print('setup')
        self.rate=rate
        self.format=data_format
        self.channels=channels
        self.device_index=device_index

        self.input_stream = None

        modelpath = os.path.join(modeldir, modelname)

        with open(labels_file,'r') as f:
            self.word_labels = yaml.safe_load(f)['labels']

        self.sess_options = ort.SessionOptions()
        self.interpreter = interpreter = ort.InferenceSession(modelpath, providers=['CPUExecutionProvider'], provider_options=[{}], sess_options=self.sess_options)
        self.input_details = self.interpreter.get_inputs()

        
    def setup(self):
        self.inference_session = None

        seconds_per_chunk = 0.5
        chunk_size = int(self.rate * seconds_per_chunk)
        self.last_chunk = None
        print('open input stream')
        self.input_stream = p.open(rate=self.rate, channels=self.channels, format=self.format, input=True, input_device_index=self.device_index, output=False, stream_callback=self.inference_callback, frames_per_buffer=chunk_size)
        print('opened..')

    def stop(self):
        self.input_stream.close()

    def calculate_features(self, audio_data, sr=PROCESSING_RATE):
        '''
        Calculate features from one second of audio data at sampling rate sr

        
        '''
        n_fft = 512
        n_mels = 64
        n_mfcc = 64

        melspec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, win_length=AudioInference.BIN_WINDOW_SIZE,  hop_length=AudioInference.BIN_WINDOW_STEP,  n_mels=n_mels,  power=2, center=True, htk=True, norm=None)
    
        # print(melspec)
        # S = lr.power_to_db(melspec)
        S = np.log(melspec + 1e-6)

        mfcc = librosa.feature.mfcc(S=S, norm='ortho', n_mfcc=n_mfcc)
        return mfcc
        


    def run_inference(self, mfcc):

        mfcc = mfcc[None,:]
        t1 = time.time_ns()//1000/1000
        result = self.interpreter.run(None, {self.input_details[0].name: mfcc})
        t2 = time.time_ns()//1000/1000
        print("Inference Time is %0.3f ms" % (t2-t1))

        if np.max(result[0][0,:]) > AudioInference.LOGIT_THRESHOLD:
            best_class = int(np.argmax(result[0][0,:]))
        else: best_class = -1
        return best_class, result
    
    def convert_audio_for_features(self, raw_input, input_rate, output_rate=PROCESSING_RATE):
        audio_data = raw_input / max([np.max(raw_input),abs(np.min(raw_input))]) #normalize to [-1:1]
            
        audio_resample = librosa.resample(audio_data.astype(np.float32), orig_sr=input_rate, target_sr=output_rate)

        return audio_resample


    def inference_callback(self, audio_buffer, frame_count, time_info, flag):
        if self.last_chunk is None:
            print('Skipping first chunk... typically takes a moment for librosa to initialize')
        else:
            t1 = time.time_ns()//1000/1000
            audio_data = np.frombuffer(self.last_chunk+audio_buffer, dtype=np.int16)


            audio_resample = self.convert_audio_for_features(audio_data, input_rate = self.rate, output_rate=AudioInference.PROCESSING_RATE)
           
            mfcc = self.calculate_features(audio_resample)
            
            t2 = time.time_ns()//1000/1000
            print("Preprocess Time is %0.3f ms" % (t2-t1))

            best_class, class_logits = self.run_inference(mfcc)
            class_name = 'unknown' if best_class < 0 else self.word_labels[best_class]
            
            print('******detected speech: ' + class_name + '******\n')
            # print(class_logits)
            # print(np.max(class_logits))
        self.last_chunk = audio_buffer

        return self.last_chunk, pyaudio.paContinue

# audio_data = stream.read(num_frames=input_rate*seconds_per_run, exception_on_overflow = False)

def main(modeldir, modelname):
    print('main')
    audio = AudioInference(modeldir=modeldir, modelname=modelname, device_index=14, )
    audio.setup()

    while (audio.input_stream.is_active()): time.sleep(5)

    audio.stop()

def test_on_file(modeldir, modelname):
    import soundfile
    audio_inf = AudioInference(modeldir=modeldir, modelname=modelname, device_index=1, )

    audio_data, sr = soundfile.read('no_0cb74144_nohash_1.wav')


    audio_resampled = audio_inf.convert_audio_for_features(audio_data, sr)
    mfcc = audio_inf.calculate_features(audio_resampled)
    
    best_class, class_logits = audio_inf.run_inference(mfcc)
    class_name = 'unknown' if best_class < 0 else audio_inf.word_labels[best_class]    
    print('******\ndetected class: ' + class_name + '\n******')

if __name__ == '__main__':
    main('.', 'matchboxnet.onnx')
    # test_on_file('./models', 'kws_ref_model_float32.tflite')