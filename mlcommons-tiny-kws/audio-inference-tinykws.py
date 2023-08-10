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
import os, time, sys
import pyaudio
import numpy as np
import librosa
import tflite_runtime.interpreter as tflite
import soundfile
import math
import scipy.fftpack

p = pyaudio.PyAudio()

class AudioInference(object):
    PROCESSING_RATE = 16000 #Hz
    NUM_MFCC_BINS = 49
    NUM_MFCC_PER_BIN = 10
    BIN_WINDOW_SIZE = int(PROCESSING_RATE * 0.03)
    BIN_WINDOW_STEP = int(PROCESSING_RATE * 0.02)
    def __init__(self, modeldir, modelname, rate=48000, data_format=pyaudio.paInt16, channels=1, device_index=1):
        '''
        @param modeldir: The directory the model is contained in
        @param modelname: the name of the modelfile within the modeldir 
        @param rate: the sampling rate of the microphone
        @param data_format: the type of the incoming audio samples
        @param channels: The number of channels supplied by the microphone. In general, only 1 should be used
        @param device_index: The index of the microphone as understood by linux. Use detect_microphone.py is this is unknown, but it should be 1 in general
        '''
        print('initializing inference session')
        self.rate=rate
        self.format=data_format
        self.channels=channels
        self.device_index=device_index

        self.input_stream = None

        modelpath = os.path.join(modeldir, modelname)

        # the set of words within this KWS model's dictionary. The ordering is important
        self.word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes", "Silence", "Unknown"]

        print('loading KWS model...')
        self.interpreter = tflite.Interpreter(modelpath, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print('loaded.')
        
    def setup(self):
        '''
        Configure the input data stream
        '''
        self.inference_session = None

        seconds_per_chunk = 0.5
        chunk_size = int(self.rate * seconds_per_chunk)
        self.last_chunk = None
        print('opening input audio stream...')
        self.input_stream = p.open(rate=self.rate, channels=self.channels, format=self.format, input=True, input_device_index=self.device_index, output=False, stream_callback=self.inference_callback, frames_per_buffer=chunk_size)
        print('opened.')

    def stop(self):
        self.input_stream.close()

    def calculate_features(self, audio_data, sr=PROCESSING_RATE):
        '''
        The default configuration of MFCC using librosa does not match what tensorflow's recommended method of calculating MFFC's.
        
        Through trial, error, and analysis, the following differences were found between the implementation in the keyword-spotting preprocessing script (which is almost identical to the TF MFCC code (https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms)) and the recommend/default librosa implementation:
            1. STFT differs when n_fft samples != window_size
            2. Mel filters in frequency space are different due to  normalization
            3. In mel-spectrogram, librosa uses power-decibel scale to get log-mel whereas tensorflow implementation uses natural log


        Since preprocessing must match to achieve the same results, this implementation of MFCC calculation changes several parameters to match how it was done in tensorflow
        
        '''
        n_fft = np.log2(AudioInference.BIN_WINDOW_SIZE)
        n_fft = 2**math.ceil(n_fft)
        
        #although the actual win_length should not be =n_fft per the preprocessing instructions, the reality is that librosa's result DOES NOT MATCH tensorflow unless n_fft=win_length
        stft = librosa.core.stft(y=audio_data.reshape((-1)),
                        n_fft=n_fft,
                        hop_length=AudioInference.BIN_WINDOW_STEP,
                        win_length=n_fft, 
                        center=False)
        
        spectrogram = np.abs(stft) # don't compute **2 for power since TF implmentation skipped this

        #calculate a mel filter transformation matrix
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
        lin_to_mel_matrix = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mel_bins, fmin=lower_edge_hertz, fmax=upper_edge_hertz, htk=True, norm=None)
        # multiply matrices to get melspectorgram
        melspectrograms = np.dot(lin_to_mel_matrix, spectrogram)

        # use natrual log instead of dB power scale, per TF implementation
        log_mel_spectrograms = np.log(melspectrograms + 1e-6)

        # DCT with orthogonal normalization convert log-mel-spectrogram to mel cepstrum coefficients (MFCC)
        mfcc = scipy.fftpack.dct(log_mel_spectrograms, axis=-2, type=2, norm='ortho')[...,:10,:]
        return mfcc

    def run_inference(self, mfcc):
        '''
        Run inference on the MFCC's that represent a second of audio
        '''
        tensor = np.zeros((1, AudioInference.NUM_MFCC_BINS, AudioInference.NUM_MFCC_PER_BIN, 1)) #shape 49,10
        tensor[0,:,:,0] = mfcc.transpose() # extend shape to agree with model expectation
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor.astype(np.float32))

        t1 = time.time_ns()//1000 / 1000
        self.interpreter.invoke()
        t2 = time.time_ns()//1000 / 1000
        print("Inference Time is %0.4f ms" % (t2-t1))

        class_probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return class_probs  
    
    def convert_audio_for_features(self, raw_input, input_rate, output_rate=PROCESSING_RATE):
        '''
        Convert the input from the microphone into a form that the primary preprocessing function (calculate features) is prepared to use

        :param raw_input: The raw buffer of audio samples
        :param input_rate: the sampling rate of the audio samples
        :param output_rate: The desired sample-rate of the output buffer of audio
        '''
        audio_data = raw_input / max([np.max(raw_input),abs(np.min(raw_input))]) #normalize to [-1:1]
            
        audio_resample = librosa.resample(audio_data.astype(np.float32), orig_sr=input_rate, target_sr=output_rate)

        return audio_resample


    def inference_callback(self, audio_buffer, frame_count, time_info, flag):
        '''
        A callback registered to pyaudio for running the main body of this application. Most parameters are unused. It is required to return and audio sample and a 'continuation' flag
        '''
        if self.last_chunk is None:
            print('Skipping first chunk')
        else:
            t1 = time.time_ns()//1000/1000
            #audio data is collected in 500 ms samples and processed in 1000 ms chunks, i.e. a 50% sliding window
            audio_data = np.frombuffer(self.last_chunk+audio_buffer, dtype=np.int16)


            audio_resample = self.convert_audio_for_features(audio_data, input_rate = self.rate, output_rate=AudioInference.PROCESSING_RATE)
           
            mfcc = self.calculate_features(audio_resample)
            
            t2 = time.time_ns()//1000/1000
            print("Preprocess Time is %0.3f ms" % (t2-t1))
            print('run inference next..')

            class_probs = self.run_inference(mfcc)

            c = np.argmax(class_probs)
            print('******\ndetected class: ' + str(self.word_labels[c]) + '\n******')

            print('finished with inference\n')

        self.last_chunk = audio_buffer

        return self.last_chunk, pyaudio.paContinue


def main(modeldir, modelname):
    print('main in audio-inference-tinykws')
    audio = AudioInference(modeldir=modeldir, modelname=modelname, device_index=14, )
    audio.setup()

    while (audio.input_stream.is_active()): time.sleep(2)

    audio.stop()

def test_on_file(modeldir, modelname, audio_filename='no_0cb74144_nohash_1.wav'):
    '''
    Test the model on an input file instead of live input from microphone. This file should be 1 second of audio long.
    '''
    audio_inf = AudioInference(modeldir=modeldir, modelname=modelname, device_index=14, )

    audio_data, sr = soundfile.read(audio_filename)


    audio_resampled = audio_inf.convert_audio_for_features(audio_data, sr)
    mfcc = audio_inf.calculate_features(audio_resampled)
    
    class_prob = audio_inf.run_inference(mfcc)
    c = np.argmax(class_prob)
    print('******\ndetected class: ' + str(audio_inf.word_labels[c]) + '\n******')

if __name__ == '__main__':
    main('./models', 'kws_ref_model_float32.tflite')
    # test_on_file('./models', 'kws_ref_model_float32.tflite')