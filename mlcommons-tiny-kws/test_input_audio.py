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
This source file is intended to test audio input using pyaudio and save collected
  audio to a .wav file. The tested device was a FIFINE K669 USB microphone with 
  48kHz sample rate
'''

import pyaudio
import numpy as np
import soundfile
import time

device_index = 1
input_rate = 48000
seconds_per_run = 5

p = pyaudio.PyAudio()

#get audio
stream = p.open(rate=input_rate, channels=1, format=pyaudio.paInt16, input=True, input_device_index=device_index) 

print('Start recording')
audio_data = stream.read(num_frames=input_rate*seconds_per_run, exception_on_overflow = False)

print('Stopped recording')

t1 = time.time_ns()
audio_formatted = np.frombuffer(audio_data, dtype=np.int16) #convert from byte stream to audio samples. Littleendian signed 16bit ints#output into floats
t2 = time.time_ns()
print(audio_formatted.shape)


print('Processed audio in %f ms' % ((t2-t1)/1e6))
print('Audio resampled')

soundfile.write('processed_audio.wav', audio_formatted, input_rate)
