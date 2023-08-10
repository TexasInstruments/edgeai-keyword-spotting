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
This source file is used to compare MFCC generation methods between tensorflow
  and librosa, since the latter was found to give different results for effectively 
  identical parameters. This file looks at each step of the preprocessing algorithm
  to understand where the two methods deviate. It is not crucial to use this file, 
  but it does provide some interesting insight into the MFCC calculation process

This is intended to run on a development machine, not an EVM / TI SoC
'''

import tensorflow as tf
import numpy as np
import librosa
import soundfile
import scipy
from matplotlib import pyplot as plt

audio_data, sr = soundfile.read('down_0c40e715_nohash_0.wav')
audio_data = audio_data.astype(np.float32)

np.set_printoptions(precision=4, suppress=True)

fft_size = 512
frame_size = 512
hop_size = 320

tf_stft  = tf.signal.stft(audio_data, frame_length=frame_size, fft_length=fft_size, frame_step=hop_size) 

tf_spectrogram = magnitudes = tf.abs(tf_stft)

lr_stft = librosa.core.stft(y=audio_data.reshape((-1)),
                                n_fft=fft_size,
                                hop_length=hop_size,
                                win_length=frame_size, 
                                center=False)
                                
lr_spectrogram = np.abs(lr_stft)



difference = np.abs(tf_spectrogram - lr_spectrogram.T)
print("STFT Differences:\nmin:", np.min(difference), 
        "max:", np.max(difference), 
        "mean:", np.mean(difference), 
        "std:", np.std(difference))
    
print('do TF melspec')
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
tf_lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, fft_size//2+1, sr,lower_edge_hertz, upper_edge_hertz)
tf_mel_spectrograms = tf.tensordot(tf_spectrogram, tf_lin_to_mel_matrix, 1)
print(tf_mel_spectrograms.shape)
tf_mel_spectrograms.set_shape(tf_spectrogram.shape[:-1].concatenate(
  tf_lin_to_mel_matrix.shape[-1:]))
print(tf_mel_spectrograms.shape)

print('do LR melspec')
# lr_melspectrograms = lr_melspectrograms.T

lr_melspectrograms = librosa.feature.melspectrogram(S=lr_spectrogram**2, center=False, sr=sr, n_fft=fft_size, fmin=lower_edge_hertz, fmax=upper_edge_hertz, n_mels=num_mel_bins, htk=True)
difference = np.abs(tf_mel_spectrograms - lr_melspectrograms.T)
print("\nmelspectrogram Differences (LR melspectrogram function):\nmin:", np.min(difference), 
        "max:", np.max(difference), 
        "mean:", np.mean(difference), 
        "std:", np.std(difference))

lr_melspectrograms = np.dot(lr_spectrogram.T, tf_lin_to_mel_matrix.numpy())
difference = np.abs(tf_mel_spectrograms - lr_melspectrograms)
print("\nmelspectrogram Differences (tf mel filters mtx):\nmin:", np.min(difference), 
        "max:", np.max(difference), 
        "mean:", np.mean(difference), 
        "std:", np.std(difference))

lr_lin_to_mel_matrix = librosa.filters.mel(sr=sr, n_fft=fft_size, n_mels=num_mel_bins, fmin=lower_edge_hertz, fmax=upper_edge_hertz, htk=True, norm=None)
lr_melspectrograms = np.dot(lr_spectrogram.T, lr_lin_to_mel_matrix.T)
lr_melspectrograms = np.dot(lr_lin_to_mel_matrix, lr_spectrogram)
difference = np.abs(tf_mel_spectrograms - lr_melspectrograms.T)
print("\nmelspectrogram Differences (filters.mel):\nmin:", np.min(difference), 
        "max:", np.max(difference), 
        "mean:", np.mean(difference), 
        "std:", np.std(difference))

print('do log mel spectrograms')
tf_log_mel_spectrograms = tf.math.log(tf_mel_spectrograms + 1e-6)
lr_log_mel_spectrograms = np.log(lr_melspectrograms + 1e-6)

difference = np.abs(tf_log_mel_spectrograms - lr_log_mel_spectrograms.T)
print("\nlog-mel Differences :\nmin:", np.min(difference), 
        "max:", np.max(difference), 
        "mean:", np.mean(difference), 
        "std:", np.std(difference))


tf_mfcc =  tf.signal.mfccs_from_log_mel_spectrograms(tf_log_mel_spectrograms)
lr_mfcc = scipy.fftpack.dct(lr_log_mel_spectrograms, axis=-2, type=2, norm='ortho')

difference = np.abs(tf_mfcc - lr_mfcc.T)
print("\nmfcc Differences :\nmin:", np.min(difference), 
        "max:", np.max(difference), 
        "mean:", np.mean(difference), 
        "std:", np.std(difference))

print(tf_mfcc[0,:])
print(lr_mfcc[0,:])



fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
tf_mfcc_plot = tf_mfcc.numpy()
print(tf_mfcc_plot.shape)
print(lr_mfcc.shape)

img1 = librosa.display.specshow(tf_mfcc_plot, x_axis='time', ax=ax[0])

ax[0].set(title='tensorflow mfcc')

fig.colorbar(img1, ax=[ax[0]])

img2 = librosa.display.specshow(lr_mfcc.T, x_axis='time', ax=ax[1])

ax[1].set(title='librosa mfcc')

fig.colorbar(img2, ax=[ax[1]])

plt.show()