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
This file is intended to run on PC, and will convert the nemo model into an equivalent ONNX format model. The dynamic sizes within the ONNX model are modified to work for 1second worth of input audio.

'''

import nemo.collections.asr as asr
import numpy as np
import os
import librosa as lr
import torch
import onnxruntime as ort
import onnx
import yaml



def get_mfcc_lr(audio_data, sampling_rate):
    '''
    nemo.collections.asr.modules.AudioToMFCCPreprocessor
        window_size: 0.025
        window_stride: 0.01
        window: hann
        n_mels: 64
        n_mfcc: 64
        n_fft: 512
    '''
    n_fft = 512
    n_mels = 64
    n_mfcc = 64
    win_len_s = 0.025
    win_step_s = 0.01

    win_size = int(sampling_rate * win_len_s)
    win_step = int(sampling_rate * win_step_s)

    melspec = lr.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_fft=n_fft, win_length=win_size,  hop_length=win_step,  n_mels=n_mels,  power=2, center=True, htk=True, norm=None)
    
    S = np.log(melspec + 1e-6)

    mfcc = lr.feature.mfcc(S=S, norm='ortho', n_mfcc=n_mfcc)
    return mfcc


def export_to_onnx(torch_model, example_data):
    '''
    Export the model to onnx. The nemo model is effectively a wrapper over the same one in pytorch
    '''

    example_data = example_data
    torch_model.export("matchboxnet-export.onnx", onnx_opset_version=11,  do_constant_folding=True, input_example=example_data)
    #the model is currently using dynamic shapes; we need those to be static
    onnx_model = onnx.load("matchboxnet-export.onnx")

    from onnx.tools import update_model_dims
    #set static dimensions
    #https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#updating-models-inputs-outputs-dimension-sizes-with-variable-length
    static_onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, {"input_name": [1, 64, 101]}, {"output_name": [1, 35]})
    
    #run shape inference to aid visualization in tools like netron (not essential)
    # https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model
    final_onnx_model = onnx.shape_inference.infer_shapes(static_onnx_model)
    onnx.save(final_onnx_model, 'matchboxnet.onnx')

def test_model(path="matchboxnet.onnx", data=None):
    '''
    Try running the ONNX model from file
    '''
    sess_options = ort.SessionOptions()

    interpreter = ort.InferenceSession(path, providers=['CPUExecutionProvider'], provider_options=[{}], sess_options=sess_options)
    input_details = interpreter.get_inputs()
    
    data = data.astype(np.float32)
    result = interpreter.run(None, {input_details[0].name: data})
    best_class = int(np.argmax(result[0][0,:]))
    return best_class, result

def main():
    # device = torch.device('cuda:0')
    device='cpu'

    # downloaded from nvidia's NGC platform
    m = asr.models.EncDecClassificationModel.restore_from('commandrecognition_en_matchboxnet3x2x64_v2.nemo')
    m.to(device) #necessary, as it defaults to CUDA if a GPU is installed with CUDA drivers
    c = m.to_config_dict()
    # print(c)
    l = c['labels']

    with open('labels.yaml', 'w') as yaml_file:
        yaml.dump({'labels':list(l)}, yaml_file)

    #verify the model works
    wavs = [f for f in os.listdir() if 'wav' in f]
    out = m.transcribe(paths2audio_files=wavs)
    print(wavs)
    print(out)

    get_feat = m.preprocessor.get_features

    audio, sr = lr.load(wavs[0], sr=16000)

    t_audio = torch.tensor(audio)
    t_len = torch.tensor(sr, dtype=torch.int32)


    torch_mfcc, _ = get_feat(t_audio, t_len)

    lr_mfcc = get_mfcc_lr(audio, sr)

    export_to_onnx(m, audio)

    best_lr, result_lr = test_model(path='matchboxnet_shape.onnx', data=lr_mfcc[None,:])
    best_torch, result_torch = test_model(path='matchboxnet_shape.onnx', data=torch_mfcc[None,:].numpy())


    print('Librosa preprocessing vs. torch gave result: "%s" vs. "%s"' % (l[best_lr], l[best_torch]))

if __name__ == '__main__': 
    main()