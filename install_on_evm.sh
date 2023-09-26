#!/bin/bash
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
echo "Running setup for audio processing. This requires a network connection"

if [[ ! -d ./portaudio ]]; then
    echo "portaudio not found; buidling portaudio"
    git clone https://github.com/PortAudio/portaudio/
    cd portaudio
    ./configure
    make -j
    make install
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    echo "built and installed portaudio to /usr/local/lib, and exported to library path"    
fi

echo "install librosa and pyaudio for audio processing and audio capture, respectively"
PROXY_INFO=""
#PROXY_INFO= " --proxy http://webproxy.ext.ti.com:80 "
pip3 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org  $PROXY_INFO install librosa pyaudio soundfile

echo "Finished setup"

