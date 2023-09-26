# Audio keyword spotting on Sitara MPU

This repository has been validated on Texas Insruments AM62x and AM62Ax microprocessors for the 8.6 and 9.0 SDKs. 

## What is this repo?

Keyword spotting allows automatic recognition of speech words within a limited vocabulary to assist human-machine interaction. This task often uses machine learning models like neural networks, and can run in a limited memory and CPU processing footprint on a microcontroller or microprocessor. 

This repo hosts python3 code for audio keyword spotting using two (tinyml-kws and matchboxnet) separate models trained on the Google Speech commands dataset (v1 and v2, respectively) running on the Am62x/AM62Ax SoC (aka 'the device') with the Linux SDK. It has been validated on these devices, but will likely run on other linux-based devices since there are very few hardware restrictions. 

## What is required

* An [AM62](https://www.ti.com/tool/SK-AM62) or [AM62A](https://www.ti.com/tool/SK-AM62A-LP) starter kit EVM
* SD card
* USB to USB-micro cable
* ethernet network connection
* USB microphone

## How to run the demo

1. Setup the SDK on an SD card for the starter kit EVM's selected device/SoC according to the respective device's user guide ([AM62x](https://dev.ti.com/tirex/explore/node?node=A__AdoyIZ2jtLBUfHZNVmgFBQ__am62x-devtools__FUz-xrs__LATEST&search=am62x) and [AM62Ax](https://dev.ti.com/tirex/explore/node?node=A__AQniYj7pI2aoPAFMxWtKDQ__am62ax-devtools__FUz-xrs__LATEST). Follow that guide until a linux terminal session is available through serial/USB or internet/SSH
2. clone this repository onto the device. This may require setting proxy variables like HTTPS_PROXY if the device is behind a firewall
3. run the 'install_on_evm.sh' script. This will download and build the portaudio dependency, and then setup a few python libraries for audio processing
4. Within the two subfolders, run either of the two python scripts starting with 'audio-inference'. Note that you will need to have the USB microphone plugged in and know the device index within linux. This is 1 by default, but the 'detect_microphone.py' script will also help identify them. 

### Words recognized

The tinyml model can recognize 12 words: Down, Go, Left, No, Off, On, Right, Stop, Up, Yes, Silence, Unknown

The matchboxnet model is capable of recognizing a larger set of words: visual, wow, learn, backward, dog, two, left, happy, nine, go, up, bed, stop, one, zero, tree, seven, on, four, bird, right, eight, no, six, forward, house, marvin, sheila, five, off, three, down, cat, follow, yes

## Support and Resources

Please direct questions to the [Processors e2e](https://e2e.ti.com/support/processors-group/processors/f/processors-forum)

* Main TI Edge AI page: [https://ti.com/edgeai](https://ti.com/edgeai)
* [TI Arm Processors](https://www.ti.com/microcontrollers-mcus-processors/arm-based-processors/overview.html)
* [ML Commons Tiny repo](https://github.com/mlcommons/tiny)
* [Nemo keyword spotting models using Matchboxnet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/commandrecognition_en_matchboxnet3x2x64_v2)