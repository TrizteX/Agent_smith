# Agent_smith

## Overview
This is the pipeline based on [Link](https://arxiv.org/pdf/1806.04558.pdf) paper. It first creates a vector embedding from your voice input using the [encoder](https://arxiv.org/pdf/1710.10467.pdf) model. This vector is used to parameterize the output of the melspectrogram which is generated using the [synthesizer](https://arxiv.org/pdf/1712.05884.pdf) model. The text you provide is used as the input for the mel spectrogram. After the final spectrogram is created a [vocoder](https://arxiv.org/pdf/1802.08435.pdf) model is used to convert it back to .flac sound file.

## Output
Check out the output.wav for results

## Setup

### 1. Install Requirements

**Python 3.6 or 3.7** is needed to run the toolbox.

A) Install PyTorch (>=1.0.1). (Install first)
B) Install ffmpeg. (Install next)
C) Run `pip install -r requirements.txt` to install the remaining necessary packages.

### 2. Download Pretrained Models
Download the latest [here](https://www.dropbox.com/sh/sask2abwgf8sbvr/AADUoC_Jq4XqBiFfiUz4lE6Ta?dl=0).

The .zip file contains Encoder, Vocoder and Synthesizer models. Open each directory and copy the "saved_models" and paste them in them inside the original Encoder, Vocoder and Synthesizer file.

pretrained.zip/
Encoder/saved_models
Vocoder/saved_models
Synthesizer/saved_models

To be copied to

Agent_smith/
Encoder/
Vocoder/
Synthesizer/

## Run

* Run `python run.py`
* It will run a test first to make sure that the setup has been done correctly
* The prompt will ask you to provide the path to the input voice file, for easier access try.wav in samples directory has been provided.
* Next you can input any sentence you wish to be enunciated as the output 
* It automatically will save the output file in the parent directory

FOR WINDOWS USERS:
If you are using windows, packages like pyaudio and ffmpeg are known to cause errors. To fix this use run.py with --no_mp3_support. Run `python run.py --no_mp3_support`
along with this, the input file must be in .WAV format only.
