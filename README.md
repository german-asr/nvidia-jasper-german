# Jasper German Speech Recognition
This repository contains scripts to train Jasper for German speech recognition.
Code was adapted from [https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper).
Code was copied from commit ``a2281e3``.

## Preparation
To get training data follow instructions in [https://github.com/ynop/megs](https://github.com/ynop/megs).
Download the LM from [https://github.com/ynop/german-asr-lm](https://github.com/ynop/german-asr-lm).

Build docker container
```
scripts/docker/build.sh
```

## Run

Launch docker container
```
scripts/docker/launch.sh \
	[german-asr-data]/data \
	[output-dir]/checkpoints \
	[output-dir]/results
```

Run training
```
scripts/train.sh
```
