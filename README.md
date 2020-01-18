# Jasper German Speech Recognition
This repository contains scripts to train Jasper for German speech recognition.
Code was adapted from [https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper).

## Preparation
To get training data follow instructions in [https://github.com/ynop/megs](https://github.com/ynop/megs).
Download the LM from [https://github.com/ynop/german-asr-lm](https://github.com/ynop/german-asr-lm).

## Results

### Best Path

| Model | Training-Data | dev_cv | test_cv | dev_tuda | test_tuda |
| ----- | ------------- | ------ | ------- | -------- | --------- |
| 3x5  | train | 27.98 | 33.18 | 21.64 | 21.88 |
| 5x5  | train | 27.10 | 31.89 | 20.82 | 21.21 |
| 7x5  | train | 24.79 | 29.13 | 18.62 | 19.29 |
| 10x5 | train | 22.08 | 26.40 | 16.58 | 16.93 |

### Beam Search with 6-gram LM

| Model | Training-Data | dev_cv | test_cv | dev_tuda | test_tuda |
| ----- | ------------- | ------ | ------- | -------- | --------- |
| 3x5  | train | 20.13 | 24.14 | 15.03 | 15.48 |
| 10x5 | train | 16.15 | 20.02 | 12.50 | 12.67 |
