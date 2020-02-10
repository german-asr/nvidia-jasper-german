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

## Results
Word error rates in %.

| Decoding                | Training-Data | dev_cv | test_cv | dev_tuda | test_tuda |
| ----------------------- | ------------- | ------ | ------- | -------- | --------- |
| Best Path               | train         | 23.95  | 28.65 | 18.13 | 18.86 |
| Beam Search (6-gram LM) | train         | 17.12  | 21.00 | 13.00 | 13.19 |

| Decoding                | Training-Data | dev_swc | test_swc | dev_voxforge | test_voxforge |
| ----------------------- | ------------- | ------ | ------- | -------- | --------- |
| Best Path               | train         | 13.97  | 11.79 | 11.14 | 10.55 |
| Beam Search (6-gram LM) | train         | 10.05  | 8.80 | 8.78 | 8.50 |
