# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.09-py3
FROM ${FROM_IMAGE_NAME}


RUN apt-get update && apt-get install -y libsndfile1 && apt-get install -y sox && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg libavc1394-0 libavc1394-dev sox libsox3 libsox-dev libsox-fmt-mp3


WORKDIR /workspace/jasper
COPY . .

RUN pip install --disable-pip-version-check -U -r requirements.txt

WORKDIR /code
