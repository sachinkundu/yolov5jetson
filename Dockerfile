FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-venv curl python3-tk

RUN python3.8 -m pip install -U pip setuptools==59.5.0

RUN git clone --recursive --branch v1.9.0 http://github.com/pytorch/pytorch

ADD pytorch-1.9-jetpack-4.5.1.patch /pytorch

WORKDIR "/pytorch"

RUN python3.8 -m pip install -r requirements.txt

RUN git submodule sync

RUN git submodule update --init --recursive --jobs 0

RUN patch -Np1 < pytorch-1.9-jetpack-4.5.1.patch

ARG USE_NCCL=0
ARG USE_DISTRIBUTED=0
ARG USE_QNNPACK=0
ARG PYTORCH_BUILD_VERSION=1.9.0
ARG USE_PYTORCH_QNNPACK=0
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
ARG PYTORCH_BUILD_NUMBER=1

RUN apt-get install -y cmake libopenblas-dev libopenmpi-dev
RUN pip3.8 install -r requirements.txt
RUN pip3.8 install scikit-build
RUN pip3.8 install ninja
RUN pip install --upgrade --force-reinstall numpy

RUN python3.8 setup.py install

WORKDIR /

ADD requirements.txt /

RUN apt remove -y python3-yaml

RUN pip3.8 install -r requirements.txt

RUN git clone https://github.com/pytorch/vision.git

WORKDIR /vision

RUN git checkout origin/release/0.10

RUN python3.8 setup.py install

WORKDIR /
