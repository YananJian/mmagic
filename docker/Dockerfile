ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDA_ALIAS="116"
ARG CUDNN="8"
ARG MMCV="2.0.0rc4"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmagic
RUN conda clean --all
RUN git clone https://github.com/YananJian/mmagic /mmagic
WORKDIR /mmagic
ENV FORCE_CUDA="1"
ENV CUDA_HOME="/usr/local/cuda"
RUN pip install openmim
RUN mim install mmcv=="2.0.0rc4"
RUN pip install -r requirements/runtime.txt
RUN pip install -r requirements/tests.txt
RUN pip install --no-cache-dir -e .

#RUN git clone https://github.com/facebookresearch/xformers.git
#RUN cd xformers && git submodule update --init --recursive && pip install -r requirements.txt && pip install -e .
RUN pip install xformers==0.0.16
RUN pip uninstall --yes diffusers && pip install diffusers==0.13.1
RUN apt-get update && apt-get install -y vim

RUN pip install jupyterlab
RUN pip install gradio
