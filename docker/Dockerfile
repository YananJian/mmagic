ARG PYTORCH="1.12.1"
#ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDA_ALIAS="113"
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
RUN git clone https://github.com/YananJian/mmagic.git /mmagic
WORKDIR /mmagic
ENV FORCE_CUDA="1"
RUN pip install openmim
RUN mim install mmcv=="2.0.0rc4"
RUN pip install -r requirements/runtime.txt
RUN pip install -r requirements/tests.txt
RUN pip install --no-cache-dir -e .

RUN git clone https://github.com/facebookresearch/xformers.git
RUN cd xformers && git submodule update --init --recursive && pip install -r requirements.txt && pip install -e .

# COMMENT OUT THE FOLLOWING LINES OT GET A CLEAN MMAGIC INSTALL
#RUN pip install git+https://github.com/facebookresearch/segment-anything.git

RUN apt-get update && apt-get install -y vim
RUN pip install jupyter notebook

#RUN pip install diffusers==0.12.1
#RUN pip install streamlit-drawable-canvas==0.8

#RUN git clone https://github.com/runwayml/stable-diffusion.git /stable-diffusion
#RUN pip install streamlit>=0.73.1


#RUN pip install --no-cache-dir torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
#RUN pip install torch-fidelity==0.3.0 torchmetrics==0.6.0 pytorch-lightning==1.4.2 omegaconf==2.1.1
#RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
#RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
#RUN pip install kornia==0.6
