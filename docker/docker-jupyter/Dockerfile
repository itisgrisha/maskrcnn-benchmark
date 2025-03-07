ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs cython matplotlib jupyter tqdm

# Install PyTorch Nightly
ARG CUDA
RUN conda install -y pytorch-nightly cudatoolkit=${CUDA} -c pytorch

# Install OpenCV
RUN conda install -y opencv -c menpo \
 && conda clean -ya

WORKDIR /root

USER root

RUN mkdir /notebooks

WORKDIR /notebooks

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/itisgrisha/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop

RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

RUN jupyter notebook --generate-config

ENV CONFIG_PATH="/root/.jupyter/jupyter_notebook_config.py"

COPY "jupyter_notebook_config.py" ${CONFIG_PATH}

ENTRYPOINT ["sh", "-c", "jupyter notebook --allow-root -y --no-browser --ip=0.0.0.0 --config=${CONFIG_PATH}"]
