FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 


#Setting up ENV
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV PYTHON_VERSION=3.6.8
ENV PATH /opt/conda/bin:$PATH


RUN apt-get update && \
    pytorchDeps='build-essential cmake git vim ca-certificates libjpeg-dev libpng-dev' && \
    condaDeps='wget bzip2 libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion' && \
    apt-get install -y --no-install-recommends $pytorchDeps $igraphDeps $condaDeps && \
    rm -rf /var/lib/apt/lists/* && \
#Install miniconda3
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
#Fixes a conda bug - https://github.com/conda/conda/issues/6030
    #conda update python && \
#Python Dependencies
#For some reason, pip works for h5py and pandas, and conda doesn't
    pip --no-cache-dir install numpy h5py cloud-volume==0.44.0 future && \
    conda install scipy pytorch==0.4.0 cython torchvision -c pytorch && \
    git clone https://github.com/seung-lab/error-detection.git && \
    conda clean -t -p && \
    apt-get purge -y --auto-remove $pytorchDeps $condaDeps #$igraphDeps #need these


WORKDIR /error-detection/tasks
ENTRYPOINT ["bash"]