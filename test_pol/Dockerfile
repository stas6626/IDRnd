FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update -y && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    yasm \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
#    opencv-data \
#    libopencv-dev \
#    libjasper1 \
#    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    software-properties-common

# Install python
RUN add-apt-repository -y ppa:jonathonf/python-3.6 \
    && add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main' \
    && apt-get update -y \
    && apt-get install -y python3.6 python3.6-dev \
    libjasper1 \
    libjasper-dev \
    python3-distutils \
    && ln -sfn /usr/bin/python3.6 /usr/local/bin/python \
    && ln -sfn /usr/bin/python3.6 /usr/bin/python3 \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir \
    astor==0.7.1 \
    audioread==2.1.6 \
    grpcio==1.20.1 \
    Markdown==3.1 \
    protobuf==3.7.1 \
    python-apt \
    Werkzeug==0.15.2 \
    catboost==0.8.1.1 \
    cycler==0.10.0 \
    decorator==4.3.0 \
    dill==0.2.8.2 \
    h5py==2.7.1 \
    imgaug==0.2.5 \
    kiwisolver==1.0.1 \
    matplotlib==2.2.2 \
    networkx==2.1 \
    numpy==1.14.3 \
    opencv-contrib-python==3.4.2.17 \
    pandas==0.23.0 \
    Pillow==5.1.0 \
    pyparsing==2.2.0 \
    python-dateutil==2.7.3 \
    pytz==2018.4 \
    PyWavelets==0.5.2 \
    scikit-image==0.13.1 \
    scikit-learn==0.19.1 \
    scipy==1.1.0 \
    setuptools==39.1.0 \
    Shapely==1.6.4.post1 \
    six==1.11.0 \
    tqdm==4.23.4 \
    keras==2.2.4 \
    tensorflow-gpu==1.13.1 \
    https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
    torchvision==0.2.2.post3 \
    xgboost==0.82 \
    lightgbm==2.2.3 \
    librosa==0.6.3 \
    kapre==0.1.4
