FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    zip \
    emacs \
    libopencv-dev \
    libffi-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev curl \
    wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYENV_ROOT /home/root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

ENV PYTHON_VERSION 3.9.10
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

RUN pip install -U pip setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH $PYTHONPATH:/work

WORKDIR /work
