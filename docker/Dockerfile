FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ARG TIMEZONE
RUN ln -sf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git zip curl wget emacs\
    build-essential libssl-dev libbz2-dev libreadline-dev \
    libopencv-dev libffi-dev libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYENV_ROOT /home/root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

ARG PYTHON_VERSION
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

RUN pip install -U pip setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH $PYTHONPATH:/work
WORKDIR /work
