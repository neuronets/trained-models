# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.0.1-gpu-jupyter

RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        ca-certificates \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir PyYAML

WORKDIR /work
LABEL maintainer="Hoda Rajaei <rajaei.hoda@gmail.com>"
