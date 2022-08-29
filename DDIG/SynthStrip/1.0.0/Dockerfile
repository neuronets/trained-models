# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        ca-certificates \
        git \
        libgomp1 \
        gcc \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# clean up
RUN rm -rf /root/.cache/pip

WORKDIR /work
LABEL maintainer="Hoda Rajaei <rajaei.hoda@gmail.com>"
