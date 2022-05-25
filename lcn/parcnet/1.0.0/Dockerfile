# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        ca-certificates \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN pip install --no-cache-dir nibabel scipy PyYAML

WORKDIR /work
LABEL maintainer="Hoda Rajaei <rajaei.hoda@gmail.com>"
