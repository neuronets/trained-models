# syntax=docker/dockerfile:1

FROM centos:7


# shell settings
WORKDIR /freesurfer

# install utils
RUN yum -y update
RUN yum -y install libgomp python3
RUN yum clean all

# python packages
RUN pip3 install scipy surfa torch==1.10.2
RUN pip3 install /external/fsmodule

# install synthstrip
RUN cp /external/mri_synthstrip /freesurfer/

# configure model
ENV FREESURFER_HOME /freesurfer

# setup rest of the env
ENV OS Linux
ENV FSF_OUTPUT_FORMAT nii.gz

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir PyYAML

# clean up
RUN rm -rf /external /root/.cache/pip

ENTRYPOINT ["python3", "/freesurfer/mri_synthstrip"]

WORKDIR /work
LABEL maintainer="Hoda Rajaei <rajaei.hoda@gmail.com>"
