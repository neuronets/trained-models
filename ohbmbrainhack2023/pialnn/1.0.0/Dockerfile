FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/miniconda3/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV PIPENV_VENV_IN_PROJECT=1
ENV JCC_JDK=/usr/lib/jvm/java-8-openjdk-amd64
RUN USE_CUDA=1
RUN CUDA_VERSION=11.6.1
RUN CUDNN_VERSION=8
RUN LINUX_DISTRO=ubuntu
RUN DISTRO_VERSION=20.04
RUN TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
> /etc/apt/apt.conf.d/keep-cache
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt-get install -y --no-install-recommends \
build-essential \
ca-certificates \
ccache \
curl \
git \
wget \
cmake \
gfortran \
libspatialindex-dev && \
apt-get install -y --no-install-recommends llvm-10 lldb-10 llvm-10-dev libllvm10 llvm-10-runtime
RUN rm -rf /var/lib/apt/lists/*
ENV PYTHON_VERSION=3.7
ENV CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
RUN curl -fsSL -v -o ~/miniconda.sh -O ${CONDA_URL} && \
chmod +x ~/miniconda.sh && \
~/miniconda.sh -b -p /opt/miniconda3 

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "pialnn", "/bin/bash", "-c"]
#RUN pip cache purge
RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-geometric
RUN pip install rtree
RUN conda install -c conda-forge libspatialindex=1.9.3
RUN conda clean -a
ENTRYPOINT ["/bin/bash"]
