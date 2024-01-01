# hash:sha256:ac48e1a4e6bb2d1baeba7568d1ec51dac94f5d77716e234f2df4fcdc678054eb
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    dask[distributed]==2021.11.1 \
    imageio==2.26.0 \
    ipython==8.11.0 \
    matplotlib==3.7.1 \
    zarr==2.14.2 \
    tqdm==2.2.3
