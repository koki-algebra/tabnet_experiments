FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# default shell
ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-l", "-c" ]

# Create the user
ARG USERNAME=pythonista
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# install basic packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
    # clear
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# install python packages
RUN conda install -c conda-forge ipywidgets jupyterlab \
    && conda install -c conda-forge pytorch-tabnet pandas matplotlib scikit-learn

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
