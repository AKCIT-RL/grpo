FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python3-opengl \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf \
    libglfw3-dev \
    libcudnn8 libcudnn8-dev \
    cmake \
    ninja-build \
    libsdl2-dev

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . .

RUN pip install poetry --upgrade
RUN poetry config virtualenvs.create false
#RUN poetry env use python3.10

RUN poetry run pip install -r requirements/requirements-all.txt