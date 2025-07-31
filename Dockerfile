FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python3-opengl \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf \
    libglfw3-dev \
    libcudnn8 libcudnn8-dev

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY pyproject.toml poetry.lock README.md ./
COPY ./cleanrl ./cleanrl
COPY ./cleanrl_utils ./cleanrl_utils
COPY ./requirements ./requirements

RUN pip install poetry --upgrade
RUN poetry env use python3.10 && \
    poetry install -E "atari mujoco"

RUN poetry run pip install -r requirements/requirements-mujoco.txt

RUN poetry run python3.10 -c "import mujoco"

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]