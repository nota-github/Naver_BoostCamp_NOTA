FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV HOME=/root
ENV APP_PATH=$HOME/Naver_BoostCamp_NOTA
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH $APP_PATH
WORKDIR $APP_PATH

COPY . $APP_PATH/
ARG DEBIAN_FRONTEND=noninteractive
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r $APP_PATH/requirements.txt
RUN apt-get update
RUN apt-get install git vim unzip wget ffmpeg libsm6 libxext6  -y

CMD ["bash"]


# docker build -t notadockerhub/np_app_segformer:latest -f docker/segformer/Dockerfile .
# docker run --name segformer_ws --shm-size=8g -it -p 32131 --gpus all -v /home:/workspace np_app_segformer:latest bash