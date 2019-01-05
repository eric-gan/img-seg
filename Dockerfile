FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y wget tar git python3 python3-pip \
    libsm6 libxext6 libxrender-dev vim less
RUN echo "set mouse=a" > ~/.vimrc

WORKDIR /
RUN git clone https://github.com/vliu15/ImgSegmentation

WORKDIR /ImgSegmentation
RUN pip3 install -y --upgrade pip
RUN pip3 install -y -r requirements.txt
RUN sh download.sh
