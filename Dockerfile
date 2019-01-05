FROM ubuntu:latest

RUN apt-get update
RUN apt-get install wget tar git python3 python3-pip

WORKDIR /
RUN git clone https://github.com/vliu15/ImgSegmentation

WORKDIR /ImgSegmentation
RUN pip3 install --upgrade pip
RUN pip3 install -y -r requirements.txt
RUN sh download.sh
