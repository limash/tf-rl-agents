FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/yak

RUN apt-get update && apt-get -y install git

RUN pip install --upgrade pip
RUN pip install tensorflow-probability tensorflow-addons
RUN pip install matplotlib pandas gym kaggle-environments dm-reverb ray[default]
RUN pip install -e git+https://github.com/limash/gym-goose.git#egg=gym_goose
