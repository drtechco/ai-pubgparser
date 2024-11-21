FROM python:3.9.20-bullseye
WORKDIR /home
COPY ./requirements.txt requirements.txt
RUN apt update && apt install ffmpeg -y
RUN pip install -r requirements.txt

