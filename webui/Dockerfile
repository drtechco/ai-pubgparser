FROM python:3.10.15-bullseye
WORKDIR /home
ENV DEBIAN_FRONTEND=noninteractive
COPY ./requirements.txt requirements.txt
RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN pip install -r requirements.txt
COPY ./main.py main.py
COPY ./kill_log.py kill_log.py
COPY ./models/ models
CMD ["python3", "main.py"]
