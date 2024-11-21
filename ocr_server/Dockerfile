FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
WORKDIR /home
# ENV PADDLE_HOME=/root/.paddleocr
RUN apt update && apt install python3-pip -y
COPY ./server_http.py server_http.py
COPY ./models /root/.paddleocr/
COPY ./requirements.txt requirements.txt
COPY ./run.sh run.sh
RUN pip install -r requirements.txt
VOLUME ./models /root/
# VOLUME ./models /home/models
RUN chmod +x run.sh
EXPOSE 3000
CMD ["python3", "server_http.py"]
