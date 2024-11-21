FROM python:3.10.15-bullseye
WORKDIR /home
ENV DEBIAN_FRONTEND=noninteractive
COPY ./requirements.txt requirements.txt
RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN pip install -r requirements.txt
COPY models/log_parser640v1.1.pt models/log_parser640v1.1.pt
COPY models/ui_det640v1.3_L.pt models/ui_det640v1.3_L.pt
COPY ./main.py main.py
CMD ["python3", "main.py"]
