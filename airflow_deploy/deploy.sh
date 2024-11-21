#! /usr/bin/bash

mkdir -p airflow_data
mkdir -p raw_videos
mkdir -p frames 
mkdir -p postgres_data

docker buildx build -f Dockerfile.airflow -t localhost:5000/airflow_custom:latest . 

docker compose up
