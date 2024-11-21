#! /usr/bin/bash

docker buildx build -f Dockerfile.airflow -t localhost:5000/airflow_custom:latest . 
