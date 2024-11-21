#! /usr/bin/bash


docker buildx build --platform linux/amd64 -t localhost:5000/auto_label:latest .
