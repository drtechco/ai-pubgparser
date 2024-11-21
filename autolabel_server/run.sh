#!/usr/bin/bash

gunicorn -w 3 -b 0.0.0.0:9999 main:app
