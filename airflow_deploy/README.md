# airflow deploy

1. create a local docker registry  at port 5000
2. modify docker-compose to fit your mounting volumes
```
volumes:
      - ./airflow_data:/opt/airflow
      - /media/hbdesk/UNTITLED/raw_video_data:/raw_video_data
      - /media/hbdesk/UNTITLED/frames_extract:/frames_extract
      - /media/hbdesk/UNTITLED/preprocessed_data:/preprocessed_data

```
3. `./deploy.sh`
