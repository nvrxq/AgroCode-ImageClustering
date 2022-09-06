#FROM python:3.9-buster
FROM tensorflow/tensorflow:latest-gpu
RUN apt update && apt install -y libgl1-mesa-glx
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN curl https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
CMD ["python3","run.py"]
