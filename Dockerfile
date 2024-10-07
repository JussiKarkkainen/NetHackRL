FROM python:3.11
# FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python3 setup.py build_ext --inplace

CMD ["DEV=1", "python3", "./src/train.py"]
