FROM python:3.10.1-slim

WORKDIR /BGREM
ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY requirements.txt /BGREM/requirements.txt
RUN pip install --upgrade pip

RUN pip install python-multipart

RUN pip install --default-timeout=1000 --no-cache-dir --upgrade -r /BGREM/requirements.txt 

COPY Bgrem_fastapi_torch.py /BGREM/Bgrem_fastapi_torch.py

COPY  model /BGREM/model
COPY U2net.pth /BGREM/U2net.pth

COPY main2.py /BGREM/main.py

EXPOSE 4000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]

