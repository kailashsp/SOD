FROM python:3.10

WORKDIR /BGREM
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get install libgl1 -y

COPY requirements.txt /BGREM/requirements.txt

RUN pip install --upgrade pip

RUN pip install python-multipart

RUN pip install --default-timeout=100 --no-cache-dir --upgrade -r /BGREM/requirements.txt 

COPY Bgrem_fasapi.py /BGREM/Bgrem_fasapi.py

COPY u2net_keras.h5 ./u2net_keras.h5

COPY main.py /BGREM/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]

