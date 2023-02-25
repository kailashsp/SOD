# frontend/Dockerfile

FROM python:3.10.1-slim

WORKDIR /app

RUN pip install streamlit==1.18.1

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui2.py"]
