FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

COPY ./src ./src
COPY ./model ./model

CMD ["python3", "src/service.py"]