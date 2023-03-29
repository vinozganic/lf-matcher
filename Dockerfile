FROM python:3.11-alpine

WORKDIR /app

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

COPY ./src ./src

CMD ["python3", "src/service.py"]