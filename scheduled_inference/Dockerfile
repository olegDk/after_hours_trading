FROM python:3.8.5-slim

RUN apt-get update && \
    apt-get -y install \
    apt-utils \
    gcc \
    libkrb5-dev

ENV PYTHONUNBUFFERED 1
ENV PROJECT_DIR /messaging

ARG REQUIREMENTS_FILE=requirements.txt

ADD requirements*.txt $PROJECT_DIR/

RUN pip --no-cache-dir install -r $PROJECT_DIR/$REQUIREMENTS_FILE

ADD . $PROJECT_DIR/

WORKDIR $PROJECT_DIR

CMD ["python", "main.py"]
