FROM python:3.10

ARG UID
ARG GID

WORKDIR /app

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt
RUN python -m textblob.download_corpora

ENV PYTHONPATH="$PYTHONPATH:/app"

EXPOSE 8888