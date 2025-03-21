FROM python:3.10.6-buster

COPY nos_paquets /nos_paquets
COPY models /models
COPY requirements.txt /requirements.txt
COPY api /api

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api_audio_dfd:app --host 0.0.0.0 --port $PORT
