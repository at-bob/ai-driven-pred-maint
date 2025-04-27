FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install requests pandas

CMD ["python", "data_streamer.py"]