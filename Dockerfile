# syntax=docker/dockerfile:1
FROM python:3.9
ENV FLASK_APP=app.py
EXPOSE 5000
EXPOSE 80
EXPOSE 443
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./app .
CMD ["flask", "run"]