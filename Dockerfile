FROM python:3.9
ENV FLASK_APP=app.py
EXPOSE 5000
EXPOSE 80
EXPOSE 443
COPY requirements.txt /home/requirements.txt
RUN ["pip", "install", "-r", "/home/requirements.txt"]
COPY ./app /home/
COPY docker-entrypoint.sh /home/docker-entrypoint.sh
RUN ["chmod", "+x", "/home/docker-entrypoint.sh"]
CMD ["/home/docker-entrypoint.sh"]