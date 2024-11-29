# this file will contain the instructions to build your Docker image
# load base image (choose wisely)
# install python requirements
# copy app files
# start uvicorn server

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
