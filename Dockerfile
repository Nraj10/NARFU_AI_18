#FROM ubuntu:latest
#LABEL authors="Nraj"
#
#ENTRYPOINT ["top", "-b"]
#
FROM python:3.9

#
WORKDIR /code

#
COPY ./module/requirements2.txt /code/requirements.txt

RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin &&\
    apt -y install libgdal-dev
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./module /code/module

#
CMD ["fastapi", "run", "module/main.py", "--port", "80"]