#FROM ubuntu:latest
#LABEL authors="Nraj"
#
#ENTRYPOINT ["top", "-b"]
#
FROM python:3.9

#
WORKDIR /code

#
COPY ./module/requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./module /code/module

#
CMD ["fastapi", "run", "module/main.py", "--port", "80"]