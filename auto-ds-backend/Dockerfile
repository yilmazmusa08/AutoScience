FROM python:3.9-alpine

WORKDIR /code

COPY requirements.txt /code/

RUN apk add gcc libc-dev libffi-dev

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]
