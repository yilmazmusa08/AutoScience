FROM python:3.10.10-alpine

WORKDIR /code

COPY requirements.txt /code/

RUN apk --update add gcc build-base freetype-dev libpng-dev openblas-dev libc-dev libffi-dev

RUN pip install --upgrade pip

RUN pip install --no-cache-dir matplotlib pandas

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]
