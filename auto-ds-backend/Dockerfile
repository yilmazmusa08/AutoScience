# Use a lightweight Python base image
FROM python:3.10.10-alpine

# Set the working directory in the container
WORKDIR /code

# Copy the base requirements.txt file into the container
COPY requirements_base.txt /code/

# Copy the additional requirements file into the container
COPY requirements_ds.txt /code/

# Install necessary system dependencies (if not already installed)
RUN apk update && apk upgrade && \
    apk add --no-cache \
    build-base \
    gcc \
    gfortran \
    libc-dev \
    libffi-dev \
    libxslt-dev \
    libxml2-dev \
    jpeg-dev \
    zlib-dev \
    libjpeg \
    make \
    patchelf \
    freetype-dev \
    libpng-dev \
    openblas-dev \
    musl-dev \
    lapack-dev \
    pkgconfig \
    freetype \
    libxft \
    libpng

# Upgrade cmake and ninja (if not already upgraded)
RUN pip install --upgrade cmake ninja

# Upgrade pip and setuptools (if not already upgraded)
RUN pip install --upgrade pip setuptools

# Install core Python packages from requirements_base.txt (if not already installed)
RUN pip install --no-cache-dir -r requirements_base.txt

# Install additional Python packages from requirements_ds.txt (if not already installed)
RUN pip install --no-cache-dir -r requirements_ds.txt

# Copy the entire project into the container
COPY . /code/

# Specify the command to run your application
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Run the command to collect static files in Django (assumes your Django project is named "autodstool")
RUN python manage.py collectstatic --noinput

# Specify the command to run your application with Gunicorn
CMD ["gunicorn", "autodstool.wsgi:application", "--bind", "0.0.0.0:8000", "--worker-class", "gevent", "--timeout", "600", "--workers", "4", "--chdir=/code"]