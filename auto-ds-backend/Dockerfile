# Use a lightweight Python base image
FROM python:3.10.10-alpine

# Set the working directory in the container
WORKDIR /code

# Copy the requirements.txt file into the container
COPY requirements.txt /code/

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

# Install Python packages from requirements.txt (if not already installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /code/

# Specify the command to run your application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]