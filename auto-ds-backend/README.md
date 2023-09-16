# Installation

```sh
sudo docker-compose up --build
```

## Apply DB migrations

```sh
sudo docker-compose run web python manage.py migrate
```

## Create Superuser

```sh
sudo docker-compose run web python manage.py createsuperuser
```

## Run

```sh
sudo docker-compose up
```