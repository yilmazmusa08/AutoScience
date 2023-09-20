@echo off

REM Set the path to your Docker Compose executable
SET DOCKER_COMPOSE=docker-compose

REM Default target
:help
if "%1"=="" (
    echo Available Make targets:
    echo   build            - Build the project in development mode
    echo   migrate          - Apply database migrations
    echo   makemigrations   - Create new database migrations
    echo   createsuperuser  - Create a superuser for Django admin
    echo   run              - Run the project in production mode
    echo   run-dev          - Run the project in development mode
    echo   test             - Run tests for the project
    echo   down             - Stop and remove containers
    echo   clean            - Remove containers, images, and volumes (use with caution)
    echo   logs             - View container logs
) else (
    call :%1
)
goto :end

:build
if not exist .\auto-ds-frontend\.env (
    copy .\auto-ds-frontend\.env.dev .\auto-ds-frontend\.env
)
%DOCKER_COMPOSE% --profile dev up --build
goto :end

:migrate
%DOCKER_COMPOSE% run backend python manage.py migrate
goto :end

:makemigrations
%DOCKER_COMPOSE% run backend python manage.py makemigrations
goto :end

:createsuperuser
%DOCKER_COMPOSE% run backend python manage.py createsuperuser
goto :end

:run
%DOCKER_COMPOSE% up -d
goto :end

:run-dev
%DOCKER_COMPOSE% --profile dev up -d
goto :end

:test
%DOCKER_COMPOSE% run backend python manage.py test
%DOCKER_COMPOSE% run frontend npm test
goto :end

:down
%DOCKER_COMPOSE% down
goto :end

:clean
%DOCKER_COMPOSE% down --rmi all -v
goto :end

:logs
%DOCKER_COMPOSE% logs

:end
