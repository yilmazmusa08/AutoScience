# Makefile for Docker Compose Operations

# Define Docker Compose executable
DOCKER_COMPOSE = sudo docker-compose

# Set the default target
.DEFAULT_GOAL := help

help:
	@echo "Available Make targets:"
	@echo "  build            - Build the project in development mode"
	@echo "  migrate          - Apply database migrations"
	@echo "  makemigrations   - Create new database migrations"
	@echo "  createsuperuser  - Create a superuser for Django admin"
	@echo "  run              - Run the project in production mode"
	@echo "  run-dev          - Run the project in development mode"
	@echo "  test             - Run tests for the project"
	@echo "  down             - Stop and remove containers"
	@echo "  clean            - Remove containers, images, and volumes (use with caution)"
	@echo "  logs             - View container logs"

build:         ## Build the project in development mode
	@if [ ! -f "./auto-ds-frontend/.env" ]; then \
		cp "./auto-ds-frontend/.env.dev" "./auto-ds-frontend/.env"; \
	fi
	$(DOCKER_COMPOSE) --profile dev up --build

migrate:       ## Apply database migrations
	$(DOCKER_COMPOSE) run backend python manage.py migrate

makemigrations: ## Create new database migrations
	$(DOCKER_COMPOSE) run backend python manage.py makemigrations

createsuperuser:  ## Create a superuser for Django admin
	$(DOCKER_COMPOSE) run backend python manage.py createsuperuser

run:           ## Run the project in production mode
	$(DOCKER_COMPOSE) up -d

run-dev:       ## Run the project in development mode
	$(DOCKER_COMPOSE) --profile dev up -d

test:   	   ## Run tests for the project
	$(DOCKER_COMPOSE) run backend python manage.py test
	$(DOCKER_COMPOSE) run frontend npm test

down:          ## Stop and remove containers
	$(DOCKER_COMPOSE) down

clean:         ## Remove containers, images, and volumes (use with caution)
	$(DOCKER_COMPOSE) down --rmi all -v

lint:  		   ## Run linting checks
    # Add linting commands here

format:        ## Format code
    # Add code formatting commands here

deploy-prod:   ## Deploy the application to production
    # Add deployment commands here

deploy-staging:   ## Deploy the application to staging
    # Add deployment commands here for staging

backup-db:     ## Create a database backup
    # Add backup commands here

restore-db:    ## Restore the database from a backup
    # Add restore commands here

logs:          ## View container logs
	$(DOCKER_COMPOSE) logs
