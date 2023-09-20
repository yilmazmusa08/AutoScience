# AutoScience

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Access URLs](#access-urls)


## Dependencies

- Docker

## Installation

To install and set up the project, follow these steps:

1. Clone the repository:

```sh
git@github.com:yilmazmusa08/AutoScience.git
cd AutoScience
```

2. Build the project:

```sh
make build
```

3. Apply Database migration:

```sh
make migrate
```

4. Create a superuser for the Django admin

```sh
make createsuperuser
```

## Configuration

You can configure the project by modifying the .env file in the frontend directory. Adjust the environment variables as needed.

## Usage

### Running the project

Run the project in production mode:

```sh
make run
```

Run the project in development mode:

```sh
make run-dev
```

## Running Tests

To run tests for the project (backend and frontend), use the following command:

```sh
make test
```

## Access URLs

Backend APIs: http://localhost/api

Django Admin: http://localhost/admin

Build frontend: http://localhost

Frontend for the development: http://localhost:3000
