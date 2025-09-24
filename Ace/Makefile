SHELL := /bin/bash

# Export all variables defined in this file to child processes.
# By default that does not happen.
.EXPORT_ALL_VARIABLES:


S3_SERVER_PORT ?= 9002
S3_SERVER_HOST ?= localhost
S3_CONSOLE_PORT ?= 9003
S3_ACCESS_KEY ?= s3_access_key
S3_SECRET_KEY ?= s3_secret_key
S3_BUCKET_CONTENT_LOCAL_PATH ?= ./tests/fixtures/s3

LOG_LEVEL=INFO
STAGE ?= test


# Run docker containers for tests
up:
	docker-compose -f ./docker-compose.yml up -d --remove-orphans

# Shutdown docker containers
down:
	docker-compose -f ./docker-compose.yml down

up-test:
	docker-compose -f ./docker-compose.test.yml up -d --remove-orphans

config-test:
	docker-compose -f ./docker-compose.test.yml config

up-test-v:
	docker-compose -f ./docker-compose.test.yml up $(TARGETS) -V --remove-orphans

down-test:
	docker-compose -f ./docker-compose.test.yml down

# Run all tests
test: up-test
	PGHOST=localhost \
	PGPORT=5432 \
	PGUSER=postgres \
	PGPASSWORD=secret \
	PGDATABASE=ace_qa \
	STS_AUTH_DISABLED=1 \
	PYTHONPATH="$PWD" \
	pytest --cov-report=term --cov-report=xml --cov-branch --cov -s
	docker-compose -f ./docker-compose.test.yml down

test-dev:
	pytest --cov-report=term --cov-report=xml --cov-branch --cov $(OPTS)

run-vendor-ranking:
	PYTHONPATH=${PWD} python ./vendor_ranking/service.py

run-ultron:
	PYTHONPATH=${PWD} python ./ultron/service.py
