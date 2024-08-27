# generate parabola curve data
parabola:
	@python scripts/prepare_data.py
split:
	@python scripts/split_parabolic_data.py

# Run pytest
test:
	poetry run pytest tests

# Docker
# 
IMAGE_NAME = free-fall-surrogate
HOST_PROJECTS_DIR := $(shell pwd)
CONTAINER_PROJECTS_DIR := /workspace/projects/free-fall-surrogate

# Build docker image
docker-build:
	docker build -t $(IMAGE_NAME) .

# Run docker container
docker-run:
	docker run --rm -it -v $(HOST_PROJECTS_DIR):$(CONTAINER_PROJECTS_DIR) $(IMAGE_NAME)
