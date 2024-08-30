# generate parabola curve data
generate_parabolic_data:
	poetry run python src/generate_parabolic_motion_data.py
split:
	poetry run python src/split_parabolic_data.py

# -------------------
# Docker
# -------------------
# Set a default valuables for docker 
IMAGE_NAME = free-fall-surrogate
HOST_PROJECTS_DIR := $(shell pwd)
CONTAINER_PROJECTS_DIR := /workspace/projects/free-fall-surrogate

# Build docker image
docker-build:
	docker build -t $(IMAGE_NAME) .

# Run docker container
docker-run:
	docker run --rm -it -v $(HOST_PROJECTS_DIR):$(CONTAINER_PROJECTS_DIR) $(IMAGE_NAME)

# -------------------
# Unit tests
# -------------------
test_parabolic_motion:
	PYTHONPATH=./src poetry run pytest ./tests/test_parabolic_motion.py