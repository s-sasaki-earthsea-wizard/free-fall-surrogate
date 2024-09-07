# generate parabola curve data
generate_parabolic_data:
	python src/generate_parabolic_motion_data.py

# split parabolic data
split:
	python src/split_parabolic_motion_data.py

# Tarin parabolic motion
train:
	python src/train_parabolic_motion.py

# Generate project summary
summary:
	python generate_project_summary.py	

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
test_parabolic_motion_generation:
	PYTHONPATH=./src pytest -s ./tests/test_parabolic_motion_generation.py
test_parabolic_motion_split:
	PYTHONPATH=./src pytest -s ./tests/test_parabolic_motion_valid_split.py
