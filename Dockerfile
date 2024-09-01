# Set the base image as the PyTorch image for Jetson Nano
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set the project name as environment variables
ENV PROJECT_NAME=free-fall-surrogate

# Create the workspace directory and set it as the working directory
RUN mkdir -p /workspace/projects/${PROJECT_NAME}
WORKDIR /workspace/projects/${PROJECT_NAME}

# Allow the Git directory to be recognized as safe
RUN git config --global --add safe.directory /workspace/projects/free-fall-surrogate

# Install pip3
RUN apt install -y \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure that 'python' points to 'python3' and 'pip' points to 'pip3'
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy the requirements files to the container
COPY requirements.txt requirements-dev.txt ./

# Install the dependencies using pip
RUN pip install -r requirements.txt -r requirements-dev.txt

# Default command when the container starts
CMD ["bash"]
