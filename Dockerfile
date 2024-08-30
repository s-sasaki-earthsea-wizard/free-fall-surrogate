# Set the base image as the PyTorch image for Jetson Nano
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Set the project name as locale environment variables
ENV PROJECT_NAME=free-fall-surrogate
ENV PYTHON_VERSION=3.10.10

# Set the timezone environment variable
ARG TZ=Asia/Tokyo
ENV TZ=${TZ}

# Install required packages, including Python 3.8 and tzdata for time zone configuration
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=$TZ apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv and its default plugins (including pyenv-virtualenv)
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"
ENV PYENV_ROOT="/root/.pyenv"

# Install Poetry using Python 3.8
RUN curl -sSL https://install.python-poetry.org | python3.8 -

# Set environment variables for pyenv and Poetry
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:/root/.local/bin:$PATH"
ENV PYENV_ROOT="/root/.pyenv"

# Make the workspace directory to mount the host directory, and set it as the working directory
RUN mkdir -p /workspace/projects/${PROJECT_NAME}
WORKDIR /workspace/projects/${PROJECT_NAME}

# Allow the Git directory to be recognized as safe
RUN git config --global --add safe.directory /workspace/projects/free-fall-surrogate

# Install Python via pyenv and set it as the global version
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv virtualenv ${PYTHON_VERSION} ${PROJECT_NAME}
RUN pyenv global ${PROJECT_NAME}

# Install dependencies by Poetry
# Copy the configuration files
COPY pyproject.toml poetry.lock ./

# Install the dependencies
RUN poetry install

# Default command when the container starts
RUN pyenv local ${PROJECT_NAME}
CMD ["bash"]
