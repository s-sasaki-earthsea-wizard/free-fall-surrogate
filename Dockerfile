# Set the base image
FROM ubuntu:20.04

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

# Set environment variables for Poetry
ENV PATH="/root/.local/bin:$PATH"

# Set the locale environment variables
ENV PROJECT_NAME=free-fall-surrogate

# Make the workspace directory to mount the host directory, and set it as the working directory
RUN mkdir -p /workspace/projects/${PROJECT_NAME}
WORKDIR /workspace/projects/${PROJECT_NAME}

# Allow the Git directory to be recognized as safe
RUN git config --global --add safe.directory /workspace/projects/free-fall-surrogate


# Install Python via pyenv
RUN pyenv install 3.10.10 && \
    pyenv global 3.10.10

# Install dependencies by Poetry
# Copy the configuration files
COPY pyproject.toml poetry.lock ./

# Set the correct Python version for Poetry environment
RUN poetry env use python3.10

# Install the dependencies
RUN poetry install

# Default command when the container starts
CMD ["bash"]
