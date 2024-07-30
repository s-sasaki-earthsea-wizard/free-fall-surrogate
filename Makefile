# generate parabola curve data
parabola:
	@python scripts/prepare_data.py
split:
	@python scripts/split_parabolic_data.py

# Install poetry
install_poetry:
	curl -sSL https://install.python-poetry.org | python3 -

# Run pytest
test:
	poetry run pytest tests

# Install pyenv
install_pyenv:
	sudo apt -y update
	sudo apt -y upgrade
	sudo apt install \
  	build-essential libssl-dev zlib1g-dev \
  	libbz2-dev libreadline-dev libsqlite3-dev curl llvm \
  	libncursesw5-dev tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
  	libopencv-dev git
	git clone https://github.com/pyenv/pyenv.git ~/.pyenv
	echo 'eval "$$(pyenv init -)"' >> ~/.bashrc