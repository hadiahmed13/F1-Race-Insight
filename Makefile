.PHONY: all setup clean test format lint etl train api dashboard start

# Default target executed when no arguments are given to make.
all: setup etl train test start

# Setup virtual environment
setup:
	python -m pip install --upgrade pip
	python -m pip install poetry
	poetry install

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Run tests with pytest
test:
	poetry run pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing

# Format code with black, isort
format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Run linting tools
lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check src/ tests/

# Run ETL
etl:
	poetry run python -m src.etl.downloader
	poetry run python -m src.etl.transformer

# Train model
train:
	poetry run python -m src.models.train

# Start API server
api:
	poetry run flask --app src.api.app run --debug

# Start Streamlit dashboard
dashboard:
	poetry run streamlit run src/dashboard/Main.py

# Start both API and Streamlit (for development)
start:
	$(MAKE) api

# Install pre-commit hooks
pre-commit-setup:
	poetry run pre-commit install 