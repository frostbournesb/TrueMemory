.PHONY: test lint install dev clean

install:
	pip install -e .

dev:
	pip install -e .[dev,all]

test:
	pytest tests/ -v

lint:
	ruff check neuromem/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
