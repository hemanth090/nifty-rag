PY ?= .venv/bin/python

.PHONY: help setup run index test clean docker

help:
	@echo "setup   create .venv and install dependencies (requirements-dev.txt)"
	@echo "run     start the HTTP server (./run_server.sh)"
	@echo "index   index data/pdfs into .pageindex and results/"
	@echo "test    run the test suite"
	@echo "clean   remove Python and pytest caches"
	@echo "docker  build the production image (nifty-rag)"

setup: .venv

.venv: requirements-dev.txt requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt

run:
	./run_server.sh

index:
	$(PY) index_pdfs.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf __pycache__ .pytest_cache tests/__pycache__

docker:
	docker build -t nifty-rag .
