install:
	pip install -e .

dev:
	pre-commit run --all-files
	pytest

start:
	streamlit run app/main.py

test:
	pytest -q
