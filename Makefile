install:
pip install -e .

dev:
ruff .
pytest

start:
streamlit run app/main.py

test:
pytest -q
