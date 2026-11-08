.PHONY: install test check app
install:
	pip install -e ".[dev,whisper,app]"
test:
	pytest -q
check:
	python scripts/check_setup.py
app:
	streamlit run app/Home.py
