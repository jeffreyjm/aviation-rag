.PHONY: setup data test
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
data:    ## download ASRS reports -> data/reports.jsonl
	.venv/bin/aviation-rag data
test:
	.venv/bin/python -m pytest -q
