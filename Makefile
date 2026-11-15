.PHONY: setup data index eval test
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
data:    ## download ASRS reports -> data/reports.jsonl
	.venv/bin/aviation-rag data
index:   ## chunk + embed (~7 min on CPU)
	.venv/bin/aviation-rag index
eval:    ## full evaluation -> eval/results/
	.venv/bin/aviation-rag eval
test:
	.venv/bin/python -m pytest -q
