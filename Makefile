# /workspace/Makefile
PYTHON ?= python
CFG    ?= configs/base.yaml
export PYTHONPATH := $(PWD)/python

.PHONY: customers

customers:
	$(PYTHON) -m cli.customers --cfg $(CFG) $(ARGS)


# (kept for later)
transactions:
	$(PYTHON) -m python.cli.transactions --cfg $(CFG) $(ARGS)

all: customers transactions

help:
	@echo "make customers [CFG=...] [ARGS='--fill-unknown Unknown']"