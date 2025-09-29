# /workspace/Makefile
PYTHON ?= python
CFG    ?= configs/base.yaml
ARGS   ?=
export PYTHONPATH := $(PWD)/python

.PHONY: customers transactions all help

customers:
	$(PYTHON) -m cli.customers --cfg $(CFG) $(ARGS)

transactions:
	$(PYTHON) -m cli.transactions --cfg $(CFG) $(ARGS)

articles:
	$(PYTHON) -m cli.articles --cfg $(CFG) $(ARGS)


all: customers transactions

help:
	@echo "make customers [CFG=...] [ARGS='--fill-unknown Unknown']"
	@echo "make transactions [CFG=...] [ARGS='--min-created 2024-06-01']"
