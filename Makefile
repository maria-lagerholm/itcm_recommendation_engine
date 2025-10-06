# /workspace/Makefile

#------variables------
PYTHON ?= python
CFG    ?= configs/base.yaml
ARGS   ?=
export PYTHONPATH := $(PWD)/python

#------phony targets------
.PHONY: customers transactions all help

#------customers------
customers:
	$(PYTHON) -m cli.customers --cfg $(CFG) $(ARGS)

#------transactions------
transactions:
	$(PYTHON) -m cli.transactions --cfg $(CFG) $(ARGS)

#------articles------
articles:
	$(PYTHON) -m cli.articles --cfg $(CFG) $(ARGS)

#------combine------
combine:
	$(PYTHON) -m cli.combine --cfg $(CFG) $(ARGS)

#------all------
all: customers transactions articles combine

#------help------
help:
	@echo "make customers [CFG=...] [ARGS='--fill-unknown Unknown']"
	@echo "make transactions [CFG=...] [ARGS='--min-created 2024-06-01']"
