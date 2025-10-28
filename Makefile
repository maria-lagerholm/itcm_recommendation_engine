# /workspace/Makefile

PYTHON ?= python
CFG    ?= configs/base.yaml
ARGS   ?=
export PYTHONPATH := $(PWD)/python

.PHONY: customers transactions articles articles_for_recs semantic_similarity basket_cf combine all help

customers:
	$(PYTHON) -m cli.customers --cfg $(CFG) $(ARGS)

articles:
	$(PYTHON) -m cli.articles --cfg $(CFG) $(ARGS)

articles_for_recs:
	$(PYTHON) -m cli.articles_for_recs --cfg $(CFG) $(ARGS)

semantic_similarity:
	$(PYTHON) -m cli.semantic_similarity --cfg $(CFG) $(ARGS)

transactions:
	$(PYTHON) -m cli.transactions --cfg $(CFG) $(ARGS)

iicf_ease:
	$(PYTHON) -m cli.iicf_ease $(ARGS)

top_same_brand:
	$(PYTHON) -m cli.top_same_brand --cfg $(CFG) $(ARGS)


combine:
	$(PYTHON) -m cli.combine --cfg $(CFG) $(ARGS)

lift:
	$(PYTHON) -m cli.lift --cfg $(CFG) $(ARGS)

all: customers articles articles_for_recs semantic_similarity transactions combine iicf_ease top_same_brand lift

help:
	@echo "make customers [CFG=...] [ARGS='--fill-unknown Unknown']"
	@echo "make transactions [CFG=...] [ARGS='--min-created 2024-06-01']"
	@echo "make articles_for_recs [CFG=...]"
	@echo "make semantic_similarity [CFG=...] [ARGS='--batch-size 64 --threads 1']"
	@echo "make combine [CFG=...]"
