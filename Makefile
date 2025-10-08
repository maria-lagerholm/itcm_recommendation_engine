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

basket_cf:
	$(PYTHON) -m cli.basket_cf --cfg $(CFG) $(ARGS)


combine:
	$(PYTHON) -m cli.combine --cfg $(CFG) $(ARGS)

all: customers articles articles_for_recs semantic_similarity transactions basket_cf combine

help:
	@echo "make customers [CFG=...] [ARGS='--fill-unknown Unknown']"
	@echo "make transactions [CFG=...] [ARGS='--min-created 2024-06-01']"
	@echo "make articles_for_recs [CFG=...]"
	@echo "make semantic_similarity [CFG=...] [ARGS='--batch-size 64 --threads 1']"
	@echo "make basket_cf [CFG=...] [ARGS='--min-item-support 10 --min-pair-support 5 --k 100 --thr 0.02 --topk 10']"
	@echo "make combine [CFG=...]"
