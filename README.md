# ITCM Recommendation Engine

A recommendation system for Åshild webstore, providing AI-powered product recommendations and data exports for dashboards.

## Features

- Automated daily data update and processing via cron.
- Reproducible environment using Docker.
- Data stored in `/workspace/data/external` and processed outputs in `/workspace/data/processed`.

## Usage

Build and start:

```bash
docker compose up -d --build itcm-recsys-prod   # Production
docker compose up -d --build itcm-recsys        # Development
```

## Pipeline

- Data is fetched automatically each day using credentials from `.secrets/`.
- Processing, cleaning, and recommendations run on schedule via cron.
- Model: [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- Results can be visualized in the [ITCM Dashboard](https://github.com/maria-lagerholm/itcm-dashboard).

To manually update data from the API (normally scheduled daily), run:

```bash
docker compose exec itcm-recsys-prod sh -lc '
  set -a; . /workspace/.secrets; set +a;
  python /workspace/scripts/update_data.py
'
```

To run the recommendation pipeline:

```bash
docker compose exec itcm-recsys-prod sh -lc '
  set -a; . /workspace/.secrets; set +a;
  python /workspace/scripts/run-pipeline.py
'
```

Make sure you have credentials in `/workspace/.secrets/ASHILD_USER`, `/workspace/.secrets/ASHILD_PASS`, and `/workspace/.secrets/ASHILD_BASE`.