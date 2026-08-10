# Brazilian Olist E-commerce Replay Platform

This project demonstrates an end-to-end e-commerce data platform using immutable Olist historical data, Kafka replay, incremental MySQL updates, feature tables, ML inference APIs, Airflow orchestration, and a React business dashboard.

## Architecture

Historical Olist CSVs (immutable) -> MySQL source tables -> immutable `event_bank_events` -> Kafka producer -> Kafka topic -> Kafka consumer -> replay tables + incremental feature store -> FastAPI APIs -> React dashboard.

Airflow DAG orchestrates health checks, event-bank build, replay batch publish, consumer verification, and summary update.

## Stack

- Backend: FastAPI, SQLAlchemy, pandas, scikit-learn model serving
- Frontend: React + Vite
- Streaming: Kafka + Zookeeper
- Databases: MySQL (business/replay), PostgreSQL (Airflow metadata)
- Orchestration: Apache Airflow
- Containers: Docker Compose

## Repository Layout

- `/backend/main.py` FastAPI APIs for replay, metrics, health, ML, Excel validation
- `/backend/services/replay_service.py` event bank, replay state, incremental feature updates
- `/backend/services/kafka_service.py` real Kafka publishing for configurable batches
- `/backend/controller/consumer.py` idempotent Kafka consumer loop
- `/backend/services/upload_service.py` immutable training dataset versioning + CSV ingest metadata
- `/backend/services/ml_service.py` model discovery, prediction, and retraining
- `/airflow/dags/replay_pipeline.py` orchestration DAG
- `/frontend/src/App.jsx` business dashboard and replay controls

## Prerequisites

- Docker + Docker Compose
- Node.js 20+
- Python 3.12+ (for local backend run)

## Configuration

1. Copy env file:
   - `cp .env.example .env`
2. Fill `.env` values:
   - `MYSQL_ROOT_PASSWORD`, `MYSQL_USER`, `MYSQL_PASSWORD`, `POSTGRES_PASSWORD`, `AIRFLOW_PASSWORD` are secrets and must not be committed
   - `MYSQL_DATABASE`, `MYSQL_HOST`, `MYSQL_PORT`, `POSTGRES_USER`, `POSTGRES_DB`, `AIRFLOW_USER`, `AIRFLOW_EMAIL` are local runtime settings
   - `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`, `KAFKA_CONSUMER_GROUP` are non-secret runtime config
   - `AIRFLOW_HEALTH_URL` is an optional non-secret backend probe URL and should point at the Airflow `/health` endpoint reachable from the backend container/process

## Start Infrastructure

```bash
docker compose up --build
```

Services:
- FastAPI: `http://localhost:8000`
- Airflow UI: `http://localhost:8080`
- MySQL: `localhost:3306`
- Kafka broker: `localhost:9092`

## Immutable Event Bank and Replay

1. Load source data (`loader` container runs `backend/quick_load.py`)
2. Build immutable event bank:
   - `POST /replay/event-bank/build`
3. Start replay batch:
   - `POST /replay/start` with `{ "batch_size": 200, "replay_speed_ms": 0 }`
4. Monitor status:
   - `GET /replay/status`
5. Pause/stop/reset:
   - `POST /replay/pause`, `POST /replay/stop`, `POST /replay/reset`

Replay state is isolated from source Olist tables in replay-specific tables.
Replay order status now progresses by event stage (`created` → `paid` → `shipped` → `delivered` / `canceled`) instead of leaking final historical state into the first replayed event.

## Dashboard and KPIs

Run frontend:

```bash
cd frontend
npm install
npm run dev
```

Dashboard shows:
- revenue, active/delivered/cancelled orders
- average delivery days
- review score
- replay processing status and failures
- recent activity
- trend snapshots
- sample ML prediction output

## ML APIs

- `GET /ml/models`
- `POST /ml/predict/{model_name}`
- `POST /ml/retrain`
- `GET /ml/training-runs`

Supported model names (if artifacts exist):
- delivery_delay
- order_cancellation
- review_prediction
- demand_forecasting
- product_recommendation
- customer_purchase_prediction

## Excel Validation API

- `POST /excel/validate`
- Accepts `.xlsx` / `.xls`
- Returns dataset detection, required-column validation, duplicate checks, null counts, inferred types, and preview rows
- Original uploaded file is never modified

## Versioned Training Dataset Uploads

- `POST /upload/validate-training-csv` validates a candidate CSV against known ML dataset signatures
- `POST /upload/training-csv` stores the uploaded CSV immutably under `backend/training_dataset_versions/<dataset>/<version>/`
- Each upload also writes a merged versioned dataset snapshot without mutating the original base CSV under `backend/ml_training_data/`
- Latest model retraining reads the newest merged dataset snapshot when one exists; otherwise it falls back to the original base dataset

## Airflow DAG

DAG: `olist_replay_orchestration`

Task flow:
1. health check
2. build event bank
3. publish Kafka batch
4. verify consumer processing
5. update metrics summary
6. generate summary
7. record completion

Retries and retry delays are configured in DAG default args.

## Troubleshooting

- If replay status is `failed`, inspect `/dashboard/summary` recent logs and consumer container logs.
- If no events are processed, ensure `consumer` container is running and Kafka is healthy.
- If ML endpoint fails, verify model pickle exists under `backend/models`.

## Current Status

Implemented and connected:
- real replay APIs
- immutable event bank creation
- Kafka producer + consumer
- idempotent processing via `processed_events`
- incremental customer/seller/product feature updates
- business dashboard APIs + frontend controls
- Airflow orchestration DAG
- versioned ML training dataset ingestion
- API-triggered retraining metadata and model version snapshots

Potential manual follow-up:
- tune replay batch defaults for your machine
- harden Kafka health checks with broker-specific runtime probes
- add deeper frontend charting if desired
