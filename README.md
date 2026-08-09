# Brazilian Ecommerce Replay Demo

This repository now demonstrates a real, replay-driven ecommerce workflow over the Olist historical dataset:

Historical CSVs → immutable event bank → replay controls → Kafka/direct processing → live database state → KPI API → React dashboard → ML artifact inspection.

## What is implemented

- **Immutable source data** in `/home/runner/work/brazilian-eccomerce-dataset/brazilian-eccomerce-dataset/backend/original_data`
- **Immutable event bank** generated into `backend/runtime/event_bank/olist_order_events.jsonl`
- **Replay API** for reset, replay, event preview, activity, and health
- **Kafka producer/consumer** using the same event payloads as the direct replay path
- **Live operational database** backed by SQLite by default or MySQL when `.env` is configured
- **Business dashboard API** with KPIs, trends, top categories, recent activity, and recent orders
- **ML artifact inspection** for the existing saved models plus sample predictions for supported models
- **Airflow DAG** that resets the live store, replays a real batch, and verifies that KPIs were populated
- **React dashboard** with centralized API calls, replay controls, loading/error states, and health diagnostics

## Repository structure

- `backend/`
  - `main.py` FastAPI application
  - `services.py` replay, KPI, ML, and event-bank services
  - `database.py` SQLAlchemy models and runtime DB configuration
  - `controller/producer.py` Kafka publisher
  - `controller/consumer.py` Kafka consumer
  - `original_data/` immutable Olist CSVs
  - `ml_training_data/` preserved feature datasets
  - `models/` existing saved ML artifacts
- `frontend/`
  - `src/api.js` centralized API client
  - `src/App.jsx` manager-facing dashboard
- `airflow/dags/ecommerce_replay_demo.py`
- `docker-compose.yml`

## Prerequisites

- Python 3.12
- Node.js 20+
- npm
- Docker + Docker Compose (for the full stack)

## Environment variables

Copy `.env.example` to `.env`.

### Required for Docker/MySQL/Airflow

- `MYSQL_ROOT_PASSWORD`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `AIRFLOW_USER`
- `AIRFLOW_PASSWORD`
- `AIRFLOW_EMAIL`

### Optional

- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_TOPIC`
- `KAFKA_CONSUMER_GROUP`
- `CORS_ORIGINS`
- `VITE_API_BASE_URL`
- `DATABASE_URL` to override the default runtime SQLite DB outside Docker

## Local backend setup

```bash
cd /home/runner/work/brazilian-eccomerce-dataset/brazilian-eccomerce-dataset/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Default local mode uses a SQLite file at `backend/runtime/ecommerce_demo.db`.

## Local frontend setup

```bash
cd /home/runner/work/brazilian-eccomerce-dataset/brazilian-eccomerce-dataset/frontend
npm install
npm run dev
```

The UI expects the backend at `http://localhost:8000` unless `VITE_API_BASE_URL` is set.

## Docker startup

```bash
cd /home/runner/work/brazilian-eccomerce-dataset/brazilian-eccomerce-dataset
cp .env.example .env
docker compose up --build
```

Services:

- Backend API: `http://localhost:8000`
- Frontend dev server: run separately with `npm run dev`
- Airflow UI: `http://localhost:8080`
- MySQL: `localhost:3306`
- Kafka broker: `localhost:9092`

## Replay workflow

### Direct replay through the API

1. `POST /api/replay/reset`
2. `POST /api/replay` with `{ "start_offset": 0, "limit": 500, "pace_ms": 0 }`
3. `GET /api/dashboard`
4. `GET /api/activity`

### Kafka workflow

Producer:

```bash
cd /home/runner/work/brazilian-eccomerce-dataset/brazilian-eccomerce-dataset/backend
python controller/producer.py --limit 500
```

Consumer:

```bash
cd /home/runner/work/brazilian-eccomerce-dataset/brazilian-eccomerce-dataset/backend
python controller/consumer.py
```

The Kafka consumer only commits offsets after the database write succeeds.

## Airflow workflow

Run the `ecommerce_replay_demo` DAG to:

1. Check backend health
2. Reset live replay state
3. Replay 250 immutable events
4. Verify that dashboard KPIs were populated

## ML workflow

- Existing trained artifacts are stored in `backend/models`
- Preserved feature tables are in `backend/ml_training_data`
- API endpoints:
  - `GET /api/ml/models`
  - `GET /api/ml/predict/{model_name}?row_index=0`

Supported sample predictions are exposed only for models with saved preprocessing artifacts and training tables.

## API overview

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/replay`
- `POST /api/replay`
- `POST /api/replay/publish`
- `POST /api/replay/reset`
- `GET /api/events`
- `GET /api/activity`
- `GET /api/ml/models`
- `GET /api/ml/predict/{model_name}`

## Troubleshooting

- If the backend cannot import ML dependencies, reinstall `backend/requirements.txt`
- If Kafka replay does not update the database, confirm the `kafka-consumer` service is healthy
- If Airflow tasks fail, verify the backend container is reachable at `http://backend:8000`
- If the frontend shows connection errors, confirm the backend is running and `VITE_API_BASE_URL` is correct

## End-to-end demonstration

1. Start the backend
2. Open the frontend dashboard
3. Reset live state
4. Replay 100–1000 events
5. Watch KPIs, trends, recent orders, and processing activity update
6. Inspect saved ML models and run a sample prediction
7. Optionally publish the same immutable events to Kafka and let the consumer apply them

## Known limitations

- The repository contains only the Brazilian ecommerce project; there is no `retail-store` project in this clone
- Excel upload is not implemented because no existing Excel workflow or training-upload path exists in this repository
- The large offline training pipeline in `backend/pipeline.py` remains preserved but is not run automatically at API startup
