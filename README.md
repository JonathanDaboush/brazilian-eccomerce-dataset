# Brazilian Ecommerce Replay Demo

This repository demonstrates a real, replay-driven ecommerce workflow over the Olist historical dataset:

Historical CSVs → immutable event bank → replay controls → Kafka/direct processing → live database state → KPI API → React dashboard → ML artifact inspection.

## What is implemented

- **Immutable source data** in `backend/original_data/` — never modified
- **Immutable event bank** generated into `backend/runtime/event_bank/olist_order_events.jsonl`
- **Replay API** for reset, replay, event preview, activity, and health
- **Kafka producer/consumer** using the same event payloads as the direct replay path
- **Live operational database** backed by SQLite by default or MySQL when `.env` is configured
- **Business dashboard API** with KPIs (revenue, delivery time, satisfaction score, customer lifetime value), monthly trends, top categories, recent activity, and recent orders
- **Feature store summary** for customer and seller aggregates exposed at `GET /api/features`
- **Excel / CSV upload endpoint** at `POST /api/upload` — validates structure and previews data without modifying originals
- **ML artifact inspection** for the saved models plus sample predictions for supported models
- **Airflow DAG** with a full orchestration pipeline: health check → reset → determine batch → validate → Kafka publish → direct DB replay → refresh analytics → verify KPIs → generate summary
- **React dashboard** with replay controls, speed options, KPI cards, trend tables, customer/seller feature store panels, upload section, and ML predictions

## Repository structure

```
backend/
  main.py                  FastAPI application
  services.py              replay, KPI, feature store, ML, upload, event-bank services
  database.py              SQLAlchemy models and runtime DB configuration
  constants.py             model targets, preprocessing configs, table mappings
  controller/
    producer.py            Kafka publisher
    consumer.py            Kafka consumer
  original_data/           immutable Olist CSVs (never modified)
  ml_training_data/        preserved feature datasets
  models/                  existing saved ML artifacts
frontend/
  src/api.js               centralized API client
  src/App.jsx              manager-facing dashboard
airflow/dags/
  ecommerce_replay_demo.py full orchestration DAG
docker-compose.yml
.env.example
```

## Prerequisites

- Python 3.12
- Node.js 20+
- npm
- Docker + Docker Compose (for the full stack with MySQL, Kafka, Airflow)

## Environment variables

Copy `.env.example` to `.env` and fill in the required values.

### Required for Docker / MySQL / Airflow

| Variable | Description |
|---|---|
| `MYSQL_ROOT_PASSWORD` | MySQL root password |
| `MYSQL_USER` | MySQL application user |
| `MYSQL_PASSWORD` | MySQL application password |
| `MYSQL_DATABASE` | MySQL database name |
| `POSTGRES_USER` | Airflow Postgres user |
| `POSTGRES_PASSWORD` | Airflow Postgres password |
| `POSTGRES_DB` | Airflow Postgres database name |
| `AIRFLOW_USER` | Airflow admin username |
| `AIRFLOW_PASSWORD` | Airflow admin password |
| `AIRFLOW_EMAIL` | Airflow admin email |

### Optional

| Variable | Description |
|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap servers (default: `kafka:9092`) |
| `KAFKA_TOPIC` | Kafka topic name (default: `olist-orders`) |
| `KAFKA_CONSUMER_GROUP` | Consumer group ID |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins |
| `VITE_API_BASE_URL` | Frontend API base URL (default: `http://localhost:8000`) |
| `DATABASE_URL` | Override the default SQLite DB path outside Docker |

## Local backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Default local mode uses a SQLite file at `backend/runtime/ecommerce_demo.db`. No MySQL or Kafka configuration required for local development.

## Local frontend setup

```bash
cd frontend
npm install
npm run dev
```

The dashboard expects the backend at `http://localhost:8000` unless `VITE_API_BASE_URL` is set in your environment.

## Docker startup

```bash
cp .env.example .env
# Edit .env and fill in required values
docker compose up --build
```

Services started:

| Service | URL |
|---|---|
| Backend API | `http://localhost:8000` |
| Airflow UI | `http://localhost:8080` |
| MySQL | `localhost:3306` |
| Kafka broker | `localhost:9092` |

Run the frontend separately: `cd frontend && npm run dev`

## Verifying each component

### Verify the database

```bash
# Local SQLite
sqlite3 backend/runtime/ecommerce_demo.db "SELECT count(*) FROM orders;"

# MySQL (Docker)
docker exec -it mysql-container mysql -u$MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE -e "SELECT count(*) FROM orders;"
```

### Verify Kafka

```bash
# Check that the broker is accessible
docker exec -it kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# List topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list

# Inspect consumer group lag
docker exec -it kafka kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group olist-replay-consumer
```

### Verify Airflow

1. Open `http://localhost:8080` and log in with `AIRFLOW_USER` / `AIRFLOW_PASSWORD`
2. Locate the `ecommerce_replay_demo` DAG
3. Trigger a manual run — all tasks should complete green

## Replay workflow

### Direct replay (API)

```bash
# Reset live state
curl -X POST http://localhost:8000/api/replay/reset

# Replay 500 events
curl -X POST http://localhost:8000/api/replay \
  -H "Content-Type: application/json" \
  -d '{"start_offset": 0, "limit": 500, "pace_ms": 0}'

# Check dashboard KPIs
curl http://localhost:8000/api/dashboard | python -m json.tool
```

### Kafka workflow

```bash
# Publish events to Kafka
cd backend
python controller/producer.py --limit 500

# In a separate terminal — consume events into the database
python controller/consumer.py
```

The consumer commits Kafka offsets only after the database write succeeds, providing at-least-once processing guarantees.

## Airflow workflow

Trigger the `ecommerce_replay_demo` DAG from the Airflow UI. The pipeline runs:

```
check_backend_health
  ↓
reset_replay_state
  ↓
determine_batch
  ↓
validate_batch
  ↓
publish_kafka_events ──── run_direct_replay_batch
                    ↘   ↗
                  refresh_analytics
                      ↓
                  verify_dashboard
                      ↓
                  generate_summary
```

## ML workflow

Trained model artifacts are stored in `backend/models/`. Preserved feature tables are in `backend/ml_training_data/`.

Supported models with sample predictions:

| Model | Target | Type |
|---|---|---|
| `delivery_delay` | `late_delivery` | Binary classification |
| `order_cancellation` | `cancelled` | Binary classification |
| `review_prediction` | `review_score` | Regression |
| `demand_forecasting` | `units_sold` | Regression |
| `customer_purchase_prediction` | `future_purchase` | Binary classification |

```bash
# List models
curl http://localhost:8000/api/ml/models

# Run a sample prediction
curl "http://localhost:8000/api/ml/predict/delivery_delay?row_index=0"
```

## API overview

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Backend health and event-bank status |
| `/api/dashboard` | GET | Full KPI snapshot with trends, categories, orders, activity |
| `/api/replay` | GET | Replay status, event bank metadata, recent batches |
| `/api/replay` | POST | Run a direct replay batch |
| `/api/replay/publish` | POST | Publish events to Kafka |
| `/api/replay/reset` | POST | Clear all live operational data |
| `/api/events` | GET | Preview event bank records |
| `/api/activity` | GET | Recent processed events |
| `/api/features` | GET | Customer and seller feature store summary |
| `/api/upload` | POST | Upload and validate a CSV / Excel file |
| `/api/ml/models` | GET | List saved ML models |
| `/api/ml/predict/{model}` | GET | Sample prediction for a supported model |

## Uploading data files

Use the **Upload data file** section in the dashboard to validate a `.csv`, `.xls`, or `.xlsx` file.
The file is read into memory only — the original is never stored or modified by the system.

## Troubleshooting

- **Backend fails to start**: confirm `backend/original_data/` contains the Olist CSV files
- **ML imports fail**: reinstall `pip install -r backend/requirements.txt`
- **Kafka replay does not update database**: confirm the `kafka-consumer` service is healthy (`docker compose ps`)
- **Airflow tasks fail**: confirm the `backend` container is reachable at `http://backend:8000` inside the Docker network
- **Frontend shows connection errors**: confirm the backend is running and `VITE_API_BASE_URL` is correct

## End-to-end demonstration

1. Start the backend (local or Docker)
2. Open the frontend dashboard at `http://localhost:5173`
3. Click **Reset live state** to clear any previous data
4. Select a replay speed and click **Replay 100 events** or **Replay 500 events**
5. Watch KPIs, monthly trends, recent orders, and processing activity update in real time
6. Scroll to **Top customers** and **Top sellers** to see the feature store populate
7. Select an ML model and click **Run sample prediction** to see an inference result
8. Upload a CSV or Excel file to validate its structure — originals are never modified
9. Open the Airflow UI at `http://localhost:8080` and trigger the `ecommerce_replay_demo` DAG

## Known limitations

- The Kafka consumer path and direct replay path both write to the same operational database; in production they would be separate
- The offline training pipeline in `backend/pipeline.py` is preserved but is not run automatically at API startup
- Prediction probabilities are shown only for classification models that expose `predict_proba`


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
