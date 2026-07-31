import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.learning_controller import build_one_page_pipeline_dashboard
from controller.analytics_controller import run_task

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.get("/pipeline-dashboard")
def run_pipeline_dashboard(context, run_task, title):
  build_one_page_pipeline_dashboard(context, run_task, title)

@app.get("/analytics")
def run_task(task_name: str, df: pd.DataFrame):
    run_task(task_name, df)