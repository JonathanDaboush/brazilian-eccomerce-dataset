from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller.test_controller import get_users
from quick_load import load_all_csv


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


@app.on_event("startup")
def startup():

    load_all_csv()


@app.get("/users")
def users():

    data = get_users()
    return {
        "users": data
    }