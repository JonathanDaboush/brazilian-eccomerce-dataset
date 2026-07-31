from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller.test_controller import get_customers


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

@app.get("/users")
def users():

    data = get_customers()
    return {
        "users": data
    }