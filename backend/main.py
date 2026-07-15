from fastapi import FastAPI

from controller.test_controller import get_users





app = FastAPI()



@app.get("/users")
def users():

    data = get_users()


    return {
        "users": data
    }