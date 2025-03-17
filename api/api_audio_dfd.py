
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Build our API

@app.get("/")
async def root():
    return {"message": "Ne pas avoir toutes ses frites dans le même sachet = ne pas avoir toute sa tête"}

# @app.get('/predict')
# def predict():
#     return {'wait': 64}
