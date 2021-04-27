from predict import *
import pandas as pd
from fastapi import FastAPI


model = NLP_Solution("base") 
app = FastAPI()


@app.get('/hello')
def hello():
    return {"hello": "world!"}


@app.get('/predict_text/{text}')
async def predict_text(text: str):
    return model.predict([text])


@app.get('/predict_text_list/{text_list}')
async def predict_text_list(message: str):
    return model.predict([message])


