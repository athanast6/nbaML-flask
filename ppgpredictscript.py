from typing import Any
from unicodedata import numeric
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect
import pandas as pd


app = Flask(__name__)


players = pd.read_csv("statsnba2023.csv",encoding="utf-8-sig",dtype=str)


def ValuePredictor(playername):
    
    playerstats = pd.DataFrame(players.loc[players['Player'] == playername])
    print(playerstats)
    predictionarray = [playerstats.iat[0,4],playerstats.iat[0,3],playerstats.iat[0,5],playerstats.iat[0,9],playerstats.iat[0,8]]
    predictionarray = np.array(predictionarray).reshape(1,-1).astype(float)
    print(predictionarray)
    loaded_model = joblib.load(open("ppgpredictormodel2.0.pkl", "rb"))
    result = loaded_model.predict(predictionarray)
    return str(result)

@app.route('/')
def home():
    if request.method == 'GET':
        return render_template("index.html")

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		playername = request.form.get("playername")
		ppgpred = str(ValuePredictor(playername))
		return render_template("index.html", prediction = ppgpred)
    