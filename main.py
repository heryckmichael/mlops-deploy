import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify

app = Flask(__name__)

colunas = [
    'prolongued_decelerations',
    'abnormal_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'histogram_mode',
    'histogram_mean'
]

def load_model(file_name = 'xgboost_modelo.json'):
    return pickle.load(open(file_name, "rb"))

modelo = load_model()

@app.route("/diagnostico/", methods=['POST'])
def get_diagnostico():
    dados = request.get_json()
    payload = np.array([dados[col] for col in colunas])

    df_input = pd.DataFrame(payload).T
    df_input.columns = colunas

    resultado = modelo.predict(df_input)
    if(resultado[0]==1) :
        res = 'Normal'
    elif(resultado[0]==2) :
        res = 'Suspeito'
    else :
        res = 'Patologico'
        
    return str(res)

@app.route("/home")
def home():
    print("Executou a rota padrão")
    return "API de predição de pontuação de credito"

app.run(debug=True)