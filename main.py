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
from flask_basicauth import BasicAuth

colunas = [
    'prolongued_decelerations',
    'abnormal_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'histogram_mode',
    'histogram_mean'
]

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = "admin"
app.config["BASIC_AUTH_PASSWORD"] = "123"

basic_auth = BasicAuth(app)


def load_model(file_name = 'xgboost_modelo.json'):
    return pickle.load(open(file_name, "rb"))

modelo = load_model()

@app.route("/diagnostico/", methods=['POST'])
@basic_auth.required
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
@basic_auth.required
def home():
    print("Executou a rota padrão")
    return "API de predição de pontuação de credito"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')