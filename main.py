import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

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

def create_model():
    print("---------CRIANDO MODELO---------")
    print("-> CARREGANDO DADOS")
    path_fetal_health = 'fetal_health.csv'
    df_fetal_health = pd.read_csv(path_fetal_health)

    print("-> TRATANDO DADOS")
    df_fetal_health_features = df_fetal_health.drop(columns='fetal_health')
    obj_norm = StandardScaler().fit(df_fetal_health_features)
    norm_df = obj_norm.transform(df_fetal_health_features)
    norm_df = pd.DataFrame(norm_df, columns=['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency'])
    df_fetal_health_target = df_fetal_health['fetal_health']

    print("-> SELECIONANDO FEATURES")
    f_classif_few_dimensions = SelectKBest(score_func=f_classif, k=5)    
    x_f_classif_few_dimensions = norm_df
    y_f_classif_few_dimensions = df_fetal_health_target
    fit_f_classif_few_dimensions = f_classif_few_dimensions.fit(x_f_classif_few_dimensions, y_f_classif_few_dimensions)
    features_f_classif_few_dimensions = fit_f_classif_few_dimensions.transform(x_f_classif_few_dimensions)
    cols_f_classif_few_features = fit_f_classif_few_dimensions.get_support(indices=True)
    X_f_classif_few_dimensions_model = norm_df.iloc[:,cols_f_classif_few_features]
    y_f_classif_few_dimensions_model = df_fetal_health_target.values
    X_train_f_classif_few_dimensions, X_test_f_classif_few_dimensions, y_train_f_classif_few_dimensions, y_test_f_classif_few_dimensions = train_test_split(X_f_classif_few_dimensions_model, y_f_classif_few_dimensions_model, test_size=0.3, stratify=y_f_classif_few_dimensions_model)

    print("-> TREINANDO MODELO")
    model_XGBClassifier_f_classif_few_dimensions = OneVsRestClassifier(XGBClassifier())
    modelo = model_XGBClassifier_f_classif_few_dimensions.fit(X_train_f_classif_few_dimensions, y_train_f_classif_few_dimensions)
 
    return modelo


modelo = create_model()

@app.route("/score/", methods=['POST'])
def get_score():
    dados = request.get_json()
    payload = np.array([dados[col] for col in colunas])

    df_input = pd.DataFrame(payload).T
    df_input.columns = colunas

    resultado = modelo.predict(df_input) 
    return str(resultado[0])

@app.route("/home")
def home():
    print("Executou a rota padrão")
    return "API de predição de pontuação de credito"

app.run(debug=True)