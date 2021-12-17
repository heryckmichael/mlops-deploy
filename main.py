# -*- coding: utf-8 -*-

'''
API Flask para Trabalho de DS para Negócios
Equipe: Daniel Soares / Heryck / Klemerson / Leonardo
'''
# Import de Libs
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

############################################
# Criação da chamada à API
############################################

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth 

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = "admin"
app.config["BASIC_AUTH_PASSWORD"] = "123"

basic_auth = BasicAuth(app)


############################################
# Função que cria o Modelo
############################################

def create_model():

    # Lendo o dataset e criando um dataframe
    print("Carregando os dados")
    path_fetal_health = '/app/fetal_health.csv'
    df_fetal_health = pd.read_csv(path_fetal_health)

    print("Tratando os dados")

    ##Normalização
    # Visualizando o dataset antes da normalização e removendo a variável target

    df_fetal_health_features = df_fetal_health.drop(columns='fetal_health')

    # Treina o algoritmo e cria o obj_norm

    obj_norm = StandardScaler().fit(df_fetal_health_features)

    # Aplica o normalizador

    norm_df = obj_norm.transform(df_fetal_health_features)

    # Transforma o numpy array para um dataframe

    norm_df = pd.DataFrame(norm_df, columns=['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency'])

    """##Seleção de features"""

    df_fetal_health_target = df_fetal_health['fetal_health']

    """**F_Classif**"""

    f_classif_few_dimensions = SelectKBest(score_func=f_classif, k=5)

    x_f_classif_few_dimensions = norm_df
    y_f_classif_few_dimensions = df_fetal_health_target

    fit_f_classif_few_dimensions = f_classif_few_dimensions.fit(x_f_classif_few_dimensions, y_f_classif_few_dimensions)

    features_f_classif_few_dimensions = fit_f_classif_few_dimensions.transform(x_f_classif_few_dimensions)

    cols_f_classif_few_features = fit_f_classif_few_dimensions.get_support(indices=True)
    norm_df.iloc[:,cols_f_classif_few_features]

    X_f_classif_few_dimensions_model = norm_df.values
    y_f_classif_few_dimensions_model = df_fetal_health_target.values

    print("Treinando o Modelo")
    X_train_f_classif_few_dimensions, X_test_f_classif_few_dimensions, y_train_f_classif_few_dimensions, y_test_f_classif_few_dimensions = train_test_split(X_f_classif_few_dimensions_model, y_f_classif_few_dimensions_model, test_size=0.3, stratify=y_f_classif_few_dimensions_model)

    model_XGBClassifier_f_classif_few_dimensions = OneVsRestClassifier(XGBClassifier())
    
    return model_XGBClassifier_f_classif_few_dimensions.fit(X_train_f_classif_few_dimensions, y_train_f_classif_few_dimensions)

    #y_pred_XGBClassifier_f_classif_few_dimensions = model_XGBClassifier_f_classif_few_dimensions.predict(X_test_f_classif_few_dimensions)

################FIM DO MODELO###############
 
modelo = create_model()
print(modelo)
# Definição de rotas para a API

#@app.route("/score/")
# def get_score()
    
@app.route("/score/<cpf>")
def show_cpf(cpf):
    print("Recebido: CPF: = %s"%cpf)
    return "CPF = %s"%cpf

@app.route("/")
def home():
    print("Executou a rota padrão")
    return "API de predição pontuação de crédito"

app.run(debug=True)

###############FIM CHAMADA À API#############