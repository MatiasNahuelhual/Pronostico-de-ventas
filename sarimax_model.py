import numpy as np
import pandas as pd
from pmdarima import auto_arima
import streamlit as st

@st.cache_data
def load_data():
    test = pd.read_csv('C:/Users/matia/Repositorios/Pronostico-de-ventas/test.csv')
    train = pd.read_csv('C:/Users/matia/Repositorios/Pronostico-de-ventas/train.csv')
    return train, test

def preprocess_data(train):
    ventas_por_semana = train.groupby('week')['num_orders'].sum().reset_index()
    ventas_por_semana.columns = ['week', 'sum_num_orders']
    return ventas_por_semana

@st.cache_resource
def train_model(ventas_por_semana):
    modelo_auto = auto_arima(ventas_por_semana['sum_num_orders'], seasonal=True, m=8)
    return modelo_auto

def make_forecast(modelo_auto, ventas_por_semana, num_semanas=10):
    pronostico = modelo_auto.predict_in_sample()
    pronostico_prox_semanas, confint = modelo_auto.predict(n_periods=num_semanas, return_conf_int=True)
    confianza_inf = confint[:, 0]
    confianza_sup = confint[:, 1]

    return pronostico, pronostico_prox_semanas, confianza_inf, confianza_sup

def plot_forecast(ventas_por_semana, pronostico, pronostico_prox_semanas, confianza_inf, confianza_sup):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 9))
    sns.set_style("darkgrid")
    sns.set_context("talk")

    indice_prox_semanas = np.arange(len(ventas_por_semana), len(ventas_por_semana) + len(pronostico_prox_semanas))

    sns.lineplot(x=ventas_por_semana['week'], y=ventas_por_semana['sum_num_orders'], label='Datos originales')
    sns.lineplot(x=ventas_por_semana['week'], y=pronostico, color='red', label='Modelo ARIMA ajustado')
    sns.lineplot(x=indice_prox_semanas, y=pronostico_prox_semanas, color='orange', label='Pronóstico próximas 10 semanas')
    plt.fill_between(indice_prox_semanas, confianza_inf, confianza_sup, color='orange', alpha=.3, label='Intervalo de confianza')

    plt.legend()
    plt.title('Pronóstico de Pedidos para las Próximas 10 Semanas')
    plt.xlabel('Semana')
    plt.ylabel('Número de Pedidos')
    plt.tight_layout()

    return plt
