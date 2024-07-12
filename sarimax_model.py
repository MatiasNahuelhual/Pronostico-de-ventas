import numpy as np
import pandas as pd
from pmdarima import auto_arima
import streamlit as st
import plotly.graph_objs as go

@st.cache_data
def load_data():
    test = pd.read_csv('test.csv')
    train = pd.read_csv('train.csv')
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
    indice_prox_semanas = np.arange(len(ventas_por_semana), len(ventas_por_semana) + len(pronostico_prox_semanas))
    
    # Crear figura
    fig = go.Figure()

    # Agregar datos originales
    fig.add_trace(go.Scatter(
        x=ventas_por_semana['week'], 
        y=ventas_por_semana['sum_num_orders'], 
        mode='lines', 
        name='Datos originales'
    ))

    # Agregar modelo ajustado
    fig.add_trace(go.Scatter(
        x=ventas_por_semana['week'], 
        y=pronostico, 
        mode='lines', 
        name='Modelo ARIMA ajustado', 
        line=dict(color='red')
    ))

    # Agregar pronóstico
    fig.add_trace(go.Scatter(
        x=indice_prox_semanas, 
        y=pronostico_prox_semanas, 
        mode='lines', 
        name='Pronóstico próximas 10 semanas', 
        line=dict(color='orange')
    ))

    # Agregar intervalo de confianza
    fig.add_trace(go.Scatter(
        x=np.concatenate([indice_prox_semanas, indice_prox_semanas[::-1]]),
        y=np.concatenate([confianza_sup, confianza_inf[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255, 165, 0, 0)'),
        showlegend=False,
        name='Intervalo de confianza'
    ))

    # Actualizar layout
    fig.update_layout(
        title='Pronóstico de Pedidos para las Próximas 10 Semanas',
        xaxis_title='Semana',
        yaxis_title='Número de Pedidos',
        template='plotly_dark'
    )

    return fig