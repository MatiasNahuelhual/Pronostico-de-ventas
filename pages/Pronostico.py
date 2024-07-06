import streamlit as st
import matplotlib.pyplot as plt
from sarimax_model import load_data, preprocess_data, train_model, make_forecast, plot_forecast

st.title('Pronóstico de Ventas con SARIMAX')

st.write('Esta aplicación predice las ventas de un comercio usando el modelo SARIMAX.')

# Cargar datos
train, test = load_data()

# Preprocesar datos
ventas_por_semana = preprocess_data(train)

# Entrenar modelo
modelo_auto = train_model(ventas_por_semana)

# Pronosticar
#num_semanas = st.slider('Selecciona el número de semanas para pronosticar:', min_value=1, max_value=52, value=10)
num_semanas = 10
pronostico, pronostico_prox_semanas, confianza_inf, confianza_sup = make_forecast(modelo_auto, ventas_por_semana, num_semanas)

# Mostrar gráfico
st.write('### Pronóstico de Ventas')
fig = plot_forecast(ventas_por_semana, pronostico, pronostico_prox_semanas, confianza_inf, confianza_sup)
st.plotly_chart(fig)

