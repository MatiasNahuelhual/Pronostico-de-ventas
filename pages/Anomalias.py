import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sarimax_model import load_data, preprocess_data, train_model, make_forecast, plot_forecast


st.title('Deteccion de valores atipicos')
st.write('La deteccion de estos valores atipicos, es decir, anomalias es de suma importancia para identificar aquellos casos en los que ocurrio un evento que no es normal, para poder estudiar la causalidad, dentro de lo posible')


# Cargar datos
train, test = load_data()

# Preprocesar datos
ventas_por_semana = preprocess_data(train)

# Entrenar modelo
modelo_auto = train_model(ventas_por_semana)

# Número de semanas para pronosticar
num_semanas = 10

# Obtener pronóstico
pronostico, pronostico_prox_semanas, confianza_inf, confianza_sup = make_forecast(modelo_auto, ventas_por_semana, num_semanas)
pronostico1 = pd.DataFrame(pronostico)


# Deteccion de anomalias
data = ventas_por_semana
data = pd.concat([data, pronostico1], axis=1)

data['residual'] = data['sum_num_orders'] - data['predicted_mean']

# Función para detectar anomalías usando Z-Score
def detect_anomalies(residuals, threshold=3):
    mean = np.mean(residuals)
    std = np.std(residuals)
    z_scores = [(res - mean) / std for res in residuals]
    anomalies = np.where(np.abs(z_scores) > threshold)
    return anomalies[0]

# Detectar anomalías en los residuos
anomalies = detect_anomalies(data['residual'])

# Ver los valores de las anomalias
anomalous_weeks = data.iloc[anomalies]

# Crear la gráfica interactiva
fig = go.Figure()

# Agregar la serie de tiempo de ventas reales
fig.add_trace(go.Scatter(x=data.index, y=data['sum_num_orders'], mode='lines', name='Ventas Reales'))

# Agregar la serie de tiempo de pronóstico
fig.add_trace(go.Scatter(x=data.index, y=data['predicted_mean'], mode='lines', name='Pronóstico'))

# Agregar las anomalías
fig.add_trace(go.Scatter(x=anomalous_weeks.index, y=anomalous_weeks['sum_num_orders'], mode='markers', name='Anomalías', marker=dict(color='red', size=10, symbol='x')))

# Configurar el layout del gráfico
fig.update_layout(title='Detección de Anomalías en Ventas', xaxis_title='Semana', yaxis_title='Ventas', legend_title='Leyenda')

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig)

# Mostrar detalles de las semanas con anomalías
st.write('Detalle de semanas con anomalías:')
st.write(anomalous_weeks)