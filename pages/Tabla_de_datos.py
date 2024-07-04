import streamlit as st
import pandas as pd
from sarimax_model import load_data, preprocess_data, train_model, make_forecast

st.title('Datos de Ventas por Semana')

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

# Crear DataFrame para pronóstico
indice_prox_semanas = pd.date_range(start=ventas_por_semana['week'].iloc[-1], periods=num_semanas + 1, freq='W')[1:]
df_pronostico = pd.DataFrame({
    'week': indice_prox_semanas,
    'sum_num_orders': pronostico_prox_semanas,
    'type': 'Pronóstico'
})

# Añadir columna de tipo a los datos históricos
ventas_por_semana['type'] = 'Histórico'

# Combinar datos históricos con pronóstico
df_combinado = pd.concat([ventas_por_semana, df_pronostico])

# Mostrar datos combinados
st.dataframe(df_combinado)