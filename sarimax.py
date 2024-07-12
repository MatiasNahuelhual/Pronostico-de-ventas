import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# Agrupar los datos por semana y sumar las ventas
ventas_por_semana = train.groupby('week')['num_orders'].sum().reset_index()
ventas_por_semana.columns = ['week', 'sum_num_orders']
ventas_por_semana

from pmdarima import auto_arima

modelo_auto = auto_arima(ventas_por_semana['sum_num_orders'], seasonal=True, m=8)  # m es la frecuencia estacional

pronostico = modelo_auto.predict_in_sample()
pronostico

# Número de semanas para pronosticar
num_semanas = 10
# Obtén las predicciones para las próximas 10 semanas
pronostico_prox_semanas = modelo_auto.predict(n_periods=num_semanas)

fitted, confint = modelo_auto.predict(n_periods=num_semanas, return_conf_int=True)

# Extrae los límites inferior y superior del intervalo de confianza
confianza_inf = confint[:, 0]
confianza_sup = confint[:, 1]

import seaborn as sns

plt.figure(figsize=(16, 9)) # 800x600 px

# Establece el estilo y el contexto de Seaborn
sns.set_style("darkgrid")
sns.set_context("talk")

# Crea un rango de índices para las próximas 10 semanas
indice_prox_semanas = np.arange(len(ventas_por_semana), len(ventas_por_semana) + num_semanas)

# Grafica los datos originales
sns.lineplot(x=ventas_por_semana['week'], y=ventas_por_semana['sum_num_orders'], label='Datos originales')

# Visualiza el modelo ARIMA ajustado
sns.lineplot(x=ventas_por_semana['week'], y=pronostico, color='red', label='Modelo ARIMA ajustado')

# Grafica las predicciones para las próximas 10 semanas
sns.lineplot(x=indice_prox_semanas, y=pronostico_prox_semanas, color='orange', label='Pronóstico próximas 10 semanas')

# Rellena el área entre los límites inferior y superior del intervalo de confianza
plt.fill_between(indice_prox_semanas, 
                 confianza_inf, 
                 confianza_sup, 
                 color='orange', alpha=.3, label='Intervalo de confianza')

plt.legend()
plt.title('Pronóstico de Pedidos para las Próximas 10 Semanas')
plt.xlabel('Semana')
plt.ylabel('Número de Pedidos')
plt.tight_layout()  # Ajusta el diseño para evitar que los elementos se superpongan
plt.show()