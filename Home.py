import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.logo("icon.png")
st.title("Home")

st.markdown(
            "### Aplicacion de pronostico de series temporales"
            )
st.markdown("En esta aplicacion se mostrara toda la informacion de las ventas del negocio con el fin de realizar un analisis descriptivo")
st.markdown("### Tarea propuesta")
st.markdown(
            'Una empresa de entrega de comidas que opera en varias ciudades, tiene varios centros logísticos en estas ciudades para enviar pedidos de comida a sus clientes quiere que usted ayude a estos centros con la previsión de la demanda para las próximas semanas, de esta manera, hará que los centros planifiquen mejor el stock de materias primas en consecuencia. Logrando así, reducir las pérdidas, optimizando los costos. Dada la información, la tarea es predecir la demanda durante las próximas 10 semanas (semanas: 146-155) para las combinaciones de comida central en el conjunto de prueba.'
            )
