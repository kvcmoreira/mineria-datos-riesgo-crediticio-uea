import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Título y descripción de la aplicación web
st.title("📊 Sistema de Evaluación de Riesgo Crediticio")
st.write("Aplicación interactiva de Minería de Datos para predecir el riesgo de impago (mora) en solicitudes de crédito.")

# Panel lateral para ingreso de datos del cliente
st.sidebar.header("📋 Datos del Solicitante")

edad = st.sidebar.slider("Edad del cliente", 18, 70, 30)
ingresos = st.sidebar.number_input("Ingresos Anuales ($)", min_value=10000, max_value=200000, value=45000, step=1000)
monto = st.sidebar.number_input("Monto del Préstamo Solicitado ($)", min_value=1000, max_value=50000, value=12000, step=500)
tasa = st.sidebar.slider("Tasa de Interés (%)", 5.0, 25.0, 11.5)
score = st.sidebar.slider("Score de Crédito (Buró)", 300, 850, 650)
antiguedad = st.sidebar.slider("Antigüedad Laboral (Años)", 0, 30, 4)

# Feature Engineering (Variables calculadas)
ratio_prestamo_ingreso = monto / ingresos if ingresos > 0 else 0
cuota_estimada_mensual = (monto * (1 + tasa / 100)) / 12

# Entrenamiento del modelo de predicción (Random Forest)
@st.cache_resource
def entrenar_modelo():
    np.random.seed(42)
    n = 1000
    X_sim = pd.DataFrame({
        'edad': np.random.randint(18, 70, size=n),
        'ingresos_anuales': np.random.randint(12000, 120000, size=n),
        'monto_prestamo': np.random.randint(1000, 35000, size=n),
        'tasa_interes': np.round(np.random.uniform(5.0, 25.0, size=n), 2),
        'score_credito': np.random.randint(300, 850, size=n),
        'antiguedad_empleo_anios': np.random.randint(0, 25, size=n),
        'ratio_prestamo_ingreso': np.random.uniform(0.05, 0.8, size=n),
        'cuota_estimada_mensual': np.random.uniform(100, 3000, size=n)
    })
    y_sim = np.random.choice([0, 1], size=n, p=[0.78, 0.22])
    
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_sim, y_sim)
    return model

modelo = entrenar_modelo()

# Datos formateados para la predicción
input_data = pd.DataFrame([[edad, ingresos, monto, tasa, score, antiguedad, ratio_prestamo_ingreso, cuota_estimada_mensual]], 
                          columns=['edad', 'ingresos_anuales', 'monto_prestamo', 'tasa_interes', 'score_credito', 'antiguedad_empleo_anios', 'ratio_prestamo_ingreso', 'cuota_estimada_mensual'])

# Botón para ejecutar la predicción
if st.button("🔍 Evaluar Riesgo Crediticio"):
    prediccion = modelo.predict(input_data)[0]
    probabilidad = modelo.predict_proba(input_data)[0][1]
    
    st.subheader("Resultado del Análisis Predictivo:")
    if prediccion == 1:
        st.error(f"⚠️ ALTO RIESGO DE MORA (Probabilidad de impago: {probabilidad*100:.2f}%)")
        st.write("Recomendación: El perfil supera los umbrales de riesgo crediticio aceptables.")
    else:
        st.success(f"✅ CRÉDITO APROBADO (Probabilidad de impago: {probabilidad*100:.2f}%)")
        st.write("Recomendación: El perfil cumple los criterios de solvencia y bajo riesgo de mora.")
