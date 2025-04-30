# Creamos el archivo de la app en el interprete principal

import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Análisis de Airbnb NY",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——— 1. CARGA DE DATOS ———
@st.cache_resource
def load_data(path='listings - New York_clean.csv'):
    # Cargamos el dataset de Airbnb, usando la columna 'id' como índice
    df = pd.read_csv(path, index_col='id')
    
    # Seleccionamos las columnas numéricas y obtenemos la lista de nombres
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    numeric_cols = numeric_df.columns.tolist()
    
    # Seleccionamos las columnas de tipo objeto (texto)
    text_df = df.select_dtypes(include=['object'])
    text_cols = text_df.columns.tolist()
    
    # Selección de la columna categórica: 'room_type'
    if 'room_type' in df.columns:
        unique_room_type = df['room_type'].unique().tolist()
    else:
        unique_room_type = []
    
    return df, numeric_cols, text_cols, unique_room_type, numeric_df

# Título principal
st.title("Dashboard de Análisis: Airbnb en Nueva York")
st.markdown("Explora los datos de alojamientos en Nueva York utilizando diferentes visualizaciones y análisis estadísticos.")

# Cargamos los datos con la estructura original
with st.spinner('Cargando datos...'):
    df, numeric_cols, text_cols, unique_room_type, numeric_df = load_data()

# ——— 2. SIDEBAR ———
st.sidebar.title("Filtros y Navegación")
st.sidebar.markdown("---")

# Selector de frame
default_page = 'Resumen general'
page = st.sidebar.radio("Secciones", 
                        ['Resumen general', 'Univariante', 'Bivariante', 'Mapa'],
                        index=['Resumen general', 'Univariante', 'Bivariante', 'Mapa'].index(default_page))

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración de Filtros")

# Filtro room_type con todas las opciones seleccionadas por defecto
sel_rooms = st.sidebar.multiselect("Tipo de habitación", 
                                  unique_room_type, 
                                  default=unique_room_type)

# Filtro precio con slider
min_price, max_price = int(df['price'].min()), int(df['price'].max())
sel_price = st.sidebar.slider("Rango de precio ($)", 
                             min_price, 
                             max_price, 
                             (min_price, max_price),
                             step=10)

# Aplicar filtros
filtered_df = df[df['room_type'].isin(sel_rooms)]
filtered_df = filtered_df[(filtered_df['price'] >= sel_price[0]) & (filtered_df['price'] <= sel_price[1])]

# Mostrar conteo de resultados
st.sidebar.markdown(f"**Resultados:** {len(filtered_df)} alojamientos")
st.sidebar.markdown("---")
st.sidebar.info("Inteligencia de Negocios")

# ——— 3. PAGES ———
if len(filtered_df) == 0:
    st.error("No hay datos que cumplan con los criterios de filtrado. Por favor, ajusta los filtros.")
else:
    if page == 'Resumen general':
        st.header("Resumen general")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio promedio", f"${filtered_df['price'].mean():.2f}")
        with col2:
            if 'review_scores_rating' in filtered_df.columns:
                st.metric("Puntuación media", f"{filtered_df['review_scores_rating'].mean():.2f}/5")
        
        with st.expander("Estadísticas numéricas", expanded=True):
            st.dataframe(filtered_df.describe().style.highlight_max(axis=0), use_container_width=True)

        with st.expander("Reporte automático general", expanded=False):
            if st.button("Generar reporte detallado"):
                with st.spinner('Generando reporte detallado...'):
                    profile = ProfileReport(filtered_df, explorative=True, minimal=True)
                    st_profile_report(profile)
            else:
                st.info("Haz clic en el botón para generar el reporte completo.")

    elif page == 'Univariante':
        st.header("Análisis univariante")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # elegir variable
            all_cols = filtered_df.columns.tolist()
            var = st.selectbox("Selecciona una variable", all_cols)
        
        with col2:
            # elegir tipo de gráfico según el tipo de variable
            if var in numeric_cols:
                chart_type = st.selectbox("Tipo de gráfica", ['Histograma', 'Caja', 'Línea'], key='numeric_chart')
            else:
                chart_type = st.selectbox("Tipo de gráfica", ['Barras', 'Pastel'], key='categorical_chart')
        
        # Mostrar estadísticas
        with st.expander("Estadísticas de la variable", expanded=True):
            st.write(filtered_df[var].describe())
        
        # Generar gráfico con indicador de carga
        with st.spinner('Generando gráfico...'):
            if var in numeric_cols:
                if chart_type == 'Histograma':
                    fig = px.histogram(filtered_df, x=var, nbins=50, 
                                       title=f"Histograma de {var}",
                                       color_discrete_sequence=['#1f77b4'])
                    fig.update_layout(bargap=0.1)
                elif chart_type == 'Caja':
                    fig = px.box(filtered_df, y=var, 
                                title=f"Box-plot de {var}",
                                color_discrete_sequence=['#ff7f0e'])
                else:
                    fig = px.line(filtered_df[var].sort_values().reset_index(), 
                                 x='id', y=var, 
                                 title=f"Línea de {var}",
                                 color_discrete_sequence=['#2ca02c'])
            else:
                vc = filtered_df[var].value_counts().reset_index()
                vc.columns = [var, 'count']
                if chart_type == 'Barras':
                    fig = px.bar(vc, x=var, y='count', 
                                title=f"Barras de {var}",
                                color_discrete_sequence=['#d62728'])
                else:
                    fig = px.pie(vc, names=var, values='count', 
                                title=f"Pastel de {var}",
                                hole=0.4)
            
            # Mejorar diseño del gráfico
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    elif page == 'Bivariante':
        st.header("Análisis bivariante")
        
        # Seleccionar tipo de análisis
        analysis_type = st.radio("Tipo de análisis", ["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"], horizontal=True)
        
        if analysis_type == "Regresión lineal simple":
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                x_var = st.selectbox("Variable eje X", numeric_cols, index=0, key="simple_x")
            
            with col2:
                y_var = st.selectbox("Variable eje Y", numeric_cols, index=min(1, len(numeric_cols)-1), key="simple_y")
            
            with col3:
                add_trend = st.checkbox("Agregar línea de tendencia", value=True)
                
                # Filtrar variables: incluir numéricas y categóricas con 8 o menos valores únicos
                valid_color_vars = ['Ninguno']
                for col in filtered_df.columns:
                    if col in numeric_cols or (col in filtered_df.select_dtypes(include=['object']).columns and filtered_df[col].nunique() <= 8):
                        valid_color_vars.append(col)
                
                color_var = st.selectbox("Color por", valid_color_vars, index=0)
            
            # Comprobar si se seleccionaron las mismas variables
            if x_var == y_var:
                st.error("Por favor selecciona variables diferentes para los ejes X e Y.")
            else:
                with st.spinner('Generando gráfico...'):
                    fig = px.scatter(
                        filtered_df, 
                        x=x_var, 
                        y=y_var,
                        color=None if color_var == 'Ninguno' else color_var,
                        trendline='ols' if add_trend else None,
                        title=f"{y_var} vs {x_var}",
                        opacity=0.7,
                        template="plotly_white"
                    )
                    
                    # Modificar color de línea de tendencia para hacerla más resaltante
                    if add_trend:
                        for trace in fig.data:
                            if hasattr(trace, 'mode') and trace.mode == 'lines':
                                trace.line.color = 'red'
                                trace.line.width = 3
                    
                    fig.update_layout(
                        height=600,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar correlación
                    corr = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
                    st.info(f"**Correlación de Pearson:** {corr:.2f}")
                    
        elif analysis_type == "Regresión lineal múltiple":
            import statsmodels.api as sm
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                y_var = st.selectbox("Variable dependiente (Y)", numeric_cols, key="multi_y")
                
            with col2:
                x_vars = st.multiselect("Variables independientes (X)", 
                                        [col for col in numeric_cols if col != y_var], 
                                        default=[numeric_cols[0]] if numeric_cols and numeric_cols[0] != y_var else [])
            
            if not x_vars:
                st.warning("Selecciona al menos una variable independiente.")
            else:
                # Eliminar filas con valores nulos en las variables seleccionadas
                model_df = filtered_df[[y_var] + x_vars].dropna()
                
                if len(model_df) == 0:
                    st.error("No hay datos suficientes para el análisis después de eliminar valores nulos.")
                else:
                    # Crear y ajustar el modelo
                    X = sm.add_constant(model_df[x_vars])
                    y = model_df[y_var]
                    model = sm.OLS(y, X).fit()
                    
                    # Mostrar resultados del modelo
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Resumen del modelo")
                        st.code(model.summary().as_text())
                    
                    with col2:
                        st.subheader("Métricas del modelo")
                        st.metric("R² ajustado", f"{model.rsquared_adj:.4f}")
                        st.metric("R²", f"{model.rsquared:.4f}")
                        st.metric("Error estándar", f"{model.mse_resid**.5:.4f}")
                    
                    # Gráfico de valores reales vs predichos
                    predictions = model.predict(X)
                    pred_df = pd.DataFrame({'Actual': y, 'Predicción': predictions})
                    
                    fig = px.scatter(pred_df, x='Actual', y='Predicción',
                                    title="Valores reales vs. predichos",
                                    template="plotly_white")
                    
                    # Añadir línea diagonal de referencia
                    fig.add_shape(
                        type='line',
                        x0=pred_df['Actual'].min(),
                        y0=pred_df['Actual'].min(),
                        x1=pred_df['Actual'].max(),
                        y1=pred_df['Actual'].max(),
                        line=dict(color='red', dash='dash')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Regresión logística
            import statsmodels.api as sm
            
            st.subheader("Regresión logística")
            
            # Seleccionar o crear una variable binaria para la regresión logística
            col1, col2 = st.columns([1, 1])
            
            with col1:
                binary_option = st.radio("Variable objetivo binaria", 
                                        ["Usar variable existente", "Binarizar variable numérica"])
            
            binary_var = None
            
            if binary_option == "Usar variable existente":
                # Encontrar variables binarias en el dataframe (con 2 valores únicos)
                binary_cols = [col for col in filtered_df.columns if filtered_df[col].nunique() == 2]
                
                if not binary_cols:
                    st.warning("No se encontraron variables binarias en el dataset")
                else:
                    binary_var = st.selectbox("Selecciona variable binaria", binary_cols)
                    
            else:  # Binarizar variable
                # Seleccionar variable para binarizar
                num_var_to_bin = st.selectbox("Variable para binarizar", numeric_cols)
                threshold = st.slider("Umbral para binarización", 
                                     float(filtered_df[num_var_to_bin].min()),
                                     float(filtered_df[num_var_to_bin].max()),
                                     float(filtered_df[num_var_to_bin].median()))
                
                # Crear la variable binaria
                binary_var = f"{num_var_to_bin}_bin"
                filtered_df[binary_var] = (filtered_df[num_var_to_bin] > threshold).astype(int)
                st.info(f"Variable creada: {binary_var} (1 si {num_var_to_bin} > {threshold}, 0 en caso contrario)")
            
            if binary_var:
                with col2:
                    x_vars = st.multiselect("Variables independientes", 
                                           [col for col in numeric_cols if col != binary_var],
                                           default=[numeric_cols[0]] if numeric_cols and numeric_cols[0] != binary_var else [])
                
                if not x_vars:
                    st.warning("Selecciona al menos una variable independiente.")
                else:
                    # Eliminar filas con valores nulos
                    model_df = filtered_df[[binary_var] + x_vars].dropna()
                    
                    if len(model_df) == 0:
                        st.error("No hay datos suficientes para el análisis después de eliminar valores nulos.")
                    else:
                        # Crear y ajustar el modelo de regresión logística
                        X = sm.add_constant(model_df[x_vars])
                        y = model_df[binary_var]
                        
                        logit_model = sm.Logit(y, X).fit(disp=0)
                        
                        # Mostrar resultados del modelo
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("Resumen del modelo")
                            st.code(logit_model.summary().as_text())
                        
                        with col2:
                            # Métricas de rendimiento
                            predictions = logit_model.predict(X)
                            pred_classes = (predictions > 0.5).astype(int)
                            
                            accuracy = np.mean(pred_classes == y)
                            st.metric("Precisión", f"{accuracy:.4f}")
                            
                            # Curva ROC
                            fpr, tpr, _ = roc_curve(y, predictions)
                            roc_auc = auc(fpr, tpr)
                            
                            fig = px.line(x=fpr, y=tpr,
                                         labels={'x': 'Tasa de falsos positivos', 'y': 'Tasa de verdaderos positivos'},
                                         title=f'Curva ROC (AUC = {roc_auc:.4f})')
                            
                            fig.add_shape(
                                type='line',
                                line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Matriz de confusión
                        st.subheader("Matriz de confusión")
                        cm = confusion_matrix(y, pred_classes)
                        cm_df = pd.DataFrame(cm, 
                                            index=['Real: 0', 'Real: 1'], 
                                            columns=['Pred: 0', 'Pred: 1'])
                        st.dataframe(cm_df)

    else:  # Mapa
        st.header("Mapa de alojamientos")
        
        if {'latitude', 'longitude'}.issubset(filtered_df.columns):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Opciones de visualización")
                color_option = st.selectbox(
                    "Color por", 
                    ['room_type', 'price', 'review_scores_rating', 'Ninguno'],
                    index=0
                )
                
                size_option = st.selectbox(
                    "Tamaño por", 
                    ['price', 'review_scores_rating', 'Constante'],
                    index=0
                )
                
                zoom = st.slider("Zoom", 8, 14, 11)
                
                st.info(f"**Mostrando:** {len(filtered_df)} alojamientos")
            
            with col1:
                with st.spinner('Cargando mapa...'):
                    fig = px.scatter_mapbox(
                        filtered_df,
                        lat='latitude', lon='longitude',
                        size=None if size_option == 'Constante' else size_option,
                        size_max=15,
                        color=None if color_option == 'Ninguno' else color_option,
                        hover_name='name' if 'name' in filtered_df.columns else None,
                        mapbox_style='open-street-map',
                        title="Distribución de alojamientos",
                        zoom=zoom,
                        height=700
                    )
                    
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se encontraron columnas 'latitude' y/o 'longitude' en el dataset.")
