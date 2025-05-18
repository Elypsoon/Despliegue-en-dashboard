# Creamos el archivo de la app en el interprete principal

import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Configuración de página
st.set_page_config(
    page_title="Análisis de Airbnb NY",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: #FF5A5F;
    }
    .section-header {
        font-weight: 600;
        margin-top: 1rem;
        border-bottom: 2px solid #FF5A5F;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ——— FUNCIONES ———
@st.cache_resource
def load_data(path='listings - New York_clean.csv'):
    """
    Carga y prepara los datos para el análisis
    
    Args:
        path (str): Ruta al archivo CSV
        
    Returns:
        tuple: DataFrame, columnas numéricas, columnas de texto, valores únicos de room_type, DataFrame numérico
    """
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

def display_general_summary(filtered_df):
    """
    Muestra el resumen general de los datos
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
    """
    st.markdown('<h2 class="section-header">Resumen general</h2>', unsafe_allow_html=True)
    
    # Métricas clave en tarjetas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precio promedio", f"${filtered_df['price'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'review_scores_rating' in filtered_df.columns:
            st.metric("Puntuación media", f"{filtered_df['review_scores_rating'].mean():.2f}/5")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total alojamientos", f"{len(filtered_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Estadísticas descriptivas
    with st.expander("Estadísticas numéricas", expanded=True):
        st.dataframe(filtered_df.describe(), use_container_width=True)

    # Reporte automático
    with st.expander("Reporte automático general", expanded=False):
        if st.button("Generar reporte detallado", key="gen_report"):
            with st.spinner('Generando reporte detallado...'):
                profile = ProfileReport(filtered_df, explorative=True, minimal=True)
                st_profile_report(profile)
        else:
            st.info("Haz clic en el botón para generar el reporte completo.")

def display_univariate_analysis(filtered_df, numeric_cols):
    """
    Muestra el análisis univariante
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
    """
    st.markdown('<h2 class="section-header">Análisis univariante</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Elegir variable
        all_cols = filtered_df.columns.tolist()
        var = st.selectbox("Selecciona una variable", all_cols)
    
    with col2:
        # Elegir tipo de gráfico según el tipo de variable
        if var in numeric_cols:
            chart_type = st.selectbox("Tipo de gráfica", ['Histograma', 'Caja', 'Línea'], key='numeric_chart')
        else:
            chart_type = st.selectbox("Tipo de gráfica", ['Barras', 'Pastel'], key='categorical_chart')
    
    # Mostrar estadísticas
    with st.expander("Estadísticas de la variable", expanded=True):
        st.write(filtered_df[var].describe())
    
    # Generar gráfico
    with st.spinner('Generando gráfico...'):
        if var in numeric_cols:
            if chart_type == 'Histograma':
                fig = px.histogram(filtered_df, x=var, nbins=50, 
                                   title=f"Histograma de {var}",
                                   color_discrete_sequence=['#FF5A5F'])
                fig.update_layout(bargap=0.1)
            elif chart_type == 'Caja':
                fig = px.box(filtered_df, y=var, 
                            title=f"Box-plot de {var}",
                            color_discrete_sequence=['#00A699'])
            else:
                fig = px.line(filtered_df[var].sort_values().reset_index(), 
                             x='id', y=var, 
                             title=f"Línea de {var}",
                             color_discrete_sequence=['#FC642D'])
        else:
            vc = filtered_df[var].value_counts().reset_index()
            vc.columns = [var, 'count']
            if chart_type == 'Barras':
                fig = px.bar(vc, x=var, y='count', 
                            title=f"Barras de {var}",
                            color_discrete_sequence=['#FF5A5F'])
            else:
                fig = px.pie(vc, names=var, values='count', 
                            title=f"Pastel de {var}",
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                            hole=0.4)
        
        # Mejorar diseño del gráfico
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def display_bivariate_analysis(filtered_df, numeric_cols):
    """
    Muestra el análisis bivariante
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
    """
    st.markdown('<h2 class="section-header">Análisis bivariante</h2>', unsafe_allow_html=True)
    
    # Seleccionar tipo de análisis
    analysis_type = st.radio("Tipo de análisis", 
                            ["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"], 
                            horizontal=True)
    
    if analysis_type == "Regresión lineal simple":
        display_simple_linear_regression(filtered_df, numeric_cols)
    elif analysis_type == "Regresión lineal múltiple":
        display_multiple_linear_regression(filtered_df, numeric_cols)
    else:  # Regresión logística
        display_logistic_regression(filtered_df, numeric_cols)

def display_simple_linear_regression(filtered_df, numeric_cols):
    """
    Muestra análisis de regresión lineal simple
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
    """
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
                        trace.line.color = '#FF5A5F'
                        trace.line.width = 3
            
            fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar correlación
            corr = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
            st.info(f"**Correlación de Pearson:** {corr:.2f}")

def display_multiple_linear_regression(filtered_df, numeric_cols):
    """
    Muestra análisis de regresión lineal múltiple
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
    """
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
            
            # Calcular coeficiente de correlación múltiple
            multiple_r = np.sqrt(model.rsquared)
            
            # Calcular coeficientes de correlación individual
            individual_correlations = {}
            for x_var in x_vars:
                corr = np.corrcoef(model_df[x_var], model_df[y_var])[0,1]
                individual_correlations[x_var] = corr
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Coeficientes de correlación")
                st.metric("Correlación múltiple (R)", f"{multiple_r:.4f}")
                
                # Mostrar correlaciones individuales
                st.subheader("Correlaciones individuales")
                for var, corr in individual_correlations.items():
                    st.metric(f"{var} vs {y_var}", f"{corr:.4f}")
            
            with col2:
                st.subheader("Valores reales vs. predichos")
                
                # Gráfico de valores reales vs predichos
                predictions = model.predict(X)
                pred_df = pd.DataFrame({'Actual': y, 'Predicción': predictions})
                
                fig = px.scatter(pred_df, x='Actual', y='Predicción',
                        template="plotly_white")
                
                # Añadir línea diagonal de referencia
                fig.add_shape(
                    type='line',
                    x0=pred_df['Actual'].min(),
                    y0=pred_df['Actual'].min(),
                    x1=pred_df['Actual'].max(),
                    y1=pred_df['Actual'].max(),
                    line=dict(color='#FF5A5F', dash='dash')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Matriz de correlaciones
            st.subheader("Matriz de correlaciones")
            
            with st.expander("Configurar matriz de correlaciones", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Seleccionar variables para la matriz de correlación
                    default_vars = [y_var] + x_vars
                    if len(default_vars) > 5:
                        default_vars = default_vars[:5]  # Limitar a 5 variables por defecto
                    
                    corr_vars = st.multiselect(
                        "Variables para correlación", 
                        numeric_cols, 
                        default=default_vars
                    )
                
                with col2:
                    corr_method = st.radio(
                        "Método de correlación",
                        ["pearson", "spearman", "kendall"],
                        horizontal=True
                    )
            
            if not corr_vars or len(corr_vars) < 2:
                st.warning("Selecciona al menos dos variables para generar la matriz de correlaciones.")
            else:
                # Calcular matriz de correlación
                corr_matrix = filtered_df[corr_vars].corr(method=corr_method)
                
                # Crear heatmap con plotly
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title=f"Matriz de correlación ({corr_method})",
                    aspect="auto",
                    labels=dict(color="Correlación")
                )
                
                fig_corr.update_layout(
                    height=max(400, len(corr_vars)*40),
                    template="plotly_white",
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
            

def display_logistic_regression(filtered_df, numeric_cols):
    """
    Muestra análisis de regresión logística
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
    """
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
                # Asegurarse de que la variable esté binarizada correctamente (0 y 1)
                if binary_var:
                    # Convertir a valores numéricos si son objetos
                    unique_values = filtered_df[binary_var].unique()
                    if filtered_df[binary_var].dtype == 'object':
                        # Crear mapeo: el primer valor será 0, el segundo 1
                        value_map = {unique_values[0]: 0, unique_values[1]: 1}
                        filtered_df[binary_var] = filtered_df[binary_var].map(value_map)
                        st.info(f"Codificación: {unique_values[0]} → 0, {unique_values[1]} → 1")
        
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
                
                try:
                    logit_model = sm.Logit(y, X).fit(disp=0)
                    
                    # Mostrar resultados del modelo
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Métricas de rendimiento
                        predictions = logit_model.predict(X)
                        pred_classes = (predictions > 0.5).astype(int)
                        
                        # Calcular métricas
                        accuracy = accuracy_score(y, pred_classes)
                        precision = precision_score(y, pred_classes, zero_division=0)
                        sensitivity = recall_score(y, pred_classes, zero_division=0)
                        
                        st.subheader("Métricas del modelo")
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric("Exactitud", f"{accuracy:.4f}")
                        with metrics_col2:
                            st.metric("Precisión", f"{precision:.4f}")
                        with metrics_col3:
                            st.metric("Sensibilidad", f"{sensitivity:.4f}")
                    
                    with col2:
                        
                        # Matriz de confusión mejorada
                        st.subheader("Matriz de confusión")
                        cm = confusion_matrix(y, pred_classes)
                        
                        # Crear matriz de confusión visual con plotly
                        cm_labels = ['Positivo', 'Negativo']
                        cm_fig = px.imshow(cm, 
                                           labels=dict(x="Predicción", y="Real", color="Cantidad"),
                                           x=cm_labels, 
                                           y=cm_labels,
                                           text_auto=True,
                                           color_continuous_scale='RdBu_r')
                        
                        cm_fig.update_layout(
                            xaxis=dict(side="top"),
                            height=400,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        
                        st.plotly_chart(cm_fig, use_container_width=True)
                except np.linalg.LinAlgError:
                    st.error("Error: Matriz singular detectada. Esto puede deberse a multicolinealidad entre las variables o a una separación perfecta en los datos.")
                    st.info("Sugerencia: Intenta seleccionar diferentes variables independientes o modifica el umbral de binarización.")
                except Exception as e:
                    st.error(f"Error al ajustar el modelo: {str(e)}")

def display_map(filtered_df):
    """
    Muestra el mapa de alojamientos
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
    """
    st.markdown('<h2 class="section-header">Mapa de alojamientos</h2>', unsafe_allow_html=True)
    
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

# ——— MAIN APP ———
def main():
    # Título principal con estilo personalizado
    st.markdown('<h1 class="main-header">Dashboard de Análisis: Airbnb en Nueva York</h1>', unsafe_allow_html=True)
    st.markdown("Explora los datos de alojamientos en Nueva York utilizando diferentes visualizaciones y análisis estadísticos.")

    # Cargamos los datos con la estructura original
    with st.spinner('Cargando datos...'):
        df, numeric_cols, text_cols, unique_room_type, numeric_df = load_data()

    # ——— SIDEBAR ———
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

    # Mostrar mensaje de error si no hay datos
    if len(filtered_df) == 0:
        st.error("No hay datos que cumplan con los criterios de filtrado. Por favor, ajusta los filtros.")
    else:
        # Renderizar la página seleccionada
        if page == 'Resumen general':
            display_general_summary(filtered_df)
        elif page == 'Univariante':
            display_univariate_analysis(filtered_df, numeric_cols)
        elif page == 'Bivariante':
            display_bivariate_analysis(filtered_df, numeric_cols)
        else:  # Mapa
            display_map(filtered_df)

if __name__ == "__main__":
    main()
