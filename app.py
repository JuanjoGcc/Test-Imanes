import streamlit as st
import pandas as pd
import altair as alt
from test import (
    cargar_datos, 
    agrupar_df_por_dia, 
    asignar_cajas,
    calcular_offset_por_dia, 
    descartar_outliers_por_porcentaje,
    obtener_dias_unicos_ordenados,
    convertir_columna_date,
    aplicar_offsets
)

def main():
    st.set_page_config(page_title="Análisis de Imanes", layout="wide")
    st.title("Análisis y Descarte de Outliers de Imanes")

    # Menú de navegación en la barra lateral
    menu = ["Carga de Datos", "Descarte de Outliers", "Visualización", "Configuraciones"]
    choice = st.sidebar.radio("Navegación", menu)

    if choice == "Carga de Datos":
        cargar_datos_seccion()
    elif choice == "Descarte de Outliers":
        descartar_outliers_seccion()
    elif choice == "Visualización":
        visualizar_datos_seccion()
    elif choice == "Configuraciones":
        configuraciones_seccion()

import streamlit as st
import pandas as pd
from test import cargar_datos, asignar_cajas

def cargar_datos_seccion():
    st.header("Carga de Datos")
    try:
        # Cargar los datos
        df = cargar_datos("data-general.csv")
        st.success("Datos cargados exitosamente.")
        
        # Asignar Numero de Caja
        df = asignar_cajas(df)
        df = convertir_columna_date(df, col_fecha='Date')

        # Crear tres columnas: izquierda, central, derecha
        col1, col2, col3 = st.columns([1, 5, 1])  # Proporciones: 1:2:1
        
        # Colocar el DataFrame en la columna central
        with col2:
            st.dataframe(df)
        
        # Guardar el DataFrame en el estado de sesión
        st.session_state['df'] = df
        
    except FileNotFoundError:
        st.error("El archivo 'data-general.csv' no fue encontrado. Por favor, verifica la ubicación.")
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")

def descartar_outliers_seccion():
    st.header("Descarte de Outliers")
    if 'df' not in st.session_state:
        st.warning("Por favor, carga los datos primero en la sección 'Carga de Datos'.")
        return

    df = st.session_state['df']
    required_columns = ['Date', 'Numero de Caja', 'B (mT)', 'Magnet ID']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Faltan columnas requeridas: {set(required_columns) - set(df.columns)}")
        return

    dias_unicos = obtener_dias_unicos_ordenados(df, col_fecha='Date')
    dias_seleccionados = st.multiselect("Selecciona los días a analizar", dias_unicos, default=dias_unicos)
    df_filtrado_por_dias = df[df['Date'].isin(dias_seleccionados)].copy()

    percent = st.slider("Porcentaje de outliers (en cada extremo)", min_value=0, max_value=10, value=5, step=1)

    if st.button("Descartar outliers"):
        df_filtrado, df_descartados = descartar_outliers_por_porcentaje(
            df_filtrado_por_dias, 
            col_campo="B (mT)", 
            percent=percent
        )
        st.write("Datos Filtrados (IN):", df_filtrado.shape[0], "registros")
        st.write("Outliers Descartados (OUT):", df_descartados.shape[0], "registros")
        magnet_ids_descartados = df_descartados['Magnet ID'].unique()
        st.write("**Magnet IDs descartados**:", magnet_ids_descartados)

        # Guardar los datos filtrados en el estado de sesión
        st.session_state['df_filtrado'] = df_filtrado
        st.session_state['df_descartados'] = df_descartados

def visualizar_datos_seccion():
    st.header("Visualización de Datos")
    if 'df_filtrado' not in st.session_state:
        st.warning("Por favor, realiza el descarte de outliers primero en la sección 'Descarte de Outliers'.")
        return

    df_filtrado = st.session_state['df_filtrado']

    st.subheader("Visualización de datos filtrados")
    chart_filtrado = alt.Chart(df_filtrado).mark_circle(size=60).encode(
        x=alt.X("Magnet ID:N", title="ID de Imán"),
        y=alt.Y("B (mT):Q", title="Campo (mT)"),
        color="Numero de Caja:N",
        tooltip=["Magnet ID", "B (mT)", "Numero de Caja", "Date"]
    ).interactive()

    st.altair_chart(chart_filtrado, use_container_width=True)

    # Botón para descargar datos filtrados
    csv_filtrado = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos filtrados",
        data=csv_filtrado,
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )

    if 'df_descartados' in st.session_state:
        df_descartados = st.session_state['df_descartados']
        csv_descartados = df_descartados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar outliers descartados",
            data=csv_descartados,
            file_name="outliers_descartados.csv",
            mime="text/csv"
        )

def configuraciones_seccion():
    st.header("Configuraciones Avanzadas")
    st.write("Aquí puedes agregar configuraciones adicionales según tus necesidades.")

if __name__ == "__main__":
    main()
