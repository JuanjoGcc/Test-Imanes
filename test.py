import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el archivo CSV en un DataFrame de pandas.
    Ajusta los separadores y decimales según corresponda.
    """
    df = pd.read_csv(ruta_csv, sep=';', decimal=',')
    
    # Ejemplo: agregar columnas extra si las necesitas
    df['Numero de Caja'] = None

    # Asegurar que 'Magnet ID' sea la columna más a la izquierda
    if 'Magnet ID' in df.columns:
        cols = ['Magnet ID'] + [col for col in df.columns if col != 'Magnet ID']
        df = df[cols]

    return df


def asignar_cajas(df: pd.DataFrame,
                  col_fecha: str = 'Date',
                  col_caja: str = 'Numero de Caja',
                  cajas: list = ['A1','A2','A3','A4','A5','A6','A7','A8','B1','B2','B3','B4','B5','B6'],
                  imanes_por_dia: dict = {1:10, 2:20, 3:30, 4:40, 5:30},
                  manual_assignments_day6: list = [
                                                    ('A1', 1751, 1784),
                                                    ('A2', 1785, 1820),
                                                    ('A3', 1821, 1853),
                                                    ('A4', 1854, 1889),
                                                    ('A5', 1890, 1928),
                                                    ('A6', 1929, 1964),
                                                    ('A7', 1965, 2000),
                                                    ('A8', 2001, 2034),
                                                    ('B1', 2035, 2036),
                                                    ('B2', 2037, 2072),
                                                    ('B3', 2073, 2108),
                                                    ('B4', 2109, 2144),
                                                    ('B5', 2145, 2180),
                                                    ('B6', 2181, 2216),
                                                    # Continúa según sea necesario
                                                ]) -> pd.DataFrame:
    """
    Asigna cada fila a una caja específica ('Numero de Caja').

    - Para los primeros 5 días, asigna automáticamente una cantidad fija de imanes por caja.
    - Para el Día 6, asigna cajas basándose en rangos de 'Magnet ID'.

    Parameters:
    - df: DataFrame con los datos.
    - col_fecha: Nombre de la columna que contiene las fechas.
    - col_caja: Nombre de la columna donde se asignarán las cajas.
    - cajas: Lista de nombres de cajas disponibles.
    - imanes_por_dia: Diccionario que mapea el número del día (1-5) a la cantidad de imanes por caja.
    - manual_assignments_day6: Lista de tuplas (nombre_caja, inicio_Magnet_ID, fin_Magnet_ID) para el Día 6.

    Returns:
    - DataFrame con la columna 'Numero de Caja' asignada.
    """
    # Paso 1: Convertir la columna 'Date' a datetime si no lo está
    if not np.issubdtype(df[col_fecha].dtype, np.datetime64):
        df[col_fecha] = pd.to_datetime(df[col_fecha], format='%a %b %d %Y')

    # Paso 2: Obtener días únicos en orden cronológico
    dias_ordenados = sorted(df[col_fecha].unique())
    print("Días únicos en orden cronológico:", dias_ordenados)

    # Paso 3: Asignar cajas automáticamente para los primeros 5 días
    for i in range(min(5, len(dias_ordenados))):
        dia_num = i + 1  # Día 1 a Día 5
        dia_actual = dias_ordenados[i]
        cant = imanes_por_dia.get(dia_num, 0)

        if cant == 0:
            print(f"No se definió cantidad de imanes para el Día {dia_num}. Salta este día.")
            continue  # Salta si no hay cantidad definida

        # Filtrar filas para el día actual
        df_dia = df[df[col_fecha] == dia_actual].copy()

        # Ordenar las filas por 'Magnet ID' (puedes ajustar este criterio si es necesario)
        df_dia_sorted = df_dia.sort_values(by='Magnet ID').reset_index()

        start = 0
        for caja in cajas:
            end = start + cant
            if start >= len(df_dia_sorted):
                break  # Termina si no hay más imanes para asignar

            # Obtener Magnet IDs para esta caja
            magnet_ids = df_dia_sorted['Magnet ID'].iloc[start:end].tolist()

            # Asignar la caja a estos Magnet IDs
            df.loc[df['Magnet ID'].isin(magnet_ids) & (df[col_fecha] == dia_actual), col_caja] = caja

            start = end  # Actualizar el inicio para la siguiente caja

        # Manejar imanes sobrantes (si los hay)
        if start < len(df_dia_sorted):
            sobrantes = df_dia_sorted['Magnet ID'].iloc[start:].tolist()
            df.loc[df['Magnet ID'].isin(sobrantes) & (df[col_fecha] == dia_actual), col_caja] = 'Sobrante'
            print(f"Se asignaron {len(sobrantes)} imanes como 'Sobrante' para el Día {dia_num}.")

    # Paso 4: Asignar cajas manualmente para el Día 6, si existe y se proporcionan asignaciones
    if len(dias_ordenados) > 5 and manual_assignments_day6:
        dia_6 = dias_ordenados[5]
        for caja, inicio_id, fin_id in manual_assignments_day6:
            mask = (df[col_fecha] == dia_6) & (df['Magnet ID'] >= inicio_id) & (df['Magnet ID'] <= fin_id)
            df.loc[mask, col_caja] = caja
        print(f"Asignaciones manuales completadas para el {dia_6}.")

    return df




def convertir_columna_date(df: pd.DataFrame, col_fecha: str = 'Date') -> pd.DataFrame:
    """
    Convierte la columna 'Date' en un tipo datetime y conserva solo la parte de la fecha,
    asumiendo que viene con un formato tipo: 'Wed Jan 8 2025' (día de la semana, mes, día numérico, año).
    Ajusta el 'format' si tu string no coincide exactamente.
    """
    df[col_fecha] = pd.to_datetime(df[col_fecha], format='%a %b %d %Y').dt.date
    return df



def obtener_dias_unicos_ordenados(df: pd.DataFrame, col_fecha: str = 'Date') -> list:
    """
    Retorna una lista de días únicos en orden cronológico (no alfabético).
    """
    # Únicos
    dias_unicos = df[col_fecha].unique()
    # Ordenarlos cronológicamente
    dias_ordenados = sorted(dias_unicos)
    return dias_ordenados

def agrupar_df_por_dia(df: pd.DataFrame, dias: list, col_fecha='Date', col_campo='B (mT)') -> dict:
    """
    Filtra el DataFrame por cada día de la lista y los guarda en un diccionario.
    La clave será 'Dia 1', 'Dia 2', etc., y el valor será un DataFrame filtrado.
    """
    data_por_dia = {}
    for i, dia in enumerate(dias):
        clave = f"Dia {i+1}"
        mask = (df[col_fecha] == dia)
        data_por_dia[clave] = df.loc[mask].copy()  # copia de la parte filtrada
    return data_por_dia


def calcular_offset_por_dia(data_por_dia: dict, col_campo='B (mT)', referencia='Dia 1') -> dict:
    """
    Calcula los offsets para alinear la media de cada día a la media del día de referencia.
    offset_por_dia[día] = mean(día_actual) - mean(día_referencia)
    """
    # Calcular la media de cada día
    medias = {}
    for clave, df_dia in data_por_dia.items():
        medias[clave] = df_dia[col_campo].mean()

    # Offset de cada día para alinear con la media del día de referencia
    media_ref = medias[referencia]
    offsets = {}
    for clave, media in medias.items():
        offsets[clave] = media - media_ref

    return offsets


def aplicar_offsets(data_por_dia: dict, offsets: dict, col_campo='B (mT)') -> dict:
    """
    Aplica los offsets calculados a cada DataFrame, restándolos del valor de B (mT).
    Crea la columna 'B (mT) Ajustado' con los valores corregidos.
    """
    data_por_dia_ajustado = {}
    for clave, df_dia in data_por_dia.items():
        df_copia = df_dia.copy()
        df_copia['B (mT) Ajustado'] = df_copia[col_campo] - offsets[clave]
        data_por_dia_ajustado[clave] = df_copia
    return data_por_dia_ajustado


def combinar_dias_en_un_df(data_por_dia_ajustado: dict) -> pd.DataFrame:
    """
    Concatena todos los días (ya ajustados) en un único DataFrame.
    """
    df_comb = pd.concat(data_por_dia_ajustado.values(), ignore_index=True)
    return df_comb


def descartar_outliers_por_porcentaje(df: pd.DataFrame, col_campo='B (mT) Ajustado', percent=5):
    """
    Descartar outliers en base a percentiles (percent% en cada extremo) sobre 'col_campo'.
    Retorna:
      - df_filtrado: DataFrame con los datos dentro de los límites.
      - df_descartados: DataFrame con los datos que quedaron fuera de los límites.
    """
    lower_limit = np.percentile(df[col_campo], percent)
    upper_limit = np.percentile(df[col_campo], 100 - percent)

    mask = (df[col_campo] >= lower_limit) & (df[col_campo] <= upper_limit)
    df_filtrado = df.loc[mask].copy()
    df_descartados = df.loc[~mask].copy()

    return df_filtrado, df_descartados


def graficar_kde_por_dias(data_por_dia_ajustado: dict, col_ajustado='B (mT) Ajustado', xlim=None, titulo="Densidades por Día Ajustadas"):
    """
    Grafica la densidad (KDE) de cada día (col_ajustado) en una sola figura.
    """
    plt.figure(figsize=(6, 4))
    for clave, df_dia in data_por_dia_ajustado.items():
        sns.kdeplot(df_dia[col_ajustado], fill=True, label=clave, alpha=0.5)
    
    plt.xlabel(col_ajustado)
    plt.ylabel("Densidad")
    plt.title(titulo)
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


def graficar_hist_con_kde(df: pd.DataFrame, col='B (mT) Ajustado', bins=22, color='blue', titulo="Histograma con KDE", xlim=None):
    """
    Grafica un histograma + KDE de la columna especificada.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color=color, bins=bins, label="Datos")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.title(titulo)
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Cargar datos
    ruta = "data-general.csv"
    df = cargar_datos(ruta)

    # 2. Convertir la columna Date a datetime para que sort sea cronológico
    df = convertir_columna_date(df, col_fecha='Date')

    # 3. Definir asignaciones manuales para el Día 6
    manual_assignments_day6 = [
        ('A1', 1751, 1784),
        ('A2', 1785, 1820),
        ('A3', 1821, 1853),
        ('A4', 1854, 1889),
        ('A5', 1890, 1928),
        ('A6', 1929, 1964),
        ('A7', 1965, 2000),
        ('A8', 2001, 2034),
        ('B1', 2035, 2036),
        ('B2', 2037, 2072),
        ('B3', 2073, 2108),
        ('B4', 2109, 2144),
        ('B5', 2145, 2180),
        ('B6', 2181, 2216),
        # Continúa según sea necesario
    ]

    # 4. Asignar Cajas (Automático para los primeros 5 días, Manual para el Día 6)
    df = asignar_cajas(df,
                       col_fecha='Date',
                       col_caja='Numero de Caja',
                       cajas=['A1','A2','A3','A4','A5','A6','A7','A8','B1','B2','B3','B4','B5','B6'],
                       imanes_por_dia={1:10, 2:20, 3:30, 4:40, 5:30},
                       manual_assignments_day6=manual_assignments_day6)



    # Guardar en aux.csv
    df.to_csv("aux.csv", index=False)

    # 3. Obtener días únicos en orden de fecha real
    dias = obtener_dias_unicos_ordenados(df, col_fecha='Date')
    print("Días únicos en orden correcto:", dias)

    # 3. Agrupar datos por día
    data_por_dia = agrupar_df_por_dia(df, dias, col_fecha='Date', col_campo='B (mT)')

    # 4. Calcular offsets para alinear con el día de referencia (Dia 1)
    offsets = calcular_offset_por_dia(data_por_dia, col_campo='B (mT)', referencia='Dia 1')
    print("Offsets calculados:", offsets)
    
    # 5. Aplicar offsets
    data_por_dia_ajustado = aplicar_offsets(data_por_dia, offsets, col_campo='B (mT)')
    
    # 6. Combinar todos los días en un DataFrame
    df_comb = combinar_dias_en_un_df(data_por_dia_ajustado)
    print("Tamaño de df_comb (todos los días ajustados):", df_comb.shape)
    
    # 7. Graficar densidades por día
    #graficar_kde_por_dias(data_por_dia_ajustado, col_ajustado='B (mT) Ajustado', titulo="Densidad (KDE) por Día - Ajustado a referencia Día 1")
    
    # 8. Graficar histograma de todos los datos combinados (ajustados)
    #graficar_hist_con_kde(df_comb, col='B (mT) Ajustado', titulo="Histograma + KDE - Todos los días ajustados")
    
    # 9. Descartar outliers (por ejemplo, 5% en cada extremo)
    df_filtrado, df_descartados = descartar_outliers_por_porcentaje(df_comb, col_campo='B (mT) Ajustado', percent=5)
    
    print("Tamaño de df_filtrado:", df_filtrado.shape)
    print("Tamaño de df_descartados:", df_descartados.shape)
    
    # 10. Ver cuáles 'Magnet ID' fueron descartados
    magnet_ids_descartados = df_descartados['Magnet ID'].unique()
    print("\nMagnet IDs descartados por outliers:", magnet_ids_descartados)
    
    # 11. Graficar histograma de datos filtrados (sin outliers)
    #graficar_hist_con_kde(df_filtrado, col='B (mT) Ajustado', color='green', titulo="Histograma + KDE - Datos Filtrados (sin outliers)")
    

if __name__ == "__main__":
    main()
