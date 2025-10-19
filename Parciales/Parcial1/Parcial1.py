import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype

CSV = "ventas-de-productores-de-glp.csv"


# Carga y exploración inicial
df = pd.read_csv(CSV, low_memory=False)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe())

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== NULOS POR COLUMNA ===")
print(df.isna().sum())

print("\n=== DUPLICADOS ===")
print(df.duplicated().sum())

print("\n=== HEAD ===")
print(df.head())

# 2) PREPARACION DE LOS DATOS

# Renombrar columna fecha_data a Fecha Data porque es la unica con guion bajo
df = df.rename(columns={"fecha_data": "Fecha Data"})

# convertir columnas numericas a objecto para analizar nulos
df["Volumen Fuera"] = df["Volumen Fuera"].astype("str")
df["Volumen Res 844/2007"] = df["Volumen Res 844/2007"].astype("str")

# tratamiento de nulos
TOKENS_NULOS = {"", " ", "   ", "-", "na", "NA", "NaN", "nan", "N/A", "Desconocido", "desconocido"}
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].replace(list(TOKENS_NULOS), pd.NA)

# convertir las columnas a numéricas
df["Volumen Fuera"] = pd.to_numeric(df["Volumen Fuera"], errors="coerce")
df["Volumen Res 844/2007"] = pd.to_numeric(df["Volumen Res 844/2007"], errors="coerce")

# convertir nulos en columnas categoricas a desconocido
df["Planta Carga"] = df["Planta Carga"].fillna("Desconocido")
df["Comprador"] = df["Comprador"].fillna("Desconocido")

# convertir nulos en columnas numericas a 0
df["Volumen Fuera"] = df["Volumen Fuera"].fillna(0)
df["Volumen Res 844/2007"] = df["Volumen Res 844/2007"].fillna(0)

# nulos tras el tratamiento
print("\n=== NULOS TRAS EL TRATAMIENTO ===")
print(df.isna().sum())

# derivacion de metricas
precio_promedio = df.groupby("Mes")["Precio Fuera Iva"].mean().round(2)
precio_ordenado = precio_promedio.sort_values(ascending=False)

print("\n=== PRECIO PROMEDIO POR MES ORDENADO DE MAYOR A MENOR ===")
print(precio_ordenado)

# Conversion de tipos columnas Object a tipado fuerte:
# Columna vendedor
df["Vendedor"] = df["Vendedor"].astype(str)
# Columna de planta de carga
df["Planta Carga"] = df["Planta Carga"].astype(str)
# Columna de tipo de producto
df["Tipo Producto"] = df["Tipo Producto"].astype(str)
# Cuit de comprador to numeric - tiene cosas como 2046330 y 20-46330
df["Cuit Comprador"] = df["Cuit Comprador"].astype(str)

# Funcion para pasar todos los cuits a XX-XXXXXXX-x
def formatear_cuit(cuit):
    cuit = str(cuit).replace("-","")
    if len(cuit) == 11:
        return f"{cuit[:2]}-{cuit[2:10]}-{cuit[10]}"
    return cuit  # Devuelve sin cambios si no tiene longitud válida
# Aplicar la funcion a la columna
df['Cuit Comprador'] = df['Cuit Comprador'].astype(str).apply(formatear_cuit)

# continuamos actualizando las columnas tipo object: Comprador
df["Comprador"] = df["Comprador"].astype(str)

# Actividad comprador
df["Actividad Comprador"] = df["Actividad Comprador"].astype(str)

# Fecha data a datetime
df["Fecha Data"] = pd.to_datetime(df["Fecha Data"])

# Mostrar nuevos tipos
print("\n=== DTYPES ===")
print("Tratamiento de tipos de datos object:")
print(df.dtypes)

# corroboramos si hubo cambios respecto a nulos
print("\n=== NULOS POR COLUMNA ===")
print(df.isna().sum())

# 3) ANÁLISIS DESCRIPTIVO

print("\n=== ANÁLISIS DESCRIPTIVO ===")

# Medidas estadísticas de columnas numéricas relevantes
columnas_numericas = ["Volumen Fuera", "Volumen Dentro", "Precio Fuera Iva", "Precio Dentro Iva"]

for col in columnas_numericas:
    print(f"\n--- {col} ---")
    print("Media:", df[col].mean().round(2))
    print("Mediana:", df[col].median().round(2))
    print("Desviación estándar:", df[col].std().round(2))
    print("Mínimo:", df[col].min())
    print("Máximo:", df[col].max().round(2))
    print("Q1:", df[col].quantile(0.25).round(2))
    print("Q3:", df[col].quantile(0.75).round(2))
    print("Rango:", df[col].max().round(2) - df[col].min().round(2))

negativos = (df["Volumen Fuera"] < 0).sum()
print("Cantidad de valores negativos en Volumen Fuera:", negativos)


# 4) VISUALIZACIÓN

print("\n=== VISUALIZACIÓN ===")
# Histograma de Volumen Fuera
plt.figure(figsize=(8,5))
sns.histplot(df["Volumen Fuera"], bins=50, kde=True)
plt.title("Distribución del Volumen Fuera")
plt.xlabel("Volumen (toneladas)")
plt.ylabel("Frecuencia")
plt.show()

# Histograma superpuesto de Precio Fuera Iva y Precio Dentro Iva
plt.figure(figsize=(8,5))
sns.histplot(df["Precio Dentro Iva"], color="blue", label="Precio Dentro", bins=40, kde=True)
sns.histplot(df["Precio Fuera Iva"], color="red", label="Precio Fuera", bins=40, kde=True)
plt.title("Comparación de precios dentro y fuera del país")
plt.xlabel("Precio (ARS)")
plt.legend()
plt.show()

# Boxplot de Precio Fuera Iva por mes
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="Mes", y="Precio Fuera Iva")
plt.title("Distribución del Precio Fuera Iva por Mes")
plt.xlabel("Mes")
plt.ylabel("Precio Fuera Iva")
plt.show()

# Scatterplot de Precio Fuera Iva vs Volumen Fuera
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Volumen Fuera", y="Precio Fuera Iva", hue="Tipo Producto", alpha=0.7)
plt.title("Relación entre Volumen y Precio (mercado externo)")
plt.xlabel("Volumen Fuera (toneladas)")
plt.ylabel("Precio Fuera Iva (ARS)")
plt.legend(title="Tipo de Producto")
plt.show()

# Heatmap de correlación de variables numéricas
plt.figure(figsize=(8,6))
corr = df[["Volumen Fuera", "Volumen Dentro", "Precio Fuera Iva", "Precio Dentro Iva"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlaciones entre variables numéricas")
plt.show()

# Pairplot de variables numéricas
sns.pairplot(df[["Volumen Fuera", "Volumen Dentro", "Precio Fuera Iva", "Precio Dentro Iva"]])
plt.suptitle("Matriz de dispersión de variables numéricas", y=1.02)
plt.show()
