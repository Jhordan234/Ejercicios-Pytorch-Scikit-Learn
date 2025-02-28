#Ejercicio Número 1

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

suma = a + b
print("Suma:", suma)

df = pd.DataFrame({'A': a, 'B': b, 'Suma': suma})
print("\nDataFrame:\n", df)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
print("\nDataFrame Normalizado:\n", df_scaled)

#Ejercicio Número 2

from sklearn.preprocessing import StandardScaler

x = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
y = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)

producto = x * y
print("Producto:\n", producto)

scaler = StandardScaler()
producto_scaled = scaler.fit_transform(producto)
print("\nProducto Normalizado:\n", producto_scaled)

#Ejercicio Número 3

from sklearn.linear_model import LinearRegression

x = np.array([4, 8, 12, 16, 20]).reshape(-1, 1)
y = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

division = x / y
print("División:\n", division)

model = LinearRegression()
model.fit(x, division)  

prediccion = model.predict([[12]])
print("\nPredicción para x=12:", prediccion)