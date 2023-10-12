# Paso 1: Importa las bibliotecas necesarias
import pandas as pd
from sklearn.linear_model import LinearRegression

# Paso 2: Carga la base de datos
data = pd.read_csv("data_v2.csv", delimiter=";")

# Paso 3: Divide los datos en variables dependientes e independientes
X = data[['habitantes', 'ingresos']]  # Variables predictoras: x1 (habitantes) y x2 (ingresos)
y = data['ingresos']  # Variable dependiente: y (ingresos)

# Paso 4: Crea y entrena el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X, y)

# Paso 5: Encuentra el intercepto de la regresión Lineal simple
intercepto = model.intercept_

# Paso 6: Encuentra los coeficientes de la regresión Lineal múltiple
coeficientes = model.coef_

print(f"Intercepto de la regresión Lineal simple: {intercepto}")
print(f"Coeficientes de la regresión Lineal múltiple (x1, x2): {coeficientes}")
