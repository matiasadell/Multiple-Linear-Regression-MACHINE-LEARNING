# Regresión Lineal Múltiple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

# Evitar la trampa de las variables ficticias
X = X[:, 1:]        # Eliminamos la primera columna

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)      # Agregamos una columna a nuestra matriz, que sean 50 filas de 1
SL = 0.05       # Siempre usar 0.05 en eliminacion hacia atras

# Se va ejecutar cada paso, y si una columna tiene mas valor que el SL la eliminamos hasta que quede una regresion lineal simple, y asi encontraremos la variable mas significativa para predecir

X_opt = X[:, [0, 1, 2, 3, 4, 5]]    # Va a agarrar todas las columnas y a cada paso va a eliminar una hasta que se quede con la mas significativa
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()     # Esta funcion le dice al dataset cual va a ser la proxima variable que se va a eliminar 
# endog es la variable que queremos predecir 
# exog representa las caracteristicas de la matriz
regression_OLS.summary()        # Me va adevolver un valor a cada una de las variables independientes, tambien los coeficientes del modelo y parametros como el r cuadrado

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()
# Variable mas significativa para la prediccion es gasto en IMAS