# IAs Utilizadas en el Codigo
# DeepSeek. (2023). DeepSeek AI models [Modelo de inteligencia artificial]. DeepSeek. https://www.deepseek.com

#Nombre: Felipe Serey Morales


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

#Carga de la fuente de datos
data = pd.read_csv('bank.csv')

#Separacion de datos de entrada y de salida
X = data[['age', 'job', 'marital', 'education', 'default', 'balance',
          'housing', 'loan', 'contact', 'day', 'month', 'duration',
          'campaign', 'pdays', 'previous', 'poutcome']]

Y  = data['y'].map({'yes' : 1, 'no' : 0})



# Separacion de variables numericas y categoricas
categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                   'loan', 'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Preprocesamiento con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(
            drop='first', handle_unknown='ignore'), categorical_cols)
    ])


#Separacion de datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#------------------------------------------------------------------------------------
#Entrenamiento del modelo de regresion,
regresion_log = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42))
])
regresion_log.fit(X_train, Y_train)
Y_pred = regresion_log.predict(X_test)

#Impresion de resultados
print("=============Resultados regresion logistica ==================")
print(f"Exactitud: {accuracy_score(Y_test,Y_pred): .4f}")
print()
print("Reporte de Clasificacion")
print(classification_report(Y_test,Y_pred))
print()

#------------------------------------------------------------------------------------
#Entrenamiento del arbol de decisiones
arbol_de_Decisiones = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                                        class_weight='balanced', random_state=42))
])
arbol_de_Decisiones.fit(X_train, Y_train)
Y_pred = arbol_de_Decisiones.predict(X_test)

#Impresion de resultados
print("=============Resultados arbol de desiciones ==================")
print(f"Exactitud: {accuracy_score(Y_test,Y_pred): .4f}")
print()
print("Reporte de Clasificacion")
print(classification_report(Y_test,Y_pred))
print()


#------------------------------------------------------------------------------------
#Entrenamiento del bosque aleatorio
bosque_aleatorio = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_depth=5, min_samples_split=20,
                                        class_weight='balanced', random_state=42))
])
bosque_aleatorio.fit(X_train, Y_train)
Y_pred = regresion_log.predict(X_test)

#Impresion de resultados
print("=============Resultados bosque aleatorio ==================")
print(f"Exactitud: {accuracy_score(Y_test,Y_pred): .4f}")
print()
print("Reporte de Clasificacion")
print(classification_report(Y_test,Y_pred))
print()






