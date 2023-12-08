import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split


from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')



# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Ajustar el modelo al conjunto de entrenamiento
rf_model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Calcular Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calcular F1-Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1:.4f}")

# Ahora, para optimizar los parámetros, puedes usar GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados por la búsqueda en cuadrícula
print("Mejores Parámetros:")
print(grid_search.best_params_)

# Obtener el mejor modelo
best_rf_model = grid_search.best_estimator_

# Predecir en el conjunto de prueba con el mejor modelo
y_pred_best = best_rf_model.predict(X_test)

# Calcular Accuracy y F1-Score con el mejor modelo
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print(f"Accuracy con el mejor modelo: {accuracy_best:.4f}")
print(f"F1-Score con el mejor modelo: {f1_best:.4f}")
