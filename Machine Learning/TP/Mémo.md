# Aide-mémoire Scikit-learn pour le ML

## 1. Importer les bibliothèques nécessaires
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
```

## 2. Charger et explorer le dataset
```python
# Exemple avec un fichier CSV
data = pd.read_csv('data.csv')
print(data.head())
```

## 3. Préparation des données
- **Sélection des caractéristiques et cible :**
```python
X = data[['feature1', 'feature2']]  # Caractéristiques
y = data['target']                   # Cible
```

- **Diviser le dataset :**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 4. Normalisation (facultatif mais recommandé)
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 5. Régression linéaire
```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
```

## 6. Régression polynomiale
```python
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

model_poly = LinearRegression()
model_poly.fit(X_poly, y_train)

X_poly_test = poly.transform(X_test)
predictions_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, predictions_poly)
print("MSE (polynomial):", mse_poly)
```

## 7. Classification linéaire
```python
class_model = LogisticRegression()
class_model.fit(X_train, y_train)
class_predictions = class_model.predict(X_test)
accuracy = accuracy_score(y_test, class_predictions)
print("Accuracy:", accuracy)
```

## 8. Classification avec SVM (Support Vector Machine)
```python
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", accuracy_svm)
```

## 9. Recherche d’hyperparamètres (Grid Search)
```python
param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
```

## 10. Validation croisée (facultatif)
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(class_model, X, y, cv=5)
print("Cross-validation scores:", scores)
```

## 11. Sauvegarder le modèle (facultatif)
```python
import joblib
joblib.dump(model, 'model.pkl')
```

## Remarques finales
- **Visualisation des résultats** : Utilisez `matplotlib` ou `seaborn` pour visualiser les résultats et la performance.
- **Exploration des données** : Toujours explorer et nettoyer vos données avant d’entraîner vos modèles.