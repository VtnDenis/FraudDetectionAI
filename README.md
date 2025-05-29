# Fraud Detection AI

## Description
Ce projet implémente un pipeline de détection de fraude basé sur l'apprentissage automatique. Il utilise des techniques de prétraitement des données, de ré-échantillonnage pour gérer les données déséquilibrées, et des modèles de classification pour prédire les transactions frauduleuses.

## Structure du projet

```
FraudDetectionAI/
├── main.py
├── README.md
├── requirements.txt
├── dataset/
│   └── creditcard.csv
├── models/
│   ├── mlpclassifier_model.joblib
│   ├── randomforest_model.joblib
│   └── xgboost_model.joblib
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── model_training.py
    ├── utils.py
```

- **main.py** : Point d'entrée du pipeline.
- **dataset/** : Contient le fichier de données `creditcard.csv`.
- **models/** : Dossier où les modèles entraînés sont sauvegardés.
- **src/** : Contient les modules pour le prétraitement des données, l'entraînement des modèles et les utilitaires.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/VtnDenis/FraudDetectionAI
   cd FraudDetectionAI
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Placez le fichier `creditcard.csv` dans le dossier `dataset/`.
2. Lancez le pipeline :
   ```bash
   python main.py
   ```
3. Les modèles entraînés seront sauvegardés dans le dossier `models/`.

## Fonctionnalités

- **Prétraitement des données** : Normalisation des montants et gestion des données déséquilibrées avec SMOTE.
- **Entraînement des modèles** : Optimisation des hyperparamètres avec `RandomizedSearchCV`.
- **Évaluation des modèles** : Calcul des métriques telles que la matrice de confusion, le rapport de classification et le score ROC AUC.

## Modèles supportés

- XGBoost
- Random Forest
- MLPClassifier

## Performances des modèles

### XGBoost

- **Confusion Matrix** :
  ```
  [[56849    15]
   [   16    82]]
  ```
- **Classification Report** :
  - Classe 0 : Précision = 0.9997, Rappel = 0.9997, F1-Score = 0.9997
  - Classe 1 : Précision = 0.8454, Rappel = 0.8367, F1-Score = 0.8410
  - Moyenne Macro : Précision = 0.9225, Rappel = 0.9182, F1-Score = 0.9204
  - Moyenne Pondérée : Précision = 0.9995, Rappel = 0.9995, F1-Score = 0.9995
- **ROC AUC Score** : 0.9881

### RandomForest

- **Confusion Matrix** :
  ```
  [[56850    14]
   [   14    84]]
  ```
- **Classification Report** :
  - Classe 0 : Précision = 0.9998, Rappel = 0.9998, F1-Score = 0.9998
  - Classe 1 : Précision = 0.8571, Rappel = 0.8571, F1-Score = 0.8571
  - Moyenne Macro : Précision = 0.9284, Rappel = 0.9284, F1-Score = 0.9284
  - Moyenne Pondérée : Précision = 0.9995, Rappel = 0.9995, F1-Score = 0.9995
- **ROC AUC Score** : 0.9897

### MLPClassifier

- **Confusion Matrix** :
  ```
  [[56678   186]
   [   13    85]]
  ```
- **Classification Report** :
  - Classe 0 : Précision = 0.9998, Rappel = 0.9967, F1-Score = 0.9982
  - Classe 1 : Précision = 0.3137, Rappel = 0.8673, F1-Score = 0.4607
  - Moyenne Macro : Précision = 0.6567, Rappel = 0.9320, F1-Score = 0.7295
  - Moyenne Pondérée : Précision = 0.9986, Rappel = 0.9965, F1-Score = 0.9973
- **ROC AUC Score** : 0.9742

## Dépendances

- Python 3.11+
- Pandas
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Joblib

## Auteur
Ce projet a été développé par Vautrin Denis.

