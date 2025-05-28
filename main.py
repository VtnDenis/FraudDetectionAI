import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split
import imblearn
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


df = pd.read_csv('./dataset/creditcard.csv')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

scaler = sk.StandardScaler()
scaler.fit(train_df[['Amount']])

train_df['Amount'] = scaler.transform(train_df[['Amount']])
val_df['Amount'] = scaler.transform(val_df[['Amount']])

x = train_df.drop('Class', axis=1)
y = train_df['Class']

print(df['Class'].value_counts(normalize=True)*100)

x_train_resampled, y_train_resampled = imblearn.over_sampling.SMOTE(random_state=42).fit_resample(x, y)

#print(pd.Series(y_train_resampled).value_counts())

#plt.xlim([0, 1000])
#df['V1'].plot(kind='hist', edgecolor='black', logy=True)

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=1  # car SMOTE a équilibré les classes
)

xgb_model.fit(x_train_resampled, y_train_resampled)

# Instancier votre modèle XGBClassifier (sans hyperparamètres ajustés)
xgb_model = XGBClassifier(random_state=42)

# Définir la grille d'hyperparamètres à tester
param_distributions = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.5, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 5]
}

# Instancier RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # Nombre d'itérations aléatoires (plus élevé=plus complet)
    scoring='roc_auc',  # Mesure d'évaluation
    cv=3,  # Validation croisée avec 3 plis
    verbose=2,  # Affichage progrès de recherche
    random_state=42,
    n_jobs=-1  # Utiliser tous les CPU disponibles
)

# Lancer la recherche sur vos données réséchantillonnées
random_search.fit(x_train_resampled, y_train_resampled)

# Afficher les meilleurs résultats
print("Meilleurs paramètres :", random_search.best_params_)
print("Meilleure performance (AUC) :", random_search.best_score_)

# Préparer les données de validation (features + target)
X_val = val_df.drop('Class', axis=1)   # Caractéristiques sans la colonne cible
y_val = val_df['Class']                # Cible réelle

# Extraire les meilleurs paramètres et entraîner un modèle final
best_params = random_search.best_params_
final_xgb_model = XGBClassifier(**best_params, random_state=42)

# Entraîner le modèle avec les données optimales
final_xgb_model.fit(x_train_resampled, y_train_resampled)

# Évaluer sur les données de validation
predictions_val = final_xgb_model.predict(X_val)
probabilities_val = final_xgb_model.predict_proba(X_val)[:, 1]

# Afficher les métriques
print("🔍 Confusion Matrix :")
print(confusion_matrix(y_val, predictions_val))

print("\n📋 Classification Report :")
print(classification_report(y_val, predictions_val, digits=4))

print(f"\n🎯 ROC AUC Score : {roc_auc_score(y_val, probabilities_val):.4f}")