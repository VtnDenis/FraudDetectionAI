from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

def train_model(model_name, x_train, y_train, x_val, y_val, random_state=42):
    estimator = None
    param_distributions = {}
    n_iter = 0

    if model_name == 'XGBoost':
        estimator = XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=random_state)
        param_distributions = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 2, 5]
        }
        n_iter = 50

    elif model_name == 'RandomForest':
        estimator = RandomForestClassifier(random_state=random_state)
        param_distributions = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        n_iter = 20

    elif model_name == 'SVC':
        estimator = SVC(random_state=random_state, probability=True)
        param_distributions = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }

        n_iter = 5

    elif model_name == 'MLPClassifier':
        estimator = MLPClassifier(random_state=random_state, max_iter=500)
        param_distributions = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
        }
        n_iter = 10

    # utilisation de RandomizedSearchCV pour optimiser les hyperparamètres
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("Démarrage de la recherche aléatoire pour les hyperparamètres du modèle:", model_name)

    # Lancer la recherche sur les données ré-échantillonnées
    random_search.fit(x_train, y_train)

    print(f"Meilleurs paramètres pour {model_name}:", random_search.best_params_)
    print(f"Meilleure performance (AUC) pour {model_name} (CV):", random_search.best_score_)

    # Entraîner le modèle final avec les meilleurs paramètres
    final_model = random_search.best_estimator_

    # Évaluation sur les données de validation
    predictions_val = final_model.predict(x_val)
    probabilities_val = final_model.predict_proba(x_val)[:, 1]

    # Collecte des métriques
    metrics = {
        'confusion_matrix': confusion_matrix(y_val, predictions_val),
        'classification_report': classification_report(y_val, predictions_val, digits=4, output_dict=True),
        'roc_auc_score': roc_auc_score(y_val, probabilities_val)
    }

    return final_model, metrics
