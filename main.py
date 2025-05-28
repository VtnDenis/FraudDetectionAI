from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.utils import save_model, print_metrics, load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def main():
    print("Démarrage du pipeline de détection de fraude...\n")

    # --- Phase 1: Chargement et prétraitement des données ---
    df = load_data()
    x_train_resampled, y_train_resampled, x_val, y_val = preprocess_data(df)

    # --- Phase 2: Entraînement et évaluation des modèles ---

    # Liste des modèles à entraîner. Décommentez/commentez pour choisir.
    models_to_train = [
        'XGBoost',
        'RandomForest',
        # 'SVC',         # SVC est très lent, décommentez avec prudence ou sur un échantillon réduit
         'MLPClassifier'
    ]

    trained_models = {}

    for model_name in models_to_train:
        # Tenter d'abord de charger le modèle s'il existe déjà
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        try:
            model = load_model(model_filename)
            print(f"Le modèle '{model_name}' a été chargé depuis le disque. Skipping training.")
            # Si chargé, on peut (optionnellement) refaire l'évaluation sur X_val
            # pour avoir les métriques affichées
            predictions_val = model.predict(x_val)
            probabilities_val = model.predict_proba(x_val)[:, 1]
            metrics = {
                'confusion_matrix': confusion_matrix(y_val, predictions_val),
                'classification_report': classification_report(y_val, predictions_val, digits=4, output_dict=True),
                'roc_auc_score': roc_auc_score(y_val, probabilities_val)
            }
            print_metrics(model_name, metrics)

        except FileNotFoundError:
            # Si le modèle n'existe pas, l'entraîner
            model, metrics = train_model(model_name, x_train_resampled, y_train_resampled, x_val, y_val)
            save_model(model, model_filename)
            print_metrics(model_name, metrics)
        trained_models[model_name] = model

    print("\nPipeline de détection de fraude terminé.")

if __name__ == "__main__":
    main()