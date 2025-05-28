import joblib
import os

def save_model(model, filename, model_dir='models/'):
    """Sauvegarde un modèle entraîné."""
    os.makedirs(model_dir, exist_ok=True) # Créer le dossier s'il n'existe pas
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé sous : {filepath}")

def load_model(filename, model_dir='models/'):
    """Charge un modèle sauvegardé."""
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier du modèle n'existe pas : {filepath}")
    model = joblib.load(filepath)
    print(f"Modèle chargé depuis : {filepath}")
    return model

def print_metrics(model_name, metrics):
    """Affiche les métriques d'évaluation de manière lisible."""
    print(f"\n--- Résultats d'évaluation pour {model_name} ---")
    print("🔍 Confusion Matrix :")
    print(metrics['confusion_matrix'])
    print("\n📋 Classification Report :")
    # Convertir le dict du classification_report en string pour un affichage propre
    report_str = ""
    for label, data in metrics['classification_report'].items():
        if isinstance(data, dict):
            report_str += f"{label: <10} "
            for metric, value in data.items():
                report_str += f"{metric}: {value:.4f} "
            report_str += "\n"
        else:
            report_str += f"{label}: {data:.4f}\n"
    print(report_str)
    print(f"\n🎯 ROC AUC Score : {metrics['roc_auc_score']:.4f}")