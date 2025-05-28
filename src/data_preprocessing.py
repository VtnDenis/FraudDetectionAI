import pandas as pd
import imblearn
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split

def load_data(file_path='dataset/creditcard.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    scaler = sk.StandardScaler()
    scaler.fit(train_df[['Amount']])

    train_df['Amount'] = scaler.transform(train_df[['Amount']])
    val_df['Amount'] = scaler.transform(val_df[['Amount']])

    # Préparer les données de validation (features + target)
    x_val = val_df.drop('Class', axis=1)  # Caractéristiques sans la colonne cible
    y_val = val_df['Class']  # Cible réelle

    x = train_df.drop('Class', axis=1)
    y = train_df['Class']

    print(df['Class'].value_counts(normalize=True) * 100)

    x_train_resampled, y_train_resampled = imblearn.over_sampling.SMOTE(random_state=42).fit_resample(x, y)

    return x_train_resampled, y_train_resampled, x_val, y_val