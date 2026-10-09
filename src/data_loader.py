import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_wine_quality():
    """Carrega o dataset Wine Quality (tinto)."""
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )
    try:
        df = pd.read_csv(url, sep=";")
        print(f"Dataset carregado da URL: {df.shape[0]} linhas, {df.shape[1]} colunas")
    except Exception as e:
        print(f"Erro ao carregar da URL ({e}). Tentando arquivo local...")
        df = pd.read_csv("data/winequality-red.csv", sep=";")
        print(f"Dataset carregado do arquivo local: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """Divide os dados em treino, validação e teste."""
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Separa treino+val do teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Separa treino da validação
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Padroniza as features (média 0, desvio 1)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler