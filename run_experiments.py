from src.data_loader import load_wine_quality, split_data, scale_features
from src.train import train_and_evaluate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def main():
    # ---------- Carregar e preparar dados ----------
    df = load_wine_quality()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    experiment_name = "wine-quality"

    # ==================================================
    # EXPERIMENTO 1 — Ridge Regression (modelo linear)
    # Pergunta: Um modelo linear simples consegue prever
    # a qualidade do vinho com bom desempenho?
    # ==================================================
    print("\n" + "=" * 60)
    print("EXPERIMENTO 1: Ridge Regression (modelo linear)")
    print("=" * 60)

    params_ridge = {"alpha": 1.0}
    model_ridge = Ridge(**params_ridge)

    metrics_ridge = train_and_evaluate(
        model=model_ridge,
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
        params=params_ridge,
        experiment_name=experiment_name,
        run_name="ridge-baseline",
    )
    print(f"RMSE teste: {metrics_ridge['test_rmse']:.4f} | R² teste: {metrics_ridge['test_r2']:.4f}")

    # ==================================================
    # EXPERIMENTO 2 — RandomForest (modelo não-linear)
    # Pergunta: Um modelo não-linear consegue capturar
    # melhor as relações entre as propriedades químicas
    # e a qualidade do vinho?
    # ==================================================
    print("\n" + "=" * 60)
    print("EXPERIMENTO 2: RandomForest (modelo não-linear)")
    print("=" * 60)

    params_rf = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    model_rf = RandomForestRegressor(**params_rf)

    metrics_rf = train_and_evaluate(
        model=model_rf,
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
        params=params_rf,
        experiment_name=experiment_name,
        run_name="random-forest",
    )
    print(f"RMSE teste: {metrics_rf['test_rmse']:.4f} | R² teste: {metrics_rf['test_r2']:.4f}")

    # ---------- Comparação final ----------
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DOS EXPERIMENTOS")
    print("=" * 60)
    print(f"{'Modelo':<20} {'RMSE teste':<12} {'MAE teste':<12} {'R² teste':<10}")
    print("-" * 60)
    print(f"{'Ridge':<20} {metrics_ridge['test_rmse']:<12.4f} {metrics_ridge['test_mae']:<12.4f} {metrics_ridge['test_r2']:<10.4f}")
    print(f"{'RandomForest':<20} {metrics_rf['test_rmse']:<12.4f} {metrics_rf['test_mae']:<12.4f} {metrics_rf['test_r2']:<10.4f}")


if __name__ == "__main__":
    main()