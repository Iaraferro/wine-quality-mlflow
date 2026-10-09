import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_evaluate(model, X_train, y_train, X_val, y_val,
                       X_test, y_test, params, experiment_name, run_name=None):
    """Treina, avalia e registra tudo no MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log dos parâmetros
        mlflow.log_params(params)

        # Treino
        model.fit(X_train, y_train)

        # Previsões
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # Métricas
        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
            "val_mae": mean_absolute_error(y_val, y_pred_val),
            "val_r2": r2_score(y_val, y_pred_val),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
             model, 
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        return metrics