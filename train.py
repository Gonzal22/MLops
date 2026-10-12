import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from data_processing import load_and_preprocess_data
from evaluation import get_metrics, save_confusion_matrix

def main():
    # REQUERIMIENTO 1: Parametrización con valores por defecto
    parser = argparse.ArgumentParser(description="Entrenamiento de Regresión Logística")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--max_iter", type=int, default=1000, help="Iteraciones máximas")
    parser.add_argument("--c_param", type=float, default=1.0, help="Inverso de la fuerza de regularización ")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Algoritmo para la optimización")
    args = parser.parse_args()

    # Carga y preparación
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "heart.csv", 
        test_size=0.2, 
        random_state=args.random_state
    )

    # Configuración de MLflow local
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Heart_Disease_Prediction_Modular")

    # REQUERIMIENTO 2: Instrumentación con MLflow
    with mlflow.start_run(run_name=f"LR_C={args.c_param}_solver={args.solver}"):
        
        # a) Registrar hiperparámetros de la corrida
        mlflow.log_params({
            "random_state": args.random_state,
            "max_iter": args.max_iter,
            "C": args.c_param,
            "solver": args.solver
        })

        # Entrenamiento
        model = LogisticRegression(
            C=args.c_param, 
            solver=args.solver, 
            max_iter=args.max_iter, 
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # b) Registrar métricas
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # c) Artefacto: Matriz de confusión
        cm_filename = "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_filename)
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename)

        # d) Modelo entrenado
        mlflow.sklearn.log_model(model, "logistic_model")
        
        print(f"Ejecución exitosa: C={args.c_param}, Solver={args.solver}. Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    main()