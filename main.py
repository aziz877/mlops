import argparse
import joblib
import mlflow
import mlflow.sklearn
import logging
import pandas as pd
from model import prepare_data, train_and_save_model, evaluate_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch

# Configurer la journalisation (logs) pour MLflow avec envoi vers Elasticsearch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])
if es.ping():
    logger.info("Connected to Elasticsearch")
else:
    logger.error("Could not connect to Elasticsearch")


def main(mode, train_path, test_path, model_filename):
    mlflow.set_experiment("mlflow_experiment")

    if mode == "prepare":
        X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
        if X_train is not None and y_train is not None:
            joblib.dump((X_train, y_train, X_test, y_test), "prepared_data.joblib")
            logger.info("Data preparation completed and saved.")
        else:
            logger.error("Error: Data preparation failed.")

    elif mode == "train":
        with mlflow.start_run() as run:
            data = joblib.load("prepared_data.joblib")
            X_train, y_train, X_test, y_test = data

            trained_model = train_and_save_model(X_train, y_train, X_test, y_test, model_filename)
            mlflow.log_param("model_filename", model_filename)

            input_example = pd.DataFrame(X_train[:1])
            signature = mlflow.models.infer_signature(X_train, trained_model.predict(X_train[:1]))
            mlflow.sklearn.log_model(trained_model, "RandomForest_Model", signature=signature, input_example=input_example)
            mlflow.register_model(f"runs:/{run.info.run_id}/RandomForest_Model", "RandomForest_Model")
            logger.info("Training completed and model registered in MLflow.")

    elif mode == "evaluate":
        with mlflow.start_run() as run:
            trained_model = joblib.load(model_filename)
            data = joblib.load("prepared_data.joblib")
            X_train, y_train, X_test, y_test = data

            if X_test is not None and y_test is not None:
                metrics = evaluate_model(trained_model, X_test, y_test)
                mlflow.log_metrics(metrics)
                es.index(index="mlflow-metrics", document=metrics)

                logger.info(f"Model evaluated. Metrics: {metrics}")

                y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                mlflow.log_metric("roc_auc", roc_auc)

                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.savefig("roc_curve.png")
                mlflow.log_artifact("roc_curve.png")
                plt.close()

                logger.info("ROC curve generated and logged in MLflow.")

            else:
                logger.warning("No test labels provided. Skipping evaluation.")

    elif mode == "full":
        with mlflow.start_run() as run:
            X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
            if X_train is None or y_train is None:
                logger.error("Error: Data preparation failed.")
                return

            trained_model = train_and_save_model(X_train, y_train, X_test, y_test, model_filename)
            mlflow.log_param("model_filename", model_filename)

            input_example = pd.DataFrame(X_train[:1])
            signature = mlflow.models.infer_signature(X_train, trained_model.predict(X_train[:1]))
            mlflow.sklearn.log_model(trained_model, "RandomForest_Model", signature=signature, input_example=input_example)
            mlflow.register_model(f"runs:/{run.info.run_id}/RandomForest_Model", "RandomForest_Model")
            logger.info("Model trained, registered, and ready in MLflow.")
            metrics = evaluate_model(trained_model, X_test, y_test)
            mlflow.log_metrics(metrics)
            es.index(index="mlflow-metrics", document=metrics)
            logger.info(f"Model evaluated. Metrics: {metrics}")
            y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            mlflow.log_metric("roc_auc", roc_auc)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig("roc_curve.png")
            mlflow.log_artifact("roc_curve.png")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for data processing, training, and evaluation with MLflow.")
    parser.add_argument("mode", choices=["prepare", "train", "evaluate", "full"], help="Mode to run the script in")
    parser.add_argument("train_path", type=str, help="Path to the training data CSV file")
    parser.add_argument("test_path", type=str, help="Path to the test data CSV file")
    parser.add_argument("--model_filename", type=str, default="modelRF.joblib", help="Filename for saving the trained model")

    args = parser.parse_args()
    main(args.mode, args.train_path, args.test_path, args.model_filename)

