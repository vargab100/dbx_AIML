# Databricks notebook source
import mlflow
mlflow.set_registry_uri("databricks-uc")

CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
mlflow.autolog()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

with mlflow.start_run(run_name='iris_gradient_boost_model') as run:
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    # Additional metrics can be logged manually if needed
    # mlflow.log_metric("test_accuracy", model.score(X_test, y_test))
best_run = mlflow.search_runs(
    order_by=['metrics.test_accuracy DESC', 'start_time DESC'],
    max_results=1
).iloc[0]

model_uri = f"runs:/{best_run.run_id}/model"
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.iris_model"

mlflow.register_model(model_uri, model_name)