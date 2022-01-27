# Import libraries
import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

# Log to the TrackingServer
mlflow.set_tracking_uri('http://mlflow.kaios.ai:5050')

# Set experiment name (if does not exist, a new experiment is created with this name)
mlflow.set_experiment(experiment_name = "mlflow_sklearn_project")

# function that calculates evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default = 1, type = float, help = "Constant that multiplies the penalty terms")
    parser.add_argument("--l1_ratio", default = 0.5, type = float, help = "The ElasticNet mixing parameter")
    args = parser.parse_args()

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    
    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]


    # Start a new MLflow run, setting it as the active run under which metrics
    # and parameters will be logged. The return value can be used as a 
    #context manager within a "with" block.
    with mlflow.start_run(run_name = 'sklearn_project'):
        mlflow.set_tag("mlflow.user", "nolan")
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        
        # log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # log model as an artifact
        # The output of the Sklearn training snippet is shown below. 
        # All the logged data is visible here along with the output model 
        # stored in the artifacts store.
        mlflow.sklearn.log_model(lr, "sklearn_model")
