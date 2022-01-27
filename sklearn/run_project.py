import mlflow
project_uri = "/home/kaios/mlflow_demo/sklearn/"

# Log to the TrackingServer
mlflow.set_tracking_uri('http://mlflow.kaios.ai:5050')

# Set experiment name (if does not exist, a new experiment is created with this name)
mlflow.set_experiment(experiment_name = "mlflow_sklearn_project")

# Run MLflow project and create a reproducible conda environment
mlflow.run(project_uri, parameters={"alpha": 0.3, 
				     "l1_ratio": 0.6})
mlflow.run(project_uri, parameters={"alpha": 0.5, 
				     "l1_ratio": 0.9})		
mlflow.run(project_uri, parameters={"alpha": 0.6, 
				     "l1_ratio": 0.5})
mlflow.run(project_uri, parameters={"alpha": 0.8, 
				     "l1_ratio": 0.2})
