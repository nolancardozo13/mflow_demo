# Import libraries    
import os
from random import random, randint
import mlflow

# Log to the TrackingServer
mlflow.set_tracking_uri('http://mlflow.kaios.ai:5050')

# Set experiment name (if does not exist, a new experiment is created with this name)
mlflow.set_experiment("mlflow_sample_code")

if __name__ == "__main__":
    # Start mlflow run  
    mlflow.start_run(run_name = 'sample_code')
    
    # By default the user is the same as your windows, linux, mac user account. 
    # However, if your user account has a random name, use the following command to set it
    # mlflow.set_tag("mlflow.user", "nolan")
    
    # set a tag for your experiment
    mlflow.set_tag("sample", "code")
    
    # Log a parameter (key-value pair)
    mlflow.log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    mlflow.log_metric("foo", random())
    mlflow.log_metric("foo", random() + 1)
    mlflow.log_metric("foo", random() + 2)
    mlflow.log_metric("foo", random() + 3)
    mlflow.log_metric("foo", random() + 4)
    
    # Log another metric and update it  
    mlflow.log_metric("oof", random())
    mlflow.log_metric("oof", random() - 1)
    mlflow.log_metric("oof", random() - 2)
    mlflow.log_metric("oof", random() - 3)
    mlflow.log_metric("oof", random() - 4)
    
    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    mlflow.log_artifacts("outputs")
    mlflow.end_run()



