import mlflow
import mlflow.sklearn
import os
import sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# Force MLflow to use pip instead of conda
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
os.environ["MLFLOW_CONDA_CREATE"] = "False"
os.environ["MLFLOW_DISABLE_ENV_CREATION"] = "True"
os.environ["MLFLOW_PYTHON_ENV_MANAGER"] = "pip"

# Override the conda environment creation function
mlflow.utils.environment._mlflow_conda_env = lambda *args, **kwargs: None

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Load the training dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create a new MLflow run
with mlflow.start_run() as run:
    # Train the model
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)
    
    # Create a sample input for signature inference
    input_example = X_train[:5]
    
    # Infer the model signature
    signature = infer_signature(X_train, rf.predict(X_train))
    
    # Create a custom environment with pip requirements
    pip_requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"mlflow=={mlflow.__version__}",
        "numpy>=1.14.3"
    ]
    
    # Log the model with explicit pip requirements and signature
    mlflow.sklearn.log_model(
        rf, 
        "random_forest_model",
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        registered_model_name="RandomForestRegressor"
    )
    
    # Log metrics manually
    mlflow.log_metric("test_score", rf.score(X_test, y_test))
