
"""
This code is a comprehensive machine learning pipeline designed to explore and evaluate multiple classifiers on a lung cancer dataset. 
The process is structured into several key stages:

	1.	Data Preprocessing: The raw dataset is prepared by encoding categorical variables and ensuring 
         that any integer columns are converted to floats, which helps handle potential missing values. 
         This step is crucial for ensuring that the models receive clean, well-structured data.
	2.	Model Training and Evaluation: 
        Seven different classifiers are tested, including Logistic Regression, Decision Tree, Random Forest, 
        SVM, AdaBoost, Naive Bayes, and XGBoost. Each model is trained and evaluated on multiple train-test splits, 
        using important metrics like accuracy, precision, and recall to gauge performance.
	3.	Experiment Tracking with MLflow: MLflow is integrated to meticulously track the entire experiment. 
        For each model, MLflow logs the mean and standard deviation of the evaluation metrics. It also saves the best-trained model, 
        capturing the model’s signature and an example input, which are essential for ensuring that the model 
        can be correctly used in production. For the XGBoost model, the code explicitly 
        saves it in JSON format to avoid warnings and ensure compatibility.

This pipeline not only automates the process of testing different models but also leverages MLflow to provide a robust and organized framework 
for experiment tracking, making it easier to compare results and manage models for future deployment.
"""

import warnings
warnings.filterwarnings("ignore", message="Distutils was imported before Setuptools")
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")

from typing import Tuple, List, Dict, Any
import pandas as pd
import sklearn as skl
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# Explicitly import required sklearn submodules
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.svm
import sklearn.naive_bayes
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.metrics
import sklearn.model_selection

# Constants
TEST_SIZE = 0.2
TRAIN_SPLITS = 3
DATASET_URL = "/data/lung_cancer_dataset/survey-lung-cancer.csv"

CLASSIFIERS_CONFIG = {
    "Logistic Regression": skl.linear_model.LogisticRegression(),
    "Decision Tree": skl.tree.DecisionTreeClassifier(),
    "Random Forest": skl.ensemble.RandomForestClassifier(),
    "SVM": skl.svm.SVC(),
    "AdaBoost": skl.ensemble.AdaBoostClassifier(algorithm="SAMME"),
    "Naive Bayes": skl.naive_bayes.GaussianNB(),
    "XGBoost": xgb.XGBClassifier()
}


metrics_list = ['accuracy', 'precision', 'recall']

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the dataset, split into features and target."""
    df['GENDER'] = skl.preprocessing.LabelEncoder().fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = skl.preprocessing.LabelEncoder().fit_transform(df['LUNG_CANCER'])
    
    # Convert integer columns to float to handle potential missing values
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype(float)
    
    X = df.drop(columns=['LUNG_CANCER'])
    y = df['LUNG_CANCER']
    return X, y

def evaluate_model(pipeline: skl.pipeline.Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """Train and evaluate the model, returning evaluation metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        'accuracy': skl.metrics.accuracy_score(y_test, y_pred),
        'precision': skl.metrics.precision_score(y_test, y_pred),
        'recall': skl.metrics.recall_score(y_test, y_pred)
    }

def run_experiment(classifier_name: str, classifier: Any, X: pd.DataFrame, y: pd.Series, train_test_splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]):
    """Run MLflow experiment for a given classifier."""
    with mlflow.start_run(run_name=classifier_name.lower().replace(" ","_")):
        pipeline = skl.pipeline.Pipeline([('scaler', skl.preprocessing.StandardScaler()), ('classifier', classifier)])
        metrics = {metric: [] for metric in metrics_list}
        
        # Evaluate on each split
        for X_train, X_test, y_train, y_test in train_test_splits:
            result = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
            for key, value in result.items():
                metrics[key].append(value)
        
        # MLFlow: Log mean and std of the metrics
        for key, values in metrics.items():
            mlflow.log_metric(f'{key}_mean', pd.Series(values).mean())
            mlflow.log_metric(f'{key}_std', pd.Series(values).std())
        
        # Train on the full dataset
        pipeline.fit(X, y)
        
        # Log the model using the appropriate MLflow function
        mlflow.sklearn.log_model(pipeline, "model")

def main() -> None:
    df = pd.read_csv(DATASET_URL)
    X, y = preprocess_data(df)
    train_test_splits = [skl.model_selection.train_test_split(X, y, test_size=TEST_SIZE, random_state=i) for i in range(TRAIN_SPLITS)]
    
    for name, classifier in CLASSIFIERS_CONFIG.items():
        run_experiment(name, classifier, X, y, train_test_splits)

if __name__ == "__main__":
    main()