import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, confusion_matrix, precision_recall_curve
from typing import Dict, Any

# Constants (file paths, model names, etc.)
MODEL_FILES = {
    'CatBoost Random Search': 'models/catboost_random.pkl',
    'XGBoost Random Search': 'models/xgboost_random.pkl',
    'LightGBM Random Search': 'models/lightgbm_random.pkl',
    'CatBoost Grid Search': 'models/catboost_grid.pkl',
    'XGBoost Grid Search': 'models/xgboost_grid.pkl',
    'LightGBM Grid Search': 'models/lightgbm_grid.pkl',
    'CatBoost Bayesian': 'models/catboost_bayes.pkl',
    'XGBoost Bayesian': 'models/xgboost_bayes.pkl',
    'LightGBM Bayesian': 'models/lightgbm_bayes.pkl'
}

# Function to load models from pickle files
def load_model(file_path: str) -> Any:
    """Load a model from a pickle file."""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to load all models
def load_all_models(model_files: Dict[str, str]) -> Dict[str, Any]:
    """Load all models from the provided dictionary of model files."""
    models = {}
    for name, file_path in model_files.items():
        models[name] = load_model(file_path)
    return models

# Function to evaluate a single model
def evaluate_model(model, X_test, y_test):
    """Evaluate a model and calculate various metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculating the confusion matrix to derive specificity
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    # AUC-PR calculation
    precision_pr, recall_pr, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_pr = auc(recall_pr, precision_pr)

    return accuracy, precision, recall, f1, auc_pr, specificity

# Function to evaluate all models
def evaluate_all_models(models: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and collect their results."""
    results = {}
    for name, model in models.items():
        accuracy, precision, recall, f1, auc_pr, specificity = evaluate_model(model, X_test, y_test)
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1 Score': f1,
            'AUC PR': auc_pr
        }
    return pd.DataFrame(results).T

# Function to save the results to CSV
def save_results_to_csv(results_df: pd.DataFrame, output_path: str):
    """Save the evaluation results to a CSV file."""
    results_df.to_csv(output_path)
    print(f"Results saved to {output_path}")

# The main function to orchestrate the process, can be called from another script
def get_evaluation(X_test_path='data/X_test.csv', y_test_path='data/y_test.csv', output_path='data/test_results.csv'):
    # Load test data
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Load all models
    models = load_all_models(MODEL_FILES)

    # Evaluate all models
    results_df = evaluate_all_models(models, X_test, y_test)

    # Save results to CSV
    save_results_to_csv(results_df, output_path)

    # Print the results
    print(results_df)

# Run the main function
if __name__ == "__main__":
    get_evaluation()
