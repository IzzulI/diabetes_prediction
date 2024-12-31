if __name__ == "__main__":
    from model_tuning_functions import lightgbm_pipeline, bayesian_optimization_tuning
else:
    from scripts.model_tuning_functions import lightgbm_pipeline, bayesian_optimization_tuning

import pandas as pd
from skopt.space import Integer, Real, Categorical
import pickle
from scipy.stats import uniform, randint


def load_data(x_train_path: str, y_train_path: str):
    """Function to load training data"""
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train, y_train

def get_lightgbm_bayesian_param_space():
    """Function to define bayesian optimization parameter space for lightgbm"""
    return {    
        'classifier__max_depth': Integer(4, 10),
        'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'classifier__n_estimators': Integer(100, 1000),
        'classifier__num_leaves': Integer(20, 150),
        'classifier__min_child_samples': Integer(10, 100),
        'classifier__subsample': Real(0.5, 1),
        'classifier__colsample_bytree': Real(0.5, 1),
        'classifier__reg_alpha': Real(0, 1),
        'classifier__reg_lambda': Real(0, 1),
        'classifier__scale_pos_weight': Categorical([1, 5, 10])
    }

def perform_bayesian_optimization(lightgbm_pipeline, param_space, X_train, y_train):
    """Function to perform bayesian optimization for model tuning"""
    return bayesian_optimization_tuning(lightgbm_pipeline, param_space, X_train, y_train)

def save_model(model, file_path: str):
    """Function to save model to a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def get_lightgbm_bayesian(x_train_path: str, y_train_path: str, model_save_path: str):
    """Master function to run the entire process"""    
    # Load the training data
    X_train, y_train = load_data(x_train_path, y_train_path)

    # Define the parameter space for bayesian optimization
    lightgbm_bayesian_param_space = get_lightgbm_bayesian_param_space()

    # Perform bayesian optimization tuning
    lightgbm_bayesian = perform_bayesian_optimization(lightgbm_pipeline, lightgbm_bayesian_param_space , X_train, y_train)

    # Save the model
    save_model(lightgbm_bayesian, model_save_path)

    # Print final model
    print(lightgbm_bayesian)

if __name__ == "__main__":
    # Call the master function with the paths to the data and the desired model save path
    get_lightgbm_bayesian('data/X_train.csv', 'data/y_train.csv', 'models/lightgbm_bayes.pkl')

