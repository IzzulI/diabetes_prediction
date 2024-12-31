if __name__ == "__main__":
    from model_tuning_functions import catboost_pipeline, bayesian_optimization_tuning
else:
    from scripts.model_tuning_functions import catboost_pipeline, bayesian_optimization_tuning

import pandas as pd
from skopt.space import Integer, Real, Categorical
import pickle
from scipy.stats import uniform, randint


def load_data(x_train_path: str, y_train_path: str):
    """Function to load training data"""
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train, y_train

def get_catboost_bayesian_param_space():
    """Function to define bayesian optimization parameter space for catboost"""
    return {    
        'classifier__iterations': (100, 1000),
        'classifier__learning_rate': (0.01, 0.3),
        'classifier__depth': (4, 10),
        'classifier__l2_leaf_reg': (1, 10),
        'classifier__border_count': (1, 255),
        'classifier__scale_pos_weight': Categorical([1, 5, 10])
    }

def perform_bayesian_optimization(catboost_pipeline, param_space, X_train, y_train):
    """Function to perform bayesian optimization for model tuning"""
    return bayesian_optimization_tuning(catboost_pipeline, param_space, X_train, y_train)

def save_model(model, file_path: str):
    """Function to save model to a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def get_catboost_bayesian(x_train_path: str, y_train_path: str, model_save_path: str):
    """Master function to run the entire process"""    
    # Load the training data
    X_train, y_train = load_data(x_train_path, y_train_path)

    # Define the parameter space for bayesian optimization
    catboost_bayesian_param_space = get_catboost_bayesian_param_space()

    # Perform bayesian optimization tuning
    catboost_bayesian = perform_bayesian_optimization(catboost_pipeline, catboost_bayesian_param_space , X_train, y_train)

    # Save the model
    save_model(catboost_bayesian, model_save_path)

    # Print final model
    print(catboost_bayesian)

if __name__ == "__main__":
    # Call the master function with the paths to the data and the desired model save path
    get_catboost_bayesian('data/X_train.csv', 'data/y_train.csv', 'models/catboost_bayes.pkl')

