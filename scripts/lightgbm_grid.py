if __name__ == "__main__":
    from model_tuning_functions import lightgbm_pipeline, grid_search_tuning
else:
    from scripts.model_tuning_functions import lightgbm_pipeline, grid_search_tuning

import pandas as pd
from skopt.space import Integer, Real, Categorical
import pickle
from scipy.stats import uniform, randint


def load_data(x_train_path: str, y_train_path: str):
    """Function to load training data"""
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train, y_train

def get_lightgbm_grid_param_space():
    """Function to define grid search parameter space for lightgbm"""
    return {
        'classifier__max_depth': [4, 5, 6],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__n_estimators': [200, 500],
        'classifier__num_leaves': [15, 31, 50, 75, 100], # Adjust num_leaves parameter
        'classifier__class_weight': ['balanced', None]
    }

def perform_grid_search(lightgbm_pipeline, param_space, X_train, y_train):
    """Function to perform grid search for model tuning"""
    return grid_search_tuning(lightgbm_pipeline, param_space, X_train, y_train)

def save_model(model, file_path: str):
    """Function to save model to a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def get_lightgbm_grid(x_train_path: str, y_train_path: str, model_save_path: str):
    """Master function to run the entire process"""    
    # Load the training data
    X_train, y_train = load_data(x_train_path, y_train_path)

    # Define the parameter space for grid search
    lightgbm_grid_param_space = get_lightgbm_grid_param_space()

    # Perform grid search tuning
    lightgbm_grid = perform_grid_search(lightgbm_pipeline, lightgbm_grid_param_space, X_train, y_train)

    # Save the model
    save_model(lightgbm_grid, model_save_path)

    # Print final model
    print(lightgbm_grid)

if __name__ == "__main__":
    # Call the master function with the paths to the data and the desired model save path
    get_lightgbm_grid('data/X_train.csv', 'data/y_train.csv', 'models/lightgbm_grid.pkl')