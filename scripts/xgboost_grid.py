if __name__ == "__main__":
    from model_tuning_functions import xgboost_pipeline, grid_search_tuning
else:
    from scripts.model_tuning_functions import xgboost_pipeline, grid_search_tuning

import pandas as pd
from skopt.space import Integer, Real, Categorical
import pickle
from scipy.stats import uniform, randint


def load_data(x_train_path: str, y_train_path: str):
    """Function to load training data"""
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train, y_train

def get_xgboost_grid_param_space():
    """Function to define grid search parameter space for xgboost"""
    return {
        'classifier__max_depth': [3, 6],  # Keeping both values for depth
        'classifier__learning_rate': [0.01, 0.1],  # Keeping both values for learning rate
        'classifier__n_estimators': [500],  # Reduced to one value
        'classifier__min_child_weight': [1, 5],  # Keeping both values for minimum child weight
        'classifier__gamma': [0],  # Reduced to one value
        'classifier__subsample': [0.75, 1.0],  # Keeping both values for subsample
        'classifier__colsample_bytree': [0.75],  # Reduced to one value
        'classifier__scale_pos_weight': [1, 5]  # Keeping both values for class imbalance
    }

def perform_grid_search(xgboost_pipeline, param_space, X_train, y_train):
    """Function to perform grid search for model tuning"""
    return grid_search_tuning(xgboost_pipeline, param_space, X_train, y_train)

def save_model(model, file_path: str):
    """Function to save model to a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def get_xgboost_grid(x_train_path: str, y_train_path: str, model_save_path: str):
    """Master function to run the entire process"""    
    # Load the training data
    X_train, y_train = load_data(x_train_path, y_train_path)

    # Define the parameter space for grid search
    xgboost_grid_param_space = get_xgboost_grid_param_space()

    # Perform grid search tuning
    xgboost_grid = perform_grid_search(xgboost_pipeline, xgboost_grid_param_space, X_train, y_train)

    # Save the model
    save_model(xgboost_grid, model_save_path)

    # Print final model
    print(xgboost_grid)

if __name__ == "__main__":
    # Call the master function with the paths to the data and the desired model save path
    get_xgboost_grid('data/X_train.csv', 'data/y_train.csv', 'models/xgboost_grid.pkl')
