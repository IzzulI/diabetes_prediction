if __name__ == "__main__":
    from model_tuning_functions import catboost_pipeline, grid_search_tuning
else:
    from scripts.model_tuning_functions import catboost_pipeline, grid_search_tuning

import pandas as pd
from skopt.space import Integer, Real, Categorical
import pickle
from scipy.stats import uniform, randint


def load_data(x_train_path: str, y_train_path: str):
    """Function to load training data"""
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train, y_train

def get_catboost_grid_param_space():
    """Function to define grid search parameter space for catboost"""
    return {
        'classifier__iterations': [200, 500],  # Keeping both values to test different iterations
        'classifier__learning_rate': [0.01, 0.1],  # Keeping both values for learning rate
        'classifier__depth': [4, 6],  # Keeping both values for depth
        'classifier__l2_leaf_reg': [1, 5],  # Keeping both values for L2 regularization
        'classifier__border_count': [32],  # Reduced to one value
        'classifier__scale_pos_weight': [1, 5]  # Keeping both values for class imbalance
    }

def perform_grid_search(catboost_pipeline, param_space, X_train, y_train):
    """Function to perform grid search for model tuning"""
    return grid_search_tuning(catboost_pipeline, param_space, X_train, y_train)

def save_model(model, file_path: str):
    """Function to save model to a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def get_catboost_grid(x_train_path: str, y_train_path: str, model_save_path: str):
    """Master function to run the entire process"""    
    # Load the training data
    X_train, y_train = load_data(x_train_path, y_train_path)

    # Define the parameter space for grid search
    catboost_grid_param_space = get_catboost_grid_param_space()

    # Perform grid search tuning
    catboost_grid = perform_grid_search(catboost_pipeline, catboost_grid_param_space, X_train, y_train)

    # Save the model
    save_model(catboost_grid, model_save_path)

    # Print final model
    print(catboost_grid)

if __name__ == "__main__":
    # Call the master function with the paths to the data and the desired model save path
    get_catboost_grid('data/X_train.csv', 'data/y_train.csv', 'models/catboost_grid.pkl')
